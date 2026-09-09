# Gradient Surgery for Multi-Task Learning

 Tianhe Yu Email: [tianheyu@cs.stanford.edu](mailto:)    Saurabh Kumar    Abhishek Gupta    Sergey Levine    Karol Hausman    Chelsea Finn    Stanford University    UC Berkeley    Robotics at Google 

###### Abstract

While deep learning and deep reinforcement learning (RL) systems have demonstrated impressive results in domains such as image classification, game playing, and robotic control, data efficiency remains a major challenge. Multi-task learning has emerged as a promising approach for sharing structure across multiple tasks to enable more efficient learning. However, the multi-task setting presents a number of optimization challenges, making it difficult to realize large efficiency gains compared to learning tasks independently. The reasons why multi-task learning is so challenging compared to single-task learning are not fully understood. In this work, we identify a set of three conditions of the multi-task optimization landscape that cause detrimental gradient interference, and develop a simple yet general approach for avoiding such interference between task gradients. We propose a form of gradient surgery that projects a task’s gradient onto the normal plane of the gradient of any other task that has a *conflicting* gradient. On a series of challenging multi-task supervised and multi-task RL problems, this approach leads to substantial gains in efficiency and performance. Further, it is model-agnostic and can be combined with previously-proposed multi-task architectures for enhanced performance.

## 1 Introduction

While deep learning and deep reinforcement learning (RL) have shown considerable promise in enabling systems to learn complex tasks, the data requirements of current methods make it difficult to learn a breadth of capabilities, particularly when all tasks are learned individually from scratch. A natural approach to such multi-task learning problems is to train a network on all tasks jointly, with the aim of discovering shared structure across the tasks in a way that achieves greater efficiency and performance than solving tasks individually. However, learning multiple tasks all at once results is a difficult optimization problem, sometimes leading to *worse* overall performance and data efficiency compared to learning tasks individually \[[42](#bib.bib42 ""), [50](#bib.bib50 "")\]. These optimization challenges are so prevalent that multiple multi-task RL algorithms have considered using independent training as a subroutine of the algorithm before distilling the independent models into a multi-tasking model \[[32](#bib.bib32 ""), [42](#bib.bib42 ""), [50](#bib.bib50 ""), [21](#bib.bib21 ""), [56](#bib.bib56 "")\], producing a multi-task model but losing out on the efficiency gains over independent training. If we could tackle the optimization challenges of multi-task learning effectively, we may be able to actually realize the hypothesized benefits of multi-task learning without the cost in final performance.

![Refer to caption](2001.06782v4/figures/pcgrad_example.png)

Figure 1: Visualization of PCGrad on a 2D multi-task optimization problem. (a) A multi-task objective landscape. (b) & (c) Contour plots of the individual task objectives that comprise (a). (d) Trajectory of gradient updates on the multi-task objective using the Adam optimizer. The gradient vectors of the two tasks at the end of the trajectory are indicated by blue and red arrows, where the relative lengths are on a log scale.(e) Trajectory of gradient updates on the multi-task objective using Adam with PCGrad. For (d) and (e), the optimization trajectory goes from black to yellow.

While there has been a significant amount of research in multi-task learning \[[6](#bib.bib6 ""), [49](#bib.bib49 "")\], the optimization challenges are not well understood. Prior work has described varying learning speeds of different tasks \[[8](#bib.bib8 ""), [26](#bib.bib26 "")\] and plateaus in the optimization landscape \[[52](#bib.bib52 "")\] as potential causes, whereas a range of other works have focused on the model architecture \[[40](#bib.bib40 ""), [33](#bib.bib33 "")\]. In this work, we instead hypothesize that one of the main optimization issues in multi-task learning arises from gradients from different tasks conflicting with one another in a way that is detrimental to making progress. We define two gradients to be conflicting if they point away from one another, i.e., have a negative cosine similarity. We hypothesize that such conflict is detrimental when a) conflicting gradients coincide with b) high positive curvature and c) a large difference in gradient magnitudes.

As an illustrative example, consider the 2D optimization landscapes of two task objectives in Figure [1](#S1.F1 "Figure 1 ‣ 1 Introduction ‣ Gradient Surgery for Multi-Task Learning")a-c. The optimization landscape of each task consists of a deep valley, a property that has been observed in neural network optimization landscapes \[[22](#bib.bib22 "")\], and the bottom of each valley is characterized by high positive curvature and large differences in the task gradient magnitudes. Under such circumstances, the multi-task gradient is dominated by one task gradient, which comes at the cost of degrading the performance of the other task. Further, due to high curvature, the improvement in the dominating task may be overestimated, while the degradation in performance of the non-dominating task may be underestimated. As a result, the optimizer struggles to make progress on the optimization objective. In Figure [1](#S1.F1 "Figure 1 ‣ 1 Introduction ‣ Gradient Surgery for Multi-Task Learning")d), the optimizer reaches the deep valley of task 1, but is unable to traverse the valley in a parameter setting where there are *conflicting gradients*, *high curvature*, and *a large difference in gradient magnitudes* (see gradients plotted in Fig. [1](#S1.F1 "Figure 1 ‣ 1 Introduction ‣ Gradient Surgery for Multi-Task Learning")d). In Section [5.3](#S5.SS3 "5.3 Empirical Analysis of the Tragic Triad ‣ 5 Experiments ‣ Gradient Surgery for Multi-Task Learning"), we find experimentally that this *tragic triad* also occurs in a higher-dimensional neural network multi-task learning problem.

The core contribution of this work is a method for mitigating gradient interference by altering the gradients directly, i.e. by performing “gradient surgery.” If two gradients are conflicting, we alter the gradients by projecting each onto the normal plane of the other, preventing the interfering components of the gradient from being applied to the network. We refer to this particular form of gradient surgery as *projecting conflicting gradients* (PCGrad). PCGrad is model-agnostic, requiring only a single modification to the application of gradients. Hence, it is easy to apply to a range of problem settings, including multi-task supervised learning and multi-task reinforcement learning, and can also be readily combined with other multi-task learning approaches, such as those that modify the architecture. We theoretically prove the local conditions under which PCGrad improves upon standard multi-task gradient descent, and we empirically evaluate PCGrad on a variety of challenging problems, including multi-task CIFAR classification, multi-objective scene understanding, a challenging multi-task RL domain, and goal-conditioned RL. Across the board, we find PCGrad leads to substantial improvements in terms of data efficiency, optimization speed, and final performance compared to prior approaches, including a more than 30% absolute improvement in multi-task reinforcement learning problems. Further, on multi-task supervised learning tasks, PCGrad can be successfully combined with prior state-of-the-art methods for multi-task learning for even greater performance.

## 2 Multi-Task Learning with PCGrad

While the multi-task problem can in principle be solved by simply applying a standard single-task algorithm with a suitable task identifier provided to the model, or a simple multi-head or multi-output model, a number of prior works \[[42](#bib.bib42 ""), [50](#bib.bib50 ""), [53](#bib.bib53 "")\] have found this learning problem to be difficult. In this section, we introduce notation, identify possible causes for the difficulty of multi-task optimization, propose a simple and general approach to mitigate it, and theoretically analyze the proposed approach.

### 2.1 Preliminaries: Problem and Notation

The goal of multi-task learning is to find parameters θ\\theta of a model fθf\_{\\theta} that achieve high average performance across all the training tasks drawn from a distribution of tasks p⁡(𝒯)p(\\mathcal{T}). More formally, we aim to solve the problem: minθ⁡𝔼𝒯i∼p⁡(𝒯)​\[ℒi​(θ)\]\\min\\limits\_{\\theta}\\mathbb{E}\_{\\mathcal{T}\_{i}\\sim p(\\mathcal{T})}\\left\[\\mathcal{L}\_{i}(\\theta)\\right\], where ℒi\\mathcal{L}\_{i} is a loss function for the ii-th task 𝒯i\\mathcal{T}\_{i} that we want to minimize. For a set of tasks, {𝒯i}\\{\\mathcal{T}\_{i}\\}, we denote the multi-task loss as ℒ⁡(θ)\=∑iℒi​(θ)\\mathcal{L}(\\theta)=\\sum\_{i}\\mathcal{L}\_{i}(\\theta), and the gradients of each task as 𝐠i\=∇ℒi​(θ)\\mathbf{g}\_{i}=\\nabla\\mathcal{L}\_{i}(\\theta) for a particular θ\\theta. (We drop the reliance on θ\\theta in the notation for brevity.) To obtain a model that solves a specific task from the task distribution p⁡(𝒯)p(\\mathcal{T}), we define a task-conditioned model fθ​(y|x,zi)f\_{\\theta}(y|x,z\_{i}), with input xx, output yy, and encoding ziz\_{i} for task 𝒯i\\mathcal{T}\_{i}, which could be provided as a one-hot vector or in any other form.

### 2.2 ​​​The Tragic Triad: Conflicting Gradients, Dominating Gradients, High Curvature

We hypothesize that a key optimization issue in multi-task learning arises from conflicting gradients, where gradients for different tasks point away from one another as measured by a negative inner product. However, conflicting gradients are not detrimental on their own. Indeed, simply averaging task gradients should provide the correct solution to descend the multi-task objective. However, there are conditions under which such conflicting gradients lead to significantly degraded performance. Consider a two-task optimization problem. If the gradient of one task is much larger in magnitude than the other, it will dominate the average gradient. If there is also high positive curvature along the directions of the task gradients, then the improvement in performance from the dominating task may be significantly overestimated, while the degradation in performance from the dominated task may be significantly underestimated. Hence, we can characterize the co-occurrence of three conditions as follows: (a) when gradients from multiple tasks are in conflict with one another (b) when the difference in gradient magnitudes is large, leading to some task gradients dominating others, and (c) when there is high curvature in the multi-task optimization landscape. We formally define the three conditions below.

###### Definition 1.

We define ϕi​j\\phi\_{ij} as the angle between two task gradients 𝐠i\\mathbf{g}\_{i} and 𝐠j\\mathbf{g}\_{j}. We define the gradients as conflicting when cos⁡ϕi​j<0\\cos\\phi\_{ij}<0.

###### Definition 2.

We define the gradient magnitude similarity between two gradients 𝐠i\\mathbf{g}\_{i} and 𝐠j\\mathbf{g}\_{j} as Φ⁡(𝐠i,𝐠j)\=2​‖𝐠i‖2​‖𝐠j‖2‖𝐠i‖22+‖𝐠j‖22.\\Phi(\\mathbf{g}\_{i},\\mathbf{g}\_{j})=\\frac{2\\|\\mathbf{g}\_{i}\\|\_{2}\\|\\mathbf{g}\_{j}\\|\_{2}}{\\|\\mathbf{g}\_{i}\\|\_{2}^{2}+\\|\\mathbf{g}\_{j}\\|\_{2}^{2}}.

When the magnitude of two gradients is the same, this value is equal to 1. As the gradient magnitudes become increasingly different, this value goes to zero.

###### Definition 3.

We define multi-task curvature as 𝐇(ℒ;θ,θ′)\=∫01∇ℒ(θ)T∇2ℒ(θ+a(θ′−θ))∇ℒ(θ)da,\\mathbf{H}(\\mathcal{L};\\theta,\\theta^{\\prime})=\\int\_{0}^{1}\\nabla\\mathcal{L}(\\theta)^{T}\\nabla^{2}\\mathcal{L}(\\theta+a(\\theta^{\\prime}-\\theta))\\nabla\\mathcal{L}(\\theta)da, which is the averaged curvature of ℒ\\mathcal{L} between θ\\theta and θ′\\theta^{\\prime} in the direction of the multi-task gradient ∇ℒ​(θ)\\nabla\\mathcal{L}(\\theta).

When 𝐇⁡(ℒ,θ,θ′)\>C\\mathbf{H}(\\mathcal{L};\\theta,\\theta^{\\prime})>C for some large positive constant CC, for model parameters θ\\theta and θ′\\theta^{\\prime} at the current and next iteration, we characterize the optimization landscape as having high curvature.

We aim to study the tragic triad and observe the presence of the three conditions through two examples. First, consider the two-dimensional optimization landscape illustrated in Fig. [1](#S1.F1 "Figure 1 ‣ 1 Introduction ‣ Gradient Surgery for Multi-Task Learning")a, where the landscape for each task objective corresponds to a deep and curved valley with large curvatures (Fig. [1](#S1.F1 "Figure 1 ‣ 1 Introduction ‣ Gradient Surgery for Multi-Task Learning")b and [1](#S1.F1 "Figure 1 ‣ 1 Introduction ‣ Gradient Surgery for Multi-Task Learning")c). The optima of this multi-task objective correspond to where the two valleys meet. More details on the optimization landscape are in Appendix [D](#A4 "Appendix D 2D Optimization Landscape Details ‣ Gradient Surgery for Multi-Task Learning"). Particular points of this optimization landscape exhibit the three described conditions, and we observe that, the Adam \[[30](#bib.bib30 "")\] optimizer stalls precisely at one of these points (see Fig. [1](#S1.F1 "Figure 1 ‣ 1 Introduction ‣ Gradient Surgery for Multi-Task Learning")d), preventing it from reaching an optimum. This provides some empirical evidence for our hypothesis. Our experiments in Section [5.3](#S5.SS3 "5.3 Empirical Analysis of the Tragic Triad ‣ 5 Experiments ‣ Gradient Surgery for Multi-Task Learning") further suggest that this phenomenon occurs in multi-task learning with deep networks. Motivated by these observations, we develop an algorithm that aims to alleviate the optimization challenges caused by conflicting gradients, dominating gradients, and high curvature, which we describe next.

### 2.3 ​​​PCGrad: Project Conflicting Gradients

Our goal is to break one condition of the tragic triad by directly altering the gradients themselves to prevent conflict. In this section, we outline our approach for altering the gradients. In the next section, we will theoretically show that de-conflicting gradients can benefit multi-task learning when dominating gradients and high curvatures are present.

To be maximally effective and widely applicable, we aim to alter the gradients in a way that allows for positive interactions between the task gradients and does not introduce assumptions on the form of the model. Hence, when gradients do not conflict, we do not change the gradients. When gradients do conflict, the goal of PCGrad is to modify the gradients for each task so as to minimize negative conflict with other task gradients, which will in turn mitigate under- and over-estimation problems arising from high curvature.

To deconflict gradients during optimization, PCGrad adopts a simple procedure: if the gradients between two tasks are in conflict, i.e. their cosine similarity is negative, we project the gradient of each task onto the normal plane of the gradient of the other task. This amounts to removing the conflicting component of the gradient for the task, thereby reducing the amount of destructive gradient interference between tasks. A pictorial description of this idea is shown in Fig. [2](#S2.F2 "Figure 2 ‣ 2.3 ​​​PCGrad: Project Conflicting Gradients ‣ 2 Multi-Task Learning with PCGrad ‣ Gradient Surgery for Multi-Task Learning"). Algorithm 1 PCGrad Update Rule 0:  Model parameters θ\\theta, task minibatch ℬ\={𝒯k}\\mathcal{B}=\\{\\mathcal{T}\_{k}\\} 1:  𝐠k←∇θℒk​(θ)​∀k\\mathbf{g}\_{k}\\leftarrow\\nabla\_{\\theta}\\mathcal{L}\_{k}(\\theta)~~\\forall k 2:  𝐠kPC←𝐠k​∀k\\mathbf{g}\_{k}^{\\text{PC}}\\leftarrow\\mathbf{g}\_{k}~~\\forall k 3:  for 𝒯i∈ℬ\\mathcal{T}\_{i}\\in\\mathcal{B} do 4:   for 𝒯j∼uniformlyℬ∖𝒯i\\mathcal{T}\_{j}\\stackrel{{\\scriptstyle\\text{\\scriptsize uniformly}}}{{\\sim}}\\mathcal{B}\\setminus\\mathcal{T}\_{i} in random order do 5:     if 𝐠iPC⋅𝐠j<0\\mathbf{g}\_{i}^{\\text{PC}}\\cdot\\mathbf{g}\_{j}<0 then 6:      // Subtract the projection of 𝐠iPC\\mathbf{g}\_{i}^{\\text{PC}} onto 𝐠j\\mathbf{g}\_{j} 7:      Set 𝐠iPC\=𝐠iPC−𝐠iPC⋅𝐠j‖𝐠j‖2​𝐠j\\mathbf{g}\_{i}^{\\text{PC}}=\\mathbf{g}\_{i}^{\\text{PC}}-\\frac{\\mathbf{g}\_{i}^{\\text{PC}}\\cdot\\mathbf{g}\_{j}}{\\|\\mathbf{g}\_{j}\\|^{2}}\\mathbf{g}\_{j} 8:  return update Δ​θ\=𝐠PC\=∑i𝐠iPC\\Delta\\theta=\\mathbf{g}^{\\text{PC}}=\\sum\_{i}\\mathbf{g}\_{i}^{\\text{PC}} ![Refer to caption](2001.06782v4/figures/grads.png)  
 Figure 2: Conflicting gradients and PCGrad. In (a), tasks ii and jj have conflicting gradient directions, which can lead to destructive interference. In (b) and (c), we illustrate the PCGrad algorithm in the case where gradients are conflicting. PCGrad projects task ii’s gradient onto the normal vector of task jj’s gradient, and vice versa. Non-conflicting task gradients (d) are not altered under PCGrad, allowing for constructive interaction. Suppose the gradient for task 𝒯i\\mathcal{T}\_{i} is 𝐠i\\mathbf{g}\_{i}, and the gradient for task 𝒯j\\mathcal{T}\_{j} is 𝐠j\\mathbf{g}\_{j}. PCGrad proceeds as follows: (1) First, it determines whether 𝐠i\\mathbf{g}\_{i} conflicts with 𝐠j\\mathbf{g}\_{j} by computing the cosine similarity between vectors 𝐠i\\mathbf{g}\_{i} and 𝐠j\\mathbf{g}\_{j}, where negative values indicate conflicting gradients. (2) If the cosine similarity is negative, we replace 𝐠i\\mathbf{g}\_{i} by its projection onto the normal plane of 𝐠j\\mathbf{g}\_{j}: 𝐠i\=𝐠i−𝐠i⋅𝐠j‖𝐠j‖2​𝐠j\\mathbf{g}\_{i}=\\mathbf{g}\_{i}-\\frac{\\mathbf{g}\_{i}\\cdot\\mathbf{g}\_{j}}{\\|\\mathbf{g}\_{j}\\|^{2}}\\mathbf{g}\_{j}. If the gradients are not in conflict, i.e. cosine similarity is non-negative, the original gradient 𝐠i\\mathbf{g}\_{i} remains unaltered. (3) PCGrad repeats this process across all of the other tasks sampled in random order from the current batch 𝒯j​∀j≠i\\mathcal{T}\_{j}~\\forall~j\\neq i, resulting in the gradient 𝐠iPC\\mathbf{g}\_{i}^{\\text{PC}} that is applied for task 𝒯i\\mathcal{T}\_{i}. We perform the same procedure for all tasks in the batch to obtain their respective gradients. The full update procedure is described in Algorithm [1](#alg1 "Algorithm 1 ‣ 2.3 ​​​PCGrad: Project Conflicting Gradients ‣ 2 Multi-Task Learning with PCGrad ‣ Gradient Surgery for Multi-Task Learning") and a discussion on using a random task order is included in Appendix [H](#A8 "Appendix H Ablation study on the task order ‣ Gradient Surgery for Multi-Task Learning").

This procedure, while simple to implement, ensures that the gradients that we apply for each task per batch interfere minimally with the other tasks in the batch, mitigating the conflicting gradient problem, producing a variant on standard first-order gradient descent in the multi-objective setting. In practice, PCGrad can be combined with any gradient-based optimizer, including commonly used methods such as SGD with momentum and Adam \[[30](#bib.bib30 "")\], by simply passing the computed update to the respective optimizer instead of the original gradient. Our experimental results verify the hypothesis that this procedure reduces the problem of conflicting gradients, and find that, as a result, learning progress is substantially improved.

### 2.4 Theoretical Analysis of PCGrad

In this section, we theoretically analyze the performance of PCGrad with two tasks:

###### Definition 4.

Consider two task loss functions ℒ1:ℝn→ℝ\\mathcal{L}\_{1}:\\mathbb{R}^{n}\\rightarrow\\mathbb{R} and ℒ2:ℝn→ℝ\\mathcal{L}\_{2}:\\mathbb{R}^{n}\\rightarrow\\mathbb{R}. We define the two-task learning objective as ℒ⁡(θ)\=ℒ1​(θ)+ℒ2​(θ)\\mathcal{L}(\\theta)=\\mathcal{L}\_{1}(\\theta)+\\mathcal{L}\_{2}(\\theta) for all θ∈ℝn\\theta\\in\\mathbb{R}^{n}, where 𝐠1\=∇ℒ1​(θ){\\mathbf{g}\_{1}}=\\nabla\\mathcal{L}\_{1}(\\theta), 𝐠2\=∇ℒ2​(θ){\\mathbf{g}\_{2}}=\\nabla\\mathcal{L}\_{2}(\\theta), and 𝐠\=𝐠1+𝐠2{\\mathbf{g}}={\\mathbf{g}\_{1}}+{\\mathbf{g}\_{2}}.

We first aim to verify that the PCGrad update corresponds to a sensible optimization procedure under simplifying assumptions. We analyze convergence of PCGrad in the convex setting, under standard assumptions in Theorem [1](#Thmtheorem1 "Theorem 1. ‣ 2.4 Theoretical Analysis of PCGrad ‣ 2 Multi-Task Learning with PCGrad ‣ Gradient Surgery for Multi-Task Learning"). For additional analysis on convergence, including the non-convex setting, with more than two tasks, and with momentum-based optimizers, see Appendices [A.1](#A1.SS1 "A.1 Proof of Theorem 1 ‣ Appendix A Proofs ‣ Gradient Surgery for Multi-Task Learning") and [A.4](#A1.SS4 "A.4 Convergence of PCGrad with Momentum-Based Gradient Descent ‣ Appendix A Proofs ‣ Gradient Surgery for Multi-Task Learning")

###### Theorem 1.

Assume ℒ1\\mathcal{L}\_{1} and ℒ2\\mathcal{L}\_{2} are convex and differentiable. Suppose the gradient of ℒ\\mathcal{L} is LL-Lipschitz with L\>0L>0. Then, the PCGrad update rule with step size t≤1Lt\\leq\\frac{1}{L} will converge to either (1) a location in the optimization landscape where cos⁡(ϕ12)\=−1\\cos(\\phi\_{12})=-1 or (2) the optimal value ℒ⁡(θ∗)\\mathcal{L}(\\theta^{\*}).

###### Proof.

See Appendix [A.1](#A1.SS1 "A.1 Proof of Theorem 1 ‣ Appendix A Proofs ‣ Gradient Surgery for Multi-Task Learning"). ∎

Theorem [1](#Thmtheorem1 "Theorem 1. ‣ 2.4 Theoretical Analysis of PCGrad ‣ 2 Multi-Task Learning with PCGrad ‣ Gradient Surgery for Multi-Task Learning") states that application of the PCGrad update in the two-task setting with a convex and Lipschitz multi-task loss function ℒ\\mathcal{L} leads to convergence to either the minimizer of ℒ\\mathcal{L} or a potentially sub-optimal objective value. A sub-optimal solution occurs when the cosine similarity between the gradients of the two tasks is exactly −1-1, i.e. the gradients directly conflict, leading to zero gradient after applying PCGrad. However, in practice, since we are using SGD, which is a noisy estimate of the true batch gradients, the cosine similarity between the gradients of two tasks in a minibatch is unlikely to be −1-1, thus avoiding this scenario. Note that, in theory, convergence may be slow if cos⁡(ϕ12)\\cos(\\phi\_{12}) hovers near −1-1. However, we don’t observe this in practice, as seen in the objective-wise learning curves in Appendix [B](#A2 "Appendix B Empirical Objective-Wise Evaluations of PCGrad ‣ Gradient Surgery for Multi-Task Learning").

Now that we have checked the sensibility of PCGrad, we aim to understand how PCGrad relates to the three conditions in the tragic triad. In particular, we derive sufficient conditions under which PCGrad achieves lower loss after one update. Here, we still analyze the two task setting, but no longer assume convexity of the loss functions.

###### Definition 5.

We define the multi-task curvature bounding measure ξ⁡(𝐠1,𝐠2)\=(1−cos2⁡ϕ12)​‖𝐠1−𝐠2‖22‖𝐠1+𝐠2‖22.\\xi({\\mathbf{g}\_{1}},{\\mathbf{g}\_{2}})=(1-\\cos^{2}\\phi\_{12})\\frac{\\|{\\mathbf{g}\_{1}}-{\\mathbf{g}\_{2}}\\|\_{2}^{2}}{\\|{\\mathbf{g}\_{1}}+{\\mathbf{g}\_{2}}\\|\_{2}^{2}}.

With the above definition, we present our next theorem:

###### Theorem 2.

Suppose ℒ\\mathcal{L} is differentiable and the gradient of ℒ\\mathcal{L} is Lipschitz continuous with constant L\>0L>0. Let θMT\\theta^{\\text{MT}} and θPCGrad\\theta^{\\text{PCGrad}} be the parameters after applying one update to θ\\theta with 𝐠{\\mathbf{g}} and PCGrad-modified gradient 𝐠PC{\\mathbf{g}}^{\\text{PC}} respectively, with step size t\>0t>0. Moreover, assume 𝐇⁡(ℒ,θ,θMT)≥ℓ​‖𝐠‖22\\mathbf{H}(\\mathcal{L};\\theta,\\theta^{\\text{MT}})\\geq\\ell\\|{\\mathbf{g}}\\|\_{2}^{2} for some constant ℓ≤L\\ell\\leq L, i.e. the multi-task curvature is lower-bounded. Then ℒ⁡(θPCGrad)≤ℒ⁡(θMT)\\mathcal{L}(\\theta^{\\text{PCGrad}})\\leq\\mathcal{L}(\\theta^{\\text{MT}}) if (a) cos⁡ϕ12≤−Φ⁡(𝐠1,𝐠2)\\cos\\phi\_{12}\\leq-\\Phi({\\mathbf{g}\_{1}},{\\mathbf{g}\_{2}}), (b) ℓ≥ξ⁡(𝐠1,𝐠2)​L\\ell\\geq\\xi({\\mathbf{g}\_{1}},{\\mathbf{g}\_{2}})L, and (c) t≥2ℓ−ξ⁡(𝐠1,𝐠2)​L.t\\geq\\frac{2}{\\ell-\\xi({\\mathbf{g}\_{1}},{\\mathbf{g}\_{2}})L}.

###### Proof.

See Appendix [A.2](#A1.SS2 "A.2 Proof of Theorem 2 ‣ Appendix A Proofs ‣ Gradient Surgery for Multi-Task Learning"). ∎

Intuitively, Theorem [2](#Thmtheorem2 "Theorem 2. ‣ 2.4 Theoretical Analysis of PCGrad ‣ 2 Multi-Task Learning with PCGrad ‣ Gradient Surgery for Multi-Task Learning") implies that PCGrad achieves lower loss value after a single gradient update compared to standard gradient descent in multi-task learning when (i) the angle between task gradients is not too small, i.e. the two tasks need to conflict sufficiently (condition (a)), (ii) the difference in magnitude needs to be sufficiently large (condition (a)), (iii) the curvature of the multi-task gradient should be large (condition (b)), (iv) and the learning rate should be big enough so that large curvature would lead to overestimation of performance improvement on the dominating task and underestimation of performance degradation on the dominated task (condition (c)). These first three points (i-iii) correspond to exactly the triad of conditions outlined in Section [2.2](#S2.SS2 "2.2 ​​​The Tragic Triad: Conflicting Gradients, Dominating Gradients, High Curvature ‣ 2 Multi-Task Learning with PCGrad ‣ Gradient Surgery for Multi-Task Learning"), while the latter condition (iv) is desirable as we hope to learn quickly. We empirically validate that the first three points, (i-iii), are frequently met in a neural network multi-task learning problem in Figure [4](#S5.F4 "Figure 4 ‣ 5.2 Multi-Task Reinforcement Learning ‣ 5 Experiments ‣ Gradient Surgery for Multi-Task Learning") in Section [5.3](#S5.SS3 "5.3 Empirical Analysis of the Tragic Triad ‣ 5 Experiments ‣ Gradient Surgery for Multi-Task Learning"). For additional analysis, including complete sufficient and necessary conditions for the PCGrad update to outperform the vanilla multi-task gradient, see Appendix [A.3](#A1.SS3 "A.3 PCGrad: Sufficient and Necessary Conditions for Loss Improvement ‣ Appendix A Proofs ‣ Gradient Surgery for Multi-Task Learning").

## 3 PCGrad in Practice

We use PCGrad in supervised learning and reinforcement learning problems with multiple tasks or goals. Here, we discuss the practical application of PCGrad to those settings.

In multi-task supervised learning, each task 𝒯i∼p⁡(𝒯)\\mathcal{T}\_{i}\\sim p(\\mathcal{T}) has a corresponding training dataset 𝒟i\\mathcal{D}\_{i} consisting of labeled training examples, i.e. 𝒟i\={(x,y)n}\\mathcal{D}\_{i}=\\{(x,y)\_{n}\\}. The objective for each task in this supervised setting is then defined as ℒi​(θ)\=𝔼(x,y)∼𝒟i​\[−log⁡fθ​(y|x,zi)\]\\mathcal{L}\_{i}(\\theta)=\\mathbb{E}\_{(x,y)\\sim\\mathcal{D}\_{i}}\\left\[-\\log f\_{\\theta}(y|x,z\_{i})\\right\], where ziz\_{i} is a one-hot encoding of task 𝒯i\\mathcal{T}\_{i}. At each training step, we randomly sample a batch of data points ℬ\\mathcal{B} from the whole dataset ⋃i𝒟i\\bigcup\_{i}\\mathcal{D}\_{i} and then group the sampled data with the same task encoding into small batches denoted as ℬi\\mathcal{B}\_{i} for each 𝒯i\\mathcal{T}\_{i} represented in ℬ\\mathcal{B}. We denote the set of tasks appearing in ℬ\\mathcal{B} as ℬ𝒯\\mathcal{B}\_{\\mathcal{T}}. After sampling, we precompute the gradient of each task in ℬ𝒯\\mathcal{B}\_{\\mathcal{T}} as ∇θℒi​(θ)\=𝔼(x,y)∼ℬi​\[−∇θ​log​fθ​(y|x,zi)\]​.\\nabla\_{\\theta}\\mathcal{L}\_{i}(\\theta)=\\mathbb{E}\_{(x,y)\\sim\\mathcal{B}\_{i}}\\left\[-\\nabla\_{\\theta}\\log f\_{\\theta}(y|x,z\_{i})\\right\]\\text{.} Given the set of precomputed gradients ∇θℒi​(θ)\\nabla\_{\\theta}\\mathcal{L}\_{i}(\\theta), we also precompute the cosine similarity between all pairs of the gradients in the set. Using the pre-computed gradients and their similarities, we can obtain the PCGrad update by following Algorithm [1](#alg1 "Algorithm 1 ‣ 2.3 ​​​PCGrad: Project Conflicting Gradients ‣ 2 Multi-Task Learning with PCGrad ‣ Gradient Surgery for Multi-Task Learning"), without re-computing task gradients nor backpropagating into the network. Since the PCGrad procedure is only modifying the gradients of shared parameters in the optimization step, it is model-agnostic and can be applied to any architecture with shared parameters. We empirically validate PCGrad with multiple architectures in Section [5](#S5 "5 Experiments ‣ Gradient Surgery for Multi-Task Learning").

For multi-task RL and goal-conditioned RL, PCGrad can be readily applied to policy gradient methods by directly updating the computed policy gradient of each task, following Algorithm [1](#alg1 "Algorithm 1 ‣ 2.3 ​​​PCGrad: Project Conflicting Gradients ‣ 2 Multi-Task Learning with PCGrad ‣ Gradient Surgery for Multi-Task Learning"), analogous to the supervised learning setting. For actor-critic algorithms, it is also straightforward to apply PCGrad: we simply replace task gradients for both the actor and the critic by their gradients computed via PCGrad. For more details on the practical implementation for RL, see Appendix [C](#A3 "Appendix C Practical Details of PCGrad on Multi-Task and Goal-Conditioned Reinforcement Learning ‣ Gradient Surgery for Multi-Task Learning").

## 4 Related Work

Algorithms for multi-task learning typically consider how to train a single model that can solve a variety of different tasks \[[6](#bib.bib6 ""), [2](#bib.bib2 ""), [49](#bib.bib49 "")\]. The multi-task formulation has been applied to many different settings, including supervised learning \[[63](#bib.bib63 ""), [35](#bib.bib35 ""), [60](#bib.bib60 ""), [53](#bib.bib53 ""), [62](#bib.bib62 "")\] and reinforcement-learning \[[17](#bib.bib17 ""), [58](#bib.bib58 "")\], as well as many different domains, such as vision \[[3](#bib.bib3 ""), [39](#bib.bib39 ""), [31](#bib.bib31 ""), [33](#bib.bib33 ""), [62](#bib.bib62 "")\], language \[[11](#bib.bib11 ""), [15](#bib.bib15 ""), [38](#bib.bib38 ""), [44](#bib.bib44 "")\] and robotics \[[45](#bib.bib45 ""), [59](#bib.bib59 ""), [25](#bib.bib25 "")\]. While multi-task learning has the promise of accelerating acquisition of large task repertoires, in practice it presents a challenging optimization problem, which has been tackled in several ways in prior work.

A number of architectural solutions have been proposed to the multi-task learning problem based on multiple modules or paths \[[19](#bib.bib19 ""), [14](#bib.bib14 ""), [40](#bib.bib40 ""), [51](#bib.bib51 ""), [46](#bib.bib46 ""), [57](#bib.bib57 ""), [46](#bib.bib46 "")\], or using attention-based architectures \[[33](#bib.bib33 ""), [37](#bib.bib37 "")\]. Our work is agnostic to the model architecture and can be combined with prior architectural approaches in a complementary fashion. A different set of multi-task learning approaches aim to decompose the problem into multiple local problems, often corresponding to each task, that are significantly easier to learn, akin to divide and conquer algorithms \[[32](#bib.bib32 ""), [50](#bib.bib50 ""), [42](#bib.bib42 ""), [56](#bib.bib56 ""), [21](#bib.bib21 ""), [13](#bib.bib13 "")\]. Eventually, the local models are combined into a single, multi-task policy using different distillation techniques (outlined in \[[27](#bib.bib27 ""), [13](#bib.bib13 "")\]). In contrast to these methods, we propose a simple and cogent scheme for multi-task learning that allows us to learn the tasks simultaneously using a single, shared model without the need for network distillation.

Similarly to our work, a number of prior approaches have observed the difficulty of optimization in multi-task learning \[[26](#bib.bib26 ""), [29](#bib.bib29 ""), [52](#bib.bib52 ""), [55](#bib.bib55 "")\]. Our work suggests that the challenge in multi-task learning may be attributed to what we describe as the tragic triad of multi-task learning (i.e., conflicting gradients, high curvature, and large gradient differences), which we address directly by introducing a simple and practical algorithm that deconflicts gradients from different tasks. Prior works combat optimization challenges by rescaling task gradients  \[[53](#bib.bib53 ""), [9](#bib.bib9 "")\]. We alter both the magnitude and direction of the gradient, which we find to be critical for good performance (see Fig. [3](#S5.F3 "Figure 3 ‣ 5.2 Multi-Task Reinforcement Learning ‣ 5 Experiments ‣ Gradient Surgery for Multi-Task Learning")). Prior work has also used the cosine similarity between gradients to define when an auxiliary task might be useful \[[16](#bib.bib16 "")\] or when two tasks are related \[[55](#bib.bib55 "")\]. We similarly use cosine similarity between gradients to determine if the gradients between a pair of tasks are in conflict. Unlike [Du et al. \[16\]](#bib.bib16 ""), we use this measure for effective multi-task learning, instead of ignoring auxiliary objectives. Overall, we empirically compare our approach to a number of these prior approaches \[[53](#bib.bib53 ""), [9](#bib.bib9 ""), [55](#bib.bib55 "")\], and observe superior performance with PCGrad.

Multiple approaches to continual learning have studied how to prevent gradient updates from adversely affecting previously-learned tasks through various forms of gradient projection \[[36](#bib.bib36 ""), [7](#bib.bib7 ""), [18](#bib.bib18 ""), [23](#bib.bib23 "")\]. These methods focus on sequential learning settings, and solve for the gradient projections using quadratic programming \[[36](#bib.bib36 "")\], only project onto the normal plane of the average gradient of past tasks \[[7](#bib.bib7 "")\], or project the current task gradients onto the orthonormal set of previous task gradients \[[18](#bib.bib18 "")\]. In contrast, our work focuses on positive transfer when simultaneously learning multiple tasks, does not require solving a QP, and *iteratively* projects the gradients of each task instead of *averaging* or only projecting the *current* task gradient. Finally, our method is distinct from and solves a different problem than the projected gradient method \[[5](#bib.bib5 "")\], which is an approach for constrained optimization that projects gradients onto the constraint manifold.

## 5 Experiments

The goal of our experiments is to study the following questions: (1) Does PCGrad make the optimization problems easier for various multi-task learning problems including supervised, reinforcement, and goal-conditioned reinforcement learning settings across different task families? (2) Can PCGrad be combined with other multi-task learning approaches to further improve performance? (3) Is the tragic triad of multi-task learning a major factor in making optimization for multi-task learning challenging? To broadly evaluate PCGrad, we consider multi-task supervised learning, multi-task RL, and goal-conditioned RL problems. We include the results on goal-conditioned RL in Appendix [F](#A6 "Appendix F Goal-Conditioned Reinforcement Learning Results ‣ Gradient Surgery for Multi-Task Learning").

During our evaluation, we tune the parameters of the baselines independently, ensuring that all methods were fairly provided with equal model and training capacity. PCGrad inherits the hyperparameters of the respective baseline method in all experiments, and has no additional hyperparameters. For more details on the experimental set-up and model architectures, see Appendix [J](#A10 "Appendix J Experiment Details ‣ Gradient Surgery for Multi-Task Learning"). The code is available online11 1 Code is released at [https://github.com/tianheyu927/PCGrad](https://github.com/tianheyu927/PCGrad "").

### 5.1 Multi-Task Supervised Learning

To answer question (1) in the supervised learning setting and question (2), we perform experiments on five standard multi-task supervised learning datasets: MultiMNIST, CityScapes, CelebA, multi-task CIFAR-100 and NYUv2. We include the results on MultiMNIST and CityScapes in Appendix [E](#A5 "Appendix E Additional Multi-Task Supervised Learning Results ‣ Gradient Surgery for Multi-Task Learning").

#P.

Architecture

Weighting

Segmentation

Depth

Surface Normal

(Higher Better)

(Lower Better)

Angle Distance

Within t∘t^{\\circ}

(Lower Better)

(Higher Better)

mIoU

Pix Acc

Abs Err

Rel Err

Mean

Median

11.25

22.5

30

Equal Weights

14.71

50.23

0.6481

0.2871

33.56

28.58

20.08

40.54

51.97

≈\\approx3

Cross-Stitch‡

Uncert. Weights∗

15.69

52.60

0.6277

0.2702

32.69

27.26

21.63

42.84

54.45

DWA†, T\=2T=2

16.11

53.19

0.5922

0.2611

32.34

26.91

21.81

43.14

54.92

Equal Weights

17.72

55.32

0.5906

0.2577

31.44

25.37

23.17

45.65

57.48

1.77

MTAN†

Uncert. Weights∗

17.67

55.61

0.5927

0.2592

31.25

25.57

22.99

45.83

57.67

DWA†, T\=2T=2

17.15

54.97

0.5956

0.2569

31.60

25.46

22.48

44.86

57.24

1.77

MTAN†\+ PCGrad (ours)

Uncert. Weights∗

20.17

56.65

0.5904

0.2467

30.01

24.83

22.28

46.12

58.77

Table 1: Three-task learning on the NYUv2 dataset: 13-class semantic segmentation, depth estimation, and surface normal prediction results. #P shows the total number of network parameters. We highlight the best performing combination of multi-task architecture and weighting in bold. The top validation scores for each task are annotated with boxes. The symbols indicate prior methods: ∗: \[[28](#bib.bib28 "")\], †: \[[33](#bib.bib33 "")\], ‡: \[[40](#bib.bib40 "")\]. Performance of other methods as reported in [Liu et al. \[33\]](#bib.bib33 ""). 

|                                               | % accuracy |
| --------------------------------------------- | ---------- |
| task specific, 1-fc \[[46](#bib.bib46 "")\]   | 42         |
| task specific, all-fc \[[46](#bib.bib46 "")\] | 49         |
| cross stitch, all-fc \[[40](#bib.bib40 "")\]  | 53         |
| routing, all-fc + WPL \[[47](#bib.bib47 "")\] | 74.7       |
| independent                                   | 67.7       |
| PCGrad (ours)                                 | 71         |
| routing-all-fc + WPL + PCGrad (ours)          | 77.5       |

Table 2: CIFAR-100 multi-task results. When combined with routing networks, PCGrad leads to a large improvement.

For CIFAR-100, we follow [Rosenbaum et al. \[46\]](#bib.bib46 "") to treat 20 coarse labels in the dataset as distinct tasks, creating a dataset with 20 tasks, with 2500 training instances and 500 test instances per task. We combine PCGrad with a powerful multi-task learning architecture, routing networks \[[46](#bib.bib46 ""), [47](#bib.bib47 "")\], by applying PCGrad only to the shared parameters. For the details of this comparison, see Appendix [J.1](#A10.SS1 "J.1 Multi-Task Supervised Learning Experiment Details ‣ Appendix J Experiment Details ‣ Gradient Surgery for Multi-Task Learning"). As shown in Table [2](#S5.T2 "Table 2 ‣ 5.1 Multi-Task Supervised Learning ‣ 5 Experiments ‣ Gradient Surgery for Multi-Task Learning"), applying PCGrad to a single network achieves 71% classification accuracy, which outperforms most of the prior methods such as cross-stitch \[[40](#bib.bib40 "")\] and independent training, suggesting that sharing representations across tasks is conducive for good performance. While routing networks achieve better performance than PCGrad on its own, they are complementary: combining PCGrad with routing networks leads to a 2.8%2.8\\% absolute improvement in test accuracy.

We also aim to use PCGrad to tackle a multi-label classfication problem, which is a commonly used benchmark for multi-task learning. In multi-label classification, given a set of attributes, the model needs to decide whether each attribute describes the input. Hence, it is essentially a binary classification problem for each attribute. We choose the CelebA dataset \[[34](#bib.bib34 "")\], which consists of 200K face images with 40 attributes. Since for each attribution, it is a binary classfication problem and thus we convert it to a 40-way multi-task learning problem following \[[53](#bib.bib53 "")\]. We use the same architecture as in \[[53](#bib.bib53 "")\].

We use the binary classification error averaged across all 4040 tasks to evaluate the performance as in \[[53](#bib.bib53 "")\]. Similar to the MultiMNIST results, we compare PCGrad to [Sener and Koltun \[53\]](#bib.bib53 "") by rerunning the open-sourced code provided in \[[53](#bib.bib53 "")\]. As shown in Table [3](#S5.T3 "Table 3 ‣ 5.1 Multi-Task Supervised Learning ‣ 5 Experiments ‣ Gradient Surgery for Multi-Task Learning"), PCGrad outperforms [Sener and Koltun \[53\]](#bib.bib53 ""), suggesting that PCGrad is effective in multi-label classification and can also improve multi-task supervised learning performance when the number of tasks is high.

|      | average classification error |
| ---- | ---------------------------- |
| 8.95 |                              |
| 8.69 |                              |

Table 3: CelebA results. We show the average classification error across all 40 tasks in CelebA. PCGrad outperforms the prior method [Sener and Koltun \[53\]](#bib.bib53 "") in this dataset.

Finally, we combine PCGrad with another state-of-art multi-task learning algorithm, MTAN \[[33](#bib.bib33 "")\], and evaluate the performance on a more challenging indoor scene dataset, NYUv2, which contains 3 tasks: 13-class semantic segmentation, depth estimation, and surface normal prediction. We compare MTAN with PCGrad to a list of methods mentioned in Appendix [J.1](#A10.SS1 "J.1 Multi-Task Supervised Learning Experiment Details ‣ Appendix J Experiment Details ‣ Gradient Surgery for Multi-Task Learning"), where each method is trained with three different weighting schemes as in \[[33](#bib.bib33 "")\], equal weighting, weight uncertainty \[[28](#bib.bib28 "")\], and DWA \[[33](#bib.bib33 "")\]. We only run MTAN with PCGrad with weight uncertainty as we find weight uncertainty as the most effective scheme for training MTAN. The results comparing Cross-Stitch, MTAN and MTAN + PCGrad are presented in Table [1](#S5.T1 "Table 1 ‣ 5.1 Multi-Task Supervised Learning ‣ 5 Experiments ‣ Gradient Surgery for Multi-Task Learning") while the full comparison can be found in Table [8](#A10.T8 "Table 8 ‣ J.4 Full CityScapes and NYUv2 Results ‣ Appendix J Experiment Details ‣ Gradient Surgery for Multi-Task Learning") in the Appendix [J.4](#A10.SS4 "J.4 Full CityScapes and NYUv2 Results ‣ Appendix J Experiment Details ‣ Gradient Surgery for Multi-Task Learning"). MTAN with PCGrad is able to achieve the best scores in 8 out of the 9 categories where there are 3 categories per task.

Our multi-task supervised learning results indicate that PCGrad can be seamlessly combined with state-of-art multi-task learning architectures and further improve their results on established supervised multi-task learning benchmarks. We include more results of PCGrad combined with more multi-task learning architectures in Appendix [I](#A9 "Appendix I Combining PCGrad with other architectures ‣ Gradient Surgery for Multi-Task Learning").

### 5.2 Multi-Task Reinforcement Learning

To answer question (2) in the RL setting, we first consider the multi-task RL problem and evaluate our algorithm on the recently proposed Meta-World benchmark \[[61](#bib.bib61 "")\]. In particular, we test all methods on the MT10 and MT50 benchmarks in Meta-World, which contain 1010 and 5050 manipulation tasks respectively shown in Figure [10](#A10.F10 "Figure 10 ‣ J.1 Multi-Task Supervised Learning Experiment Details ‣ Appendix J Experiment Details ‣ Gradient Surgery for Multi-Task Learning"). in Appendix [J.2](#A10.SS2 "J.2 Multi-Task Reinforcement Learning Experiment Details ‣ Appendix J Experiment Details ‣ Gradient Surgery for Multi-Task Learning").

![Refer to caption](2001.06782v4/figures/PCGrad_mt10_mt50_mag_dir_gradnorm.png)

Figure 3: For the two plots on the left, we show learning curves on MT10 and MT50 respectively. PCGrad significantly outperforms the other methods in terms of both success rates and data efficiency. In the rightmost plot, we present the ablation study on only using the magnitude and the direction of gradients modified by PCGrad and a comparison to GradNorm \[[8](#bib.bib8 "")\]. PCGrad outperforms both ablations and GradNorm, indicating the importance of modifying both the gradient directions and magnitudes in multi-task learning.

The results are shown in left two plots in Figure [3](#S5.F3 "Figure 3 ‣ 5.2 Multi-Task Reinforcement Learning ‣ 5 Experiments ‣ Gradient Surgery for Multi-Task Learning"). PCGrad combined with SAC learns all tasks with the best data efficiency and successfully solves all of the 1010 tasks in MT10 and about 7070% of the 5050 tasks in MT50. Training a single SAC policy and a multi-head policy is unable to acquire half of the skills in both MT10 and MT50, suggesting that eliminating gradient interference across tasks can significantly boost performance of multi-task RL. Training independent SAC agents is able to eventually solve all tasks in MT10 and 7070% of the tasks in MT50, but requires about 22 millions and 1515 millions more samples than PCGrad with SAC in MT10 and MT50 respectively, implying that applying PCGrad can result in leveraging shared structure among tasks that expedites multi-task learning. As noted by [Yu et al. \[61\]](#bib.bib61 ""), these tasks involve distinct behavior motions, which makes learning all tasks with a single policy challenging as demonstrated by poor baseline performance. The ability to learn these tasks together opens the door for a number of interesting extensions to meta-learning and generalization to novel task families.

![Refer to caption](2001.06782v4/figures/theorem2_analysis.png)

Figure 4: An empirical analysis of the theoretical conditions discussed in Theorem [2](#Thmtheorem2 "Theorem 2. ‣ 2.4 Theoretical Analysis of PCGrad ‣ 2 Multi-Task Learning with PCGrad ‣ Gradient Surgery for Multi-Task Learning"), showing the first 100100 iterations of training on two RL tasks, reach and press button top. Left: The estimated value of the multi-task curvature. We observe high multi-task curvatures exist throughout training, providing evidence for condition (b) in Theorem [2](#Thmtheorem2 "Theorem 2. ‣ 2.4 Theoretical Analysis of PCGrad ‣ 2 Multi-Task Learning with PCGrad ‣ Gradient Surgery for Multi-Task Learning"). Middle: The solid lines show the percentage gradients with positive cosine similarity between two task gradients, while the dotted lines and dashed lines show the percentage of iterations in which condition (a) and the implication of condition (b) (ξ⁡(𝐠1,𝐠2)≤1\\xi({\\mathbf{g}\_{1}},{\\mathbf{g}\_{2}})\\leq 1) in Theorem [2](#Thmtheorem2 "Theorem 2. ‣ 2.4 Theoretical Analysis of PCGrad ‣ 2 Multi-Task Learning with PCGrad ‣ Gradient Surgery for Multi-Task Learning") are held respectively, among iterations when the cosine similarity is negative. Right: The average return of each task achieved by SAC and SAC combined with PCGrad. From the plots in the Middle and on the Right, we can tell that condition (a) holds most of the time for both Adam and Adam combined with PCGrad when they haven’t solved Task 2 and as soon as Adam combined PCGrad starts to learn Task 2, the percentage of condition (a) held starts to decline. This observation suggests that condition (a) is a key factor for PCGrad excelling in multi-task learning. 

Since the PCGrad update affects both the gradient direction and the gradient magnitude, we perform an ablation study that tests two variants of PCGrad: (1) only applying the gradient direction corrected with PCGrad while keeping the gradient magnitude unchanged and (2) only applying the gradient magnitude computed by PCGrad while keeping the gradient direction unchanged. We further run a direction comparison to GradNorm \[[8](#bib.bib8 "")\], which also scales only the magnitudes of the task gradients. As shown in the rightmost plot in Figure [3](#S5.F3 "Figure 3 ‣ 5.2 Multi-Task Reinforcement Learning ‣ 5 Experiments ‣ Gradient Surgery for Multi-Task Learning"), both variants and GradNorm perform worse than PCGrad and the variant where we only vary the gradient magnitude is much worse than PCGrad. This emphasizes the importance of the orientation change, which is particularly notable as multiple prior works only alter gradient magnitudes \[[8](#bib.bib8 ""), [53](#bib.bib53 "")\]. We also notice that the variant of PCGrad where only the gradient magnitudes change achieves comparable performance to GradNorm, which suggests that it is important to modify both the gradient directions and magnitudes to eliminate interference and achieve good multi-task learning results. Finally, to test the importance of keeping positive cosine similarities between tasks for positive transfer, we compare PCGrad to a recently proposed method in \[[55](#bib.bib55 "")\] that regularizes cosine similarities of different task gradients towards 00. PCGrad outperforms [Suteu and Guo \[55\]](#bib.bib55 "") by a large margin. We leave details of the comparison to Appendix [G](#A7 "Appendix G Comparison to CosReg ‣ Gradient Surgery for Multi-Task Learning").

### 5.3 Empirical Analysis of the Tragic Triad

Finally, to answer question (1), we compare the performance of standard multi-task SAC and the multi-task SAC with PCGrad. We evaluate each method on two tasks, reach and press button top, in the Meta-World \[[61](#bib.bib61 "")\] benchmark. As shown in the leftmost plot in Figure [4](#S5.F4 "Figure 4 ‣ 5.2 Multi-Task Reinforcement Learning ‣ 5 Experiments ‣ Gradient Surgery for Multi-Task Learning"), we plot the multi-task curvature, which is computed as 𝐇(ℒ;θt,θt+1)\=⋅\[ℒ(θt+1)−ℒ(θt)−∇θtℒ(θt)T(θt+1−θt)\]\\mathbf{H}(\\mathcal{L};\\theta^{t}\\!,\\!\\theta^{t+1}\\!)\\!=\\!2\\!\\cdot\\!\\left\[\\mathcal{L}(\\theta^{t+1})\\!\\!-\\!\\!\\mathcal{L}(\\theta^{t})\\!\\!-\\!\\!\\nabla\_{\\theta^{t}}\\mathcal{L}(\\theta^{t})^{T}\\!(\\theta^{t+1}\\!\\!-\\!\\theta^{t})\\right\] by Taylor’s Theorem where ℒ\\mathcal{L} is the multi-task loss, and θt\\theta^{t} and θt+1\\theta^{t+1} are the parameters at iteration tt and t+1t+1. During the training process, the multi-task curvature stays positive and is increasing for both Adam and Adam combined PCGrad, suggesting that condition (b) in Theorem [2](#Thmtheorem2 "Theorem 2. ‣ 2.4 Theoretical Analysis of PCGrad ‣ 2 Multi-Task Learning with PCGrad ‣ Gradient Surgery for Multi-Task Learning") that the multi-task curvature is lower bounded by some positive value is widely held empirically. To further analyze conditions in Theorem [2](#Thmtheorem2 "Theorem 2. ‣ 2.4 Theoretical Analysis of PCGrad ‣ 2 Multi-Task Learning with PCGrad ‣ Gradient Surgery for Multi-Task Learning") empirically, we plot the percentage of condition (a) (i.e. conflicting gradients) and the implication of condition (b) (ξ⁡(𝐠1,𝐠2)≤1\\xi({\\mathbf{g}\_{1}},{\\mathbf{g}\_{2}})\\leq 1) in Theorem [2](#Thmtheorem2 "Theorem 2. ‣ 2.4 Theoretical Analysis of PCGrad ‣ 2 Multi-Task Learning with PCGrad ‣ Gradient Surgery for Multi-Task Learning") being held among the total number of iterations where the cosine similarity is negative in the plot in the middle of Figure [4](#S5.F4 "Figure 4 ‣ 5.2 Multi-Task Reinforcement Learning ‣ 5 Experiments ‣ Gradient Surgery for Multi-Task Learning"). Along with the plot on the right in Figure [4](#S5.F4 "Figure 4 ‣ 5.2 Multi-Task Reinforcement Learning ‣ 5 Experiments ‣ Gradient Surgery for Multi-Task Learning"), which presents the average return of the two tasks during training, we can see that while Adam and Adam with PCGrad haven’t received reward signal from Task 2, condition (a) and the implication of condition (b) stay held and as soon as Adam with PCGrad begins to solve Task 2, the percentage of condition (a) and the implication of condition (b) being held start to decrease. Such a pattern suggests that conflicting gradients, high curvatures and dominating gradients indeed produce considerable challenges in optimization before multi-task learner gains any useful learning signal, which also implies that the tragic triad may indeed be the determining factor where PCGrad can lead to better performance gain over standard multi-task learning in practice.

## 6 Conclusion

In this work, we identified a set of conditions that underlies major challenges in multi-task optimization: conflicting gradients, high positive curvature, and large gradient differences. We proposed a simple algorithm (PCGrad) to mitigate these challenges via “gradient surgery.” PCGrad provides a simple solution to mitigating gradient interference, which substantially improves optimization performance. We provide simple didactic examples and subsequently show significant improvement in optimization for a variety of multi-task supervised learning and reinforcement learning problems. We show that, when some optimization challenges of multi-task learning are alleviated by PCGrad, we can obtain hypothesized benefits in efficiency and asymptotic performance of multi-task settings.

While we studied multi-task supervised learning and multi-task reinforcement learning in this work, we suspect the problem of conflicting gradients to be prevalent in a range of other settings and applications, such as meta-learning, continual learning, multi-goal imitation learning \[[10](#bib.bib10 "")\], and multi-task problems in natural language processing applications \[[38](#bib.bib38 "")\]. Due to its simplicity and model-agnostic nature, we expect that applying PCGrad in these domains to be a promising avenue for future investigation. Further, the general idea of gradient surgery may be an important ingredient for alleviating a broader class of optimization challenges in deep learning, such as the challenges in the stability challenges in two-player games \[[48](#bib.bib48 "")\] and multi-agent optimizations \[[41](#bib.bib41 "")\]. We believe this work to be a step towards simple yet general techniques for addressing some of these challenges.

## Broader Impact

#### Applications and Benefits.

Despite recent success, current deep learning and deep RL methods mostly focus on tackling a single specific task from scratch. Prior methods have proposed methods that can perform multiple tasks, but they often yield comparable or even higher data complexity compared to learning each task individually. Our method enables deep learning systems that mitigate inferences between differing tasks and thus achieves data-efficient multi-task learning. Since our method is general and simple to apply to various problems, there are many possible real-world applications, including but not limited to computer vision systems, autonomous driving, and robotics. For computer vision systems, our method can be used to develop algorithms that enable efficient classification, instance and semantics segmentation and object detection at the same time, which could improve performances of computer vision systems by reusing features obtained from each task and lead to a leap in real-world domains such as autonomous driving. For robotics, there are many situations where multi-task learning is needed. For example, surgical robots are required to perform a wide range of tasks such as stitching and removing tumour from the patient’s body. Kitchen robots should be able to complete multiple chores such as cooking and washing dishes at the same time. Hence, our work represents a step towards making multi-task reinforcement learning more applicable to those settings.

#### Risks.

However, there are potential risks that apply to all machine learning and reinforcement learning systems including ours, including but not limited to safety, reward specification in RL which is often difficult to acquire in the real world, bias in supervised learning systems due to the composition of training data, and compute/data-intensive training procedures. For example, safety issues arise when autonomous driving cars fail to generalize to out-of-distribution data, which leads to crashing or even hurting people. Moreover, reward specification in RL is generally inaccessible in the real world, making RL unable to scale to real robots. In supervised learning domains, learned models could inherit the bias that exists in the training dataset. Furthermore, training procedures of ML models are generally compute/data-intensive, which cause inequitable access to these models. Our method is not immune to these risks. Hence, we encourage future research to design more robust and safe multi-task RL algorithms that can prevent unsafe behaviors. It is also important to push research in self-supervised and unsupervised multi-task RL in order to resolve the issue of reward specification. For supervised learning, we recommend researchers to publish their trained multi-task learning models to make access to those models equitable to everyone in field and develop new datasets that can mitigate biases and also be readily used in multi-task learning.

## Acknowledgments and Disclosure of Funding

The authors would like to thank Annie Xie for reviewing an earlier draft of the paper, Eric Mitchell for technical guidance, and Aravind Rajeswaran and Deirdre Quillen for helpful discussions. Tianhe Yu is partially supported by Intel Corporation. Saurabh Kumar is supported by an NSF Graduate Research Fellowship and the Stanford Knight Hennessy Fellowship. Abhishek Gupta is supported by an NSF Graduate Research Fellowship. Chelsea Finn is a CIFAR Fellow in the Learning in Machines and Brains program.

## References

*   \[1\] Vijay Badrinarayanan, Alex Kendall, and Roberto Cipolla. Segnet: A deep convolutional encoder-decoder architecture for image segmentation. *IEEE transactions on pattern analysis and machine intelligence*, 39(12), 2017.
*   \[2\] Bart Bakker and Tom Heskes. Task clustering and gating for bayesian multitask learning. *J. Mach. Learn. Res.*, 2003.
*   \[3\] Hakan Bilen and Andrea Vedaldi. Integrated perception with recurrent multi-task neural networks. In *Advances in Neural Information Processing Systems*, 2016.
*   \[4\] Stephen Boyd and Lieven Vandenberghe. *Convex optimization*. Cambridge university press, 2004.
*   \[5\] Paul H Calamai and Jorge J Moré. Projected gradient methods for linearly constrained problems. *Mathematical programming*, 39(1), 1987.
*   \[6\] Rich Caruana. Multitask learning. *Machine Learning*, 1997.
*   \[7\] Arslan Chaudhry, Marc’Aurelio Ranzato, Marcus Rohrbach, and Mohamed Elhoseiny. Efficient lifelong learning with a-gem. *arXiv:1812.00420*, 2018.
*   \[8\] Zhao Chen, Vijay Badrinarayanan, Chen-Yu Lee, and Andrew Rabinovich. Gradnorm: Gradient normalization for adaptive loss balancing in deep multitask networks. *arXiv:1711.02257*, 2017.
*   \[9\] Zhao Chen, Vijay Badrinarayanan, Chen-Yu Lee, and Andrew Rabinovich. Gradnorm: Gradient normalization for adaptive loss balancing in deep multitask networks. In *International Conference on Machine Learning*, 2018.
*   \[10\] Felipe Codevilla, Matthias Miiller, Antonio López, Vladlen Koltun, and Alexey Dosovitskiy. End-to-end driving via conditional imitation learning. In *International Conference on Robotics and Automation (ICRA)*, 2018.
*   \[11\] Ronan Collobert and Jason Weston. A unified architecture for natural language processing: Deep neural networks with multitask learning. In *International Conference on Machine Learning*, 2008.
*   \[12\] Marius Cordts, Mohamed Omran, Sebastian Ramos, Timo Rehfeld, Markus Enzweiler, Rodrigo Benenson, Uwe Franke, Stefan Roth, and Bernt Schiele. The cityscapes dataset for semantic urban scene understanding. In *Proceedings of the IEEE conference on computer vision and pattern recognition*, pages 3213–3223, 2016.
*   \[13\] Wojciech M. Czarnecki, Razvan Pascanu, Simon Osindero, Siddhant M. Jayakumar, Grzegorz Swirszcz, and Max Jaderberg. Distilling policy distillation. In *International Conference on Artificial Intelligence and Statistics, AISTATS*, 2019.
*   \[14\] Coline Devin, Abhishek Gupta, Trevor Darrell, Pieter Abbeel, and Sergey Levine. Learning modular neural network policies for multi-task and multi-robot transfer. *CoRR*, abs/1609.07088, 2016.
*   \[15\] Daxiang Dong, Hua Wu, Wei He, Dianhai Yu, and Haifeng Wang. Multi-task learning for multiple language translation. In *Annual Meeting of the Association for Computational Linguistics and the 7th International Joint Conference on Natural Language Processing (Volume 1: Long Papers)*, 2015.
*   \[16\] Yunshu Du, Wojciech M. Czarnecki, Siddhant M. Jayakumar, Razvan Pascanu, and Balaji Lakshminarayanan. Adapting auxiliary losses using gradient similarity. *CoRR*, abs/1812.02224, 2018.
*   \[17\] Lasse Espeholt, Hubert Soyer, Rémi Munos, Karen Simonyan, Volodymyr Mnih, Tom Ward, Yotam Doron, Vlad Firoiu, Tim Harley, Iain Dunning, Shane Legg, and Koray Kavukcuoglu. IMPALA: scalable distributed deep-rl with importance weighted actor-learner architectures. In *International Conference on Machine Learning*, 2018.
*   \[18\] Mehrdad Farajtabar, Navid Azizan, Alex Mott, and Ang Li. Orthogonal gradient descent for continual learning. *arXiv preprint arXiv:1910.07104*, 2019.
*   \[19\] Chrisantha Fernando, Dylan Banarse, Charles Blundell, Yori Zwols, David Ha, Andrei A. Rusu, Alexander Pritzel, and Daan Wierstra. Pathnet: Evolution channels gradient descent in super neural networks. *CoRR*, abs/1701.08734, 2017. URL [http://arxiv.org/abs/1701.08734](http://arxiv.org/abs/1701.08734 "").
*   \[20\] William Fulton. Eigenvalues, invariant factors, highest weights, and schubert calculus. *Bulletin of the American Mathematical Society*, 37(3):209–249, 2000.
*   \[21\] Dibya Ghosh, Avi Singh, Aravind Rajeswaran, Vikash Kumar, and Sergey Levine. Divide-and-conquer reinforcement learning. *CoRR*, abs/1711.09874, 2017.
*   \[22\] Ian J Goodfellow, Oriol Vinyals, and Andrew M Saxe. Qualitatively characterizing neural network optimization problems. *arXiv:1412.6544*, 2014.
*   \[23\] Yunhui Guo, Mingrui Liu, Tianbao Yang, and Tajana Rosing. Improved schemes for episodic memory-based lifelong learning, 2020.
*   \[24\] Tuomas Haarnoja, Aurick Zhou, Pieter Abbeel, and Sergey Levine. Soft actor-critic: Off-policy maximum entropy deep reinforcement learning with a stochastic actor. *International Conference on Machine Learning*, 2018.
*   \[25\] Karol Hausman, Jost Tobias Springenberg, Ziyu Wang, Nicolas Heess, and Martin Riedmiller. Learning an embedding space for transferable robot skills. 2018.
*   \[26\] Matteo Hessel, Hubert Soyer, Lasse Espeholt, Wojciech Czarnecki, Simon Schmitt, and Hado van Hasselt. Multi-task deep reinforcement learning with popart. In *Proceedings of the AAAI Conference on Artificial Intelligence*, volume 33, 2019.
*   \[27\] Geoffrey Hinton, Oriol Vinyals, and Jeff Dean. Distilling the knowledge in a neural network. *arXiv:1503.02531*, 2015.
*   \[28\] Alex Kendall, Yarin Gal, and Roberto Cipolla. Multi-task learning using uncertainty to weigh losses for scene geometry and semantics. In *Computer Vision and Pattern Recognition*, 2018a.
*   \[29\] Alex Kendall, Yarin Gal, and Roberto Cipolla. Multi-task learning using uncertainty to weigh losses for scene geometry and semantics. In *CVPR*, 2018b.
*   \[30\] Diederik P Kingma and Jimmy Ba. Adam: A method for stochastic optimization. *arXiv:1412.6980*, 2014.
*   \[31\] Iasonas Kokkinos. Ubernet: Training a universal convolutional neural network for low-, mid-, and high-level vision using diverse datasets and limited memory. In *Computer Vision and Pattern Recognition*, 2017.
*   \[32\] Sergey Levine, Chelsea Finn, Trevor Darrell, and Pieter Abbeel. End-to-end training of deep visuomotor policies. *The Journal of Machine Learning Research*, 17(1), 2016.
*   \[33\] Shikun Liu, Edward Johns, and Andrew J. Davison. End-to-end multi-task learning with attention. *CoRR*, abs/1803.10704, 2018.
*   \[34\] Ziwei Liu, Ping Luo, Xiaogang Wang, and Xiaoou Tang. Deep learning face attributes in the wild. In *Proceedings of the IEEE international conference on computer vision*, pages 3730–3738, 2015.
*   \[35\] Mingsheng Long and Jianmin Wang. Learning multiple tasks with deep relationship networks. *arXiv:1506.02117*, 2, 2015.
*   \[36\] David Lopez-Paz and Marc’Aurelio Ranzato. Gradient episodic memory for continual learning. In *Advances in Neural Information Processing Systems*, 2017.
*   \[37\] Kevis-Kokitsi Maninis, Ilija Radosavovic, and Iasonas Kokkinos. Attentive single-tasking of multiple tasks. *CoRR*, abs/1904.08918, 2019.
*   \[38\] Bryan McCann, Nitish Shirish Keskar, Caiming Xiong, and Richard Socher. The natural language decathlon: Multitask learning as question answering. *arXiv:1806.08730*, 2018.
*   \[39\] Ishan Misra, Abhinav Shrivastava, Abhinav Gupta, and Martial Hebert. Cross-stitch networks for multi-task learning. In *Computer Vision and Pattern Recognition*, 2016a.
*   \[40\] Ishan Misra, Abhinav Shrivastava, Abhinav Gupta, and Martial Hebert. Cross-stitch networks for multi-task learning. In *Conference on Computer Vision and Pattern Recognition, CVPR*, 2016b.
*   \[41\] Angelia Nedic and Asuman Ozdaglar. Distributed subgradient methods for multi-agent optimization. *IEEE Transactions on Automatic Control*, 54(1), 2009.
*   \[42\] Emilio Parisotto, Jimmy Lei Ba, and Ruslan Salakhutdinov. Actor-mimic: Deep multitask and transfer reinforcement learning. *arXiv:1511.06342*, 2015.
*   \[43\] Boris T Polyak. Some methods of speeding up the convergence of iteration methods. *USSR Computational Mathematics and Mathematical Physics*, 4(5):1–17, 1964.
*   \[44\] Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, and Ilya Sutskever. Language models are unsupervised multitask learners. *OpenAI Blog*, 1(8), 2019.
*   \[45\] Martin Riedmiller, Roland Hafner, Thomas Lampe, Michael Neunert, Jonas Degrave, Tom Van de Wiele, Volodymyr Mnih, Nicolas Heess, and Jost Tobias Springenberg. Learning by playing-solving sparse reward tasks from scratch. *arXiv:1802.10567*, 2018.
*   \[46\] Clemens Rosenbaum, Tim Klinger, and Matthew Riemer. Routing networks: Adaptive selection of non-linear functions for multi-task learning. *International Conference on Learning Representations (ICLR)*, 2018.
*   \[47\] Clemens Rosenbaum, Ignacio Cases, Matthew Riemer, and Tim Klinger. Routing networks and the challenges of modular and compositional computation. *arXiv:1904.12774*, 2019.
*   \[48\] Kevin Roth, Aurelien Lucchi, Sebastian Nowozin, and Thomas Hofmann. Stabilizing training of generative adversarial networks through regularization. In *Advances in Neural Information Processing Systems*, 2017.
*   \[49\] Sebastian Ruder. An overview of multi-task learning in deep neural networks. *arXiv:1706.05098*, 2017.
*   \[50\] Andrei A. Rusu, Sergio Gomez Colmenarejo, Çaglar Gülçehre, Guillaume Desjardins, James Kirkpatrick, Razvan Pascanu, Volodymyr Mnih, Koray Kavukcuoglu, and Raia Hadsell. Policy distillation. In *International Conference on Learning Representations, ICLR*, 2016a.
*   \[51\] Andrei A Rusu, Neil C Rabinowitz, Guillaume Desjardins, Hubert Soyer, James Kirkpatrick, Koray Kavukcuoglu, Razvan Pascanu, and Raia Hadsell. Progressive neural networks. *arXiv:1606.04671*, 2016b.
*   \[52\] Tom Schaul, Diana Borsa, Joseph Modayil, and Razvan Pascanu. Ray interference: a source of plateaus in deep reinforcement learning. *arXiv:1904.11455*, 2019.
*   \[53\] Ozan Sener and Vladlen Koltun. Multi-task learning as multi-objective optimization. In *Advances in Neural Information Processing Systems*, 2018.
*   \[54\] Yuekai Sun. Notes on first-order methods for minimizing smooth functions, 2015. [https://web.stanford.edu/class/msande318/notes/notes-first-order-smooth.pdf](https://web.stanford.edu/class/msande318/notes/notes-first-order-smooth.pdf "").
*   \[55\] Mihai Suteu and Yike Guo. Regularizing deep multi-task networks using orthogonal gradients. *arXiv preprint arXiv:1912.06844*, 2019.
*   \[56\] Yee Teh, Victor Bapst, Wojciech M Czarnecki, John Quan, James Kirkpatrick, Raia Hadsell, Nicolas Heess, and Razvan Pascanu. Distral: Robust multitask reinforcement learning. In *Advances in Neural Information Processing Systems*, 2017.
*   \[57\] Simon Vandenhende, Bert De Brabandere, and Luc Van Gool. Branched multi-task networks: Deciding what layers to share. *CoRR*, abs/1904.02920, 2019.
*   \[58\] Aaron Wilson, Alan Fern, Soumya Ray, and Prasad Tadepalli. Multi-task reinforcement learning: a hierarchical bayesian approach. In *International Conference on Machine Learning*, 2007.
*   \[59\] Markus Wulfmeier, Abbas Abdolmaleki, Roland Hafner, Jost Tobias Springenberg, Michael Neunert, Tim Hertweck, Thomas Lampe, Noah Siegel, Nicolas Heess, and Martin Riedmiller. Regularized hierarchical policies for compositional transfer in robotics. *arXiv:1906.11228*, 2019.
*   \[60\] Yongxin Yang and Timothy M Hospedales. Trace norm regularised deep multi-task learning. *arXiv:1606.04038*, 2016.
*   \[61\] Tianhe Yu, Deirdre Quillen, Zhanpeng He, Ryan Julian, Karol Hausman, Chelsea Finn, and Sergey Levine. Meta-world: A benchmark and evaluation for multi-task and meta-reinforcement learning. *arXiv:1910.10897*, 2019.
*   \[62\] Amir R Zamir, Alexander Sax, William Shen, Leonidas J Guibas, Jitendra Malik, and Silvio Savarese. Taskonomy: Disentangling task transfer learning. In *Computer Vision and Pattern Recognition*, 2018.
*   \[63\] Zhanpeng Zhang, Ping Luo, Chen Change Loy, and Xiaoou Tang. Facial landmark detection by deep multi-task learning. In *European conference on computer vision*. Springer, 2014.

Appendix

## Appendix A Proofs

### A.1 Proof of Theorem 1

See [1](#Thmtheorem1 "Theorem 1. ‣ 2.4 Theoretical Analysis of PCGrad ‣ 2 Multi-Task Learning with PCGrad ‣ Gradient Surgery for Multi-Task Learning")

###### Proof.

We will use the shorthand ||⋅||||\\cdot|| to denote the L2L\_{2}-norm and ∇ℒ\=∇θℒ\\nabla\\mathcal{L}=\\nabla\_{\\theta}\\mathcal{L}, where θ\\theta is the parameter vector. Following Definition [1](#Thmdefinition1 "Definition 1. ‣ 2.2 ​​​The Tragic Triad: Conflicting Gradients, Dominating Gradients, High Curvature ‣ 2 Multi-Task Learning with PCGrad ‣ Gradient Surgery for Multi-Task Learning") and [4](#Thmdefinition4 "Definition 4. ‣ 2.4 Theoretical Analysis of PCGrad ‣ 2 Multi-Task Learning with PCGrad ‣ Gradient Surgery for Multi-Task Learning"), let 𝐠𝟏\=∇ℒ1{\\mathbf{g\_{1}}}=\\nabla\\mathcal{L}\_{1}, 𝐠𝟐\=∇ℒ2{\\mathbf{g\_{2}}}=\\nabla\\mathcal{L}\_{2}, 𝐠\=∇ℒ\=𝐠𝟏+𝐠𝟐{\\mathbf{g}}=\\nabla\\mathcal{L}={\\mathbf{g\_{1}}}+{\\mathbf{g\_{2}}}, and ϕ12\\phi\_{12} be the angle between 𝐠𝟏{\\mathbf{g\_{1}}} and 𝐠𝟐{\\mathbf{g\_{2}}}.

At each PCGrad update, we have two cases: c​o​s​(ϕ12)≥0cos(\\phi\_{12})\\geq 0 or cos⁡(ϕ12)<0\\cos(\\phi\_{12})<0.

If cos⁡(ϕ12)≥0\\cos(\\phi\_{12})\\geq 0, then we apply the standard gradient descent update using t≤1Lt\\leq\\frac{1}{L}, which leads to a strict decrease in the objective function value ℒ⁡(θ)\\mathcal{L}(\\theta) (since it is also convex) unless ∇ℒ​(θ)\=0\\nabla\\mathcal{L}(\\theta)=0, which occurs only when θ\=θ∗\\theta=\\theta^{\*} \[[4](#bib.bib4 "")\].

In the case that cos⁡(ϕ12)<0\\cos(\\phi\_{12})<0, we proceed as follows:

Our assumption that ∇ℒ\\nabla\\mathcal{L} is Lipschitz continuous with constant LL implies that ∇2ℒ​(θ)−L​I\\nabla^{2}\\mathcal{L}(\\theta)-LI is a negative semi-definite matrix. Using this fact, we can perform a quadratic expansion of ℒ\\mathcal{L} around ℒ⁡(θ)\\mathcal{L}(\\theta) and obtain the following inequality:

ℒ⁡(θ+)\\displaystyle\\mathcal{L}(\\theta^{+})

≤ℒ(θ)+∇ℒ(θ)T(θ+−θ)+12∇2ℒ(θ)||θ+−θ||2\\displaystyle\\leq\\mathcal{L}(\\theta)+\\nabla\\mathcal{L}(\\theta)^{T}(\\theta^{+}-\\theta)+\\frac{1}{2}\\nabla^{2}\\mathcal{L}(\\theta)||\\theta^{+}-\\theta||^{2}

≤ℒ(θ)+∇ℒ(θ)T(θ+−θ)+12L||θ+−θ||2\\displaystyle\\leq\\mathcal{L}(\\theta)+\\nabla\\mathcal{L}(\\theta)^{T}(\\theta^{+}-\\theta)+\\frac{1}{2}L||\\theta^{+}-\\theta||^{2}

Now, we can plug in the PCGrad update by letting θ+\=θ−t⋅(𝐠−𝐠𝟏⋅𝐠𝟐‖𝐠𝟏‖2​𝐠𝟏−𝐠𝟏⋅𝐠𝟐‖𝐠𝟐‖2​𝐠𝟐)\\theta^{+}=\\theta-t\\cdot({\\mathbf{g}}-\\frac{{\\mathbf{g\_{1}}}\\cdot{\\mathbf{g\_{2}}}}{||{\\mathbf{g\_{1}}}||^{2}}{\\mathbf{g\_{1}}}-\\frac{{\\mathbf{g\_{1}}}\\cdot{\\mathbf{g\_{2}}}}{||{\\mathbf{g\_{2}}}||^{2}}{\\mathbf{g\_{2}}}). We then get:

ℒ⁡(θ+)\\displaystyle\\mathcal{L}(\\theta^{+})

≤ℒ⁡(θ)+t⋅𝐠T​(−𝐠+𝐠𝟏⋅𝐠𝟐‖𝐠𝟏‖2​𝐠𝟏+𝐠𝟏⋅𝐠𝟐‖𝐠𝟐‖2​𝐠𝟐)+12​L​t2​‖𝐠−𝐠𝟏⋅𝐠𝟐‖𝐠𝟏‖2​𝐠𝟏−𝐠𝟏⋅𝐠𝟐‖𝐠𝟐‖2​𝐠𝟐‖2\\displaystyle\\leq\\mathcal{L}(\\theta)+t\\cdot{\\mathbf{g}}^{T}(-{\\mathbf{g}}+\\frac{{\\mathbf{g\_{1}}}\\cdot{\\mathbf{g\_{2}}}}{||{\\mathbf{g\_{1}}}||^{2}}{\\mathbf{g\_{1}}}+\\frac{{\\mathbf{g\_{1}}}\\cdot{\\mathbf{g\_{2}}}}{||{\\mathbf{g\_{2}}}||^{2}}{\\mathbf{g\_{2}}})+\\frac{1}{2}Lt^{2}||{\\mathbf{g}}-\\frac{{\\mathbf{g\_{1}}}\\cdot{\\mathbf{g\_{2}}}}{||{\\mathbf{g\_{1}}}||^{2}}{\\mathbf{g\_{1}}}-\\frac{{\\mathbf{g\_{1}}}\\cdot{\\mathbf{g\_{2}}}}{||{\\mathbf{g\_{2}}}||^{2}}{\\mathbf{g\_{2}}}||^{2}

(Expanding, using the identity 𝐠\=𝐠𝟏+𝐠𝟐{\\mathbf{g}}={\\mathbf{g\_{1}}}+{\\mathbf{g\_{2}}})

\=ℒ(θ)+t(−||𝐠𝟏||2−||𝐠𝟐||2+(𝐠𝟏⋅𝐠𝟐)2‖𝐠𝟏‖2+(𝐠𝟏⋅𝐠𝟐)2‖𝐠𝟐‖2)+12Lt2||𝐠𝟏+𝐠𝟐\\displaystyle=\\mathcal{L}(\\theta)+t\\left(-||{\\mathbf{g\_{1}}}||^{2}-||{\\mathbf{g\_{2}}}||^{2}+\\frac{({\\mathbf{g\_{1}}}\\cdot{\\mathbf{g\_{2}}})^{2}}{||{\\mathbf{g\_{1}}}||^{2}}+\\frac{({\\mathbf{g\_{1}}}\\cdot{\\mathbf{g\_{2}}})^{2}}{||{\\mathbf{g\_{2}}}||^{2}}\\right)+\\frac{1}{2}Lt^{2}||{\\mathbf{g\_{1}}}+{\\mathbf{g\_{2}}}

−𝐠𝟏⋅𝐠𝟐‖𝐠𝟏‖2𝐠𝟏−𝐠𝟏⋅𝐠𝟐‖𝐠𝟐‖2𝐠𝟐||2\\displaystyle-\\frac{{\\mathbf{g\_{1}}}\\cdot{\\mathbf{g\_{2}}}}{||{\\mathbf{g\_{1}}}||^{2}}{\\mathbf{g\_{1}}}-\\frac{{\\mathbf{g\_{1}}}\\cdot{\\mathbf{g\_{2}}}}{||{\\mathbf{g\_{2}}}||^{2}}{\\mathbf{g\_{2}}}||^{2}

(Expanding further and re-arranging terms)

\=ℒ⁡(θ)−(t−12​L​t2)​(‖𝐠𝟏‖2+‖𝐠𝟐‖2−(𝐠𝟏⋅𝐠𝟐)2‖𝐠𝟏‖2−(𝐠𝟏⋅𝐠𝟐)2‖𝐠𝟐‖2)\\displaystyle=\\mathcal{L}(\\theta)-(t-\\frac{1}{2}Lt^{2})(||{\\mathbf{g\_{1}}}||^{2}+||{\\mathbf{g\_{2}}}||^{2}-\\frac{({\\mathbf{g\_{1}}}\\cdot{\\mathbf{g\_{2}}})^{2}}{||{\\mathbf{g\_{1}}}||^{2}}-\\frac{({\\mathbf{g\_{1}}}\\cdot{\\mathbf{g\_{2}}})^{2}}{||{\\mathbf{g\_{2}}}||^{2}})

−L​t2​(𝐠𝟏⋅𝐠𝟐−(𝐠𝟏⋅𝐠𝟐)2‖𝐠𝟏‖2​‖𝐠𝟐‖2​𝐠𝟏⋅𝐠𝟐)\\displaystyle-Lt^{2}({\\mathbf{g\_{1}}}\\cdot{\\mathbf{g\_{2}}}-\\frac{({\\mathbf{g\_{1}}}\\cdot{\\mathbf{g\_{2}}})^{2}}{||{\\mathbf{g\_{1}}}||^{2}||{\\mathbf{g\_{2}}}||^{2}}{\\mathbf{g\_{1}}}\\cdot{\\mathbf{g\_{2}}})

(Using the identity cos⁡(ϕ12)\=𝐠𝟏⋅𝐠𝟐‖𝐠𝟏‖​‖𝐠𝟐‖\\cos(\\phi\_{12})=\\frac{{\\mathbf{g\_{1}}}\\cdot{\\mathbf{g\_{2}}}}{||{\\mathbf{g\_{1}}}||||{\\mathbf{g\_{2}}}||})

\=ℒ⁡(θ)−(t−12​L​t2)​\[(1−cos2⁡(ϕ12))​‖𝐠𝟏‖2+(1−cos2⁡(ϕ12))​‖𝐠𝟐‖2\]\\displaystyle=\\mathcal{L}(\\theta)-(t-\\frac{1}{2}Lt^{2})\[(1-\\cos^{2}(\\phi\_{12}))||{\\mathbf{g\_{1}}}||^{2}+(1-\\cos^{2}(\\phi\_{12}))||{\\mathbf{g\_{2}}}||^{2}\]

−L​t2​(1−cos2⁡(ϕ12))​‖𝐠𝟏‖​‖𝐠𝟐‖​cos⁡(ϕ12)\\displaystyle-Lt^{2}(1-\\cos^{2}(\\phi\_{12}))||{\\mathbf{g\_{1}}}||||{\\mathbf{g\_{2}}}||\\cos(\\phi\_{12})

(1)

(Note that cos⁡(ϕ12)<0\\cos(\\phi\_{12})<0 so the final term is non-negative)

Using t≤1Lt\\leq\\frac{1}{L}, we know that −(1−12​L​t)\=12​L​t−1≤12​L​(1/L)−1\=−12-(1-\\frac{1}{2}Lt)=\\frac{1}{2}Lt-1\\leq\\frac{1}{2}L(1/L)-1=\\frac{-1}{2} and L​t2≤tLt^{2}\\leq t.

Plugging this into the last expression above, we can conclude the following:

ℒ⁡(θ+)\\displaystyle\\mathcal{L}(\\theta^{+})

≤ℒ⁡(θ)−12​t​\[(1−cos2⁡(ϕ12))​‖𝐠𝟏‖2+(1−cos2⁡(ϕ12))​‖𝐠𝟐‖2\]\\displaystyle\\leq\\mathcal{L}(\\theta)-\\frac{1}{2}t\[(1-\\cos^{2}(\\phi\_{12}))||{\\mathbf{g\_{1}}}||^{2}+(1-\\cos^{2}(\\phi\_{12}))||{\\mathbf{g\_{2}}}||^{2}\]

−t⁡(1−cos2⁡(ϕ12))​‖𝐠𝟏‖​‖𝐠𝟐‖​cos⁡(ϕ12)\\displaystyle-t(1-\\cos^{2}(\\phi\_{12}))||{\\mathbf{g\_{1}}}||||{\\mathbf{g\_{2}}}||\\cos(\\phi\_{12})

\=ℒ⁡(θ)−12​t​(1−cos2⁡(ϕ12))​\[‖𝐠𝟏‖2+2​‖𝐠𝟏‖​‖𝐠𝟐‖​cos⁡(ϕ12)+‖𝐠𝟐‖2\]\\displaystyle=\\mathcal{L}(\\theta)-\\frac{1}{2}t(1-\\cos^{2}(\\phi\_{12}))\[||{\\mathbf{g\_{1}}}||^{2}+2||{\\mathbf{g\_{1}}}||||{\\mathbf{g\_{2}}}||\\cos(\\phi\_{12})+||{\\mathbf{g\_{2}}}||^{2}\]

\=ℒ⁡(θ)−12​t​(1−cos2⁡(ϕ12))​\[‖𝐠𝟏‖2+2​𝐠𝟏⋅𝐠𝟐+‖𝐠𝟐‖2\]\\displaystyle=\\mathcal{L}(\\theta)-\\frac{1}{2}t(1-\\cos^{2}(\\phi\_{12}))\[||{\\mathbf{g\_{1}}}||^{2}+2{\\mathbf{g\_{1}}}\\cdot{\\mathbf{g\_{2}}}+||{\\mathbf{g\_{2}}}||^{2}\]

\=ℒ⁡(θ)−12​t​(1−cos2⁡(ϕ12))​‖𝐠𝟏+𝐠𝟐‖2\\displaystyle=\\mathcal{L}(\\theta)-\\frac{1}{2}t(1-\\cos^{2}(\\phi\_{12}))||{\\mathbf{g\_{1}}}+{\\mathbf{g\_{2}}}||^{2}

\=ℒ⁡(θ)−12​t​(1−cos2⁡(ϕ12))​‖𝐠‖2\\displaystyle=\\mathcal{L}(\\theta)-\\frac{1}{2}t(1-\\cos^{2}(\\phi\_{12}))\\|{\\mathbf{g}}\\|^{2}

If cos⁡(ϕ12)\>−1\\cos(\\phi\_{12})>-1, then 12​t​(1−cos2⁡(ϕ12))​‖𝐠‖2\\frac{1}{2}t(1-\\cos^{2}(\\phi\_{12}))\\|{\\mathbf{g}}\\|^{2} will always be positive unless 𝐠\=0{\\mathbf{g}}=0. This inequality implies that the objective function value strictly decreases with each iteration where cos⁡(ϕ12)\>−1\\cos(\\phi\_{12})>-1.

Hence repeatedly applying PCGrad process can either reach the optimal value ℒ⁡(θ)\=ℒ⁡(θ∗)\\mathcal{L}(\\theta)=\\mathcal{L}(\\theta^{\*}) or cos⁡(ϕ12)\=−1\\cos(\\phi\_{12})=-1, in which case 12​t​(1−cos2⁡(ϕ12))​‖𝐠‖2\=0\\frac{1}{2}t(1-\\cos^{2}(\\phi\_{12}))\\|{\\mathbf{g}}\\|^{2}=0. Note that this result only holds when we choose tt to be small enough, i.e. t≤1Lt\\leq\\frac{1}{L}.

∎

###### Corollary 1.

Assume the nn objectives ℒ1,ℒ2,…,ℒn\\mathcal{L}\_{1},\\mathcal{L}\_{2},...,\\mathcal{L}\_{n} are convex and differentiable. Suppose the gradient of ℒ\\mathcal{L} is Lipschitz continuous with constant L\>0L>0. Assume that cos⁡(𝐠,𝐠PC)≥12\\cos({\\mathbf{g}},{\\mathbf{g}}^{\\text{PC}})\\geq\\frac{1}{2}. Then, the PCGrad update rule with step size t≤1Lt\\leq\\frac{1}{L} will converge to either (1) a location in the optimization landscape where cos(𝐠i,𝐠j)\=−1∀i,j\\cos({\\mathbf{g}}\_{i},{\\mathbf{g}}\_{j})=-1\\forall i,j or (2) the optimal value ℒ⁡(θ∗)\\mathcal{L}(\\theta^{\*}).

###### Proof.

Our assumption that ∇ℒ\\nabla\\mathcal{L} is Lipschitz continuous with constant LL implies that ∇2ℒ​(θ)−L​I\\nabla^{2}\\mathcal{L}(\\theta)-LI is a negative semi-definite matrix. Using this fact, we can perform a quadratic expansion of ℒ\\mathcal{L} around ℒ⁡(θ)\\mathcal{L}(\\theta) and obtain the following inequality:

ℒ⁡(θ+)\\displaystyle\\mathcal{L}(\\theta^{+})

≤ℒ(θ)+∇ℒ(θ)T(θ+−θ)+12∇2ℒ(θ)||θ+−θ||2\\displaystyle\\leq\\mathcal{L}(\\theta)+\\nabla\\mathcal{L}(\\theta)^{T}(\\theta^{+}-\\theta)+\\frac{1}{2}\\nabla^{2}\\mathcal{L}(\\theta)||\\theta^{+}-\\theta||^{2}

≤ℒ(θ)+∇ℒ(θ)T(θ+−θ)+12L||θ+−θ||2\\displaystyle\\leq\\mathcal{L}(\\theta)+\\nabla\\mathcal{L}(\\theta)^{T}(\\theta^{+}-\\theta)+\\frac{1}{2}L||\\theta^{+}-\\theta||^{2}

Now, we can plug in the PCGrad update by letting θ+\=θ−t⋅𝐠PC\\theta^{+}=\\theta-t\\cdot{\\mathbf{g}}^{\\text{PC}}. We then get:

ℒ⁡(θ+)\\displaystyle\\mathcal{L}(\\theta^{+})

≤ℒ⁡(θ)−t⋅𝐠T​𝐠PC+12​L​t2​‖𝐠PC‖2\\displaystyle\\leq\\mathcal{L}(\\theta)-t\\cdot{\\mathbf{g}}^{T}{\\mathbf{g}}^{\\text{PC}}+\\frac{1}{2}Lt^{2}||{\\mathbf{g}}^{\\text{PC}}||^{2}

(Using the assumption that cos⁡(𝐠,𝐠PC)≥12\\cos({\\mathbf{g}},{\\mathbf{g}}^{\\text{PC}})\\geq\\frac{1}{2}.)

≤ℒ⁡(θ)−12​t​‖𝐠‖⋅‖𝐠PC‖+12​L​t2​‖𝐠PC‖2\\displaystyle\\leq\\mathcal{L}(\\theta)-\\frac{1}{2}t||{\\mathbf{g}}||\\cdot||{\\mathbf{g}}^{\\text{PC}}||+\\frac{1}{2}Lt^{2}||{\\mathbf{g}}^{\\text{PC}}||^{2}

≤ℒ⁡(θ)−12​t​‖𝐠‖⋅‖𝐠PC​‖+12​L​t2‖​𝐠PC‖⋅‖𝐠‖\\displaystyle\\leq\\mathcal{L}(\\theta)-\\frac{1}{2}t||{\\mathbf{g}}||\\cdot||{\\mathbf{g}}^{\\text{PC}}||+\\frac{1}{2}Lt^{2}||{\\mathbf{g}}^{\\text{PC}}||\\cdot||{\\mathbf{g}}||

Note that −12​t||𝐠||⋅||𝐠PC​‖+12​L​t2‖​𝐠PC||⋅||𝐠||≤0-\\frac{1}{2}t||{\\mathbf{g}}||\\cdot||{\\mathbf{g}}^{\\text{PC}}||+\\frac{1}{2}Lt^{2}||{\\mathbf{g}}^{\\text{PC}}||\\cdot||{\\mathbf{g}}||\\leq 0 when t≤1Lt\\leq\\frac{1}{L}. Further, when t<1Lt<\\frac{1}{L}, −12​t||𝐠||⋅||𝐠PC​‖+12​L​t2‖​𝐠PC||⋅||𝐠||\=0-\\frac{1}{2}t||{\\mathbf{g}}||\\cdot||{\\mathbf{g}}^{\\text{PC}}||+\\frac{1}{2}Lt^{2}||{\\mathbf{g}}^{\\text{PC}}||\\cdot||{\\mathbf{g}}||=0 if and only if ‖𝐠‖\=0||{\\mathbf{g}}||=0 or ‖𝐠PC‖\=0||{\\mathbf{g}}^{\\text{PC}}||=0.

Hence repeatedly applying PCGrad process can either reach the optimal value ℒ⁡(θ)\=ℒ⁡(θ∗)\\mathcal{L}(\\theta)=\\mathcal{L}(\\theta^{\*}) or a location in the optimization landscape where cos⁡(𝐠i,𝐠j)\=−1\\cos({\\mathbf{g}}\_{i},{\\mathbf{g}}\_{j})=-1 for all pairs of tasks i,ji,j. Note that this result only holds when we choose tt to be small enough, i.e. t≤1Lt\\leq\\frac{1}{L}. ∎

###### Proposition 1.

Assume ℒ1\\mathcal{L}\_{1} and ℒ2\\mathcal{L}\_{2} are differentiable but possibly non-convex. Suppose the gradient of ℒ\\mathcal{L} is Lipschitz continuous with constant L\>0L>0. Then, the PCGrad update rule with step size t≤1Lt\\leq\\frac{1}{L} will converge to either (1) a location in the optimization landscape where cos⁡(ϕ12)\=−1\\cos(\\phi\_{12})=-1 or (2) find a θk\\theta\_{k} that is almost a stationary point.

###### Proof.

Following Definition [1](#Thmdefinition1 "Definition 1. ‣ 2.2 ​​​The Tragic Triad: Conflicting Gradients, Dominating Gradients, High Curvature ‣ 2 Multi-Task Learning with PCGrad ‣ Gradient Surgery for Multi-Task Learning") and [4](#Thmdefinition4 "Definition 4. ‣ 2.4 Theoretical Analysis of PCGrad ‣ 2 Multi-Task Learning with PCGrad ‣ Gradient Surgery for Multi-Task Learning"), let 𝐠𝟏k\=∇ℒ1{\\mathbf{g\_{1}}}\_{k}=\\nabla\\mathcal{L}\_{1} at iteration kk, 𝐠𝟐k\=∇ℒ2{\\mathbf{g\_{2}}}\_{k}=\\nabla\\mathcal{L}\_{2} at iteration kk, and 𝐠k\=∇ℒ\=𝐠𝟏k+𝐠𝟐k{\\mathbf{g}}\_{k}=\\nabla\\mathcal{L}={\\mathbf{g\_{1}}}\_{k}+{\\mathbf{g\_{2}}}\_{k} at iteration kk, and ϕ12,k\\phi\_{12,k} be the angle between 𝐠𝟏k{\\mathbf{g\_{1}}}\_{k} and 𝐠𝟐k{\\mathbf{g\_{2}}}\_{k}.

From the proof of Theorem 1, when cos⁡(ϕ12,k)<0\\cos(\\phi\_{12},k)<0 we have:

‖𝐠k‖2≤2t​ℒ⁡(θk−1)−ℒ⁡(θk)(1−cos2⁡(ϕ12,k)).||{\\mathbf{g}}\_{k}||^{2}\\leq\\frac{2}{t}\\frac{\\mathcal{L}(\\theta\_{k-1})-\\mathcal{L}(\\theta\_{k})}{(1-\\cos^{2}(\\phi\_{12,k}))}.

Thus, we have:

min0≤k≤K⁡‖𝐠k‖2\\displaystyle\\min\_{0\\leq k\\leq K}||{\\mathbf{g}}\_{k}||^{2}

≤1K​∑i\=0K−1‖𝐠i‖2\\displaystyle\\leq\\frac{1}{K}\\sum\_{i=0}^{K-1}||{\\mathbf{g}}\_{i}||^{2}

≤2K​t​∑i\=0K−1ℒ⁡(θi−1)−ℒ⁡(θi)(1−cos2⁡(ϕ12,i))\\displaystyle\\leq\\frac{2}{Kt}\\sum\_{i=0}^{K-1}\\frac{\\mathcal{L}(\\theta\_{i-1})-\\mathcal{L}(\\theta\_{i})}{(1-\\cos^{2}(\\phi\_{12,i}))}

If at any iteration, cos⁡(ϕ12,k)\=−1\\cos(\\phi\_{12,k})=-1, then the optimization will stop at that point. If ∀k∈\[0,K\]\\forall k\\in\[0,K\], cos⁡(ϕ12,k)≥α\>−1\\cos(\\phi\_{12,k})\\geq\\alpha>-1, then, we have:

min0≤k≤K⁡‖𝐠k‖2\\displaystyle\\min\_{0\\leq k\\leq K}||{\\mathbf{g}}\_{k}||^{2}

≤2K⁡(1−α2)​t​∑i\=0K−1(ℒ⁡(θi−1)−ℒ⁡(θi))\\displaystyle\\leq\\frac{2}{K(1-\\alpha^{2})t}\\sum\_{i=0}^{K-1}(\\mathcal{L}(\\theta\_{i-1})-\\mathcal{L}(\\theta\_{i}))

\=2K⁡(1−α2)​t​(ℒ⁡(θ0)−ℒ⁡(θK))\\displaystyle=\\frac{2}{K(1-\\alpha^{2})t}(\\mathcal{L}(\\theta\_{0})-\\mathcal{L}(\\theta\_{K}))

≤2K⁡(1−α2)​t​(ℒ⁡(θ0)−ℒ∗).\\displaystyle\\leq\\frac{2}{K(1-\\alpha^{2})t}(\\mathcal{L}(\\theta\_{0})-\\mathcal{L}^{\*}).

where ℒ∗\\mathcal{L}^{\*} is the minimal function value. ∎

Note that the convergence rate of PCGrad in the non-convex setting largely depends on the value of α\\alpha and generally how small cos⁡(ϕ12,k)\\cos(\\phi\_{12,k}) is on average.

### A.2 Proof of Theorem 2

See [2](#Thmtheorem2 "Theorem 2. ‣ 2.4 Theoretical Analysis of PCGrad ‣ 2 Multi-Task Learning with PCGrad ‣ Gradient Surgery for Multi-Task Learning")

###### Proof.

Note that θMT\=θ−t⋅𝐠\\theta^{\\text{MT}}=\\theta-t\\cdot{\\mathbf{g}} and θPCGrad\=θ−t⁡(𝐠−𝐠𝟏⋅𝐠𝟐‖𝐠𝟏‖2​𝐠𝟏−𝐠𝟏⋅𝐠𝟐‖𝐠𝟐‖2​𝐠𝟐)\\theta^{\\text{PCGrad}}=\\theta-t({\\mathbf{g}}-\\frac{{\\mathbf{g\_{1}}}\\cdot{\\mathbf{g\_{2}}}}{||{\\mathbf{g\_{1}}}||^{2}}{\\mathbf{g\_{1}}}-\\frac{{\\mathbf{g\_{1}}}\\cdot{\\mathbf{g\_{2}}}}{||{\\mathbf{g\_{2}}}||^{2}}{\\mathbf{g\_{2}}}). Based on the condition that 𝐇⁡(ℒ,θ,θMT)≥ℓ​‖𝐠‖22\\mathbf{H}(\\mathcal{L};\\theta,\\theta^{\\text{MT}})\\geq\\ell\\|{\\mathbf{g}}\\|\_{2}^{2}, we first apply Taylor’s Theorem to ℒ⁡(θMT)\\mathcal{L}(\\theta^{\\text{MT}}) and obtain the following result:

ℒ⁡(θMT)\\displaystyle\\mathcal{L}(\\theta^{\\text{MT}})

\=ℒ⁡(θ)+𝐠T​(−t​𝐠)+∫01(−t​𝐠)T​∇2ℒ​(θ+a⋅(−t​𝐠))2​(−t​𝐠)​𝑑a\\displaystyle=\\mathcal{L}(\\theta)+{\\mathbf{g}}^{T}(-t{\\mathbf{g}})+\\int\_{0}^{1}(-t{\\mathbf{g}})^{T}\\frac{\\nabla^{2}\\mathcal{L}(\\theta+a\\cdot(-t{\\mathbf{g}}))}{2}(-t{\\mathbf{g}})da

≥ℒ⁡(θ)+𝐠T​(−t​𝐠)+t2⋅12​ℓ⋅‖𝐠‖22\\displaystyle\\geq\\mathcal{L}(\\theta)+{\\mathbf{g}}^{T}(-t{\\mathbf{g}})+t^{2}\\cdot\\frac{1}{2}\\ell\\cdot\\|{\\mathbf{g}}\\|\_{2}^{2}

\=ℒ⁡(θ)−t​‖𝐠‖22+12​ℓ​t2​‖𝐠‖22\\displaystyle=\\mathcal{L}(\\theta)-t\\|{\\mathbf{g}}\\|\_{2}^{2}+\\frac{1}{2}\\ell t^{2}\\|{\\mathbf{g}}\\|\_{2}^{2}

\=ℒ⁡(θ)+(12​ℓ​t2−t)​‖𝐠‖22\\displaystyle=\\mathcal{L}(\\theta)+(\\frac{1}{2}\\ell t^{2}-t)\\|{\\mathbf{g}}\\|\_{2}^{2}

(2)

where the first inequality follows from Definition [3](#Thmdefinition3 "Definition 3. ‣ 2.2 ​​​The Tragic Triad: Conflicting Gradients, Dominating Gradients, High Curvature ‣ 2 Multi-Task Learning with PCGrad ‣ Gradient Surgery for Multi-Task Learning") and the assumption 𝐇⁡(ℒ,θ,θMT)≥ℓ​‖𝐠‖22\\mathbf{H}(\\mathcal{L};\\theta,\\theta^{\\text{MT}})\\geq\\ell\\|{\\mathbf{g}}\\|\_{2}^{2}. From equation [1](#A1.E1 "Equation 1 ‣ Proof. ‣ A.1 Proof of Theorem 1 ‣ Appendix A Proofs ‣ Gradient Surgery for Multi-Task Learning"), we have the simplified upper bound for ℒ⁡(θPCGrad)\\mathcal{L}(\\theta^{\\text{PCGrad}}):

ℒ⁡(θPCGrad)\\displaystyle\\mathcal{L}(\\theta^{\\text{PCGrad}})

≤ℒ(θ)−(1−cos2ϕ12)\[(t−12Lt2)⋅(∥𝐠𝟏∥22+∥𝐠𝟏∥22)+Lt2∥𝐠𝟏∥2∥𝐠𝟐∥2cosϕ12\]\\displaystyle\\leq\\mathcal{L}(\\theta)-(1-\\cos^{2}\\phi\_{12})\[(t-\\frac{1}{2}Lt^{2})\\cdot(\\|{\\mathbf{g\_{1}}}\\|\_{2}^{2}+\\|{\\mathbf{g\_{1}}}\\|\_{2}^{2})+Lt^{2}\\|{\\mathbf{g\_{1}}}\\|\_{2}\\|{\\mathbf{g\_{2}}}\\|\_{2}\\cos\\phi\_{12}\]

(3)

Apply Equation [2](#A1.E2 "Equation 2 ‣ Proof. ‣ A.2 Proof of Theorem 2 ‣ Appendix A Proofs ‣ Gradient Surgery for Multi-Task Learning") and Equation [3](#A1.E3 "Equation 3 ‣ Proof. ‣ A.2 Proof of Theorem 2 ‣ Appendix A Proofs ‣ Gradient Surgery for Multi-Task Learning") and we have the following inequality:

ℒ⁡(θMT)−ℒ⁡(θPCGrad)≥ℒ⁡(θ)+(12​ℓ​t2−t)​‖𝐠‖22−ℒ⁡(θ)\\displaystyle\\mathcal{L}(\\theta^{\\text{MT}})-\\mathcal{L}(\\theta^{\\text{PCGrad}})\\geq\\mathcal{L}(\\theta)+(\\frac{1}{2}\\ell t^{2}-t)\\|{\\mathbf{g}}\\|\_{2}^{2}-\\mathcal{L}(\\theta)

+(1−cos2ϕ12)\[(t−12Lt2)(∥𝐠𝟏∥22+∥𝐠𝟐∥22)+Lt2∥𝐠𝟏∥2∥𝐠𝟐∥2cosϕ12\]\\displaystyle+(1-\\cos^{2}\\phi\_{12})\[(t-\\frac{1}{2}Lt^{2})(\\|{\\mathbf{g\_{1}}}\\|\_{2}^{2}+\\|{\\mathbf{g\_{2}}}\\|\_{2}^{2})+Lt^{2}\\|{\\mathbf{g\_{1}}}\\|\_{2}\\|{\\mathbf{g\_{2}}}\\|\_{2}\\cos\\phi\_{12}\]

\=(12ℓt2−t)∥𝐠𝟏+𝐠𝟐∥22+(1−cos2ϕ12)\[(t−12Lt2)⋅(∥𝐠𝟏∥22+∥𝐠𝟐∥22)+Lt2∥𝐠𝟏∥2∥𝐠𝟐∥2cosϕ12\]\\displaystyle=(\\frac{1}{2}\\ell t^{2}-t)\\|{\\mathbf{g\_{1}}}+{\\mathbf{g\_{2}}}\\|\_{2}^{2}\\!\\!+\\!\\!(1-\\cos^{2}\\phi\_{12})\\!\\!\\left\[(t-\\frac{1}{2}Lt^{2})\\!\\cdot\\!(\\|{\\mathbf{g\_{1}}}\\|\_{2}^{2}\\!+\\!\\|{\\mathbf{g\_{2}}}\\|\_{2}^{2})\\!+\\!Lt^{2}\\|{\\mathbf{g\_{1}}}\\|\_{2}\\|{\\mathbf{g\_{2}}}\\|\_{2}\\cos\\phi\_{12}\\right\]

\=(12∥𝐠𝟏+𝐠𝟐∥22ℓ−1−cos2⁡ϕ122(∥𝐠𝟏∥22+∥𝐠𝟐∥22−2∥𝐠𝟏∥2∥𝐠𝟐∥2cosϕ12)L)t2\\displaystyle=\\left(\\frac{1}{2}\\|{\\mathbf{g\_{1}}}+{\\mathbf{g\_{2}}}\\|\_{2}^{2}\\ell-\\frac{1-\\cos^{2}\\phi\_{12}}{2}(\\|{\\mathbf{g\_{1}}}\\|\_{2}^{2}+\\|{\\mathbf{g\_{2}}}\\|\_{2}^{2}-2\\|{\\mathbf{g\_{1}}}\\|\_{2}\\|{\\mathbf{g\_{2}}}\\|\_{2}\\cos\\phi\_{12})L\\right)t^{2}

−((∥𝐠𝟏∥22+∥𝐠𝟐∥22)cos2ϕ12+2∥𝐠𝟏∥2∥𝐠𝟐∥2cosϕ12)t\\displaystyle-\\left((\\|{\\mathbf{g\_{1}}}\\|\_{2}^{2}+\\|{\\mathbf{g\_{2}}}\\|\_{2}^{2})\\cos^{2}\\phi\_{12}+2\\|{\\mathbf{g\_{1}}}\\|\_{2}\\|{\\mathbf{g\_{2}}}\\|\_{2}\\cos\\phi\_{12}\\right)t

\=(12​‖𝐠𝟏+𝐠𝟐‖22​ℓ−1−cos2⁡ϕ122​‖𝐠𝟏−𝐠𝟐‖22​L)​t2\\displaystyle=\\left(\\frac{1}{2}\\|{\\mathbf{g\_{1}}}+{\\mathbf{g\_{2}}}\\|\_{2}^{2}\\ell-\\frac{1-\\cos^{2}\\phi\_{12}}{2}\\|{\\mathbf{g\_{1}}}-{\\mathbf{g\_{2}}}\\|\_{2}^{2}L\\right)t^{2}

−((∥𝐠𝟏∥22+∥𝐠𝟐∥22)cos2ϕ12+2∥𝐠𝟏∥2∥𝐠𝟐∥2cosϕ12)t\\displaystyle-\\left((\\|{\\mathbf{g\_{1}}}\\|\_{2}^{2}+\\|{\\mathbf{g\_{2}}}\\|\_{2}^{2})\\cos^{2}\\phi\_{12}+2\\|{\\mathbf{g\_{1}}}\\|\_{2}\\|{\\mathbf{g\_{2}}}\\|\_{2}\\cos\\phi\_{12}\\right)t

\=t⋅\[(12∥𝐠𝟏+𝐠𝟐∥22ℓ−1−cos2⁡ϕ122(∥𝐠𝟏−𝐠𝟐∥22)L)t\\displaystyle=t\\cdot\\left\[\\left(\\frac{1}{2}\\|{\\mathbf{g\_{1}}}+{\\mathbf{g\_{2}}}\\|\_{2}^{2}\\ell-\\frac{1-\\cos^{2}\\phi\_{12}}{2}(\\|{\\mathbf{g\_{1}}}-{\\mathbf{g\_{2}}}\\|\_{2}^{2})L\\right)t\\right.

−((∥𝐠𝟏∥22+∥𝐠𝟐∥22)cos2ϕ12+2∥𝐠𝟏∥2∥𝐠𝟐∥2cosϕ12)\]\\displaystyle\\left.-\\left((\\|{\\mathbf{g\_{1}}}\\|\_{2}^{2}+\\|{\\mathbf{g\_{2}}}\\|\_{2}^{2})\\cos^{2}\\phi\_{12}+2\\|{\\mathbf{g\_{1}}}\\|\_{2}\\|{\\mathbf{g\_{2}}}\\|\_{2}\\cos\\phi\_{12}\\right)\\right\]

(4)

Since cos⁡ϕ12≤−Φ⁡(𝐠𝟏,𝐠𝟐)\=−2​‖𝐠𝟏‖2​‖𝐠𝟐‖2‖𝐠𝟏‖22+‖𝐠𝟐‖22\\cos\\phi\_{12}\\leq-\\Phi({\\mathbf{g\_{1}}},{\\mathbf{g\_{2}}})=-\\frac{2\\|{\\mathbf{g\_{1}}}\\|\_{2}\\|{\\mathbf{g\_{2}}}\\|\_{2}}{\\|{\\mathbf{g\_{1}}}\\|\_{2}^{2}+\\|{\\mathbf{g\_{2}}}\\|\_{2}^{2}} and ℓ≥ξ⁡(𝐠𝟏,𝐠𝟐)\=(1−cos2⁡ϕ12)​(‖𝐠𝟏−𝐠𝟐‖22)‖𝐠𝟏+𝐠𝟐‖22​L\\ell\\geq\\xi({\\mathbf{g\_{1}}},{\\mathbf{g\_{2}}})=\\frac{(1-\\cos^{2}\\phi\_{12})(\\|{\\mathbf{g\_{1}}}-{\\mathbf{g\_{2}}}\\|\_{2}^{2})}{\\|{\\mathbf{g\_{1}}}+{\\mathbf{g\_{2}}}\\|\_{2}^{2}}L, we have

12​‖𝐠𝟏+𝐠𝟐‖22​ℓ−1−cos2⁡ϕ122​‖𝐠𝟏−𝐠𝟐‖22​L≥0\\frac{1}{2}\\|{\\mathbf{g\_{1}}}+{\\mathbf{g\_{2}}}\\|\_{2}^{2}\\ell-\\frac{1-\\cos^{2}\\phi\_{12}}{2}\\|{\\mathbf{g\_{1}}}-{\\mathbf{g\_{2}}}\\|\_{2}^{2}L\\geq 0

and

(∥𝐠𝟏∥22+∥𝐠𝟐∥22)cos2ϕ12+2∥𝐠𝟏∥2∥𝐠𝟐∥2cosϕ12≥0.(\\|{\\mathbf{g\_{1}}}\\|\_{2}^{2}+\\|{\\mathbf{g\_{2}}}\\|\_{2}^{2})\\cos^{2}\\phi\_{12}+2\\|{\\mathbf{g\_{1}}}\\|\_{2}\\|{\\mathbf{g\_{2}}}\\|\_{2}\\cos\\phi\_{12}\\geq 0.

By the condition that t≥2ℓ−ξ⁡(𝐠𝟏,𝐠𝟐)​L\=2ℓ−(1−cos2⁡ϕ12)​‖𝐠𝟏−𝐠𝟐‖22‖𝐠𝟏+𝐠𝟐‖22​Lt\\geq\\frac{2}{\\ell-\\xi({\\mathbf{g\_{1}}},{\\mathbf{g\_{2}}})L}=\\frac{2}{\\ell-\\frac{(1-\\cos^{2}\\phi\_{12})\\|{\\mathbf{g\_{1}}}-{\\mathbf{g\_{2}}}\\|\_{2}^{2}}{\\|{\\mathbf{g\_{1}}}+{\\mathbf{g\_{2}}}\\|\_{2}^{2}}L} and monotonicity of linear functions, we have the following:

ℒ(θMT)−ℒ(θPCGrad)≥\[(12∥𝐠𝟏+𝐠𝟐∥22ℓ−−cos2⁡ϕ122⋅∥𝐠𝟏−𝐠𝟐∥22L)⋅2ℓ−(1−cos2⁡ϕ12)​‖𝐠𝟏−𝐠𝟐‖22‖𝐠𝟏+𝐠𝟐‖22​L\\displaystyle\\mathcal{L}(\\theta^{\\text{MT}})\\!-\\!\\mathcal{L}(\\theta^{\\text{PCGrad}})\\geq\[\\left(\\frac{1}{2}\\|{\\mathbf{g\_{1}}}\\!+\\!{\\mathbf{g\_{2}}}\\|\_{2}^{2}\\ell-\\frac{1\\!-\\!\\cos^{2}\\phi\_{12}}{2}\\!\\cdot\\!\\|{\\mathbf{g\_{1}}}\\!-\\!{\\mathbf{g\_{2}}}\\|\_{2}^{2}L\\right)\\cdot\\frac{2}{\\ell\\!-\\!\\frac{(1-\\cos^{2}\\phi\_{12})\\|{\\mathbf{g\_{1}}}-{\\mathbf{g\_{2}}}\\|\_{2}^{2}}{\\|{\\mathbf{g\_{1}}}+{\\mathbf{g\_{2}}}\\|\_{2}^{2}}L}

−((∥𝐠𝟏∥22+∥𝐠𝟐∥22)cos2ϕ12+2∥𝐠𝟏∥2∥𝐠𝟐∥2cosϕ12)\]⋅t\\displaystyle-\\left((\\|{\\mathbf{g\_{1}}}\\|\_{2}^{2}+\\|{\\mathbf{g\_{2}}}\\|\_{2}^{2})\\cos^{2}\\phi\_{12}+2\\|{\\mathbf{g\_{1}}}\\|\_{2}\\|{\\mathbf{g\_{2}}}\\|\_{2}\\cos\\phi\_{12}\\right)\]\\cdot t

\=\[∥𝐠𝟏+𝐠𝟐∥22⋅(ℓ−(1−cos2⁡ϕ12)⋅‖𝐠𝟏−𝐠𝟐‖22‖𝐠𝟏+𝐠𝟐‖22)L)⋅1ℓ−(1−cos2⁡ϕ12)​‖𝐠𝟏−𝐠𝟐‖22‖𝐠𝟏+𝐠𝟐‖22​L\\displaystyle=\[\\|{\\mathbf{g\_{1}}}+{\\mathbf{g\_{2}}}\\|\_{2}^{2}\\cdot\\left(\\ell-\\frac{(1-\\cos^{2}\\phi\_{12})\\cdot\\|{\\mathbf{g\_{1}}}-{\\mathbf{g\_{2}}}\\|\_{2}^{2}}{\\|{\\mathbf{g\_{1}}}+{\\mathbf{g\_{2}}}\\|\_{2}^{2}})L\\right)\\cdot\\frac{1}{\\ell-\\frac{(1-\\cos^{2}\\phi\_{12})\\|{\\mathbf{g\_{1}}}-{\\mathbf{g\_{2}}}\\|\_{2}^{2}}{\\|{\\mathbf{g\_{1}}}+{\\mathbf{g\_{2}}}\\|\_{2}^{2}}L}

−((∥𝐠𝟏∥22+∥𝐠𝟐∥22)cos2ϕ12+2∥𝐠𝟏∥2∥𝐠𝟐∥2cosϕ12)\]⋅t\\displaystyle-\\left((\\|{\\mathbf{g\_{1}}}\\|\_{2}^{2}+\\|{\\mathbf{g\_{2}}}\\|\_{2}^{2})\\cos^{2}\\phi\_{12}+2\\|{\\mathbf{g\_{1}}}\\|\_{2}\\|{\\mathbf{g\_{2}}}\\|\_{2}\\cos\\phi\_{12}\\right)\]\\cdot t

\=\[∥𝐠𝟏+𝐠𝟐∥22−((∥𝐠𝟏∥22+∥𝐠𝟐∥22)cos2ϕ12+2∥𝐠𝟏∥2∥𝐠𝟐∥2cosϕ12)\]⋅t\\displaystyle=\\left\[\\|{\\mathbf{g\_{1}}}+{\\mathbf{g\_{2}}}\\|\_{2}^{2}-\\left((\\|{\\mathbf{g\_{1}}}\\|\_{2}^{2}+\\|{\\mathbf{g\_{2}}}\\|\_{2}^{2})\\cos^{2}\\phi\_{12}+2\\|{\\mathbf{g\_{1}}}\\|\_{2}\\|{\\mathbf{g\_{2}}}\\|\_{2}\\cos\\phi\_{12}\\right)\\right\]\\cdot t

\=\[∥𝐠𝟏∥22+∥𝐠𝟐∥22+2∥𝐠𝟏∥2∥𝐠𝟐∥2cosϕ12−((∥𝐠𝟏∥22+∥𝐠𝟐∥22)cos2ϕ12+2∥𝐠𝟏∥2∥𝐠𝟐∥2cosϕ12)\]⋅t\\displaystyle=\\left\[\\|{\\mathbf{g\_{1}}}\\|\_{2}^{2}\\!+\\!\\|{\\mathbf{g\_{2}}}\\|\_{2}^{2}\\!+\\!2\\|{\\mathbf{g\_{1}}}\\|\_{2}\\|{\\mathbf{g\_{2}}}\\|\_{2}\\!\\cos\\phi\_{12}\\!-\\!\\left((\\|{\\mathbf{g\_{1}}}\\|\_{2}^{2}\\!+\\!\\|{\\mathbf{g\_{2}}}\\|\_{2}^{2})\\cos^{2}\\!\\phi\_{12}\\!+\\!2\\|{\\mathbf{g\_{1}}}\\|\_{2}\\|{\\mathbf{g\_{2}}}\\|\_{2}\\!\\cos\\phi\_{12}\\right)\\right\]\\cdot t

\=(1−cos2⁡ϕ12)​(‖𝐠𝟏‖22+‖𝐠𝟐‖22)⋅t\\displaystyle=(1-\\cos^{2}\\phi\_{12})(\\|{\\mathbf{g\_{1}}}\\|\_{2}^{2}+\\|{\\mathbf{g\_{2}}}\\|\_{2}^{2})\\cdot t

≥0\\displaystyle\\geq 0

∎

### A.3 PCGrad: Sufficient and Necessary Conditions for Loss Improvement

Beyond the sufficient conditions shown in Theorem [2](#Thmtheorem2 "Theorem 2. ‣ 2.4 Theoretical Analysis of PCGrad ‣ 2 Multi-Task Learning with PCGrad ‣ Gradient Surgery for Multi-Task Learning"), we also present the sufficient and necessary conditions under which PCGrad achieves lower loss after one gradient update in Theorem [3](#Thmtheorem3 "Theorem 3. ‣ A.3 PCGrad: Sufficient and Necessary Conditions for Loss Improvement ‣ Appendix A Proofs ‣ Gradient Surgery for Multi-Task Learning") in the two-task setting.

###### Theorem 3.

Suppose ℒ\\mathcal{L} is differentiable and the gradient of ℒ\\mathcal{L} is Lipschitz continuous with constant L\>0L>0. Let θMT\\theta^{\\text{MT}} and θPCGrad\\theta^{\\text{PCGrad}} be the parameters after applying one update to θ\\theta with 𝐠{\\mathbf{g}} and PCGrad-modified gradient 𝐠PC{\\mathbf{g}}^{\\text{PC}} respectively, with step size t\>0t>0. Moreover, assume 𝐇⁡(ℒ,θ,θMT)≥ℓ​‖𝐠‖22\\mathbf{H}(\\mathcal{L};\\theta,\\theta^{\\text{MT}})\\geq\\ell\\|{\\mathbf{g}}\\|\_{2}^{2} for some constant ℓ≤L\\ell\\leq L, i.e. the multi-task curvature is lower-bounded. Then ℒ⁡(θPCGrad)≤ℒ⁡(θMT)\\mathcal{L}(\\theta^{\\text{PCGrad}})\\leq\\mathcal{L}(\\theta^{\\text{MT}}) if and only if

*   •

```
−Φ⁡(𝐠𝟏,𝐠𝟐)≤cos⁡ϕ12<0-\\Phi({\\mathbf{g\_{1}}},{\\mathbf{g\_{2}}})\\leq\\cos\\phi\_{12}<0
```
*   •

```
ℓ≤ξ⁡(𝐠𝟏,𝐠𝟐)​L\\ell\\leq\\xi({\\mathbf{g\_{1}}},{\\mathbf{g\_{2}}})L
```
*   •

```
0<t≤(∥𝐠𝟏∥22+∥𝐠𝟐∥22)cos2ϕ12+2∥𝐠𝟏∥2∥𝐠𝟐∥2cosϕ1212​‖𝐠𝟏+𝐠𝟐‖22​ℓ−1−cos2⁡ϕ122​(‖𝐠𝟏−𝐠𝟐‖22)​L0\<t\\leq\\frac{(\\|{\\mathbf{g\_{1}}}\\|^{2}\_{2}+\\|{\\mathbf{g\_{2}}}\\|^{2}\_{2})\\cos^{2}\\phi\_{12}+2\\|{\\mathbf{g\_{1}}}\\|\_{2}\\|{\\mathbf{g\_{2}}}\\|\_{2}\\cos\\phi\_{12}}{\\frac{1}{2}\\|{\\mathbf{g\_{1}}}+{\\mathbf{g\_{2}}}\\|\_{2}^{2}\\ell-\\frac{1-\\cos^{2}\\phi\_{12}}{2}(\\|{\\mathbf{g\_{1}}}-{\\mathbf{g\_{2}}}\\|\_{2}^{2})L}
```
or

*   •

```
cos⁡ϕ12≤−Φ⁡(𝐠𝟏,𝐠𝟐)\\cos\\phi\_{12}\\leq-\\Phi({\\mathbf{g\_{1}}},{\\mathbf{g\_{2}}})
```
*   •

```
ℓ≥ξ⁡(𝐠𝟏,𝐠𝟐)​L\\ell\\geq\\xi({\\mathbf{g\_{1}}},{\\mathbf{g\_{2}}})L
```
*   •

```
t≥(∥𝐠𝟏∥22+∥𝐠𝟐∥22)cos2ϕ12+2∥𝐠𝟏∥2∥𝐠𝟐∥2cosϕ1212​‖𝐠𝟏+𝐠𝟐‖22​ℓ−1−cos2⁡ϕ122​(‖𝐠𝟏−𝐠𝟐‖22)​L.t\\geq\\frac{(\\|{\\mathbf{g\_{1}}}\\|^{2}\_{2}+\\|{\\mathbf{g\_{2}}}\\|^{2}\_{2})\\cos^{2}\\phi\_{12}+2\\|{\\mathbf{g\_{1}}}\\|\_{2}\\|{\\mathbf{g\_{2}}}\\|\_{2}\\cos\\phi\_{12}}{\\frac{1}{2}\\|{\\mathbf{g\_{1}}}+{\\mathbf{g\_{2}}}\\|\_{2}^{2}\\ell-\\frac{1-\\cos^{2}\\phi\_{12}}{2}(\\|{\\mathbf{g\_{1}}}-{\\mathbf{g\_{2}}}\\|\_{2}^{2})L}.
```
###### Proof.

To show the necessary conditions, from Equation [4](#A1.E4 "Equation 4 ‣ Proof. ‣ A.2 Proof of Theorem 2 ‣ Appendix A Proofs ‣ Gradient Surgery for Multi-Task Learning"), all we need is

t⋅\[(12∥𝐠𝟏+𝐠𝟐∥22ℓ−1−cos2⁡ϕ122(∥𝐠𝟏−𝐠𝟐∥22)L)t\\displaystyle t\\cdot\[(\\frac{1}{2}\\|{\\mathbf{g\_{1}}}+{\\mathbf{g\_{2}}}\\|\_{2}^{2}\\ell-\\frac{1-\\cos^{2}\\phi\_{12}}{2}(\\|{\\mathbf{g\_{1}}}-{\\mathbf{g\_{2}}}\\|\_{2}^{2})L)t

−((∥𝐠𝟏∥22+∥𝐠𝟐∥22)cos2ϕ12+2∥𝐠𝟏∥2∥𝐠𝟐∥2cosϕ12)\]≥0\\displaystyle-\\left((\\|{\\mathbf{g\_{1}}}\\|\_{2}^{2}+\\|{\\mathbf{g\_{2}}}\\|\_{2}^{2})\\cos^{2}\\phi\_{12}+2\\|{\\mathbf{g\_{1}}}\\|\_{2}\\|{\\mathbf{g\_{2}}}\\|\_{2}\\cos\\phi\_{12}\\right)\]\\geq 0

(5)

Since t≥0t\\geq 0, it reduces to show

(12​‖𝐠𝟏+𝐠𝟐‖22​ℓ−1−cos2⁡ϕ122​(‖𝐠𝟏−𝐠𝟐‖22)​L)​t\\displaystyle(\\frac{1}{2}\\|{\\mathbf{g\_{1}}}+{\\mathbf{g\_{2}}}\\|\_{2}^{2}\\ell-\\frac{1-\\cos^{2}\\phi\_{12}}{2}(\\|{\\mathbf{g\_{1}}}-{\\mathbf{g\_{2}}}\\|\_{2}^{2})L)t

−((∥𝐠𝟏∥22+∥𝐠𝟐∥22)cos2ϕ12+2∥𝐠𝟏∥2∥𝐠𝟐∥2cosϕ12)≥0\\displaystyle-\\left((\\|{\\mathbf{g\_{1}}}\\|\_{2}^{2}+\\|{\\mathbf{g\_{2}}}\\|\_{2}^{2})\\cos^{2}\\phi\_{12}+2\\|{\\mathbf{g\_{1}}}\\|\_{2}\\|{\\mathbf{g\_{2}}}\\|\_{2}\\cos\\phi\_{12}\\right)\\geq 0

(6)

For Equation [6](#A1.E6 "Equation 6 ‣ Proof. ‣ A.3 PCGrad: Sufficient and Necessary Conditions for Loss Improvement ‣ Appendix A Proofs ‣ Gradient Surgery for Multi-Task Learning") to hold while ensuring that t≥0t\\geq 0, there are two cases:

*   •

```
12​‖𝐠𝟏+𝐠𝟐‖22​ℓ−(1−cos2⁡ϕ12)​(‖𝐠𝟏‖22+‖𝐠𝟐‖22)​L≥0\\frac{1}{2}\\|{\\mathbf{g\_{1}}}+{\\mathbf{g\_{2}}}\\|\_{2}^{2}\\ell-(1-\\cos^{2}\\phi\_{12})(\\|{\\mathbf{g\_{1}}}\\|\_{2}^{2}+\\|{\\mathbf{g\_{2}}}\\|\_{2}^{2})L\\geq 0,
(∥𝐠𝟏∥22+∥𝐠𝟐∥22)cos2ϕ12+2∥𝐠𝟏∥2∥𝐠𝟐∥2cosϕ12≥0(\\|{\\mathbf{g\_{1}}}\\|\_{2}^{2}+\\|{\\mathbf{g\_{2}}}\\|\_{2}^{2})\\cos^{2}\\phi\_{12}+2\\|{\\mathbf{g\_{1}}}\\|\_{2}\\|{\\mathbf{g\_{2}}}\\|\_{2}\\cos\\phi\_{12}\\geq 0,
t≥(∥𝐠𝟏∥22+∥𝐠𝟐∥22)cos2ϕ12+2∥𝐠𝟏∥2∥𝐠𝟐∥2cosϕ1212​‖𝐠𝟏+𝐠𝟐‖22​ℓ−1−cos2⁡ϕ122​(‖𝐠𝟏−𝐠𝟐‖22)​Lt\\geq\\frac{(\\|{\\mathbf{g\_{1}}}\\|^{2}\_{2}+\\|{\\mathbf{g\_{2}}}\\|^{2}\_{2})\\cos^{2}\\phi\_{12}+2\\|{\\mathbf{g\_{1}}}\\|\_{2}\\|{\\mathbf{g\_{2}}}\\|\_{2}\\cos\\phi\_{12}}{\\frac{1}{2}\\|{\\mathbf{g\_{1}}}+{\\mathbf{g\_{2}}}\\|\_{2}^{2}\\ell-\\frac{1-\\cos^{2}\\phi\_{12}}{2}(\\|{\\mathbf{g\_{1}}}-{\\mathbf{g\_{2}}}\\|\_{2}^{2})L}
```
*   •

```
12​‖𝐠𝟏+𝐠𝟐‖22​ℓ−(1−cos2⁡ϕ12)​(‖𝐠𝟏‖22+‖𝐠𝟐‖22)​L≤0\\frac{1}{2}\\|{\\mathbf{g\_{1}}}+{\\mathbf{g\_{2}}}\\|\_{2}^{2}\\ell-(1-\\cos^{2}\\phi\_{12})(\\|{\\mathbf{g\_{1}}}\\|\_{2}^{2}+\\|{\\mathbf{g\_{2}}}\\|\_{2}^{2})L\\leq 0,
(∥𝐠𝟏∥22+∥𝐠𝟐∥22)cos2ϕ12+2∥𝐠𝟏∥2∥𝐠𝟐∥2cosϕ12≤0(\\|{\\mathbf{g\_{1}}}\\|\_{2}^{2}+\\|{\\mathbf{g\_{2}}}\\|\_{2}^{2})\\cos^{2}\\phi\_{12}+2\\|{\\mathbf{g\_{1}}}\\|\_{2}\\|{\\mathbf{g\_{2}}}\\|\_{2}\\cos\\phi\_{12}\\leq 0,
t≥(∥𝐠𝟏∥22+∥𝐠𝟐∥22)cos2ϕ12+2∥𝐠𝟏∥2∥𝐠𝟐∥2cosϕ1212​‖𝐠𝟏+𝐠𝟐‖22​ℓ−1−cos2⁡ϕ122​(‖𝐠𝟏−𝐠𝟐‖22)​Lt\\geq\\frac{(\\|{\\mathbf{g\_{1}}}\\|^{2}\_{2}+\\|{\\mathbf{g\_{2}}}\\|^{2}\_{2})\\cos^{2}\\phi\_{12}+2\\|{\\mathbf{g\_{1}}}\\|\_{2}\\|{\\mathbf{g\_{2}}}\\|\_{2}\\cos\\phi\_{12}}{\\frac{1}{2}\\|{\\mathbf{g\_{1}}}+{\\mathbf{g\_{2}}}\\|\_{2}^{2}\\ell-\\frac{1-\\cos^{2}\\phi\_{12}}{2}(\\|{\\mathbf{g\_{1}}}-{\\mathbf{g\_{2}}}\\|\_{2}^{2})L}
```
, which can be simplified to

*   •

```
cos⁡ϕ12≤−2​‖𝐠𝟏‖2​‖𝐠𝟐‖2‖𝐠𝟏‖22+‖𝐠𝟐‖22\=−Φ⁡(𝐠𝟏,𝐠𝟐)\\cos\\phi\_{12}\\leq-\\frac{2\\|{\\mathbf{g\_{1}}}\\|\_{2}\\|{\\mathbf{g\_{2}}}\\|\_{2}}{\\|{\\mathbf{g\_{1}}}\\|\_{2}^{2}+\\|{\\mathbf{g\_{2}}}\\|\_{2}^{2}}=-\\Phi({\\mathbf{g\_{1}}},{\\mathbf{g\_{2}}}),
ℓ≥(1−cos2⁡ϕ12)​(‖𝐠𝟏‖22+‖𝐠𝟐‖22)‖𝐠𝟏+𝐠𝟐‖22​L\=ξ⁡(𝐠𝟏,𝐠𝟐)\\ell\\geq\\frac{(1-\\cos^{2}\\phi\_{12})(\\|{\\mathbf{g\_{1}}}\\|\_{2}^{2}+\\|{\\mathbf{g\_{2}}}\\|\_{2}^{2})}{\\|{\\mathbf{g\_{1}}}+{\\mathbf{g\_{2}}}\\|\_{2}^{2}}L=\\xi({\\mathbf{g\_{1}}},{\\mathbf{g\_{2}}}),
t≥(∥𝐠𝟏∥22+∥𝐠𝟐∥22)cos2ϕ12+2∥𝐠𝟏∥2∥𝐠𝟐∥2cosϕ1212​‖𝐠𝟏+𝐠𝟐‖22​ℓ−1−cos2⁡ϕ122​(‖𝐠𝟏−𝐠𝟐‖22)​Lt\\geq\\frac{(\\|{\\mathbf{g\_{1}}}\\|^{2}\_{2}+\\|{\\mathbf{g\_{2}}}\\|^{2}\_{2})\\cos^{2}\\phi\_{12}+2\\|{\\mathbf{g\_{1}}}\\|\_{2}\\|{\\mathbf{g\_{2}}}\\|\_{2}\\cos\\phi\_{12}}{\\frac{1}{2}\\|{\\mathbf{g\_{1}}}+{\\mathbf{g\_{2}}}\\|\_{2}^{2}\\ell-\\frac{1-\\cos^{2}\\phi\_{12}}{2}(\\|{\\mathbf{g\_{1}}}-{\\mathbf{g\_{2}}}\\|\_{2}^{2})L}
```
*   •

```
−2​‖𝐠𝟏‖2​‖𝐠𝟐‖2‖𝐠𝟏‖22+‖𝐠𝟐‖22\=−Φ⁡(𝐠𝟏,𝐠𝟐)≤cos⁡ϕ12<0-\\frac{2\\|{\\mathbf{g\_{1}}}\\|\_{2}\\|{\\mathbf{g\_{2}}}\\|\_{2}}{\\|{\\mathbf{g\_{1}}}\\|\_{2}^{2}+\\|{\\mathbf{g\_{2}}}\\|\_{2}^{2}}=-\\Phi({\\mathbf{g\_{1}}},{\\mathbf{g\_{2}}})\\leq\\cos\\phi\_{12}<0,
ℓ≤(1−cos2⁡ϕ12)​(‖𝐠𝟏‖22+‖𝐠𝟐‖22)‖𝐠𝟏+𝐠𝟐‖22​L\=ξ⁡(𝐠𝟏,𝐠𝟐)\\ell\\leq\\frac{(1-\\cos^{2}\\phi\_{12})(\\|{\\mathbf{g\_{1}}}\\|\_{2}^{2}+\\|{\\mathbf{g\_{2}}}\\|\_{2}^{2})}{\\|{\\mathbf{g\_{1}}}+{\\mathbf{g\_{2}}}\\|\_{2}^{2}}L=\\xi({\\mathbf{g\_{1}}},{\\mathbf{g\_{2}}}),
0<t≤(∥𝐠𝟏∥22+∥𝐠𝟐∥22)cos2ϕ12+2∥𝐠𝟏∥2∥𝐠𝟐∥2cosϕ1212​‖𝐠𝟏+𝐠𝟐‖22​ℓ−1−cos2⁡ϕ122​(‖𝐠𝟏−𝐠𝟐‖22)​L0\<t\\leq\\frac{(\\|{\\mathbf{g\_{1}}}\\|^{2}\_{2}+\\|{\\mathbf{g\_{2}}}\\|^{2}\_{2})\\cos^{2}\\phi\_{12}+2\\|{\\mathbf{g\_{1}}}\\|\_{2}\\|{\\mathbf{g\_{2}}}\\|\_{2}\\cos\\phi\_{12}}{\\frac{1}{2}\\|{\\mathbf{g\_{1}}}+{\\mathbf{g\_{2}}}\\|\_{2}^{2}\\ell-\\frac{1-\\cos^{2}\\phi\_{12}}{2}(\\|{\\mathbf{g\_{1}}}-{\\mathbf{g\_{2}}}\\|\_{2}^{2})L}.
```
The sufficient conditions hold as we can plug the conditions to RHS of Equation [6](#A1.E6 "Equation 6 ‣ Proof. ‣ A.3 PCGrad: Sufficient and Necessary Conditions for Loss Improvement ‣ Appendix A Proofs ‣ Gradient Surgery for Multi-Task Learning") and achieve non-negative result. ∎

### A.4 Convergence of PCGrad with Momentum-Based Gradient Descent

In this subsection, we show convergence of PCGrad with momentum-based methods, which is more aligned with our practical implementation. In our analysis, we consider the heavy ball method \[[43](#bib.bib43 "")\] as follows:

θk+1←θk−αk∇ℒ(θk)+βk(θk−θk−1)\\theta\_{k+1}\\leftarrow\\theta\_{k}-\\alpha\_{k}\\nabla\\mathcal{L}(\\theta\_{k})+\\beta\_{k}(\\theta\_{k}-\\theta\_{k-1})

where kk denotes the kk-th step and αk\\alpha\_{k} and βk\\beta\_{k} are step sizes for the gradient and momentum at step kk respectively. We now present our theorem.

###### Theorem 4.

Assume ℒ1\\mathcal{L}\_{1} and ℒ2\\mathcal{L}\_{2} are μ1\\mu\_{1}\- and μ2\\mu\_{2}-strongly convex and also L1L\_{1}\- and L2L\_{2}-smooth respectively where μ1,μ2,L1,L2\>0\\mu\_{1},\\mu\_{2},L\_{1},L\_{2}>0. Define ϕ12k\\phi^{k}\_{12} as the angle between two task gradients 𝐠𝟏​(θk){\\mathbf{g\_{1}}}{(\\theta\_{k})} and 𝐠𝟐​(θk){\\mathbf{g\_{2}}}{(\\theta\_{k})} and define Rk\=‖𝐠𝟏​(θk)‖‖𝐠𝟐​(θk)‖R\_{k}=\\frac{\\|{\\mathbf{g\_{1}}}(\\theta\_{k})\\|}{\\|{\\mathbf{g\_{2}}}(\\theta\_{k})\\|}. Denote μk\=(1−cos⁡ϕ12k/Rk)​μ1+(1−cos⁡ϕ12k⋅Rk)​μ2\\mu\_{k}=(1-\\cos\\phi^{k}\_{12}/R\_{k})\\mu\_{1}+(1-\\cos\\phi^{k}\_{12}\\cdot R\_{k})\\mu\_{2} and Lk\=(1−cos⁡ϕ12k/Rk)​L1+(1−cos⁡ϕ12k⋅Rk)​L2L\_{k}=(1-\\cos\\phi^{k}\_{12}/R\_{k})L\_{1}+(1-\\cos\\phi^{k}\_{12}\\cdot R\_{k})L\_{2} Then, the PCGrad update rule of the heavy ball method with step sizes αk\=4Lk+μk\\alpha\_{k}=\\frac{4}{\\sqrt{L\_{k}}+\\sqrt{\\mu\_{k}}} and βk\=max⁡{|1−αk​μk|,|1−αk​Lk|}2\\beta\_{k}=\\max\\{|1-\\sqrt{\\alpha\_{k}\\mu\_{k}}|,|1-\\sqrt{\\alpha\_{k}L\_{k}}|\\}^{2} will converge linearly to either (1) a location in the optimization landscape where cos⁡(ϕ12k)\=−1\\cos(\\phi^{k}\_{12})=-1 or (2) the optimal value ℒ⁡(θ∗)\\mathcal{L}(\\theta^{\*}).

###### Proof.

We first observe that the PCGrad-modified gradient 𝐠PC{\\mathbf{g}}^{\\text{PC}} has the following identity:

𝐠PC\\displaystyle{\\mathbf{g}}^{\\text{PC}}

\=𝐠−𝐠𝟏⋅𝐠𝟐‖𝐠𝟏‖2​𝐠𝟏−𝐠𝟏⋅𝐠𝟐‖𝐠𝟐‖2​𝐠𝟐\\displaystyle={\\mathbf{g}}-\\frac{{\\mathbf{g\_{1}}}\\cdot{\\mathbf{g\_{2}}}}{||{\\mathbf{g\_{1}}}||^{2}}{\\mathbf{g\_{1}}}-\\frac{{\\mathbf{g\_{1}}}\\cdot{\\mathbf{g\_{2}}}}{||{\\mathbf{g\_{2}}}||^{2}}{\\mathbf{g\_{2}}}

\=(1−𝐠𝟏⋅𝐠𝟐‖𝐠𝟏‖2)​𝐠𝟏+(1−𝐠𝟏⋅𝐠𝟐‖𝐠𝟐‖2)​𝐠𝟐\\displaystyle=(1-\\frac{{\\mathbf{g\_{1}}}\\cdot{\\mathbf{g\_{2}}}}{||{\\mathbf{g\_{1}}}||^{2}}){\\mathbf{g\_{1}}}+(1-\\frac{{\\mathbf{g\_{1}}}\\cdot{\\mathbf{g\_{2}}}}{||{\\mathbf{g\_{2}}}||^{2}}){\\mathbf{g\_{2}}}

\=(1−𝐠𝟏⋅𝐠𝟐‖𝐠𝟏‖​‖𝐠𝟐‖​‖𝐠𝟐‖‖𝐠𝟏‖)​𝐠𝟏+(1−𝐠𝟏⋅𝐠𝟐‖𝐠𝟏‖​‖𝐠𝟐‖​‖𝐠𝟏‖‖𝐠𝟐‖)​𝐠𝟐\\displaystyle=(1-\\frac{{\\mathbf{g\_{1}}}\\cdot{\\mathbf{g\_{2}}}}{\\|{\\mathbf{g\_{1}}}\\|\\|{\\mathbf{g\_{2}}}\\|}\\frac{\\|{\\mathbf{g\_{2}}}\\|}{\\|{\\mathbf{g\_{1}}}\\|}){\\mathbf{g\_{1}}}+(1-\\frac{{\\mathbf{g\_{1}}}\\cdot{\\mathbf{g\_{2}}}}{\\|{\\mathbf{g\_{1}}}\\|\\|{\\mathbf{g\_{2}}}\\|}\\frac{\\|{\\mathbf{g\_{1}}}\\|}{\\|{\\mathbf{g\_{2}}}\\|}){\\mathbf{g\_{2}}}

\=(1−cos⁡ϕ12/R)​𝐠𝟏+(1−cos⁡ϕ12⋅R)​𝐠𝟐.\\displaystyle=(1-\\cos\\phi\_{12}/R){\\mathbf{g\_{1}}}+(1-\\cos\\phi\_{12}\\cdot R){\\mathbf{g\_{2}}}.

(7)

Applying Equation [7](#A1.E7 "Equation 7 ‣ Proof. ‣ A.4 Convergence of PCGrad with Momentum-Based Gradient Descent ‣ Appendix A Proofs ‣ Gradient Surgery for Multi-Task Learning"), we can write the PCGrad update rule of the heavy ball method in matrix form as follows:

‖\[θk+1−θ∗θk−θ∗\]‖2\\displaystyle\\left\\|\\begin{bmatrix}\\theta\_{k+1}-\\theta^{\*}\\\\ \\theta\_{k}-\\theta^{\*}\\end{bmatrix}\\right\\|\_{2}

\=‖\[θk+βk​(θk−θk−1)−θ∗θk−θ∗\]−αk​\[𝐠PC​(θk)0\]‖2\\displaystyle=\\left\\|\\begin{bmatrix}\\theta\_{k}+\\beta\_{k}(\\theta\_{k}-\\theta\_{k-1})-\\theta^{\*}\\\\ \\theta\_{k}-\\theta^{\*}\\end{bmatrix}-\\alpha\_{k}\\begin{bmatrix}{\\mathbf{g}}^{\\text{PC}}(\\theta\_{k})\\\\ 0\\end{bmatrix}\\right\\|\_{2}

\=‖\[θk+βk​(θk−θk−1)−θ∗θk−θ∗\]\\displaystyle=\\left\\|\\begin{bmatrix}\\theta\_{k}+\\beta\_{k}(\\theta\_{k}-\\theta\_{k-1})-\\theta^{\*}\\\\ \\theta\_{k}-\\theta^{\*}\\end{bmatrix}\\right.

−αk\[(1−cos⁡ϕ12k/Rk)​𝐠𝟏​(θk)+(1−cos⁡ϕ12k⋅Rk)​𝐠𝟐​(θk)0\]‖2\\displaystyle\\left.-\\alpha\_{k}\\begin{bmatrix}(1-\\cos\\phi^{k}\_{12}/R\_{k}){\\mathbf{g\_{1}}}(\\theta\_{k})+(1-\\cos\\phi^{k}\_{12}\\cdot R\_{k}){\\mathbf{g\_{2}}}(\\theta\_{k})\\\\ 0\\end{bmatrix}\\right\\|\_{2}

\=‖\[θk+βk​(θk−θk−1)−θ∗θk−θ∗\]\\displaystyle=\\left\\|\\begin{bmatrix}\\theta\_{k}+\\beta\_{k}(\\theta\_{k}-\\theta\_{k-1})-\\theta^{\*}\\\\ \\theta\_{k}-\\theta^{\*}\\end{bmatrix}\\right.

−αk\[\[(1−cos⁡ϕ12k/Rk)​∇2ℒ1​(zk)+(1−cos⁡ϕ12k⋅Rk)​∇2ℒ2​(zk′)\]​(θk−θ∗)0\]‖2\\displaystyle\\left.-\\alpha\_{k}\\begin{bmatrix}\\left\[(1-\\cos\\phi^{k}\_{12}/R\_{k})\\nabla^{2}\\mathcal{L}\_{1}(z\_{k})+(1-\\cos\\phi^{k}\_{12}\\cdot R\_{k})\\nabla^{2}\\mathcal{L}\_{2}(z^{\\prime}\_{k})\\right\](\\theta\_{k}-\\theta^{\*})\\\\ 0\\end{bmatrix}\\right\\|\_{2}

for some ​zk,zk′​ on the line segment between ​θk​ and ​θ∗\\displaystyle\\text{for some }z\_{k},z^{\\prime}\_{k}\\text{ on the line segment between }\\theta\_{k}\\text{ and }\\theta^{\*}

\=‖\[(1+βk)​I−αk​Hk−βk​II0\]​\[θk−θ∗θk−1−θ∗\]‖2\\displaystyle=\\left\\|\\begin{bmatrix}(1+\\beta\_{k})I-\\alpha\_{k}H\_{k}&-\\beta\_{k}I\\\\ I&0\\end{bmatrix}\\begin{bmatrix}\\theta\_{k}-\\theta^{\*}\\\\ \\theta\_{k-1}-\\theta^{\*}\\end{bmatrix}\\right\\|\_{2}

≤‖\[(1+βk)​I−αk​Hk−βk​II0\]‖2​‖\[θk−θ∗θk−1−θ∗\]‖2\\displaystyle\\leq\\left\\|\\begin{bmatrix}(1+\\beta\_{k})I-\\alpha\_{k}H\_{k}&-\\beta\_{k}I\\\\ I&0\\end{bmatrix}\\right\\|\_{2}\\left\\|\\begin{bmatrix}\\theta\_{k}-\\theta^{\*}\\\\ \\theta\_{k-1}-\\theta^{\*}\\end{bmatrix}\\right\\|\_{2}

where Hk\=(1−cos⁡ϕ12k/Rk)​∇2ℒ1​(zk)+(1−cos⁡ϕ12k⋅Rk)​∇2ℒ2​(zk′)H\_{k}=(1-\\cos\\phi^{k}\_{12}/R\_{k})\\nabla^{2}\\mathcal{L}\_{1}(z\_{k})+(1-\\cos\\phi^{k}\_{12}\\cdot R\_{k})\\nabla^{2}\\mathcal{L}\_{2}(z^{\\prime}\_{k}).

By strong convexity and smoothness of ℒ1\\mathcal{L}\_{1} and ℒ2\\mathcal{L}\_{2}, we have the eigenvalues of ∇2ℒ1​(zk)\\nabla^{2}\\mathcal{L}\_{1}(z\_{k}) are between μ1\\mu\_{1} and L1L\_{1}. Similarly, the eigenvalues of ∇2ℒ2​(zk′)\\nabla^{2}\\mathcal{L}\_{2}(z^{\\prime}\_{k}) are between μ2\\mu\_{2} and L2L\_{2}. Thus the eigenvalues of HkH\_{k} are between μk\=(1−cos⁡ϕ12k/Rk)​μ1+(1−cos⁡ϕ12k⋅Rk)​μ2\\mu\_{k}=(1-\\cos\\phi^{k}\_{12}/R\_{k})\\mu\_{1}+(1-\\cos\\phi^{k}\_{12}\\cdot R\_{k})\\mu\_{2} and Lk\=(1−cos⁡ϕ12k/Rk)​L1+(1−cos⁡ϕ12k⋅Rk)​L2L\_{k}=(1-\\cos\\phi^{k}\_{12}/R\_{k})L\_{1}+(1-\\cos\\phi^{k}\_{12}\\cdot R\_{k})L\_{2} \[[20](#bib.bib20 "")\]. Hence following Lemma 3.1 in  \[[54](#bib.bib54 "")\], we have

‖\[(1+βk)​I−αk​Hk−βk​II0\]‖2≤max⁡{|1−αk​μk|,|1−αk​Lk|}.\\left\\|\\begin{bmatrix}(1+\\beta\_{k})I-\\alpha\_{k}H\_{k}&-\\beta\_{k}I\\\\ I&0\\end{bmatrix}\\right\\|\_{2}\\leq\\max\\{|1-\\sqrt{\\alpha\_{k}\\mu\_{k}}|,|1-\\sqrt{\\alpha\_{k}L\_{k}}|\\}.

Thus we have

‖\[θk+1−θ∗θk−θ∗\]‖2\\displaystyle\\left\\|\\begin{bmatrix}\\theta\_{k+1}-\\theta^{\*}\\\\ \\theta\_{k}-\\theta^{\*}\\end{bmatrix}\\right\\|\_{2}

≤max⁡{|1−αk​μk|,|1−αk​Lk|}​‖\[θk−θ∗θk−1−θ∗\]‖2\\displaystyle\\leq\\max\\{|1-\\sqrt{\\alpha\_{k}\\mu\_{k}}|,|1-\\sqrt{\\alpha\_{k}L\_{k}}|\\}\\left\\|\\begin{bmatrix}\\theta\_{k}-\\theta^{\*}\\\\ \\theta\_{k-1}-\\theta^{\*}\\end{bmatrix}\\right\\|\_{2}

\=κk−1κk+1​‖\[θk−θ∗θk−1−θ∗\]‖2\\displaystyle=\\frac{\\sqrt{\\kappa\_{k}}-1}{\\sqrt{\\kappa\_{k}}+1}\\left\\|\\begin{bmatrix}\\theta\_{k}-\\theta^{\*}\\\\ \\theta\_{k-1}-\\theta^{\*}\\end{bmatrix}\\right\\|\_{2}

(8)

≤‖\[θk−θ∗θk−1−θ∗\]‖2\\displaystyle\\leq\\left\\|\\begin{bmatrix}\\theta\_{k}-\\theta^{\*}\\\\ \\theta\_{k-1}-\\theta^{\*}\\end{bmatrix}\\right\\|\_{2}

where κk\=Lkμk\\kappa\_{k}=\\frac{L\_{k}}{\\mu\_{k}} and Equation [8](#A1.E8 "Equation 8 ‣ Proof. ‣ A.4 Convergence of PCGrad with Momentum-Based Gradient Descent ‣ Appendix A Proofs ‣ Gradient Surgery for Multi-Task Learning") follows from substitution αk\=4Lk+μk\\alpha\_{k}=\\frac{4}{\\sqrt{L\_{k}}+\\sqrt{\\mu\_{k}}}. Hence PCGrad with heavy ball method converges linearly if cos⁡ϕ12k≠−1\\cos\\phi^{k}\_{12}\\neq-1. ∎

## Appendix B Empirical Objective-Wise Evaluations of PCGrad

In this section, we visualize the per-task training loss and validation loss curves respectively on NYUv2. The goal of measuring objective-wise performance is to study the convergence of PCGrad in practice, particularly amidst the possibility of slow convergence due to cosine similarities near -1, as discussed in Section [2.4](#S2.SS4 "2.4 Theoretical Analysis of PCGrad ‣ 2 Multi-Task Learning with PCGrad ‣ Gradient Surgery for Multi-Task Learning").

We show the objective-wise evaluation results on NYUv2 in Figure [5](#A2.F5 "Figure 5 ‣ Appendix B Empirical Objective-Wise Evaluations of PCGrad ‣ Gradient Surgery for Multi-Task Learning"). For evaluations on NYUv2, PCGrad + MTAN attains similar training convergence rate compared to MTAN in three tasks in NYUv2 while converging faster and achieving lower validation loss in 2 out of 3 tasks. Note that in task 0 of the NYUv2 dataset, both methods seem to overfit, suggesting a better regularization scheme for this domain.

In general, these results suggest that PCGrad has a regularization effect on supervised multi-task learning, rather than an improvement on optimization speed or convergence. We hypothesize that this regularization is caused by PCGrad leading to greater sharing of representations across tasks, such that the supervision for one task better regularizes the training of another. This regularization effect seems notably different from the effect of PCGrad on reinforcement learning problems, where PCGrad dramatically improves training performance. This suggests that multi-task supervised learning and multi-task reinforcement learning problems may have distinct challenges.

Figure 5: Empirical objective-wise evaluations on NYUv2. On the top row, we show the objective-wise training learning curves and on the bottom row, we show the objective-wise validation learning curves. PCGrad+MTAN converges with a similar rate compared to MTAN in training and for validation losses, PCGrad+MTAN converges faster and obtains a lower final validation loss in two out of three tasks. This result corroborate that in practice, PCGrad does not exhibit the potential slow convergence problem shown in Theorem [1](#Thmtheorem1 "Theorem 1. ‣ 2.4 Theoretical Analysis of PCGrad ‣ 2 Multi-Task Learning with PCGrad ‣ Gradient Surgery for Multi-Task Learning").

## Appendix C Practical Details of PCGrad on Multi-Task and Goal-Conditioned Reinforcement Learning

In our experiments, we apply PCGrad to the soft actor-critic (SAC) algorithm \[[24](#bib.bib24 "")\], an off-policy RL method. In SAC, we employ a Q-learning style gradient to compute the gradient of the Q-function network, Qϕ​(s,a,zi)Q\_{\\phi}(s,a,z\_{i}), often known as the critic, and a reparameterization-style gradient to compute the gradient of the policy network πθ​(a|s,zi)\\pi\_{\\theta}(a|s,z\_{i}), often known as the actor. For sampling, we instantiate a set of replay buffers {𝒟i}𝒯i∼p⁡(𝒯)\\{\\mathcal{D}\_{i}\\}\_{\\mathcal{T}\_{i}\\sim p(\\mathcal{T})}. Training and data collection are alternated throughout training. During a data collection step, we run the policy πθ\\pi\_{\\theta} on all the tasks 𝒯i∼p⁡(𝒯)\\mathcal{T}\_{i}\\sim p(\\mathcal{T}) to collect an equal number of paths for each task and store the paths of each task 𝒯i\\mathcal{T}\_{i} into the corresponding replay buffer 𝒟i\\mathcal{D}\_{i}. At each training step, we sample an equal amount of data from each replay buffer 𝒟i\\mathcal{D}\_{i} to form a stratified batch. For each task 𝒯i∼p⁡(𝒯)\\mathcal{T}\_{i}\\sim p(\\mathcal{T}), the parameters of the critic θ\\theta are optimized to minimize the soft Bellman residual:

JQ(i)​(ϕ)\\displaystyle J^{(i)}\_{Q}(\\phi)

\=𝔼(st,at,zi)∼𝒟i​\[Qϕ​(st,at,zi)−(r⁡(st,at,zi)+γ​Vϕ¯​(st+1,zi))\]​,\\displaystyle=\\mathbb{E}\_{(s\_{t},a\_{t},z\_{i})\\sim\\mathcal{D}\_{i}}\\left\[Q\_{\\phi}(s\_{t},a\_{t},z\_{i})-(r(s\_{t},a\_{t},z\_{i})+\\gamma V\_{\\bar{\\phi}}(s\_{t+1},z\_{i}))\\right\]\\text{,}

(9)

Vϕ¯​(st+1,zi)\\displaystyle V\_{\\bar{\\phi}}(s\_{t+1},z\_{i})

\=𝔼at+1∼πθ​\[Qϕ¯​(st+1,at+1,zi)−α​log⁡πθ​(at+1|st+1,zi)\]​,\\displaystyle=\\mathbb{E}\_{a\_{t+1}\\sim\\pi\_{\\theta}}\\left\[Q\_{\\bar{\\phi}}(s\_{t+1},a\_{t+1},z\_{i})-\\alpha\\log\\pi\_{\\theta}(a\_{t+1}|s\_{t+1},z\_{i})\\right\]\\text{,}

(10)

where γ\\gamma is the discount factor, ϕ¯\\bar{\\phi} are the delayed parameters, and α\\alpha is a learnable temperature that automatically adjusts the weight of the entropy term. For each task 𝒯i∼p⁡(𝒯)\\mathcal{T}\_{i}\\sim p(\\mathcal{T}), the parameters of the policy πθ\\pi\_{\\theta} are trained to minimize the following objective

Jπ(i)​(θ)\\displaystyle J^{(i)}\_{\\pi}(\\theta)

\=𝔼st∼𝒟i​\[𝔼OPENat∼πθ​(at|st,zi))​\[α​log⁡πθ​(at|st,zi)−Qϕ​(st,at,zi)\]\]​.\\displaystyle=\\mathbb{E}\_{s\_{t}\\sim\\mathcal{D}\_{i}}\\left\[\\mathbb{E}\_{a\_{t}\\sim\\pi\_{\\theta}(a\_{t}|s\_{t},z\_{i}))}\\left\[\\alpha\\log\\pi\_{\\theta}(a\_{t}|s\_{t},z\_{i})-Q\_{\\phi}(s\_{t},a\_{t},z\_{i})\\right\]\\right\]\\text{.}

(11)

We compute ∇ϕJQ(i)​(ϕ)\\nabla\_{\\phi}J^{(i)}\_{Q}(\\phi) and ∇θJπ(i)​(θ)\\nabla\_{\\theta}J^{(i)}\_{\\pi}(\\theta) for all 𝒯i∼p⁡(𝒯)\\mathcal{T}\_{i}\\sim p(\\mathcal{T}) and apply PCGrad to both following Algorithm [1](#alg1 "Algorithm 1 ‣ 2.3 ​​​PCGrad: Project Conflicting Gradients ‣ 2 Multi-Task Learning with PCGrad ‣ Gradient Surgery for Multi-Task Learning").

In the context of SAC specifically, we also propose to learn the temperature α\\alpha for adjusting entropy of the policy on a per-task basis. This allows the method to control the entropy of the multi-task policy per-task. The motivation is that if we use a single learnable temperature for adjusting entropy of the multi-task policy πθ​(a|s,zi)\\pi\_{\\theta}(a|s,z\_{i}), SAC may stop exploring once all easier tasks are solved, leading to poor performance on tasks that are harder or require more exploration. To address this issue, we propose to learn the temperature on a per-task basis as mentioned in Section [3](#S3 "3 PCGrad in Practice ‣ Gradient Surgery for Multi-Task Learning"), i.e. using a parametrized model to represent αψ​(zi)\\alpha\_{\\psi}(z\_{i}). This allows the method to control the entropy of πθ​(a|s,zi)\\pi\_{\\theta}(a|s,z\_{i}) per-task. We optimize the parameters of αψ​(zi)\\alpha\_{\\psi}(z\_{i}) using the same constrained optimization framework as in \[[24](#bib.bib24 "")\].

When applying PCGrad to goal-conditioned RL, we represent p⁡(𝒯)p(\\mathcal{T}) as a distribution of goals and let ziz\_{i} be the encoding of a goal. Similar to the multi-task supervised learning setting discussed in Section [3](#S3 "3 PCGrad in Practice ‣ Gradient Surgery for Multi-Task Learning"), PCGrad may be combined with various architectures designed for multi-task and goal-conditioned RL \[[19](#bib.bib19 ""), [14](#bib.bib14 "")\], where PCGrad operates on the gradients of shared parameters, leaving task-specific parameters untouched.

## Appendix D 2D Optimization Landscape Details

To produce the 2D optimization visualizations in Figure [1](#S1.F1 "Figure 1 ‣ 1 Introduction ‣ Gradient Surgery for Multi-Task Learning"), we used a parameter vector θ\=\[θ1,θ2\]∈ℝ2\\theta=\[\\theta\_{1},\\theta\_{2}\]\\in\\mathbbm{R}^{2} and the following task loss functions:

ℒ1​(θ)\=20​log⁡(max⁡(|.5​θ1+tanh⁡(θ2)|,0.000005))\\displaystyle\\mathcal{L}\_{1}(\\theta)=20\\log(\\max(|.5\\theta\_{1}+\\tanh(\\theta\_{2})|,0.000005))

ℒ2​(θ)\=25​log⁡(max⁡(|.5​θ1−tanh⁡(θ2)+2|,0.000005))\\displaystyle\\mathcal{L}\_{2}(\\theta)=25\\log(\\max(|.5\\theta\_{1}-\\tanh(\\theta\_{2})+2|,0.000005))

The multi-task objective is ℒ⁡(θ)\=ℒ1​(θ)+ℒ2​(θ)\\mathcal{L}(\\theta)=\\mathcal{L}\_{1}(\\theta)+\\mathcal{L}\_{2}(\\theta). We initialized θ\=\[0.5,−3\]\\theta=\[0.5,-3\] and performed 500,000 gradient updates to minimize ℒ\\mathcal{L} using the Adam optimizer with learning rate 0.0010.001. We compared using Adam for each update to using Adam in conjunction with the PCGrad method presented in Section [2.3](#S2.SS3 "2.3 ​​​PCGrad: Project Conflicting Gradients ‣ 2 Multi-Task Learning with PCGrad ‣ Gradient Surgery for Multi-Task Learning").

## Appendix E Additional Multi-Task Supervised Learning Results

We present our multi-task supervised learning results on MultiMNIST and CityScapes here.

#### MultiMNIST.

Following the same set-up of [Sener and Koltun \[53\]](#bib.bib53 ""), for each image, we sample a different one uniformly at random. Then we put one of the image on the top left and the other one on the bottom right. The two tasks in the multi-task learning problem are to classify the digits on the top left (task-L) and bottom right (task-R) respectively. We construct such 6060K examples. We combine PCGrad with the same backbone architecture used in \[[53](#bib.bib53 "")\] and compare its performance to [Sener and Koltun \[53\]](#bib.bib53 "") by running the open-sourced code provided in \[[53](#bib.bib53 "")\]. As shown in Table [4](#A5.T4 "Table 4 ‣ MultiMNIST. ‣ Appendix E Additional Multi-Task Supervised Learning Results ‣ Gradient Surgery for Multi-Task Learning"), PCGrad results 0.13% and 0.55% improvement over \[[53](#bib.bib53 "")\] in left and right digit accuracy respectively.

|       | left digit | right digit |
| ----- | ---------- | ----------- |
| 96.45 | 94.95      |             |
| 96.58 | 95.50      |             |

Table 4: MultiMNIST results. PCGrad achieves improvements over the approach by [Sener and Koltun \[53\]](#bib.bib53 "") in both left and right digit classfication accuracy.

#### CityScapes.

The CityScapes dataset \[[12](#bib.bib12 "")\] contains 19 classes of street-view images resized to 128×256128\\times 256. There are two tasks in this dataset: semantic segmentation and depth estimation. Following the setup in [Liu et al. \[33\]](#bib.bib33 ""), we pair the depth estimation task with semantic segmentation using the coarser 7 categories instead of the finer 19 classes in the original CityScapes dataset. Similar to NYUv2 evaluations described in Section [5](#S5 "5 Experiments ‣ Gradient Surgery for Multi-Task Learning"), we also combine PCGrad with MTAN \[[33](#bib.bib33 "")\] and compare it to a range of methods discussed in Appendix [J.1](#A10.SS1 "J.1 Multi-Task Supervised Learning Experiment Details ‣ Appendix J Experiment Details ‣ Gradient Surgery for Multi-Task Learning"). For the combination of PCGrad and MTAN, we only use equal weighting as discussed in \[[33](#bib.bib33 "")\] as we find it working well in practice. We present the results in Table [5](#A5.T5 "Table 5 ‣ CityScapes. ‣ Appendix E Additional Multi-Task Supervised Learning Results ‣ Gradient Surgery for Multi-Task Learning"). As shown in Table [5](#A5.T5 "Table 5 ‣ CityScapes. ‣ Appendix E Additional Multi-Task Supervised Learning Results ‣ Gradient Surgery for Multi-Task Learning"), PCGrad + MTAN outperforms MTAN in three out of four scores while obtaining the top scores in both mIoU abd pixel accuracy for the semantic segmentation task, suggesting the effectiveness of PCGrad on realistic image datasets. We also provide the full results including three different weighting schemes in Table [7](#A10.T7 "Table 7 ‣ J.4 Full CityScapes and NYUv2 Results ‣ Appendix J Experiment Details ‣ Gradient Surgery for Multi-Task Learning") in Appendix [J.4](#A10.SS4 "J.4 Full CityScapes and NYUv2 Results ‣ Appendix J Experiment Details ‣ Gradient Surgery for Multi-Task Learning").

#P.

Architecture

Segmentation

Depth

(Higher Better)

(Lower Better)

mIoU

Pix Acc

Abs Err

Rel Err

2

One Task

51.09

90.69

0.0158

34.17

3.04

STAN

51.90

90.87

0.0145

27.46

1.75

Split, Wide

50.17

90.63

0.0167

44.73

2

Split, Deep

49.85

88.69

0.0180

43.86

3.63

Dense

51.91

90.89

0.0138

27.21

≈\\approx2

Cross-Stitch \[[39](#bib.bib39 "")\]

50.08

90.33

0.0154

34.49

1.65

MTAN

53.04

91.11

0.0144

33.63

1.65

PCGrad+MTAN (Ours)

53.59

91.45

0.0171

31.34

Table 5: We present the 7-class semantic segmentation and depth estimation results on CityScapes dataset. We use #P to denote the number of parameters of the network. We use box and bold text to highlight the method that achieves the best validation score for each task. As seen in the results, PCGrad+MTAN with equal weights outperforms MTAN with equal weights in three out of four scores while achieving the top score both scores in the segmentation task.

## Appendix F Goal-Conditioned Reinforcement Learning Results

For our goal-conditioned RL evaluation, we adopt the goal-conditioned robotic pushing task with a Sawyer robot where the goals are represented as the concatenations of the initial positions of the puck to be pushed and the its goal location, both of which are uniformly sampled (details in Appendix [J.3](#A10.SS3 "J.3 Goal-conditioned Experiment Details ‣ Appendix J Experiment Details ‣ Gradient Surgery for Multi-Task Learning")). We also apply the temperature adjustment strategy as discussed in Section [3](#S3 "3 PCGrad in Practice ‣ Gradient Surgery for Multi-Task Learning") to predict the temperature for entropy term given the goal. We summarize the results in Figure [6](#A6.F6 "Figure 6 ‣ Appendix F Goal-Conditioned Reinforcement Learning Results ‣ Gradient Surgery for Multi-Task Learning"). PCGrad with SAC achieves better performance in terms of average distance to the goal position, while the vanilla SAC agent is struggling to successfully accomplish the task. This suggests that PCGrad is able to ease the RL optimization problem also when the task distribution is continuous.

![Refer to caption](2001.06782v4/figures/PCGrad_push_random_goals_init_hardest_itr600_3_random_seed_nopa_legend.png)

Figure 6: We present the goal-conditioned RL results. PCGrad outperforms vanilla SAC in terms of both average distance the goal and data efficiency.

## Appendix G Comparison to CosReg

We compare PCGrad to a prior method CosReg \[[55](#bib.bib55 "")\], which adds a regularization term to force the cosine similarity between gradients of two different tasks to stay 00. PCGrad achieves much better average success rate in MT10 benchmark as shown in Figure [7](#A7.F7 "Figure 7 ‣ Appendix G Comparison to CosReg ‣ Gradient Surgery for Multi-Task Learning"). Hence, while it’s important to reduce interference between tasks, it’s also crucial to keep the task gradients that enjoy positive cosine similarities in order to ensure sharing across tasks.

![Refer to caption](2001.06782v4/figures/PCGrad_mt10_itr900_pcgrad_ablation_nopa_cosine.png)

Figure 7: Comparison between PCGrad and CosReg \[[55](#bib.bib55 "")\]. PCGrad outperforms CosReg, suggesting that we should both reduce the interference and keep shared structure across tasks.

## Appendix H Ablation study on the task order

As stated on line 4 in Algorithm [1](#alg1 "Algorithm 1 ‣ 2.3 ​​​PCGrad: Project Conflicting Gradients ‣ 2 Multi-Task Learning with PCGrad ‣ Gradient Surgery for Multi-Task Learning"), we sample the tasks from the batch and randomly shuffle the order of the tasks before performing the update steps in PCGrad. With random shuffling, we make PCGrad symmetric w.r.t. the task order in expectation. In Figure [8](#A8.F8 "Figure 8 ‣ Appendix H Ablation study on the task order ‣ Gradient Surgery for Multi-Task Learning"), we observe that PCGrad with a random task order achieves better performance between PCGrad with a fixed task order in the setting of MT50 where the number of tasks is large and the conflicting gradient phenomenon is much more likely to happen.

![Refer to caption](2001.06782v4/figures/PCGrad_mt50_itr400_random_order_nopa_ablation.png)

Figure 8: Ablation study on using a fixed task order during PCGrad. PCGrad with a random task order does significantly better PCGrad with a fixed task order in MT50 benchmark.

## Appendix I Combining PCGrad with other architectures

In this subsection, we test whether PCGrad can improve performances when combined with more methods. In Table [6](#A9.T6 "Table 6 ‣ Appendix I Combining PCGrad with other architectures ‣ Gradient Surgery for Multi-Task Learning"), we find that PCGrad does improve the performance in all four metrics of the three tasks on the NYUv2 dataset when combined with Cross-Stitch \[[39](#bib.bib39 "")\] and Dense. In Figure [9](#A9.F9 "Figure 9 ‣ Appendix I Combining PCGrad with other architectures ‣ Gradient Surgery for Multi-Task Learning"), we also show that PCGrad + Multi-head SAC outperforms Multi-head SAC on its own. These results suggest that PCGrad can be flexibly combined with any multi-task learning architectures to further improve performance.

| Method                | Segmentation | Depth  | Surface Normal |       |
| --------------------- | ------------ | ------ | -------------- | ----- |
| Cross-Stitch          | 15.69        | 0.6277 | 32.69          | 21.63 |
| Cross-Stitch + PCGrad | 18.14        | 0.5805 | 31.38          | 21.75 |
| Dense                 | 16.48        | 0.6282 | 31.68          | 21.73 |
| Dense + PCGrad        | 18.08        | 0.5850 | 30.17          | 23.29 |

Table 6: We show the performances of PCGrad combined with other methods on three-task learning on the NYUv2 dataset, where PCGrad further improves the results of prior multi-task learning architectures. 

![Refer to caption](2001.06782v4/figures/PCGrad_mt10_itr900_multihead_pcgrad.png)  

Figure 9: We show the comparison between Multi-head SAC and Multi-head SAC + PCGrad on MT10. Multi-head SAC + PCGrad outperforms Multi-head SAC, suggesting that PCGrad can improves the performance of multi-headed architectures in the multi-task RL settings.

## Appendix J Experiment Details

### J.1 Multi-Task Supervised Learning Experiment Details

For all the multi-task supervised learning experiments, PCGrad converges within 12 hours on a NVIDIA TITAN RTX GPU while the vanilla models without PCGrad converge within 8 hours. PCGrad consumes at most 10 GB memory on GPU while the vanilla method consumes 6GB on GPU among all experiments.

For our CIFAR-100 multi-task experiment, we adopt the architecture used in \[[47](#bib.bib47 "")\], which is a convolutional neural network that consists of 3 convolutional layers with 160160 3×33\\times 3 filters each layer and 2 fully connected layers with 320320 hidden units. As for experiments on the NYUv2 dataset, we follow \[[33](#bib.bib33 "")\] to use SegNet \[[1](#bib.bib1 "")\] as the backbone architecture.

We use five algorithms as baselines in the CIFAR-100 multi-task experiment: task specific-1-fc \[[46](#bib.bib46 "")\]: a convolutional neural network shared across tasks except that each task has a separate last fully-connected layer, task specific-1-fc \[[46](#bib.bib46 "")\] : all the convolutional layers shared across tasks with separate fully-connected layers for each task, cross stitch-all-fc \[[40](#bib.bib40 "")\]: one convolutional neural network per task along with cross-stitch units to share features across tasks, routing-all-fc + WPL \[[47](#bib.bib47 "")\]: a network that employs a trainable router trained with multi-agent RL algorithm (WPL) to select trainable functions for each task, independent: training separate neural networks for each task.

For comparisons on the NYUv2 dataset, we consider 5 baselines: Single Task, One Task: the vanilla SegNet used for single-task training, Single Task, STAN \[[33](#bib.bib33 "")\]: the single-task version of MTAN as mentioned below, Multi-Task, Split, Wide / Deep \[[33](#bib.bib33 "")\]: the standard SegNet shared for all three tasks except that each task has a separate last layer for final task-specific prediction with two variants Wide and Deep specified in  \[[33](#bib.bib33 "")\], Multi-Task Dense: a shared network followed by separate task-specific networks, Multi-Task Cross-Stitch \[[40](#bib.bib40 "")\]: similar to the baseline used in CIFAR-100 experiment but with SegNet as the backbone, MTAN \[[33](#bib.bib33 "")\]: a shared network with a soft-attention module for each task.

![Refer to caption](2001.06782v4/figures/MT50.png)

Figure 10: The 5050 tasks of MT50 from Meta-World \[[61](#bib.bib61 "")\]. MT10 is a subset of these tasks, which includes reach, push, pick & place, open drawer, close drawer, open door, press button top, open window, close window, and insert peg inside. 

### J.2 Multi-Task Reinforcement Learning Experiment Details

Our reinforcement learning experiments all use the SAC \[[24](#bib.bib24 "")\] algorithm as the base algorithm, where the actor and the critic are represented as 6-layer fully-connected feedforward neural networks for all methods. The numbers of hidden units of each layer of the neural networks are 160160, 300300 and 200200 for MT10, MT50 and goal-conditioned RL respectively. For the multi-task RL experiments, PCGrad + SAC converges in 1 day (5M simulation steps) and 5 days (20M simulation steps) on the MT10 and MT50 benchmarks respectively on a NVIDIA TITAN RTX GPU while vanilla SAC converges in 12 hours and 3 days on the two benchmarks respectively. PCGrad + SAC consumes 1 GB and 6 GB memory on GPU on the MT10 and MT50 benchmarks respectively while the vanilla SAC consumes 0.5 GB and 3 GB respectively.

In the case of multi-task reinforcement learning, we evaluate our algorithm on the recently proposed Meta-World benchmark \[[61](#bib.bib61 "")\]. This benchmark includes a variety of simulated robotic manipulation tasks contained in a shared, table-top environment with a simulated Sawyer arm (visualized in Fig. [10](#A10.F10 "Figure 10 ‣ J.1 Multi-Task Supervised Learning Experiment Details ‣ Appendix J Experiment Details ‣ Gradient Surgery for Multi-Task Learning")). In particular, we use the multi-task benchmarks MT10 and MT50, which consists of the 10 tasks and 50 tasks respectively depicted in Fig. [10](#A10.F10 "Figure 10 ‣ J.1 Multi-Task Supervised Learning Experiment Details ‣ Appendix J Experiment Details ‣ Gradient Surgery for Multi-Task Learning") that require diverse strategies to solve them, which makes them difficult to optimize jointly with a single policy. Note that MT10 is a subset of MT50. At each data collection step, we collect 600600 samples for each task, and at each training step, we sample 128128 datapoints per task from corresponding replay buffers. We measure success according to the metrics used in the Meta-World benchmark where the reported the success rates are averaged across tasks. For all methods, we apply the temperature adjustment strategy as discussed in Section [3](#S3 "3 PCGrad in Practice ‣ Gradient Surgery for Multi-Task Learning") to learn a separate alpha term per task as the task encoding in MT10 and MT50 is just a one-hot encoding.

On the multi-task and goal-conditioned RL domain, we apply PCGrad to the vanilla SAC algorithm with task encoding as part of the input to the actor and the critic as described in Section [3](#S3 "3 PCGrad in Practice ‣ Gradient Surgery for Multi-Task Learning") and compare PCGrad to the vanilla SAC without PCGrad and training actors and critics for each task individually (Independent).

### J.3 Goal-conditioned Experiment Details

We use the pushing environment from the Meta-World benchmark \[[61](#bib.bib61 "")\] as shown in Figure [10](#A10.F10 "Figure 10 ‣ J.1 Multi-Task Supervised Learning Experiment Details ‣ Appendix J Experiment Details ‣ Gradient Surgery for Multi-Task Learning"). In this environment, the table spans from \[−0.4,0.2\]\[-0.4,0.2\] to \[0.4,1.0\]\[0.4,1.0\] in the 2D space. To construct the goals, we sample the intial positions of the puck from the range \[−0.2,0.6\]\[-0.2,0.6\] to \[0.2,0.7\]\[0.2,0.7\] on the table and the goal positions from the range \[−0.2,0.85\]\[-0.2,0.85\] to \[0.2,0.95\]\[0.2,0.95\] on the table. The goal is represented as a concatenation of the initial puck position and the goal position. Since in the goal-conditioned setting, the task distribution is continuous, we sample a minibatch of 99 goals and 128128 samples per goal at each training iteration and also sample 600600 samples per goal in the minibatch at each data collection step.

### J.4 Full CityScapes and NYUv2 Results

We provide the full comparison on the CityScapes and NYUv2 datasets in Table [7](#A10.T7 "Table 7 ‣ J.4 Full CityScapes and NYUv2 Results ‣ Appendix J Experiment Details ‣ Gradient Surgery for Multi-Task Learning") and Table [8](#A10.T8 "Table 8 ‣ J.4 Full CityScapes and NYUv2 Results ‣ Appendix J Experiment Details ‣ Gradient Surgery for Multi-Task Learning") respectively.

#P.

Architecture

Weighting

Segmentation

Depth

(Higher Better)

(Lower Better)

mIoU

Pix Acc

Abs Err

Rel Err

2

One Task

n.a.

51.09

90.69

0.0158

34.17

3.04

STAN

n.a.

51.90

90.87

0.0145

27.46

Equal Weights

50.17

90.63

0.0167

44.73

1.75

Split, Wide

Uncert. Weights \[[28](#bib.bib28 "")\]

51.21

90.72

0.0158

44.01

DWA, T\=2T=2

50.39

90.45

0.0164

43.93

Equal Weights

49.85

88.69

0.0180

43.86

2

Split, Deep

Uncert. Weights \[[28](#bib.bib28 "")\]

48.12

88.68

0.0169

39.73

DWA, T\=2T=2

49.67

88.81

0.0182

46.63

Equal Weights

51.91

90.89

0.0138

27.21

3.63

Dense

Uncert. Weights \[[28](#bib.bib28 "")\]

51.89

91.22

0.0134

25.36

DWA, T\=2T=2

51.78

90.88

0.0137

26.67

Equal Weights

50.08

90.33

0.0154

34.49

≈\\approx2

Cross-Stitch \[[39](#bib.bib39 "")\]

Uncert. Weights \[[28](#bib.bib28 "")\]

50.31

90.43

0.0152

31.36

DWA, T\=2T=2

50.33

90.55

0.0153

33.37

Equal Weights

53.04

91.11

0.0144

33.63

1.65

MTAN

Uncert. Weights \[[28](#bib.bib28 "")\]

53.86

91.10

0.0144

35.72

DWA, T\=2T=2

53.29

91.09

0.0144

34.14

1.65

PCGrad+MTAN (Ours)

Equal Weights

53.59

91.45

0.0171

31.34

Table 7: We present the 7-class semantic segmentation and depth estimation results on CityScapes dataset. We use #P to denote the number of parameters of the network, and the best performing variant of each architecture is highlighted in bold. We use box to highlight the method that achieves the best validation score for each task. As seen in the results, PCGrad+MTAN with equal weights outperforms MTAN with equal weights in three out of four scores while achieving the top score in pixel accuracy across all methods.

Type

#P.

Architecture

Weighting

Segmentation

Depth

Surface Normal

(Higher Better)

(Lower Better)

Angle Distance

Within t∘t^{\\circ}

(Lower Better)

(Higher Better)

mIoU

Pix Acc

Abs Err

Rel Err

Mean

Median

11.25

22.5

30

Single Task

3

One Task

n.a.

15.10

51.54

0.7508

0.3266

31.76

25.51

22.12

45.33

57.13

4.56

STAN†

n.a.

15.73

52.89

0.6935

0.2891

32.09

26.32

21.49

44.38

56.51

Multi Task

Equal Weights

15.89

51.19

0.6494

0.2804

33.69

28.91

18.54

39.91

52.02

1.75

Split, Wide

Uncert. Weights∗

15.86

51.12

0.6040

0.2570

32.33

26.62

21.68

43.59

55.36

DWA†, T\=2T=2

16.92

53.72

0.6125

0.2546

32.34

27.10

20.69

42.73

54.74

Equal Weights

13.03

41.47

0.7836

0.3326

38.28

36.55

9.50

27.11

39.63

2

Split, Deep

Uncert. Weights∗

14.53

43.69

0.7705

0.3340

35.14

32.13

14.69

34.52

46.94

DWA†, T\=2T=2

13.63

44.41

0.7581

0.3227

36.41

34.12

12.82

31.12

43.48

Equal Weights

16.06

52.73

0.6488

0.2871

33.58

28.01

20.07

41.50

53.35

4.95

Dense

Uncert. Weights∗

16.48

54.40

0.6282

0.2761

31.68

25.68

21.73

44.58

56.65

DWA†, T\=2T=2

16.15

54.35

0.6059

0.2593

32.44

27.40

20.53

42.76

54.27

Equal Weights

14.71

50.23

0.6481

0.2871

33.56

28.58

20.08

40.54

51.97

≈\\approx3

Cross-Stitch‡

Uncert. Weights∗

15.69

52.60

0.6277

0.2702

32.69

27.26

21.63

42.84

54.45

DWA†, T\=2T=2

16.11

53.19

0.5922

0.2611

32.34

26.91

21.81

43.14

54.92

Equal Weights

17.72

55.32

0.5906

0.2577

31.44

25.37

23.17

45.65

57.48

1.77

MTAN†

Uncert. Weights∗

17.67

55.61

0.5927

0.2592

31.25

25.57

22.99

45.83

57.67

DWA†, T\=2T=2

17.15

54.97

0.5956

0.2569

31.60

25.46

22.48

44.86

57.24

1.77

MTAN† + PCGrad (ours)

Uncert. Weights∗

20.17

56.65

0.5904

0.2467

30.01

24.83

22.28

46.12

58.77

Table 8: We present the full results on three tasks on the NYUv2 dataset: 13-class semantic segmentation, depth estimation, and surface normal prediction results. #P shows the total number of network parameters. We highlight the best performing combination of multi-task architecture and weighting in bold. The top validation scores for each task are annotated with boxes. The symbols indicate prior methods: ∗: \[[28](#bib.bib28 "")\], †: \[[33](#bib.bib33 "")\], ‡: \[[40](#bib.bib40 "")\]. Performance of other methods taken from \[[33](#bib.bib33 "")\]. 

[](javascript:toggleReadingMode\(\); "Disable reading mode, show header and footer")