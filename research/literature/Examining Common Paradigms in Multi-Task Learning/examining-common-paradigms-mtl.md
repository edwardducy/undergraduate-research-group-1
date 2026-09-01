# Examining Common Paradigms in Multi-Task Learning

Cathrin Elich 1 , 2 3 , Lukas Kirchdorfer 1 4 , Jan M. K¨ohler ∗ , 1 , and Lukas Schott ∗ , 1

<sup>1</sup> Bosch Center for Artificial Intelligence

<sup>2</sup> Max Planck Institute for Intelligent Systems, T¨ubingen, Germany <sup>3</sup> Max Planck ETH Center for Learning Systems <sup>4</sup> University of Mannheim

cathrin.elich@tuebingen.mpg.de, {jan.koehler,lukas.schott }@bosch.com ‡Work done during an internship at Bosch. ∗ Joint senior authors.

Abstract. While multi-task learning (MTL) has gained significant attention in recent years, its underlying mechanisms remain poorly understood. Recent methods did not yield consistent performance improvements over single task learning (STL) baselines, underscoring the importance of gaining more profound insights about challenges specific to MTL. In our study, we investigate paradigms in MTL in the context of STL: First, the impact of the choice of optimizer has only been mildly investigated in MTL. We show the pivotal role of common STL tools such as the Adam optimizer in MTL empirically in various experiments. To further investigate Adam's effectiveness, we theoretical derive a partial loss-scale invariance under mild assumptions. Second, the notion of gradient conflicts has often been phrased as a specific problem in MTL. We explore the role of gradient conflicts in MTL and compare it to STL. For angular gradient alignment we find no evidence that this is a unique problem in MTL. We emphasize differences in gradient magnitude as the main distinguishing factor. Overall, we find surprising similarities between STL and MTL suggesting to consider methods from both fields in a broader context.

Keywords: Multi-task learning · Deep Learning · Computer Vision

### 1 Introduction

Multi-task learning (MTL) is gaining significance in the deep learning literature and in industry applications. Especially, tasks like autonomous driving and robotics necessitate real-time execution of neural networks while obeying constraints of limited computational resources. Consequently, there is a demand for neural networks capable of simultaneously inferring multiple tasks [\[19,](#page-13-0) [26\]](#page-14-0).

In a seminal study, Caruana [\[4\]](#page-12-0) highlights both advantages and challenges in MTL. On the one hand, certain tasks can exhibit a symbiotic relationship, resulting in a mutual performance enhancement when trained together. On the other hand, conflicts between tasks can arise and decrease the performance when trained jointly, also known as negative transfer .

Several approaches have been suggested to mitigate the issue of negative transfer among tasks during network training. Our study focuses on two main branches in the literature: First, gradient magnitude methods which incorporate weights to scale task-specific losses to achieve an adequate balance between tasks. Second, gradient alignment methods which aim to resolve conflicts in gradient vectors that may arise between tasks within a shared network backbone.

The effectiveness of the proposed MTL methods remain uncertain in the literature. Upon comparing various studies, it becomes evident that there is no definitive approach that consistently performs well across different settings [\[48\]](#page-15-0). This observation has been reinforced in more recent studies where competitive performance was achieved through plain unitary scaling in combination with common regularization methods [\[25\]](#page-14-1) or tuned task weighting [\[50\]](#page-15-1).

The current understanding of MTL still lacks a deeper comprehension of its underlying mechanisms. To address this gap, our study aims to examine commonly held paradigms, such as the choice of optimizer, as well as the notion of gradient alignment and gradient magnitudes. Our contributions are:

- The impact of off-the-shelf optimizers has received little attention in MTL benchmarks. We evaluate the Adam [\[22\]](#page-13-1) optimizer and demonstrate its favorable performance over SGD+momentum in various experiments.
- We provide a potential explanation for Adam's effectiveness in MTL by theoretically demonstrating a partial invariance w.r.t. to different loss scalings. Similarly, we derive a full invariance for an optimal variation of the wellestablished used method of uncertainty weighting [\[21\]](#page-13-2).
- So far gradient alignment conflicts have mostly been considered between different tasks [\[7,](#page-12-1) [20,](#page-13-3) [31,](#page-14-2) [56\]](#page-16-0). We present empirical evidence that conflicts arising from gradient alignment between tasks are not exclusive and can even be more pronounced between different samples within a task.
- Corroborating the methods proposed to balance gradient magnitude conflicts in MTL [\[21,](#page-13-2) [32,](#page-14-3) [34,](#page-14-4) [50\]](#page-15-1), we confirm that gradient magnitudes pose a challenge between tasks and is less pronounced between samples within a task.
- We examine the presumption of increased robustness on corrupted data as a result of MTL [\[24,](#page-13-4) [37\]](#page-14-5). We find light evidence that a higher number of tasks can result in improved transferability. Due to page limitations, we moved these results to App. [A6,](#page-35-0) focusing the more compelling findings in the main text.

Overall, we provide a vast set of experiments and theoretical insights which contribute to a more comprehensive understanding of MTL in computer vision.

# 2 Related Work

Work in multi-task learning (MTL) can be roughly divided into three fields:

Network architectures focus on the question of how features should be shared across tasks, e.g. [\[34,](#page-14-4) [36,](#page-14-6) [38,](#page-14-7) [51\]](#page-15-2). Multi-task optimization (MTO) aims to resolve imbalances and conflicts of tasks during MTL. Task affinities examine a grouping of tasks that should be learned together to benefit from the joint training [\[12,](#page-13-5)[47\]](#page-15-3). A general overview of recent works in MTL can be found in [\[43,](#page-15-4) [48\]](#page-15-0). Our work focuses on MTO, which we review more thoroughly in the following.

Gradient magnitude methods prevent the dominance of individual tasks by balancing them with task-specific weights. One line of works are loss-weighting methods. Here, weights are determined before any (task-wise) gradient computation and are used for a weighted aggregation of the tasks' losses. These methods consider either the task uncertainty (UW) [\[21\]](#page-13-2), rate of change of the losses (DWA, FAMO) [\[30,](#page-14-8) [34\]](#page-14-4), the tasks' difficulty (DTP) [\[14\]](#page-13-6), validation performance by applying meta-learning (MOML, Auto-λ) [\[33,](#page-14-9) [53\]](#page-15-5), or randomly chosen task weights (RLW) [\[28\]](#page-14-10). In line with these, the geometric mean of task losses has been used to handle the different convergence rates of the tasks [\[8\]](#page-12-2). An advantage of theses methods is their computational efficiency as the gradient needs to be computed only once for the aggregated loss. Alternatively, other methods consider the task-specific gradients directly, e.g., by normalizing them (Grad-Norm) [\[6\]](#page-12-3) or propose a hybrid balancing between task-wise loss and gradient scaling (IMTL, DB-MTL) [\[27,](#page-14-11) [32\]](#page-14-3). Furthermore, there are several adaptions for the multiple-gradient descent algorithm (MGDA) [\[10\]](#page-12-4), e.g. for applying it efficiently in deep learning setups [\[44\]](#page-15-6) or by introducing a stochastic gradient correction [\[11\]](#page-12-5). Recently, task-wise gradient weights have been estimated by treating MTL as a bargaining problem (Nash-MTL) [\[40\]](#page-14-12), or considering a stability criterion (Aligned-MTL) [\[45\]](#page-15-7). Crucially, all gradient magnitude methods consider scalar weightings of task-wise gradients within the backbone and/or heads. They do not modify the alignment of task-specific gradient vectors.

Gradient alignment methods perform more profound vector manipulations on the task-wise gradients w.r.t. to the network weights of a shared backbone before aggregating them. The underlying assumption indicates conflicting gradients as a major problem in MTL. To address this, GradDrop [\[7\]](#page-12-1) randomly drops gradient components in the case of opposing signs. PCGrad [\[56\]](#page-16-0) proposes to circumvent problems of conflicting gradients by projecting them onto each other's normal plane. Following this idea, Liu et al. [\[31\]](#page-14-2) propose CAGrad to converge to a minimum of the average loss instead of any point on the Pareto front. RotoGrad [\[20\]](#page-13-3) rotates gradients at the intersection of the heads and backbone to improve their alignment. Shi et al. [\[46\]](#page-15-8) propose to alter the network architecture based on the occurrence of layer-wise gradient conflicts. Lastly, [\[41\]](#page-15-9) use separate optimizers such as SGD and SGD+momentum per task. This is extended to AdaGrad, RMSProp and Adam in AdaTask [\[52\]](#page-15-10).

Recent studies question the effectiveness of optimization-based methods in MTL. Xin et al. [\[50\]](#page-15-1) execute an extensive hyperparameter search to show that simple scalar task-weighting performs equivalent or superior to many aforementioned multi-task optimization methods. Their hyperparameter search not only include the task-weights, but also common deep learning parameters such as the learning rate and regularization. Concurrently, Kurin et al. [\[25\]](#page-14-1) empirically show that fixed task-weights combined with regularization and stabilization techniques yield to equivalent performance compared to sophisticated multi-task optimization methods. Following these, Royer et al. [\[42\]](#page-15-11) examine the role of model capacity for MTL performance as well as the occurrences of gradient conflicts. We extend these critical studies. In particular, we theoretically and empirically demonstrate that the choice of optimizer is crucial and could potentially help to explain discrepancies found in prior studies [\(4.1\)](#page-4-0). We further specifically distinguish between gradient conflicts between tasks and samples [\(4.2\)](#page-9-0).

### 3 Problem Statement

Multi-task learning addresses the problem of learning a set of T tasks simultaneously (see e.g. [\[4,](#page-12-0) [48\]](#page-15-0)). It is noteworthy that this setup is occasionally also referred to as multi-label or multi-target learning [\[57\]](#page-16-1) Importantly, this study does not incorporate multi-input data. We consider a supervised learning setup, use a shared backbone architecture, and learn all tasks together. Formally, given input data X , the goal is to learn a function fθ(x) which maps a point x ∈ X to each task label y<sup>t</sup> with t = 1, .., T. The trainable parameters θ = {ϕ, ψ1:<sup>T</sup> } consist of shared parameters ϕ and task-specific parameters ψt. Training a task t is associated with the loss Lt(fθ(x); θ), e.g., a regression or classification loss. We denote respective gradients on the shared and task-specific parameters with g ϕ <sup>t</sup> = ∇ϕLt, and g ψ <sup>t</sup> = ∇ψLt. When training on multiple tasks, the shared parameters ϕ needs to be updated w.r.t. all task-wise gradients g ϕ <sup>t</sup> which requires an appropriate aggregation. A simple solution is to uniformly sum up the task losses L = P <sup>t</sup> L<sup>t</sup> which is referred to as Equal Weighting (EW). However, as tasks might be competing against each other, this can result in negative transfer and thus sub-optimal solutions. One way to deal with this difficulty is to adapt the magnitude of task-specific gradients. This can be achieved by weighting tasks during training, e.g., by scaling different losses L = P <sup>t</sup> αtLt, where α<sup>t</sup> ≥ 0. Note that the α<sup>t</sup> can change during training. Furthermore, the weighing can also be performed on gradient level to distinguish between shared and task-specific gradients. We refer to those approaches as gradient magnitude methods. Interestingly, the relationship between loss weights, network updates and learning rate also depends on the optimizer. We show a derivation for SGD and Adam in Appendix [A1.2.](#page-18-0) Additionally to adapting the gradient magnitude, one can directly adapt the alignment of task-wise gradient vectors within the shared backbone g˜ <sup>ϕ</sup> = h(g<sup>1</sup> ϕ , ..., g<sup>T</sup> ϕ ).

In practice, an optimum for θ that yields best performance on all tasks often does not exist. Instead, improving performance on some task often yields a performance decrease in another task. To still enable a comparison across network instances in MTL, an instance θ ∗ is called to be Pareto optimal, if there is no other θ ′ such that Lt(θ ′ ) ≤ Lt(θ ∗ ) ∀t with strict inequality in at least one task. The Pareto front consists of the Pareto optimal solutions.

### 4 Experiments and results

In this section we perform several experiments to gain a more profound understanding of multi-task learning (MTL) in computer vision by questioning common paradigms. We compare the impact of Adam and SGD in MTL in Sec. [4.1](#page-4-0) and examine the process of gradient similarity in different settings in Sec. [4.2.](#page-9-0) Throughout this evaluation, we repeatedly make use of common setups, which we will specify as follows and in more detail in App. [A3.](#page-25-0)

Datasets: For our experiments, we consider three different datasets that are commonly used for evaluating MTL in computer vision: Cityscapes [\[9\]](#page-12-6) contains images of urban street scenes. In line with previous work, we consider the tasks of semantic segmentation (7 classes) and depth estimation. NYUv2 [\[39\]](#page-14-13) is an indoor dataset for scene understanding which was recorded over 464 different scenes across three different cities. Besides semantic segmentation (13-class) and depth estimation, it also contains the task of surface normal prediction. CelebA [\[35\]](#page-14-14) consists of 200K face images which are labeled with 40 binary attributes.

Networks: We use network architectures with hard-parameter sharing which consist of a shared backbone and task-specific heads. For the dense prediction tasks on Cityscapes and NYUv2, we compare SegNet [\[1\]](#page-12-7) and DeepLabV3+ [\[5\]](#page-12-8). Experiments on CelebA are performed on a ResNet-18 [\[15\]](#page-13-7) with an additional single linear layer for each head.

Training: For each method, we follow the loss or gradient aggregation as described in the related work, e.g., for equal weighting all task-specific losses are simply summed up to compute the joint network gradients. The learning rate is tuned separately for each approach. We use the validation set performance of the ∆<sup>m</sup> metric as early stopping criteria. The ∆<sup>m</sup> metric [\[36\]](#page-14-6) measures the average relative task performance drop of a method m compared to the single-task baseline b using the same backbone and is computed as ∆<sup>m</sup> = 1 T P<sup>T</sup> <sup>t</sup>=1(−1)<sup>l</sup><sup>t</sup> (Mm,t −Mb,t)/Mb,t where <sup>l</sup><sup>t</sup> = 1 if a higher value means better for measure M·,t of some task metric t, and 0 otherwise.

#### <span id="page-4-0"></span>4.1 Effectiveness of Adam in multi-task learning

Examined paradigm: The impact of the choice of standard optimizer is often disregarded and varies across studies (overview in Tab. [A1\)](#page-26-0) when comparing MTL methods. For instance, Adam [\[22\]](#page-13-1) was successfully used to show that random/constant weighting of tasks' losses performs competitive compared to MTO methods [\[25,](#page-14-1) [28,](#page-14-10) [50\]](#page-15-1). In contrast, many methods proposing adaptive, taskspecific weighting methods [\[21,](#page-13-2) [32\]](#page-14-3) use stochastic gradient descent with momentum (SGD+mom). In recent works, the optimizer choice converged to Adam and a fixed learning rate schedule [\[31,](#page-14-2)[34,](#page-14-4)[45,](#page-15-7)[56\]](#page-16-0) without a comparison to SGD+mom.

In this part of our study, we investigate the impact of Adam and SGD+mom in conjunction with common MTO methods. We identify the choice of optimizer as a crucial confounder in the experimental setup. Compared to SGD+mom, we find that the Adam optimizer itself is a quite effective baseline in MTL and can be regarded as a loss weighting method from a theoretical viewpoint.

Toy Task Experiment To get a first impression of the impact of the optimizer and common hyperparameters such as the learning rate, we investigate the impact of Adam and plain gradient descent (GD) in a simple toy task.

<span id="page-5-0"></span>

Fig. 1: Toy task experiment from CAGrad [\[31\]](#page-14-2) for different learning rates and optimizers. Consistent with results from [\[50\]](#page-15-1), we observe that the choice of the learning rate is crucial even for this toy optimization problem. Moreover, it becomes apparent, that selecting Adam over simple gradient decent (GD) yields superior results. The contour lines depict the 2D loss landscape; the optimization trajectories are colored from red to yellow for 100k iteration steps from three different starting points (seeds).

<span id="page-5-1"></span>Table 1: Maximum number of iterations for all seeds in the toy task experiment from [\[31\]](#page-14-2) to reach the global minimum for varying MTO method, learning rate, and optimizer combination. EW+Adam often shows the fastest convergence to the global minimum. '-' denotes that not all seeds converged within 100k iterations. As reported in [\[31\]](#page-14-2), PCGrad often only converges to a point on the Pareto Front. We highlight the best run for each learning rate over all MTO methods.

Approach: We repeat the experiment of Liu et al. [\[31\]](#page-14-2) using their original implementation but further test different learning rates and optimizers. They motivate their gradient alignment method CAGrad with a simple toy optimization problem in which their method reliably converges to the minimum of the average loss, while other MTO approaches would either get stuck (e.g., EW) or only converge to any point on the Pareto front (e.g., PCGrad [\[56\]](#page-16-0), MGDA [\[44\]](#page-15-6)).

Result: For higher learning rates with Adam optimizer, even equal weighting (EW) reaches the global optimum (cf. Fig. [1,](#page-5-0) e.g., EW+Adam, lr=0.05) and often converges even faster than dedicated MTO methods (Tab. [1\)](#page-5-1). Note, original results were shown for learning rate 0.001 using Adam and were, thus, in favor of CAGrad. Results for additional learning rates are reported in Tab. [A3.](#page-30-0)

Conclusion: The choice of optimizer appears to be more important on the success of the outcome of this experiment than the choice of MTO method, as Adam converges considerably faster and more reliably than GD. Also, tuning the learning rate is a relevant factor, however, especially in MTL with differently scaled losses, a single suitable learning rate for all tasks does often not exist.

<span id="page-6-0"></span>Table 2: Number of Pareto optimal (PO) experiments using either Adam or SGD+mom. as optimizer. Models trained with Adam are consistently more often on the Pareto front compared to those trained with SGD+mom. The number of Adam-based runs that are not dominated by any SGD-based run (PO w.r.t. SGD) is even higher, while the reverse does not apply.

<span id="page-6-1"></span>![](_page_6_Figure_4.jpeg)

Fig. 2: Parallel coordinate plot over all experiments on Cityscapes. We distinguish between experiments using SGD+mom and Adam optimizer. Experiments that reached Pareto front performance are drawn with higher saturation. We observe that Adam clearly outperforms the usage of SGD+mom.

Experiments on Cityscapes and NYUv2 We test the effectiveness of Adam and its role as a confounder in common MTL datasets for various MTO methods.

Approach: We compare Adam and SGD+mom in combination with any MTO method from equal weighting (EW), uncertainty weighting (UW) [\[21\]](#page-13-2), random loss weighting (RLW) [\[28\]](#page-14-10), PCGrad [\[56\]](#page-16-0), CAGrad [\[31\]](#page-14-2), IMTL [\[32\]](#page-14-3) and Aligned-MTL [\[45\]](#page-15-7), for which we used the implementation from [\[29\]](#page-14-15), as well as MTL-IO [\[41\]](#page-15-9) and AdaTask [\[52\]](#page-15-10). We distinguish between any combination of dataset {Cityscapes [\[9\]](#page-12-6), NYUv2 [\[39\]](#page-14-13)} and network architecture {SegNet [\[1\]](#page-12-7), DeepLabV3 [\[5\]](#page-12-8)}. We run experiments for ten different initial learning rates from [0.5, 0.1, 0.05, ..., 0.00001] and select the best one w.r.t. to the validation performance. More details are described in App. [A3.2.](#page-25-1) As different models and parameter setups can show preference towards different tasks and metrics, we are interested in those models which are Pareto optimal (PO).

Results: We observe over all experimental setups that Adam performs favorably over SGD+mom (Tab. [2\)](#page-6-0). This especially holds true for experiments on Cityscapes where the Pareto front for both network architectures only consists of Adam-based models. Moreover, an even larger number of Adam-based models is not dominated by any model trained with SGD+mom (PO w.r.t. SGD). For NYUv2, Adam still performs stronger but SGD+mom. also occasionally delivers a PO result. For the individual metrics, the predominance of Adam is further visualized in a parallel coordinate plot in Figs. [2](#page-6-1) and [A4.](#page-29-0) Bold lines indicate the overall Pareto optimal experiments (PO full).

In App. [A4,](#page-28-0) we further report best ∆<sup>m</sup> results for common MTO methods in combination with Adam or SGD+mom (Tabs. [A4](#page-31-0) to [A7\)](#page-32-0). Again, Adam boosts the overall performance across methods. Furthermore, when comparing the ranking of MTO methods w.r.t. the ∆<sup>m</sup> metric, we see that the order can change based on the choice of optimizer, e.g., for Cityscapes with SegNet the best method with Adam is UW but with SGD+mom it is CAGrad. This underlines the importance of the choice of optimizer as a confounder in the experimental setup. Noteworthy, EW with Adam yields Pareto optimal results in three of the four setups (cf. Tab. [A2\)](#page-29-1) and is not dominated by any specialized MTO method trained in combination with SGD+mom for all dataset and network combinations. This supports claims questioning the effectiveness of specific MTO methods [\[25,](#page-14-1) [50\]](#page-15-1). Nonetheless, looking at the ∆<sup>m</sup> metric and individual metrics, we see that sometimes with a small relative performance drop on one metric, significant gains on another metric can be achieved (e.g., Cityscapes+sem.seg. and depth for UW vs EW).

Conclusion: Not only a well-tuned learning rate but also the optimizer is crucial for MTL performance. In a fair and extensive experimental comparison, we were able to show that Adam shows superior performance in MTL setup compared to SGD+mom.

The reasonable effectiveness of Adam in the context of uncertainty weighting We show that Adam's mechanism to estimate a parameter-specific learning rate is partially loss-scale invariant and hypothesize that this could contribute to Adams effectiveness in MTL. We demonstrate this partial invariance theoretically and empirically. Furthermore, a full loss-scale invariance can also be shown under mild assumptions for UW [\[21\]](#page-13-2), which is among the most prevalent loss weighting method in the literature, and related similar variant [\[27\]](#page-14-11).

The loss-scale invariance of UW can be shown by assuming an optimal solution for the σ values similar to [\[23\]](#page-13-8). This assumption is mild as this is a 1-dimensional convex optimization problem for each σ. The invariance can be demonstrated by inserting the analytical solution starting from UW. For example, assuming a Laplacian distribution (this can be shown for other distributions as well), we have

min σ<sup>t</sup> 1 σt L<sup>t</sup> + log σ<sup>t</sup> ⇒ σ<sup>t</sup> = L<sup>t</sup> (1)

The left hand side shows the typical form of UW, as shown for a Gaussian in [\[21,](#page-13-2) eq.(5)]. Here, L<sup>t</sup> is a task-specific loss and σ<sup>t</sup> is a scalar parameter that is usually learned. Plugging back the optimal solution for σt, we get

L = X t Lt sg[Lt] + c, (2)

where sg is the stop-gradient operator and c is a constant that can be omitted during optimization. Given this, we directly see the invariance w.r.t. loss-scalings. For instance, with L<sup>1</sup> → α1L<sup>1</sup> and L<sup>2</sup> → α2L2, the derivative of the total loss L remains unchanged. As this invariance is shown on the loss-level, it holds for all gradient updates w.r.t. the head and backbone. Intuitively, this could explain why UW performs strongly in the context of various loss scalings such as measuring depth in centimeters or meters. Further details, are in App. [A1.](#page-17-0)

Similarly, for Adam, we can prove a partial scale invariance of losses in MTL that holds for the parameters of network heads. As before, we assume a hydralike network architecture with a shared backbone and task-specific heads. We start with the parameter-update rule from Adam and scale the corresponding losses L<sup>t</sup> → αtLt. When only considering the parameters of the corresponding heads ψt, the scalings α<sup>t</sup> cancel out

ψt,i = ψt,i−<sup>1</sup> − γ q α t ˆv ′ t <sup>α</sup>tmˆ′ t . (3)

Thus, for the network heads, we see a similar effect as for optimal UW that different scalings do not impact the network update. However, this does not hold for the backbone. The full derivation is shown in App. [A1.](#page-17-0) We confirm empirically in a handcrafted loss-scaling experiment in App. [A2](#page-21-0) and Figs. [A1](#page-23-0) and [A2](#page-24-0) that SGD does not offer any scaling invariance, whereas Adam involves the invariance property for the heads. The optimal UW demonstrates a scaling invariance for the heads and the backbone.

We would like to note that our derivation for Adam is only valid for constant αt, e.g., measuring depth in different units or unitary weightings [\[25,](#page-14-1) [50\]](#page-15-1). In case of dynamic loss weights that are not constant (e.g., UW), the weights do not cancel out fully due to the accumulation of gradient histories within Adam. Nonetheless, this has profound implications for loss weighting methods that are used in conjunction with Adam. For instance, when turning off the history within Adam (by setting β1,<sup>2</sup> = 0) and having a fixed backbone, all loss weighting methods, such as UW, RLW, and others, become equivalent to equal weighting.

Additional ablations to our previous experiments suppport the relevance of invariance in MTL (cf. Tabs. [A4](#page-31-0) to [A7\)](#page-32-0). First, we compare to signSGD+mom [\[3\]](#page-12-9) which only updates on the sign of gradients and is therefore trivially scaleinvariant in the heads. In a direct comparison with SGD+mom, we observe a superiority of signSGD for a majority of tested setups. Next, we applied taskspecific Adam optimizers as in AdaTask [\[52\]](#page-15-10) for a full loss-scale invariance and to allow an estimate of task-specific momentum and squared gradient accumulation. This is Pareto dominant over plain Adam+EW in almost all cases and significantly improves the ∆<sup>m</sup> metric.

Conclusion: In the context of MTL, we derive and measure a full loss-scale invariance for an optimal UW and a partial invariance for Adam. This partial invariance does not hold for SGD+mom and could explain, among other properties, the effectiveness of Adam in MTL. Furthermore, when comparing different loss weighting methods, it is crucial to be aware of the influence of the optimizer.

#### <span id="page-9-0"></span>4.2 Investigating gradient conflicts between tasks and samples

Examined paradigm: The field of MTL strongly focuses on resolving conflicts between tasks, especially from a perspective of gradient conflicts [\[20,](#page-13-3) [31,](#page-14-2) [56\]](#page-16-0). In computer vision, tasks are often defined on a conceptional level such as segmentation and depth (Cityscapes), or recognizing multiple attributes (CelebA). However, in principle, conflicts can not only occur between tasks but also between samples within a task.

Fig. 3: High intra-task diversity can mimic MTL. We argue that in an extreme case, even recognizing a single cat in multiple images could be considered MTL. For instance, in one image, the cat could be hiding behind a plant and only revealing its eyes, requiring a neural network to recognize the cat solely based on the eyes. In other images, the cat might only reveal its paws, front of a bright window, or might be tired and curled up into a furry ball because we took so many pictures. This would require a paw, shape or fur classifier. Thus, a neural network is required to recognize multiple attributes to reliably recognize our cat. We note that this is conceptually similar to the commonly [\[44,](#page-15-6) [56\]](#page-16-0) considered MTL dataset CelebA [\[35\]](#page-14-14) which requires attribute detection such as wavy hair, mustache or hat, but within one image.

![](_page_9_Picture_4.jpeg)

Motivated by this example, we would like to quantify inter-task and intersample conflicts in common datasets from a perspective of the MTO literature, which inspects gradient conflicts in neural networks. In particular, we challenge the sole focus on inter-task gradients conflicts in MTL. While several works follow the idea of overcoming gradient conflicts in MTL [\[20,](#page-13-3) [31,](#page-14-2) [46\]](#page-15-8), their appearance has only been mildly investigated so far.

Prerequisite: We compare gradients w.r.t. network weights for different tasks t and samples x<sup>i</sup> . The alignment of two gradients g, g ′ on the shared parameters, e.g., of task a and task b, is compared with the cosine similarity

Scos(g, g ′ ) = cos(ϕ) = <sup>g</sup> · <sup>g</sup> ′ ∥g∥∥g ′∥ . (4)

Thus, two gradients are in conflict, if their cosine similarity is smaller than zero [\[56\]](#page-16-0). In particular, Scos is 1/−1 if gradients point in the same/opposite direction and 0 in case of orthogonal directions. The gradient magnitude similarity

Smag(g, g ′ ) = <sup>2</sup>∥g∥<sup>2</sup> · ∥<sup>g</sup> ′∥<sup>2</sup> ∥g∥ 2 <sup>2</sup> + ∥g ′∥ 2 2 (5)

as defined in [\[56\]](#page-16-0), yields values close to 1 for gradients of similar magnitude, or close to 0 for large discrepancies in magnitude. High dissimilarity in both gradient direction and magnitude is presumed to be a common MTL problem.

Approach: During the training on aforementioned datasets, we examine gradient similarity across two different setups: (1) between gradients of different tasks with respect to a single sample (inter-task), e.g., g = ∇ϕL<sup>0</sup> (fθ(xi)) and g ′ = ∇ϕL<sup>1</sup> (fθ(xi)); and (2) between gradients corresponding to the same task

<span id="page-10-0"></span>![](_page_10_Figure_1.jpeg)

Fig. 4: Gradient similarities and conflicts for different datasets and network architectures over training epochs. For each dataset and network combination, we report (from left to right) gradient cosine similarity, gradient magnitude similarity, and the ratio of conflicting gradient parameters w.r.t. gradient pairs corresponding to either inter-samples (fixed task) or inter-tasks (fixed sample). We report mean (solid line), standard deviation (shaded area), upper (97.5%) and lower (2.5%) percentile (dotted line) within an epoch. Overall, the direction conflicts are similar (first / last column), whereas the magnitude differences are more pronounced in MTL (middle column).

but different samples within a batch (inter-sample), e.g., g = ∇ϕL<sup>t</sup> (fθ(x0)) and g ′ = ∇ϕL<sup>t</sup> (fθ(x1)). For both setups, we compute the gradient cosine similarity and gradient magnitude similarity as well as the ratio of conflicting gradient parameters. We are aware that our comparison between samples and tasks is not direct. Nonetheless, it serves as a coarse indicator to estimate their impact during network training. Implementation details are in App. [A3.](#page-25-0)

Results: We show the evolvement of the gradient similarity measures over epochs in Fig. [4.](#page-10-0) Surprisingly, when comparing inter-sample (red line) and intertask (blue line), we find no consistent evidence for gradient alignment conflicts (left column) to be an exclusive problem of having multiple tasks. For instance, for Cityscapes, the variation of gradient alignment is fully encapsulated within the spread we observe in inter-sample variation (task is fixed). For CelebA, the converse seems to be mostly the case. Furthermore, the choice of network architecture and distribution of task-specific and shared parameters (SegNet vs. DeepLabV3) can have a large influence on the spread of the cosine-similarity. Both architectures have roughly a similar number of shared-parameters. However, DeepLabV3 has a higher number of task-specific parameters which seems to reduce the variance in conflicts for both inter-sample and inter-task (row one vs. two). In line with these observations, we found a similar number of conflicting gradient parameters (third column) for both inter-sample and inter-task comparisons among all experiments.

For gradient magnitude similarities (middle column), we observe a clearer pattern. The similarity in magnitudes are continuously (in the mean) less pronounced for the inter-task setup compared to inter-samples (blue line is below red one in all settings). Interestingly, the relative difference between the two setups remains similar over training which justifies the choice of fixed scalar task weightings as done in [\[50\]](#page-15-1). Further measures can be found in Figures [A5](#page-33-0) to [A7.](#page-34-0)

Conclusion: We find that the difficulty of MTL (inter-task and inter-sample) as opposed to STL (inter-sample only) is predominantly due to differences in gradient magnitudes. Balancing different magnitudes is tackled in the literature, e.g., [\[21,](#page-13-2) [50\]](#page-15-1) The problem of conflicting gradients has been typically associated with task-specific conflicts [\[20,](#page-13-3) [31,](#page-14-2) [56\]](#page-16-0), here, we find that gradient alignment conflicts can actually be even more pronounced between samples. On the one hand, these observations are along the same lines as findings by Royer et al. [\[42\]](#page-15-11) who reason that 'correcting conflicting gradients [between tasks] at every training iteration can be superfluous'. On the other hand, gradient-alignment methods in MTL could be considered not only in the context of task-specific conflicts but also for conflicts between samples. Interestingly, previous work has explored the potential benefit of not only learning weights per task but also per sample in the dataset [\[49\]](#page-15-12). While our experiments show a relatively high similarity in gradient magnitude across samples and, thus, don't motivate a sample-wise loss weighting, this could, however, be due to only little disruptive noise within data samples which has been the main motivation of [\[49\]](#page-15-12).

### 5 Conclusion and outlook

This study aims to enhance our understanding of multi-task learning (MTL) in computer vision, providing valuable insights for future research as well as guidance for implementations of real-world applications.

We show that common optimization methods from single task learning (STL) like the Adam optimizer are effective in MTL problems. Next, we compare gradient conflicts during training between tasks and samples. While gradient magnitudes are a specific problem between tasks (MTL) and thus justify the need for multi-task specific methods for automatic loss weighting, we find the variability in gradient alignment to be similar between samples and tasks. Thus, we encourage a more unified viewpoint in which specific MTO methods are also considered in single-task problems and vice versa.

Beyond our work, we encourage to improve the understanding of challenges and paradigms specific to MTL. For instance, our understanding of task (and sample) specific capacity allocation within a network and how best to tune it to custom requirements, is still not thoroughly understood. Often task-weights are increased to assign more importance to a task which is in contrast to tuning the learning rate per task where a smaller learning rate can be beneficial. Thus, we require further investigations and disentanglement of these two concepts.

# Acknowledgments

We thank Claudia Blaiotta, Martin Rapp, Frank R. Schmidt, Leonhard Hennicke, and Bastian Bischoff for their feedback and valuable discussions. Cathrin Elich thanks her supervisors, J¨org St¨uckler and Marc Pollefeys, for enabling the opportunity to pursue an internship during her Ph.D. studies.

The Bosch Group is carbon neutral. Administration, manufacturing and research activities do no longer leave a carbon footprint. This also includes GPU clusters on which the experiments have been performed.

### References

- <span id="page-12-7"></span>1. Badrinarayanan, V., Kendall, A., Cipolla, R.: SegNet: A Deep Convolutional Encoder-Decoder Architecture for Image Segmentation. IEEE Transactions on Pattern Analysis and Machine Intelligence (2017)
- <span id="page-12-10"></span>2. Beery, S., Van Horn, G., Perona, P.: Recognition in terra incognita. In: Proceedings of the European conference on computer vision (ECCV). pp. 456–473 (2018)
- <span id="page-12-9"></span>3. Bernstein, J., Wang, Y.X., Azizzadenesheli, K., Anandkumar, A.: signSGD: Compressed optimisation for non-convex problems. In: Proceedings of the 35th International Conference on Machine Learning. Proceedings of Machine Learning Research, vol. 80, pp. 560–569. PMLR (10–15 Jul 2018)
- <span id="page-12-0"></span>4. Caruana, R.: Multitask learning. Machine learning 28, 41–75 (1997)
- <span id="page-12-8"></span>5. Chen, L.C., Zhu, Y., Papandreou, G., Schroff, F., Adam, H.: Encoder-decoder with atrous separable convolution for semantic image segmentation. In: Computer Vision – ECCV 2018 (2018)
- <span id="page-12-3"></span>6. Chen, Z., Badrinarayanan, V., Lee, C., Rabinovich, A.: Gradnorm: Gradient normalization for adaptive loss balancing in deep multitask networks. In: Dy, J.G., Krause, A. (eds.) Proc. of ICML. Proceedings of Machine Learning Research, vol. 80, pp. 793–802. PMLR (2018)
- <span id="page-12-1"></span>7. Chen, Z., Ngiam, J., Huang, Y., Luong, T., Kretzschmar, H., Chai, Y., Anguelov, D.: Just pick a sign: Optimizing deep multitask models with gradient sign dropout. In: Larochelle, H., Ranzato, M., Hadsell, R., Balcan, M., Lin, H. (eds.) Advances in Neural Information Processing Systems 33: Annual Conference on Neural Information Processing Systems 2020, NeurIPS 2020, December 6-12, 2020, virtual (2020)
- <span id="page-12-2"></span>8. Chennupati, S., Sistu, G., Yogamani, S., A Rawashdeh, S.: Multinet++: Multistream feature aggregation and geometric loss strategy for multi-task learning. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops (2019)
- <span id="page-12-6"></span>9. Cordts, M., Omran, M., Ramos, S., Rehfeld, T., Enzweiler, M., Benenson, R., Franke, U., Roth, S., Schiele, B.: The cityscapes dataset for semantic urban scene understanding. In: 2016 IEEE Conference on Computer Vision and Pattern Recognition, CVPR 2016, Las Vegas, NV, USA, June 27-30, 2016. pp. 3213–3223. IEEE Computer Society (2016).<https://doi.org/10.1109/CVPR.2016.350>
- <span id="page-12-4"></span>10. D´esid´eri, J.A.: Multiple-gradient descent algorithm (mgda) for multiobjective optimization. Comptes Rendus Mathematique 350, 313–318 (2012)
- <span id="page-12-5"></span>11. Fernando, H.D., Shen, H., Liu, M., Chaudhury, S., Murugesan, K., Chen, T.: Mitigating gradient bias in multi-objective learning: A provably convergent approach.

- In: The Eleventh International Conference on Learning Representations, ICLR 2023, Kigali, Rwanda, May 1-5, 2023 (2023)
- <span id="page-13-5"></span>12. Fifty, C., Amid, E., Zhao, Z., Yu, T., Anil, R., Finn, C.: Efficiently identifying task groupings for multi-task learning. In: Ranzato, M., Beygelzimer, A., Dauphin, Y.N., Liang, P., Vaughan, J.W. (eds.) Advances in Neural Information Processing Systems 34: Annual Conference on Neural Information Processing Systems 2021, NeurIPS 2021, December 6-14, 2021, virtual. pp. 27503–27516 (2021)
- <span id="page-13-9"></span>13. Geirhos, R., Jacobsen, J.H., Michaelis, C., Zemel, R., Brendel, W., Bethge, M., Wichmann, F.A.: Shortcut learning in deep neural networks. Nature Machine Intelligence 2(11), 665–673 (2020)
- <span id="page-13-6"></span>14. Guo, M., Haque, A., Huang, D.A., Yeung, S., Fei-Fei, L.: Dynamic task prioritization for multitask learning. In: Proceedings of the European Conference on Computer Vision (ECCV) (2018)
- <span id="page-13-7"></span>15. He, K., Zhang, X., Ren, S., Sun, J.: Deep residual learning for image recognition. In: 2016 IEEE Conference on Computer Vision and Pattern Recognition, CVPR 2016, Las Vegas, NV, USA, June 27-30, 2016. pp. 770–778. IEEE Computer Society (2016).<https://doi.org/10.1109/CVPR.2016.90>
- <span id="page-13-12"></span>16. Hendrycks, D., Dietterich, T.G.: Benchmarking neural network robustness to common corruptions and perturbations. In: Proc. of ICLR. OpenReview.net (2019)
- <span id="page-13-11"></span>17. Hu, Z., Zhao, Z., Yi, X., Yao, T., Hong, L., Sun, Y., Chi, E.: Improving multi-task generalization via regularizing spurious correlation. Advances in Neural Information Processing Systems 35, 11450–11466 (2022)
- <span id="page-13-10"></span>18. Ilyas, A., Santurkar, S., Tsipras, D., Engstrom, L., Tran, B., Madry, A.: Adversarial examples are not bugs, they are features. In: Wallach, H.M., Larochelle, H., Beygelzimer, A., d'Alch´e-Buc, F., Fox, E.B., Garnett, R. (eds.) Advances in Neural Information Processing Systems 32: Annual Conference on Neural Information Processing Systems 2019, NeurIPS 2019, December 8-14, 2019, Vancouver, BC, Canada. pp. 125–136 (2019)
- <span id="page-13-0"></span>19. Ishihara, K., Kanervisto, A., Miura, J., Hautam¨aki, V.: Multi-task learning with attention for end-to-end autonomous driving. In: IEEE Conference on Computer Vision and Pattern Recognition Workshops, CVPR Workshops 2021, virtual, June 19-25, 2021. pp. 2902–2911. Computer Vision Foundation / IEEE (2021). <https://doi.org/10.1109/CVPRW53098.2021.00325>
- <span id="page-13-3"></span>20. Javaloy, A., Valera, I.: Rotograd: Gradient homogenization in multitask learning. In: Proc. of ICLR. OpenReview.net (2022)
- <span id="page-13-2"></span>21. Kendall, A., Gal, Y., Cipolla, R.: Multi-task learning using uncertainty to weigh losses for scene geometry and semantics. In: 2018 IEEE Conference on Computer Vision and Pattern Recognition, CVPR 2018, Salt Lake City, UT, USA, June 18-22, 2018. pp. 7482–7491. IEEE Computer Society (2018). <https://doi.org/10.1109/CVPR.2018.00781>
- <span id="page-13-1"></span>22. Kingma, D.P., Ba, J.: Adam: A method for stochastic optimization. In: Bengio, Y., LeCun, Y. (eds.) Proc. of ICLR (2015)
- <span id="page-13-8"></span>23. Kirchdorfer, L., Elich, C., Kutsche, S., Stuckenschmidt, H., Schott, L., K¨ohler: Analytical uncertainty-based loss weighting in multi-task learning. In: German Conference on Pattern Recognition (2024)
- <span id="page-13-4"></span>24. Klingner, M., Bar, A., Fingscheidt, T.: Improved noise and attack robustness for semantic segmentation by using multi-task training with self-supervised depth estimation. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops. pp. 320–321 (2020)

- <span id="page-14-1"></span>25. Kurin, V., De Palma, A., Kostrikov, I., Whiteson, S., Kumar, M.P.: In Defense of the Unitary Scalarization for Deep Multi-Task Learning. In: Neural Information Processing Systems (2022)
- <span id="page-14-0"></span>26. Lee, D.G.: Fast drivable areas estimation with multi-task learning for real-time autonomous driving assistant. Applied Sciences 11(22), 10713 (2021)
- <span id="page-14-11"></span>27. Lin, B., Jiang, W., Ye, F., Zhang, Y., Chen, P., Chen, Y.C., Liu, S., Kwok, J.T.: Dual-balancing for multi-task learning (2023)
- <span id="page-14-10"></span>28. Lin, B., YE, F., Zhang, Y., Tsang, I.: Reasonable Effectiveness of Random Weighting: A Litmus Test for Multi-Task Learning. Transactions on Machine Learning Research (2022)
- <span id="page-14-15"></span>29. Lin, B., Zhang, Y.: LibMTL: A Python Library for Multi-Task Learning. ArXiv preprint abs/2203.14338 (2022)
- <span id="page-14-8"></span>30. Liu, B., Feng, Y., Stone, P., Liu, Q.: Famo: Fast adaptive multitask optimization. In: Oh, A., Naumann, T., Globerson, A., Saenko, K., Hardt, M., Levine, S. (eds.) Advances in Neural Information Processing Systems. vol. 36, pp. 57226–57243. Curran Associates, Inc. (2023)
- <span id="page-14-2"></span>31. Liu, B., Liu, X., Jin, X., Stone, P., Liu, Q.: Conflict-averse gradient descent for multi-task learning. In: Ranzato, M., Beygelzimer, A., Dauphin, Y.N., Liang, P., Vaughan, J.W. (eds.) Advances in Neural Information Processing Systems 34: Annual Conference on Neural Information Processing Systems 2021, NeurIPS 2021, December 6-14, 2021, virtual. pp. 18878–18890 (2021)
- <span id="page-14-3"></span>32. Liu, L., Li, Y., Kuang, Z., Xue, J., Chen, Y., Yang, W., Liao, Q., Zhang, W.: Towards impartial multi-task learning. In: Proc. of ICLR. OpenReview.net (2021)
- <span id="page-14-9"></span>33. Liu, S., James, S., Davison, A.J., Johns, E.: Auto-Lambda: Disentangling Dynamic Task Relationships. Transactions on Machine Learning Research (2022)
- <span id="page-14-4"></span>34. Liu, S., Johns, E., Davison, A.J.: End-to-end multi-task learning with attention. In: IEEE Conference on Computer Vision and Pattern Recognition, CVPR 2019, Long Beach, CA, USA, June 16-20, 2019. pp. 1871–1880. Computer Vision Foundation / IEEE (2019).<https://doi.org/10.1109/CVPR.2019.00197>
- <span id="page-14-14"></span>35. Liu, Z., Luo, P., Wang, X., Tang, X.: Deep learning face attributes in the wild. In: 2015 IEEE International Conference on Computer Vision, ICCV 2015, Santiago, Chile, December 7-13, 2015. pp. 3730–3738. IEEE Computer Society (2015). <https://doi.org/10.1109/ICCV.2015.425>
- <span id="page-14-6"></span>36. Maninis, K., Radosavovic, I., Kokkinos, I.: Attentive single-tasking of multiple tasks. In: IEEE Conference on Computer Vision and Pattern Recognition, CVPR 2019, Long Beach, CA, USA, June 16-20, 2019. pp. 1851–1860. Computer Vision Foundation / IEEE (2019).<https://doi.org/10.1109/CVPR.2019.00195>
- <span id="page-14-5"></span>37. Mao, C., Gupta, A., Nitin, V., Ray, B., Song, S., Yang, J., Vondrick, C.: Multitask learning strengthens adversarial robustness. In: Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part II 16. pp. 158–174. Springer (2020)
- <span id="page-14-7"></span>38. Misra, I., Shrivastava, A., Gupta, A., Hebert, M.: Cross-stitch networks for multitask learning. In: 2016 IEEE Conference on Computer Vision and Pattern Recognition, CVPR 2016, Las Vegas, NV, USA, June 27-30, 2016. pp. 3994–4003. IEEE Computer Society (2016).<https://doi.org/10.1109/CVPR.2016.433>
- <span id="page-14-13"></span>39. Nathan Silberman, Derek Hoiem, P.K., Fergus, R.: Indoor segmentation and support inference from rgbd images. In: ECCV (2012)
- <span id="page-14-12"></span>40. Navon, A., Shamsian, A., Achituve, I., Maron, H., Kawaguchi, K., Chechik, G., Fetaya, E.: Multi-task learning as a bargaining game. In: Chaudhuri, K., Jegelka, S., Song, L., Szepesv´ari, C., Niu, G., Sabato, S. (eds.) International Conference on

- Machine Learning, ICML 2022, 17-23 July 2022, Baltimore, Maryland, USA. Proceedings of Machine Learning Research, vol. 162, pp. 16428–16446. PMLR (2022)
- <span id="page-15-9"></span>41. Pascal, L., Michiardi, P., Bost, X., Huet, B., Zuluaga, M.A.: Improved optimization strategies for deep multi-task networks. ArXiv preprint abs/2109.11678 (2021)
- <span id="page-15-11"></span>42. Royer, A., Blankevoort, T., Bejnordi, B.E.: Scalarization for multi-task and multidomain learning at scale. In: Thirty-seventh Conference on Neural Information Processing Systems (2023)
- <span id="page-15-4"></span>43. Ruder, S.: An overview of multi-task learning in deep neural networks. ArXiv preprint abs/1706.05098 (2017)
- <span id="page-15-6"></span>44. Sener, O., Koltun, V.: Multi-task learning as multi-objective optimization. In: Bengio, S., Wallach, H.M., Larochelle, H., Grauman, K., Cesa-Bianchi, N., Garnett,
  - R. (eds.) Advances in Neural Information Processing Systems 31: Annual Conference on Neural Information Processing Systems 2018, NeurIPS 2018, December 3-8, 2018, Montr´eal, Canada. pp. 525–536 (2018)
- <span id="page-15-7"></span>45. Senushkin, D., Patakin, N., Kuznetsov, A., Konushin, A.: Independent component alignment for multi-task learning. In: IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2023, Vancouver, BC, Canada, June 17-24, 2023. IEEE (2023)
- <span id="page-15-8"></span>46. Shi, G., Li, Q., Zhang, W., Chen, J., Wu, X.M.: Recon: Reducing Conflicting Gradients From the Root For Multi-Task Learning. In: The Eleventh International Conference on Learning Representations (2023)
- <span id="page-15-3"></span>47. Standley, T., Zamir, A.R., Chen, D., Guibas, L.J., Malik, J., Savarese, S.: Which tasks should be learned together in multi-task learning? In: Proc. of ICML. Proceedings of Machine Learning Research, vol. 119, pp. 9120–9132. PMLR (2020)
- <span id="page-15-0"></span>48. Vandenhende, S., Georgoulis, S., Van Gansbeke, W., Proesmans, M., Dai, D., Van Gool, L.: Multi-task learning for dense prediction tasks: A survey. IEEE Transactions on Pattern Analysis and Machine Intelligence (2021). <https://doi.org/10.1109/TPAMI.2021.3054719>
- <span id="page-15-12"></span>49. Vasu, P.K.A., Saxena, S., Tuzel, O.: Instance-level task parameters: A robust multitask weighting framework. CoRR abs/2106.06129 (2021)
- <span id="page-15-1"></span>50. Xin, D., Ghorbani, B., Garg, A., Firat, O., Gilmer, J.: Do Current Multi-Task Optimization Methods in Deep Learning Even Help? In: Neural Information Processing Systems (2022)
- <span id="page-15-2"></span>51. Xu, D., Ouyang, W., Wang, X., Sebe, N.: Pad-net: Multi-tasks guided predictionand-distillation network for simultaneous depth estimation and scene parsing. In: 2018 IEEE Conference on Computer Vision and Pattern Recognition, CVPR 2018, Salt Lake City, UT, USA, June 18-22, 2018. pp. 675–684. IEEE Computer Society (2018).<https://doi.org/10.1109/CVPR.2018.00077>
- <span id="page-15-10"></span>52. Yang, E., Pan, J., Wang, X., Yu, H., Shen, L., Chen, X., Xiao, L., Jiang, J., Guo, G.: Adatask: A task-aware adaptive learning rate approach to multi-task learning. Proceedings of the AAAI Conference on Artificial Intelligence 37(9), 10745–10753 (2023)
- <span id="page-15-5"></span>53. Ye, F., Lin, B., Yue, Z., Guo, P., Xiao, Q., Zhang, Y.: Multi-objective meta learning. In: Advances in Neural Information Processing Systems. vol. 34 (2021)
- <span id="page-15-14"></span>54. Yeo, T., Kar, O.F., Zamir, A.: Robustness via cross-domain ensembles. In: 2021 IEEE/CVF International Conference on Computer Vision, ICCV 2021, Montreal, QC, Canada, October 10-17, 2021. pp. 12169–12179. IEEE (2021). <https://doi.org/10.1109/ICCV48922.2021.01197>
- <span id="page-15-13"></span>55. Yu, F., Koltun, V., Funkhouser, T.A.: Dilated residual networks. In: 2017 IEEE Conference on Computer Vision and Pattern Recognition, CVPR 2017, Hon-

- olulu, HI, USA, July 21-26, 2017. pp. 636–644. IEEE Computer Society (2017). <https://doi.org/10.1109/CVPR.2017.75>
- <span id="page-16-0"></span>56. Yu, T., Kumar, S., Gupta, A., Levine, S., Hausman, K., Finn, C.: Gradient surgery for multi-task learning. In: Larochelle, H., Ranzato, M., Hadsell, R., Balcan, M., Lin, H. (eds.) Advances in Neural Information Processing Systems 33: Annual Conference on Neural Information Processing Systems 2020, NeurIPS 2020, December 6-12, 2020, virtual (2020)
- <span id="page-16-1"></span>57. Zhang, Y., Yang, Q.: A survey on multi-task learning. IEEE Transactions on Knowledge and Data Engineering 34(12), 5586–5609 (2022). <https://doi.org/10.1109/TKDE.2021.3070203>
- <span id="page-16-2"></span>58. Zhao, H., Shi, J., Qi, X., Wang, X., Jia, J.: Pyramid scene parsing network. In: 2017 IEEE Conference on Computer Vision and Pattern Recognition, CVPR 2017, Honolulu, HI, USA, July 21-26, 2017. pp. 6230–6239. IEEE Computer Society (2017).<https://doi.org/10.1109/CVPR.2017.660>

### Examining Common Paradigms in Multi-Task Learning -Supplementary Material-

### <span id="page-17-0"></span>A1 Theoretical insights into multi-task learning dynamics

In this section, we aim to explain the success of the Adam optimizer [\[22\]](#page-13-1) by relating it to uncertainty weighting [\[21\]](#page-13-2). We show partial invariances w.r.t. prior task-weights for the Adam optimizer and full invariances for the uncertainty weighting under mild assumptions. We further show that for SGD + momentum no invariance can be observed. Instead, the loss-weight can be seen as a taskspecific learning rate which is not the case for the Adam optimizer. Previous literature on weighting methods in MTL did not explicitly show how task-weighting methods are affected by different optimizers.

### A1.1 Uncertainty weighting (UW): Full loss-scale invariance

In UW [\[21\]](#page-13-2), the homoscedastic uncertainty[<sup>5</sup>](#page-17-1) σ<sup>t</sup> to weight task t is learned by gradient descent. However, we can also analytically compute the optimal uncertainty weights in each iteration instead of learning them using gradient descent as done in [\[23\]](#page-13-8). The minimization objective depends on the underlying loss function and likelihood. For simplicity, we show the derivation exemplary for the L<sup>1</sup> loss. It is straight-forward to derive the same for a Gaussian and other distributions. The objective of uncertainty weighting is given as

min σ<sup>t</sup> 1 σt L<sup>t</sup> + log σ<sup>t</sup> (6)

with L<sup>t</sup> = |y − f<sup>W</sup> (x)| which can be derived from a log likelihood of a Laplace distribution p(y|f<sup>W</sup> (x), σ) = <sup>1</sup> 2σ exp(− |y−f<sup>W</sup> (x)| σ ). Taking the derivative and solving for σ<sup>t</sup> results in an analytically optimal solution:

∂ ∂σ<sup>t</sup> 1 σt L<sup>t</sup> + log σ<sup>t</sup> = − 1 σ t L<sup>t</sup> + 1 σt (7) − 1 σ 2 t L<sup>t</sup> + 1 σt != 0 <sup>⇒</sup> <sup>σ</sup><sup>t</sup> <sup>=</sup> <sup>L</sup><sup>t</sup> (8)

with σ<sup>t</sup> > 0. As the optimization problem is convex and just one dimensional, assuming an optimal log-sigma is a mild assumption. Plugging the optimal solution back into the original uncertainty weighting, we get

L = X t 1 sg[Lt] L<sup>t</sup> + log p sg[Lt], (9)

where we denote sg as the stopgradient operator.

Since there is no gradient for the second part of the loss, it can be simplified such that

<span id="page-17-1"></span><sup>5</sup> In Kendall et al., this is termed the aleatoric homoscedastic uncertainty. However, as the task weights vary over the course of training and also with respect to the model capacity, it is technically not only the aleatoric uncertainty but also encapsulates further components such as model capacity and amount of data seen.

L = X t Lt sg[Lt] . (10)

Assuming task-specific weights αt, we get

L = X t αtL<sup>t</sup> αtsg[Lt] = X t Lt sg[Lt] (11)

<span id="page-18-1"></span>Therefore, the optimal uncertainty weighting is invariant w.r.t. task-specific loss-scalings, as each scaling cancels out.

#### <span id="page-18-0"></span>A1.2 SGD: No loss-scale invariance and relationship of learning rate and task weights on a gradient level

Unlike for optimal UW, we show that the SGD update rule does not show any invariances and that task-weights are essentially task-specific learning rates. Instead, task-weights and learning rate are interacting hyperparameters and thus cannot be viewed in isolation.

The parameter update rule in neural networks optimized with SGD is

θ<sup>i</sup> = θi−<sup>1</sup> − γ ∂ ∂θi−<sup>1</sup> L, (12)

where the network parameters in iteration i are defined as θ<sup>i</sup> , γ is the learning rate and L = P <sup>t</sup> αiL<sup>t</sup> .

In the case of uniform task weights (EW), α = αi∀i, we have

θ<sup>i</sup> = θi−<sup>1</sup> − γ ∂ ∂θi−<sup>1</sup> X i αL<sup>t</sup> = θi−<sup>1</sup> − γα ∂ ∂θi−<sup>1</sup> X i Lt (13)

Here, task weight and learning rate are interchangeable. In particular, increasing the weight α by a constant factor c has the same effect as increasing the learning rate by a factor c.

In the case of non-uniform task weights α<sup>i</sup> , the parameter update is

θ<sup>i</sup> = θi−<sup>1</sup> − γ ∂ ∂θi−<sup>1</sup> X i αtL<sup>t</sup> = θi−<sup>1</sup> − ∂ ∂θi−<sup>1</sup> X i γαtL<sup>t</sup> (14)

As the learning rate can be included in the task-specific weight, it follows that task weighting is interchangeable to assigning task-specific learning rates. Tasks with a higher weight α<sup>i</sup> have a proportionally higher parameter update step and vice versa.

While this holds for SGD and SGD + momentum, it does not apply to optimizers such as Adam, Adagrad, or RMSProp. We demonstrate this for Adam in the following subsection.

#### A1.3 Adam: Partial loss-scale invariance

Similarly to the invariance demonstrated for optimal UW, we derive a partial invariance for Adam. In their work, Kingma and Ba [\[22\]](#page-13-1) have already shown that the magnitudes of the parameter updates using Adam are invariant to rescaling the gradients. Our novelty lies in demonstrating this invariance property in the context of MTL and its impact on different MTO methods. For Adam, we claim that the magnitude of task-specific weights only affects the backbone and cancels out for the heads.

We consider the standard MTL model setting with a shared backbone and task-specific heads. In this analysis, we assume a frozen backbone and only look at the task-specific parameters ψ<sup>t</sup> of task t whose loss L<sup>t</sup> is scaled by αt, such that L<sup>t</sup> → αtLt. The parameter update of one head is independent of the other heads as the derivative of the losses w.r.t. the other tasks is 0:

∂ ∂ψt,i−<sup>1</sup> L<sup>j</sup> = 0 for t ̸= j. (15)

The general update rule for parameters ψ at time step i using Adam is

ψ<sup>i</sup> = ψi−<sup>1</sup> − γ √ vˆ<sup>i</sup> + ϵ mˆ<sup>i</sup> , (16)

where m<sup>i</sup> = β1mi−<sup>1</sup> + (1 − β1)g<sup>i</sup> and v<sup>i</sup> = β2vi−<sup>1</sup> + (1 − β2)g 2 i . To counteract the bias towards 0, the moments are corrected as ˆm<sup>i</sup> = m<sup>i</sup> 1−β i and ˆv<sup>i</sup> = vi 1−β i .

For task-specific parameters ψt, task weights α<sup>t</sup> linearly scale the first moment mt,i

mt,i = β1mt,i−<sup>1</sup> + (1 − β1)gt,i = β1mt,i−<sup>1</sup> + (1 − β1) ∂ ∂ψt,i−<sup>1</sup> αtL<sup>t</sup> = β1mt,i−<sup>1</sup> + (1 − β1)α<sup>t</sup> ∂ ∂ψt,i−<sup>1</sup> Lt = β1mt,i−<sup>1</sup> + (1 − β1)αtg ′ t,i (17)

and quadratically scale the second moment vt,i

vt,i = β2vt,i−<sup>1</sup> + (1 − β2)g 2 t,i <sup>=</sup> <sup>β</sup>2vt,i−<sup>1</sup> + (1 <sup>−</sup> <sup>β</sup>2)( <sup>∂</sup> ∂ψt,i−<sup>1</sup> αtLt) 2 = β2vt,i−<sup>1</sup> + (1 − β2)α 2 t ( ∂ ∂ψt,i−<sup>1</sup> Lt) 2 = β2vt,i−<sup>1</sup> + (1 − β2)α 2 t g ′2 t,i, (18)

where g ′ t,i is the gradient of the unscaled loss L<sup>t</sup> w.r.t. the task-specific parameters for task t. As this holds for iteration i and because we have mt,<sup>1</sup> = αg′ t,<sup>1</sup> + 0 respectively vt,<sup>1</sup> = α 2 t g ′2 t,<sup>1</sup> + 0 with mt,<sup>0</sup> = 0, vt,<sup>0</sup> = 0 at the first iteration, this holds for any iteration step. We can thus rewrite ˆmt,i = αtmˆ ′ t,i and ˆvt,i = α 2 t vˆ′ t,i.

Plugging this back into the update rule, we get

ψt,i = ψt,i−<sup>1</sup> − γ √ vˆti mˆ t,i = ψt,i−<sup>1</sup> − γ q α<sup>2</sup> t vˆ′ t,i ✚α✚tmˆ′ t,i (19)

<span id="page-20-0"></span>where the loss-scaling α<sup>t</sup> cancels out. Therefore, the parameters of the taskspecific heads are invariant to loss-scalings using Adam.

This partial invariance is a highly desired property as there is a fundamental trade-off between tuning the learning rate and manual task weights. Given Adams invariance for the head, the weighting only affects the backbone. Thus the learning rate can be set for the parameters of the head independent of the loss weights. With the loss weights, we can prioritize tasks in the backbone and therefore walk along the Pareto front as empirically shown by [\[50\]](#page-15-1).

The invariance, however, does not hold anymore when the backbone parameters θ are updated as well. As we have

m<sup>i</sup> = β1mi−<sup>1</sup> + (1 − β1) ∂ ∂θi−<sup>1</sup> X t αtL<sup>t</sup> = β1mi−<sup>1</sup> + (1 − β1) X t αtg ′ t,i (20)

and

<sup>v</sup><sup>i</sup> <sup>=</sup> <sup>β</sup>1vi−<sup>1</sup> + (1 <sup>−</sup> <sup>β</sup>1)( <sup>∂</sup> ∂θi−<sup>1</sup> X t αtLt) 2 <sup>=</sup> <sup>β</sup>1vi−<sup>1</sup> + (1 <sup>−</sup> <sup>β</sup>1)(X t αtg ′ t,i) 2 (21)

we conclude that the task weights a<sup>t</sup> linearly affect the first moment m<sup>i</sup> , while having a quadratic effect on the update of the second moment v<sup>i</sup> .

Note that for both task-heads only as well as the backbone, we have a full invariance in case of independent optimizers, e.g., one Adam optimizer per task similar to [\[41,](#page-15-9)[52\]](#page-15-10). However, naive implementations scale poorly (in terms of computational complexity) with the number of tasks here.

In the following experiments, we provide empirical evidence for our finding that a) Adam offers loss-scale invariance for the parameters of the task-specific heads, and b) Adam offers loss-scale invariance for all network parameters (backbone and heads) if β1,<sup>2</sup> = 0.

# <span id="page-21-0"></span>A2 Empirical Confirmation of scale invariances in Adam and Optimal Uncertainty Weighting

In the prior section, we derived theoretical results for loss-scale (partial) invariance within multi-task learning for the Adam optimizer and uncertainty weighting. In this section, we confirm this invariance empirically with a toy task.

Experimental Setup We consider a two-task toy experiment in which we look at the gradient magnitudes with different combinations of Adam, SGD, EW, optimal uncertainty weighting (UW-O), and loss-scalings. To generate the data, we sample scalar input values from a uniform distribution; the outputs are just scalings of the input. We apply a simple neural network which consists of a shared backbone (two layers with LeakyReLU as non-linearity and 20 neurons per hidden layer) and two heads for the two tasks, each consisting again of two layers. Both task measure the depth but in different units using the L1-loss.

We provide two settings: In the first one, depth is measured on the same scale. In the second setting, one depth loss is scaled by 10x (e.g., measured in cm instead of deci-meters) and one other loss is scaled by 0.1 (e.g., measured in meters instead of deci-meters). For each setting, we test various combinations of loss weighting and optimizer combinations.

The 8 different experiments are:

- EW using SGD
- EW using SGD with scalings 10 · Lseg, 0.01 · Ldep
- EW using Adam
- EW using Adam with scalings 10 · Lseg and 0.01 · Ldep
- UW-O) using SGD
- UW-O) using SGD with scalings 10 · Lseg and 0.01 · Ldep
- EW using separate Adam optimizers per task
- EW using separate Adam optimizers per task with scalings 10·Lseg, 0.01·Ldep

To better control for different factors of influence, we first perform the first 6 of the listed experiments with a fixed backbone, i.e., we do not update the parameters in the backbone but only in the heads. Afterward, we show all 8 experiments trained with a network where all parameters (including the backbone) are updated. This allows us to verify if our theoretical derivations regarding the (partial) loss-scaling invariance of Adam and UW-O also hold in practice, and compare this to the SGD optimizer.

Note that we only care about the invariance and did not tune any hyperparameters for performance.

Results for fixed backbone Figure [A1](#page-23-0) shows the losses, the scaled losses (by loss weighting method), the gradient magnitudes as well as the gradient update magnitudes for both heads along the 100 epochs of training with a fixed backbone. Regarding SGD, we can observe that the equal weighting experiment differs from its scaled variant along all 8 dimensions. This is because SGD does not offer any loss-scaling invariance. As expected, at the beginning of the training the gradient update magnitude of the first depth head parameters with the scaled loss (dotted line) is by a factor of 10 higher than the unscaled (solid line) one. The same effect applies to the gradient update magnitude of the second depth head parameters, but with a factor of 0.01.

In contrast, Adam is loss-scale invariant. We can observe that the unscaled (solid line) and the scaled version (dotted line) have equal gradient update magnitudes in the last row. Note that practically due to an epsilon = 10−<sup>8</sup> parameter in the denominator and float precision a slight divergence would occur with larger number of epochs. This result confirms our theoretical finding in equation [19.](#page-20-0) We skip the experiment of separated Adam optimizers per task because it would be equivalent to this version given a fixed backbone.

Lastly, we want to investigate the invariance properties of UW-O. We compare the scaled (dotted line) and unscaled (solid line) version of UW-O with the SGD optimizer. As expected, the gradients, as well as the gradient updates, match in both heads.

In the following, let's investigate whether the observed results still hold if we also consider the update of the backbone parameters.

Results for free backbone Figure [A2](#page-24-0) shows the scaled losses, the gradient magnitudes as well as the gradient update magnitudes in the backbone and the depth heads along the 100 epochs of training with a free backbone. Again, the loss-scalings affect the gradient magnitudes using SGD. This applies to both backbone and heads.

When looking at the Adam experiments, we can observe that it is partly loss-scale invariant by looking at the first iteration in the heads. However, due to different updates in the backbone, the networks behave different in both settings (scaled and unscaled loses). Furthermore, when implementing task-specific optimizers, we can observe that not only the gradient update magnitudes in the task heads, but also in the backbone match between the scaled (dotted line) and the unscaled (solid line) variant. Thus, all network parameters are invariant to loss-scalings when using separate Adam optimizers. This confirms our theoretical results.

Along the lines of our theoretical findings, we can observe that UW-O offers scaling-invariance across the whole network as the gradients as well as the gradient updates match among the two variants in the backbone and in both heads. This empirical observation matches our theoretical derivation in equation [11.](#page-18-1)

<span id="page-23-0"></span>![](_page_23_Figure_2.jpeg)

Fig. A1: Invariances within the neural network for a frozen backbone. Comparing the effect of loss-scalings in a toy experiment with two tasks. For each optimizer and loss weighting combination, we run two settings with a) loss L1 and loss L2 are equally weighted or b) L1 is scaled by 10x and L2 by 0.1. For each setting, we measure the SGD + momentum and Adam optimizer with no post weighting (EW) and SGD + momentum with optimimal uncertainty weighting. We show the scaled losses, gradient magnitudes, and gradient update magnitudes in the the two task heads and keep the backbone frozen. While SGD does not offer any loss-scaling invariance, Adam makes the gradient updates of the head parameters invariant to scales confirming our derivation (red lines overlap in lowest row). Equivalently, for UW-O we also observe the theoretically derived invariances (green lines overlap in lowest row)

<span id="page-24-0"></span>![](_page_24_Figure_1.jpeg)

Fig. A2: Invariances within the neural network for a learnable backbone. Comparing the effect of loss-scalings in a toy experiment with two tasks. For each optimizer and loss weighting combination, we run two settings with a) loss L1 and loss L2 are equally weighted or b) L1 is scaled by 10x and L2 by 0.1. For each setting, we measure the SGD + momentum and Adam optimizer with no post weighting (EW) and SGD + momentum with optimimal uncertainty weighting. Additionally, we implement independent Adam optimizer per task. We show the scaled losses, gradient magnitudes, and gradient update magnitudes in the backbone(first row) and the the two task heads (2nd and 3rd row). Neither Adam, nor SGD show invariances if the backbone is trained as well. UW-O is still invriant (green lines are overlapping). We revoke Adam's inveriance by implementing separate optimizers per task (lowerst black lines are overlapping).

# <span id="page-25-0"></span>A3 Implementation Details

In this section, we explain the applied settings used for the reported experiments in more detail. In particular, we describe the handling of the different datasets in App. [A3.1](#page-25-2) and provide further information on the applied training procedures in App. [A3.2.](#page-25-3) Our chosen experimental setups are designed to follow previous work and mainly inspirited by [\[29,](#page-14-15) [31,](#page-14-2) [50\]](#page-15-1). However, we found that the experimental setup would vary widely across different works in the field of multi-task learning as can be seen in Table [A1.](#page-26-0) We use a uniform setup for each dataset independent of the choice of network and MTO.

### <span id="page-25-2"></span>A3.1 Datasets

Cityscapes [\[9\]](#page-12-6) We make use of the official split of the dataset which consists of 2975 training and 500 validation scenes. Similar to [\[50\]](#page-15-1), we denote 595 random samples from the training split as validation data and report test results on the original validation split. We further follow the pre-processing scheme from [\[34\]](#page-14-4) of re-scaling images to 128x256 pixels and use inverse depth labels. During training, we apply random scaling and cropping for data augmentation[<sup>6</sup>](#page-25-4) . Following previous work [\[31\]](#page-14-2) for number of epochs and learning rate schedule, we train for 300 epochs and decrease the learning rate by a factor of 0.5 every 100 epochs. The batch size is setto 64, similar to [\[50\]](#page-15-1). We only consider a fixed weight decay of 10−<sup>5</sup> for all datasets and experiments as we found varying this parameter had only little influence in initial experiments.

NYUv2 [\[39\]](#page-14-13) From the 795 official training images we use 159 for our validation split as in [\[29\]](#page-14-15) and report test performance on the official 654 test images. Similar to [\[34\]](#page-14-4), we re-size the images to 288x384 pixels. Training is run for 200 epochs with a batch size of 8. We apply the same data augmentation and learning rate schedule as for Cityscapes.

CelebA [\[35\]](#page-14-14) We re-size images to 64x64 pixels as done in [\[28\]](#page-14-10) and consider the original split of 162,770/19,867/19,962 for training, validation, and testing. We set the batch size to 512, train for 100 epochs, and halve the learning rates every 30 epochs.

#### <span id="page-25-3"></span>A3.2 Training

<span id="page-25-1"></span>Effectiveness of Adam in MTL. All presented results are based on performing early stopping w.r.t. ∆<sup>m</sup> metric on the validation set. For this, we further trained single-task learning (STL) models for each experiment combination (dataset and network) using the respective network architecture except for the missing head(s). We trained the models using Adam and any learning rate from {0.01, 0.005, ..., 0.00005}. The training was stopped early based on the validation

<span id="page-25-4"></span><sup>6</sup> https://github.com/Cranial-XIX/CAGrad

<span id="page-26-0"></span>Table A1: Original experiment setup as reported in respective papers. We note a high variation regarding the choice of network, optimizer, and other hyper-parameters among the different works.

loss. Reported scores in Tabs. [A4](#page-31-0) to [A7](#page-32-0) are computed as the mean of the models' performance that were initialized with the three different seeds.

Our implementation for all experiments is based on the LibMTL library [\[29\]](#page-14-15).

Gradient Similarity. Our gradient similarity experiments were conducted on the best performing hyper-parameter configuration for EW from the previous extensive evaluation. Over the full training, gradient similarity measures are computed every five iteration steps and summarized per epoch. To make the computation effort more feasible in case of settings with large batch size or high number of tasks, we randomly select eight samples or tasks respectively and consider corresponding gradients in these cases.

#### A3.3 Loss functions

Cityscapes For the task of semantic segmentation, we employ a pixel-wise crossentropy loss:

LCE = X C c=1 y c · log(p c ) (22)

where C is the number of classes, y <sup>c</sup> ∈ {0, 1} indicates the ground truth class, and p c is the predicted probability for class c which results from computing the softmax for output logits z c , p <sup>c</sup> = exp(z c P ) <sup>c</sup>=1 exp(z c) . This loss is averaged over the image.

For depth estimation, we utilize the L<sup>1</sup> loss:

Ldepth = ∥y − yˆ∥<sup>1</sup> (23)

where y, yˆ indicate ground truth and prediction, respectively. Pixels with invalid depth value in the ground truth data are ignored. It is noteworthy that these two types of losses are not balanced when used directly without modification.

NYUv2 The tasks of semantic segmentation and depth estimation are trained using the same loss functions as described for Cityscapes. In addition, we compute the cosine loss on the (normalized) surface normal maps:

Lnormal = 1 − cos θ = 1 − y · yˆ ∥y∥∥yˆ∥ (24)

where y, yˆ are the ground truth and predicted normal maps. Similar to the Cityscapes setup, the combination of these loss functions is not balanced per default.

CelebA To learn to predict multiple attributes, we use a binary cross entropy loss for the individual classes:

LCE,bin = −[y log(p) + (1 − y) log(1 − p)] (25)

Although all these losses have a similar scale, their impact varies based on the difficulty of the individual tasks and the number of available samples displaying the respective attribute.

#### A3.4 Evaluation criteria

In this study, we primarily focus on Pareto optimal solution to acknowledge that different configurations may lead to varying preferences for the learned tasks. However, it is important to note that not every point on the Pareto front is relevant in practice, especially when one metric significantly dominates while others are close to chance level. Moreover, specific real-world applications can have a stronger, pre-defined prioritization of one or a few sub-tasks which requires a relative weighting of the tasks' performances.

Additionally, we further employ the ∆<sup>m</sup> metric which offers a simple option to directly compare the performance of two models using a single scalar. This metric further indicates the relative performance compared to the single-tasks models.

Note that for our initial toy task experiment (Sec. [4.1\)](#page-4-0), we consider the original setting from [\[31\]](#page-14-2) which optimzes the global minimum of the two loss functions.

### <span id="page-28-0"></span>A4 Additional results on comparison between Adam and SGD

We present additional evaluation results for our comparison between optimizers for MTL. In Figure [A3,](#page-28-1) we compare the ∆<sup>m</sup> metric performance between the usage of Adam and SGD+mom. Fig. [A4](#page-29-0) shows additional parallel coordinate plots for NYUv2 and both choices of networks. In Table [A2,](#page-29-1) we count for each used MTO method the number of experiment runs that are located on the Pareto front w.r.t. each setup. Best performing quantitative results for all MTOs can be found in Tabs. [A4](#page-31-0) to [A7.](#page-32-0)

We further show extended results on the toy task by Liu et al. [\[31\]](#page-14-2) for more learning rates in Tab. [A3.](#page-30-0)

<span id="page-28-1"></span>![](_page_28_Figure_4.jpeg)

Fig. A3: Mean ∆<sup>m</sup> metric for experiments run on Cityscapes and NYUv2 with SegNet and DeepLabV3. We compare the performance of the best hyperparameter setting for every MTO method using either Adam (left) or SGD+Momentum (right) (lower is better). Every MTO is associated with a different line color/style. On Cityscapes, there is a large difference for the ∆<sup>m</sup> score for Adam compared to SGD+Momentum, especially for UW, IMTL, and CAGrad. Therefore, for this setup, the result depends more on the optimizer than on the MTO method. On the NYUv2 dataset this observation weakens. Adam still achieves the lowest ∆<sup>m</sup> scores across different MTO methods (except for SegNet with UW and IMTL), though, besides chosing Adam, it is also important to select the appropriate MTO method.

<span id="page-29-0"></span>![](_page_29_Figure_1.jpeg)

Fig. A4: Parallel coordinate plot over all experiments on NYUv2 We distinguish between experiments using SGD+mom and Adam optimizer. Experiments that reached Pareto front performance are drawn with higher saturation. Similiar to results on Cityscapes in the main paper (Fig. [2\)](#page-6-1) we observe a dominance of Adam albeit, here, we also have some experiments using SGD+Mom. on the overall Pareto front.

<span id="page-29-1"></span>Table A2: Count of Pareto optimal experiments for each MTO method. We found no single MTO method to be clearly superior over all combinations of dataset and networks. Total numbers can be compared to Table [2](#page-6-0)

<span id="page-30-0"></span>Table A3: Number of iterations after which all seeds in toy task experiment from CAGrad [\[31\]](#page-14-2) have reached the global minimum for different learning rates and optimizer. We show results for additional learning rates compared to the main paper. The maximum iteration number over all three seeds for each MTO method / learning rate / optimizer combination is reported. If not all seeds converged to the global minimum within 100k iteration steps, we denote it as '-'. In several setups, EW+Adam converges fastest to the global minimum. Especially for small learning rates, CAGrad performs advantageous compared to EW. As reported in previous work, we found that PCGrad often would converge only to some point on the Pareto Front. The best and second best run for each learning rate over all MTO methods are indicated via font type.

<sup>\*</sup>LR used for results in [\[31\]](#page-14-2) with Adam

<span id="page-31-0"></span>Table A4: Results for different MTO methods and optimizers on Cityscapes [\[9\]](#page-12-6) using SegNet [\[1\]](#page-12-7). The best score per metric is highlighted for each MTO method as well as over all methods and optimizers. While different MTO methods perform best over the distinct metrics, models trained with Adam outperform those based on SGD+mom in most direct comparisons. On the overall ∆<sup>m</sup> metric, Adam shows superior performance for all MTO methods, in some cases even with a high margin. Best performance for each metric was also achieved by using Adam.

Table A5: Results for different MTO methods and optimizers on Cityscapes [\[9\]](#page-12-6) using DeepLabV3+ [\[5\]](#page-12-8). The best score per metric is highlighted for each MTO method as well as over all methods and optimizers. Adam is Pareto dominant over SGD+mom in a direct pairwise comparison across all MTO methods. Here, we would also like to highlight that MTL can outperform STL as suggested by [\[4\]](#page-12-0).

Table A6: Results for different MTO methods and optimizers on NYUv2 [\[39\]](#page-14-13) using SegNet [\[1\]](#page-12-7). The best score per metric is highlighted for each MTO method as well as over all methods and optimizers. Using Adam yields in superior performance in the majority of cases, both when considering the individual metrics and the overall ∆<sup>m</sup> metric. We note that ∆<sup>m</sup> is more effected by the normal task due to the higher number of corresponding metrics as can be observed in the case of UW.

<span id="page-32-0"></span>Table A7: Results for different MTO methods and optimizers on NYUv2 [\[39\]](#page-14-13) using DeepLabV3+ [\[5\]](#page-12-8). The best score per metric is highlighted for each MTO method as well as over all methods and optimizers. We note a full dominance of Adam over SGD+mom on both the depth and normal tasks as well as on the ∆<sup>m</sup> metric. Overall, best results for all metrics were also achieved using Adam as optimizer.

# A5 Additional gradient alignment results

We report extended evaluation on gradient simlarity in MTL and STL.

Alternatively to a single sample/ task, we consider the average gradient in Fig. [A5.](#page-33-0) In Fig. [A6,](#page-34-1) we differentiate between conflicting and supporting gradient pairs when evaluating the cosine similarity. Figure [A7](#page-34-0) shows the evaluation of the scalar product as an combined measure of similarity in gradient direction and magnitude.

<span id="page-33-0"></span>![](_page_33_Figure_4.jpeg)

Fig. A5: Gradient similarities when averaging over batch/ losses. In contrast to results shown in Fig. [4,](#page-10-0) we compute either the average gradient over all tasks when comparing inter-samples or the average gradient over all samples within the batch for the comparison between inter-tasks. We observed a lower variance in some cases (e.g. Cityscapes+Segnet, grad. cosine similarity) which we trace back on noisy gradients being averaged out. Overall, we obtain the same findings as for a direct gradient comparison.

<span id="page-34-1"></span>![](_page_34_Figure_1.jpeg)

<span id="page-34-0"></span>![](_page_34_Figure_3.jpeg)

### <span id="page-35-0"></span>A6 Robustness of multi-task representations on corrupted data

In this additonal part of our analysis, we investigate whether features learned for multiple tasks generalize better to corrupted data compared to those learned for single tasks only.

Motivation: In his seminal paper, Caruana gives preliminary evidence that MTL provides stronger features and avoids spurious correlations (referred to better attribute selection) [\[4\]](#page-12-0). More recently, spurious correlations have often been directly connected with robustness [\[13,](#page-13-9) [18\]](#page-13-10). Results from current literature on the robustness of MTL features are mixed. While MTL is stated to increase the adversarial- and noise-robustness over STL [\[24,](#page-13-4) [37,](#page-14-5) [54\]](#page-15-14), others argue features selected by MTL could be more likely to be non-causal and, therefore, less robust [\[2,](#page-12-10) [17\]](#page-13-11). Here, we further examine whether MTL features lead to better robustness. We would like to nuance that we do not consider the transferability of representations, e.g., to new tasks, but solely focus on the claim that the MTL trained features are more robust w.r.t. different inputs.

Approach: In our experiment, we treat the common corruptions [\[16\]](#page-13-12) as downstream task and compare the performance after fine-tuning the heads on corrupted data while freezing the pre-trained STL/MTL backbone. While this differs from the typical OOD setup, here, it allows us to explore whether MTL or STL yield more robust representation for corrupted data.

We select models trained on clean data with the best performing hyperparameter configuration from previous experiments and fine-tune their heads on corrupted data. Following this, we compare the test performance of models trained in the multi-task setup to those that were learned for a single-task only. We use the perturbation modes proposed by Hendrycks et al. [\[16\]](#page-13-12) which include different variants of noise, blur, and weather conditions and apply five levels of severity. We randomly select corruption and severity level for each data sample during fine-tuning and create a full corrupted version of the test data considering all proposed corruptions and perturbation levels.

To quantify the robustness of single- and multi-task models, we first compute the individual task metrics M (e.g., mIoU) per task t for a STL and MTL network. Next, we compute the relative performance when each model is faced with corrupted data. Lastly, we calculate the difference of relative performances of the MTL compared to the STL model. In detail, over all corruption modes C and levels of severity S we have

δ<sup>t</sup> = 1 |C| · |S| X c∈C X s∈S (−1)<sup>p</sup>(t) δt,c,s (26)

<span id="page-35-1"></span>with δt,c,s = MMT L,corrupted t,c,s MMT L,clean t − MST L,corrupted t,c,s MST L,clean t

where p(t) = 1 if a higher value on task t corresponds to better performance and p(t) = 0 otherwise. This metric yields δ<sup>t</sup> < 0 if the MTL model was able to handle data corruption better. If the STL model is less impacted, we get δ<sup>t</sup> > 0.

<span id="page-36-0"></span>![](_page_36_Figure_1.jpeg)

Fig. A8: Transfer to out-of-distribution data for MTL and STL. For every task and respective metrics, we show the difference over relative performance decrease over all corruption modes averaged over five levels of severity and three runs. EW was used to train the MTL model on uncorrupted data. We color blocks in case either STL or MTL is able to handle the respective corruption better for all metrics of one task. Regarding the Cityscapes dataset, the performance on both task would strongly benefit in MTL setup. A similar behavior can be seen for NYUv2+DeepLabV3. Using SegNet on NYUv2, however, shows preferences towards STL features. Overall, we see a minor indication that MTL result in features that would generalize better to corrupted data.

Results: Figure [A8](#page-36-0) shows δ<sup>t</sup> for different corruption types for all network architectures and datasets with EW averaged over five corruption levels and

<span id="page-37-0"></span>Table A8: Out-Of Distribution transfer on corrupted Cityscapes [\[9\]](#page-12-6) dataset for different networks and MTO methods. We report difference between relative performance decrease for single-task and multi-task learning averaged over all modes of corruption and all levels of severity (cf. Equation [\(26\)](#page-35-1)). A value lower than zero indicates a better generalization capability of the MTL backbone, a positive value displays that the STL backbone shows a lower decrease when evaluated on the corrupted data. Results are averaged over runs for three seeds for both multi-task and single-task models. Overall, we observe a slight benefit in performance for the depth task when training for multiple tasks.

three seeds. Considering the combination DeepLabV3 and Cityscapes, on the semantic segmentation tasks, the STL models show a slightly lower decrease in performance on the corrupted data than MTL (δ<sup>t</sup> > 0 more often; shaded in red), indicating that the features learned for these respective tasks can better generalize to corrupted data. In contrast, the MTL model shows significant better relative performance on the depth task (δ<sup>t</sup> < 0 more often; shaded green). Comparing these observations to other dataset+network combinations, we find a strong robustness of MTL features for both setups of CityScape+SegNet and NYUv2+DeepLabV3 over all tasks. However, evaluation on NYUv2+SegNet resulted on average in better performance on STL features, especially for the depth task. Furthermore, we see little evidence for general higher robustness against certain types of corruption (e.g., higher robustness against weather conditions) for either MTL or STL across all setups.

The results of other MTO methods (Tabs. [A8](#page-37-0) and [A9\)](#page-38-0) indicate that it depends less on the used method but more on the choice of dataset and network architecture whether some tasks would benefit from MTL for learning more robust features. Averaged absolute scores can be found in Tabs. [A10](#page-39-0) and [A11.](#page-39-1)

Conclusion: Our experiments show that MTL can result in learning more robust features, either for a subset of tasks or even all. However, we could not observe a uniform pattern whether certain tasks consistently benefit from MTL. Instead, it depends on the task, the type of corruption, the network, and the dataset whether MTL or STL is superior towards corrupted data. Whether there

<span id="page-38-0"></span>Table A9: Out-Of Distribution transfer on corrupted NYUv2 [\[39\]](#page-14-13) dataset for different multi-task optimization methods. We report difference between relative performance decrease for single-task and multi-task learning averaged over all modes of corruption and all levels of severity (cf. Equation [\(26\)](#page-35-1)). A value lower than zero indicates a better generalization capability of the MTL backbone, a positive value displays that the STL backbone shows a lower decrease when evaluated on the corrupted data. Results are averaged over runs for three seeds for both multi-task and single-task models. While for DeepLabV3 largely benefits from MTL, this is not the case for SegNet. Over both networks, EW profits shows lowest relative performance decrease among all MTO methods. Interestingly we found that even for different metrics corresponding to the same task, either the multi-task or single-task learning model would show lower decrease in performance on the corrupted data.

is a general pattern, we leave to further research. We further cannot fully confirm the outcome of [\[24\]](#page-13-4) as only two of our setups have indicated that the segmentation task can be more robust in the MTL setting. Controversial to the claim of [\[37\]](#page-14-5), our evaluation shows that none of the MTL approaches, even IMTL, PC-Grad or CAGrad which adjust the gradients, yields consistent values of δ<sup>t</sup> < 0 which would have shown an advantage of certain MTO methods over STL.

<span id="page-39-0"></span>

Table A10: Results for evaluating on corrupted Cityscapes [\[9\]](#page-12-6). Scores are averaged over all corruption modes, level of severity and three seeds.

<span id="page-39-1"></span>

Table A11: Results for evaluating on corrupted NYUv2 [\[39\]](#page-14-13). Scores are averaged over all corruption modes, level of severity and three seeds.