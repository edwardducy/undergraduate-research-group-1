# Dual-Balancing for Multi-Task Learning

Baijiong Lin<sup>a,b</sup>, Weisen Jiang<sup>c</sup>, Feiyang Ye<sup>d</sup>, Yu Zhang<sup>d</sup>, Pengguang Chen<sup>e</sup>, Ying-Cong Chen<sup>a,b,f,∗</sup>, Shu Liu<sup>e,∗</sup>, Ivor W. Tsang<sup>g</sup>, James T. Kwok<sup>f</sup>

<sup>a</sup>The Hong Kong University of Science and Technology

(Guangzhou), Guangzhou, 510000, China

<sup>b</sup>HKUST(GZ) - SmartMore Joint Lab, Guangzhou, 510000, China

<sup>c</sup>The Chinese University of Hong Kong, Hong Kong, 999077, China

<sup>d</sup>Southern University of Science and Technology, Shenzhen, 518055, China

<sup>e</sup>SmartMore, Shenzhen, 518000, China

<sup>f</sup>The Hong Kong University of Science and Technology, Hong Kong, 999077, China

<sup>g</sup>Centre for Frontier AI Research, A<sup>∗</sup>STAR, 138632, Singapore

## Abstract

Multi-task learning aims to learn multiple related tasks simultaneously and has achieved great success in various fields. However, the disparity in loss and gradient scales among tasks often leads to performance compromises, and the balancing of tasks remains a significant challenge. In this paper, we propose Dual-Balancing Multi-Task Learning (DB-MTL) to achieve task balancing from both the loss and gradient perspectives. Specifically, DB-MTL achieves loss-scale balancing by performing logarithm transformation on each task loss, and rescales gradient magnitudes by normalizing all task gradients to comparable magnitudes using the maximum gradient norm. Extensive experiments on a number of benchmark datasets demonstrate that DB-MTL consistently performs better than the current state-of-the-art.

Keywords: multi-task learning, loss balancing, gradient balancing.

## 1. Introduction

Multi-task learning (MTL) [1, 2, 3] jointly learns multiple related tasks using a single model, improving parameter-eficiency and inference speed compared to learning a separate model for each task. By sharing the model, MTL can extract common knowledge to improve each task’s performance. It has demonstrated its superiority in various fields, such as computer vision [4, 5, 6, 7, 8], natural language processing [9, 10, 11, 12, 13], and recommendation systems [14, 15, 16, 17].

To learn multiple tasks simultaneously, equal weighting (EW) [2] is a straightforward method that minimizes the sum of task losses with equal task weights. However, it usually sufers from the challenging task balancing problem [4, 18], in which some tasks perform well while others do not [19]. To alleviate this problem, a number of methods have been recently proposed by dynamically tuning the task weights. They can be categorized as loss balancing [20, 21, 22, 23, 24, 25] and gradient balancing [26, 27, 28, 29, 30, 31, 13, 32, 33]. Loss balancing methods balance the tasks based on the learning speed [21] or validation performance [22, 24, 25] at the loss level, while gradient balancing methods balance the gradients by mitigating gradient conflicts [29] or enforcing gradient norms to be close [26] at the gradient level. However, recently, multiple extensive empirical studies [18, 34, 35] demonstrate that the performance of these existing methods is still unsatisfactory, indicating that task balancing is still an open problem.

To mitigate the task balancing problem, in this paper, we consider simultaneously balancing both the loss scales (at the loss level) and gradient magnitudes (at the gradient level). Since the loss scales/gradient magnitudes among tasks can be diferent, those with large values can dominate the update direction of the model, causing unsatisfactory performance on some other tasks [19, 31]. Therefore, we propose a simple yet efective Dual-Balancing Multi-Task Learning (DB-MTL) method that consists of both loss-scale and gradient-magnitude balancing. First, we perform a logarithm transformation on each task loss to make all task losses have a similar scale. This is non-parametric and can recover the loss transformation in IMTL-L [31]. We find that the logarithm transformation also benefits existing gradient balancing methods. Second, we normalize all task gradients to the same magnitude as the maximum gradient norm. This is training-free and guarantees all gradients’ magnitude are the same compared with GradNorm [26]. Empirically, we find that the magnitude of normalized gradients plays an important role in performance, and setting it as the maximum gradient norm among tasks performs the best. Extensive experiments are performed on a number of benchmark datasets. Results demonstrate that DB-MTL consistently outperforms the current state-of-the-art.

Our contributions can be summarized as follows:

1. We propose DB-MTL, a novel dual-balancing approach that simultaneously addresses both loss-scale and gradient-magnitude imbalances in multi-task learning through:

• A parameter-free logarithm transformation for loss-scale balancing that efectively equalizes loss scales across tasks;  
• A maximum-norm gradient normalization strategy that rescales all task gradients to comparable magnitudes for balanced model updates.

2. We conduct extensive experiments across diverse benchmarks demonstrating that DB-MTL consistently outperforms state-of-the-art MTL methods.

Notations. For clarity, we summarize the key notations used throughout this paper. We use $T$ to denote the number of tasks, $\mathcal { D } _ { t }$ for the training dataset of task $t , \theta$ and $\{ \psi _ { t } \} _ { t = 1 } ^ { T }$ for task-sharing and task-specific parameters respectively, $\gamma _ { t }$ for task weights, and $\ell _ { t }$ for the loss function of task t. $\mathbf { g } _ { t , k }$ and $\tilde { \bf g } _ { k }$ represent the gradient and aggregated gradient at iteration $k _ { : }$ , with $\alpha _ { k }$ as the scaling factor.

## 2. Related Works

In an MTL problem with $T$ tasks, we aim to learn a model from $\{ \mathcal { D } _ { t } \} _ { t = 1 } ^ { T }$ where $\mathcal { D } _ { t }$ is the training dataset of task t. The MTL model parameters can be divided into two parts: (i) task-sharing parameter $\theta ,$ and (ii) taskspecific parameters $\{ \psi _ { t } \} _ { t = 1 } ^ { T }$ . For example, in computer vision tasks, $\pmb \theta$ usually represents a feature encoder $( \mathrm { e . g . }$ , ResNet [36]) to extract common features among tasks, while $\psi _ { t }$ corresponds to the task-specific output module (e.g., a fully-connected layer). For parameter eficiency, $\pmb \theta$ contains most of the MTL model parameters, and is crucial to the performance.

Let $\ell _ { t } ( \mathcal { D } _ { t } ; \pmb { \theta } , \psi _ { t } )$ be the loss on task t’s data $\mathcal { D } _ { t }$ using parameter $( \pmb \theta , \psi _ { t } )$ The training objective of MTL is $\begin{array} { r } { \sum _ { t = 1 } ^ { T } \gamma _ { t } \ell _ { t } ( \mathcal { D } _ { t } ; \pmb { \theta } , \psi _ { t } ) } \end{array}$ , where $\gamma _ { t }$ is the weight for task t. Equal weighting (EW) [2] is a simple MTL approach that sets $\gamma _ { t } =$ 1 for all tasks. However, EW usually sufers from the task balancing problem in which some tasks have unsatisfactory performance [19]. To improve its performance, many other MTL methods have been proposed to dynamically tune the task weights $\{ \gamma _ { t } \} _ { i = 1 } ^ { T }$ during training. They can be categorized as loss balancing, gradient balancing, or hybrid balancing.

## 2.1. Loss Balancing Methods

This approach weights the task losses with $\{ \gamma _ { t } \} _ { i = 1 } ^ { T }$ that are computed dynamically. $\{ \gamma _ { t } \} _ { i = : } ^ { T }$ afect the update of both the task-sharing parameter θ and task-specific parameter $\{ \psi _ { t } \} _ { t = 1 } ^ { T }$ . They can be set based on measures such as homoscedastic uncertainty [20], learning speed [21], validation performance [22, 24], and improvable gap [37]. Alternatively, IMTL-L [31] encourages the weighted losses $\{ \gamma _ { t } \ell _ { t } ( \mathcal { D } _ { t } ; \pmb { \theta } , \psi _ { t } ) \} _ { t = 1 } ^ { T }$ to have similar loss scale across all tasks by transforming each loss $\ell _ { t } ( \mathcal { D } _ { t } ; \pmb { \theta } , \psi _ { t } )$ as $e ^ { s _ { t } } \ell _ { t } ( \mathcal { D } _ { t } ; \pmb { \theta } , \pmb { \psi } _ { t } ) - s _ { t }$ , where $\{ s _ { t } \} _ { t = 1 } ^ { T }$ are learnable parameters and obtained by gradient descent at each iteration.

## 2.2. Gradient Balancing Methods

The update of the task-sharing parameter $\pmb \theta$ depends on all task gradients $\{ \nabla _ { \pmb { \theta } } \ell _ { t } ( \mathcal { D } _ { t } ; \pmb { \theta } , \psi _ { t } ) \} _ { t = 1 } ^ { T }$ . Thus, gradient balancing methods aim to aggregate all task gradients in diferent manners. For example, MGDA [27] formulates MTL as a multi-objective optimization problem and selects the aggregated gradient with the minimum norm [38]. CAGrad [30] improves MGDA by constraining the aggregated gradient to be around the average gradient. MoCo [33] mitigates the bias in MGDA by introducing a momentum-like gradient estimate and a regularization term. GradNorm [26] learns task weights to scale the task gradients to similar magnitudes. PCGrad [29] projects the gradient of one task onto the normal plane of the other if their gradients conflict. GradVac [13] aligns the gradients regardless of whether the gradients conflict or not. GradDrop [28] randomly masks out gradient values with inconsistent signs. IMTL-G [31] learns task weights to enforce the aggregated gradient to have equal projections on each task gradient. Nash-MTL [32] formulates gradient aggregation as a Nash bargaining game.

For most gradient balancing methods (such as PCGrad [29], CAGrad [30], MoCo [33], GradDrop [28], and IMTL-G [31]), the task weight $\gamma _ { t }$ only afects update of the task-sharing parameter θ, while in some other gradient balancing methods (such as MGDA [27], GradNorm [26], and Nash-MTL [32]), the task weight $\gamma _ { t }$ afects the update of both the task-sharing and task-specific parameters.

## 2.3. Hybrid Balancing Methods

As loss balancing and gradient balancing are complementary, these two types of methods can be combined to achieve better performance. In this approach, the task weight $\gamma _ { t }$ is obtained as the product of the loss and gradient balancing weights. For example, the first hybrid balancing method IMTL [31] combines IMTL-L with IMTL-G. Subsequently, various combinations [18, 37, 25] of loss/gradient balancing methods demonstrate performance improvements. In this paper, we propose DB-MTL that combines the logarithm transformation (for loss balancing) and the maximum-norm gradient normalization (for gradient balancing).

## 3. Proposed Method

In this section, we alleviate the task balancing problem from both the loss and gradient perspectives. First, we balance all loss scales by performing logarithm transformation on each task’s loss (Section 3.1). Next, we achieve gradient-magnitude balancing by normalizing each task’s gradient to the same magnitude as the maximum gradient norm (Section 3.2). The procedure, called DB-MTL (Dual-Balancing Multi-Task Learning), is shown in Algorithm 1.

## 3.1. Scale-Balancing Loss Transformation

Tasks with diferent types of loss functions usually have diferent scales, leading to the task balancing problem. For example, in the NYUv2 dataset [39], the cross-entropy loss, $L _ { 1 }$ loss, and cosine loss are used as the loss functions of the semantic segmentation, depth estimation, and surface normal prediction tasks, respectively. As observed in [19, 29, 32] and also in our experimental results in Tables 1 and 6, surface normal prediction is afected by the other two tasks (semantic segmentation and depth estimation), causing MTL methods like EW to perform unsatisfactorily.

When prior knowledge of the loss scales is available, we can choose $\{ s _ { t } ^ { \star } \} _ { t = 1 } ^ { T }$ such that $\{ s _ { \pm } ^ { \star } \ell _ { t } ( \mathcal { D } _ { t } ; \pmb \theta , \psi _ { t } ) \} _ { t = 1 } ^ { T }$ have the same scale, and then minimize the total loss $\begin{array} { r l } { ~ } & { { } \sum _ { t = 1 } ^ { T } s _ { t } ^ { \star } \ell _ { t } ( \mathcal { D } _ { t } ; \pmb { \theta } , \psi _ { t } ) } \end{array}$ . Previous methods [20, 21, 31, 22] implicitly learn $\{ s _ { t } ^ { \star } \} _ { t = 1 } ^ { T }$ when learning the task weights $\{ \gamma _ { t } \} _ { t = 1 } ^ { T }$ . However, obviously the optimal $\{ s _ { t } ^ { \star } \} _ { t = 1 } ^ { T }$ cannot be obtained during training.

Without the availability of $\{ s _ { t } ^ { \star } \} _ { t = 1 } ^ { T }$ , the logarithm transformation can be used to alleviate the loss scale problem. Specifically, we transform each task’s loss $\ell _ { t } ( \mathcal { D } _ { t } ; \pmb { \theta } , \psi _ { t } )$ to log $\ell _ { t } ( \mathcal { D } _ { t } ; \pmb { \theta } , \psi _ { t } )$ , and then minimize $\textstyle \sum _ { t = 1 } ^ { T }$ log $\ell _ { t } ( \mathcal { D } _ { t } ; \pmb { \theta } , \psi _ { t } )$ . Since log(·) can compress the range of its input, it can reduce the loss scale gap between diferent tasks.

Algorithm 1 Dual-Balancing Multi-Task Learning.

<div class="mineru-algorithm" style="white-space: pre-wrap; font-family:monospace;">
Require: numbers of iterations $K$, learning rate $\eta$, tasks $\{\mathcal{D}_t\}_{t=1}^T$, $\epsilon = 10^{-8}$, $\beta$;
randomly initialize $\boldsymbol{\theta}_0, \{\boldsymbol{\psi}_{t,0}\}_{t=1}^T$;
initialize $\hat{\mathbf{g}}_{t,-1} = \mathbf{0}$, for all $t$;
for $k = 0, \ldots, K - 1$ do
    for $t = 1, \ldots, T$ do
        sample a mini-batch dataset $\mathcal{B}_{t,k}$ from $\mathcal{D}_t$;
        $\mathbf{g}_{t,k} = \nabla_{\boldsymbol{\theta}_k} \log(\ell_t(\mathcal{B}_{t,k}; \boldsymbol{\theta}_k, \boldsymbol{\psi}_{t,k}) + \epsilon)$;
        compute $\hat{\mathbf{g}}_{t,k} = \beta \hat{\mathbf{g}}_{t,k-1} + (1 - \beta) \mathbf{g}_{t,k}$;
    end for
    compute $\tilde{\mathbf{g}}_k = \alpha_k \sum_{t=1}^T \frac{\hat{\mathbf{g}}_{t,k}}{\|\hat{\mathbf{g}}_{t,k}\|_2 + \epsilon}$, where $\alpha_k = \max_{1 \leq t \leq T} \| \hat{\mathbf{g}}_{t,k} \|_2$;
    update task-sharing parameter by $\boldsymbol{\theta}_{k+1} = \boldsymbol{\theta}_k - \eta \tilde{\mathbf{g}}_k$;
    for $t = 1, \ldots, T$ do
        $\boldsymbol{\psi}_{t,k+1} = \boldsymbol{\psi}_{t,k} - \eta \nabla_{\boldsymbol{\psi}_{t,k}} \log(\ell_t(\mathcal{B}_{t,k}; \boldsymbol{\theta}_k, \boldsymbol{\psi}_{t,k}) + \epsilon)$;
    end for
end for
Return $\boldsymbol{\theta}_K, \{\boldsymbol{\psi}_{t,K}\}_{t=1}^T$.
</div>

IMTL-L [31] tackles the loss scale issue using a transformed loss $e ^ { s _ { t } } \ell _ { t } ( \mathcal { D } _ { t } ; \pmb { \theta } , \psi _ { t } ) -$ $s _ { t }$ , where $s _ { t }$ is a learnable parameter for the t-th task and approximately solved by one-step gradient descent at every iteration. The following Proposition 3.1 shows that IMTL-L is equivalent to the logarithm transformation when $s _ { t }$ is the exact minimizer in each iteration.

Proposition 3.1. For $x > 0 , \log ( x ) = \operatorname* { m i n } _ { s } e ^ { s } x - s - 1$

Proof. Define an auxiliary function $f ( s ) = e ^ { s } x - s - 1$ . It is easy to show that $\begin{array} { r } { \frac { \mathrm { d } f ( s ) } { \mathrm { d } s } = e ^ { s } x - 1 } \end{array}$ and $\begin{array} { r } { \frac { \mathrm { d } ^ { 2 } f ( s ) } { \mathrm { d } s ^ { 2 } } = e ^ { s } x > 0 } \end{array}$ . Thus, $f ( s )$ is convex. By the first-order optimal condition [40], let $e ^ { s ^ { \star } } x - 1 = 0$ , the global minimizer is solved as $s ^ { \star } = - \log ( x )$ . Therefore, $f ( s ^ { \star } ) = e ^ { s ^ { \star } } x - s ^ { \star } - 1 = e ^ { - \log ( x ) } x + \log ( x ) - 1 =$ log(x), where we finish the proof. □

Compared to IMTL-L, the logarithm transformation does not require additional parameters and computational cost during training. Thus, the logarithm transformation is simpler and more efective than IMTL-L.

## 3.2. Magnitude-Balancing Gradient Normalization

In addition to the task losses, task gradients also sufer from the scale issue. As the update direction of θ is obtained by uniformly averaging all task gradients, it may be dominated by the large task gradients, causing sub-optimal performance [29, 30].

A simple approach is to normalize task gradients to the same magnitude. As computing the batch gradient is computationally expensive, mini-batch stochastic gradient descent is often used in practice. Specifically, at iteration k, we sample a mini-batch $\boldsymbol { B } _ { t , k }$ from $\mathcal { D } _ { t }$ for the t-th task (step 5 in Algorithm 1) and compute the mini-batch gradient $\mathbf { g } _ { t , k } = \nabla _ { \pmb { \theta } _ { k } }$ log $\ell _ { t } ( B _ { t , k } ; \pmb { \theta } _ { k } , \pmb { \psi } _ { t , k } )$ (step 6 in Algorithm 1). Exponential moving average (EMA), which is popularly used in adaptive gradient methods (e.g., RMSProp [41], AdaDelta [42], and Adam [43]), is used to estimate $\mathbb { E } _ { B _ { t , k } \sim \mathcal { D } _ { t } } \nabla _ { \pmb { \theta } _ { k } }$ log $\ell _ { t } ( B _ { t , k } ; \pmb { \theta } _ { k } , \pmb { \psi } _ { t , k } )$ dynamically (step 7 in Algorithm 1) as

$$
\hat {\mathbf {g}} _ {t, k} = \beta \hat {\mathbf {g}} _ {t, k - 1} + (1 - \beta) \mathbf {g} _ {t, k}, \tag {1}
$$

where $\beta \in ( 0 , 1 )$ controls the forgetting rate. After obtaining the task gradients $\{ \hat { \mathbf { g } } _ { t , k } \} _ { t = 1 } ^ { T }$ , we normalize them to have the same $\ell _ { 2 } { \mathrm { - n o r m } }$ , and compute the aggregated gradient as

$$
\tilde {\mathbf {g}} _ {k} = \alpha_ {k} \sum_ {t = 1} ^ {T} \frac {\hat {\mathbf {g}} _ {t , k}}{\| \hat {\mathbf {g}} _ {t , k} \| _ {2}}, \tag {2}
$$

where $\alpha _ { k }$ is a scaling factor controlling the update magnitude. After normalization, all tasks contribute with comparable magnitudes to the update direction.

The choice of $\alpha _ { k }$ is critical in alleviating the task balancing problem. Intuitively, when some tasks have large gradient norms and others have small gradient norms, the first group of tasks has not yet converged while the second group of tasks has almost converged. The current model $\pmb { \theta } _ { k }$ is undesirable and can cause the task balancing problem as not all tasks have converged. Hence, $\alpha _ { k }$ should be large to escape this undesirable solution. On the other hand, when all task gradient norms are small, model $\pmb { \theta } _ { k }$ is close to a stationary solution for all tasks, and $\alpha _ { k }$ should be small so that the solution will no longer change. Thus, we choose $\alpha _ { k } = \operatorname* { m a x } _ { 1 \leq t \leq T } \| \hat { \mathbf { g } } _ { t , k } \| _ { 2 }$ , i.e., $\alpha _ { k }$ is small if and only if all the task gradient norms are small.

After scaling the losses and gradients, the task-sharing parameter is updated as $\pmb { \theta } _ { k + 1 } = \pmb { \theta } _ { k } - \eta \tilde { \mathbf { g } } _ { k }$ (step 10), where $\eta > 0$ is the learning rate. For the task-specific parameters $\{ \psi _ { t , k } \} _ { t = 1 } ^ { T }$ , as the update of each of them only depends on the corresponding task gradient separately, their gradients do not sufer from the gradient scaling issue. Hence, the update for task-specific parameters is simply $\psi _ { t , k + 1 } = \psi _ { t , k } - \eta \nabla _ { \psi _ { t , k } }$ log $\ell _ { t } ( B _ { t , k } ; \pmb { \theta } _ { k } , \psi _ { t , k } )$ (steps 11-13).

GradNorm [26] also aims to learn $\{ \gamma _ { t } \} _ { t = 1 } ^ { T }$ so that the scaled gradients have similar norms. However, it has two problems. First, alternating the updates of model parameters and task weights cannot guarantee all task gradients have the same magnitude in each iteration. Second, as will be seen from Figure 6 in Section 4.5, the choice of the update magnitude $\alpha _ { k }$ can significantly afect performance. However, this is not considered in GradNorm.

## 4. Experiments

In this section, we empirically evaluate the proposed DB-MTL on a number of tasks, including scene understanding (Section 4.1), molecular property prediction (Section 4.2), and image classification (Section 4.3).

## 4.1. Evaluation on Scene Understanding

Datasets. Following RLW [18], CAGrad [30], and Nash-MTL [32], the following two scene understanding datasets are used: (i) NYUv2 [39], which is an indoor scene understanding dataset. It has 3 tasks (13-class semantic segmentation, depth estimation, and surface normal prediction) with 795 training and 654 testing images. (ii) Cityscapes [44], which is an urban scene understanding dataset. It has 2 tasks (7-class semantic segmentation and depth estimation) with 2, 975 training and 500 testing images.

Baselines. The proposed DB-MTL is compared with a number of MTL baselines, including (i) equal weighting (EW) [2]; (ii) GLS [45], which minimizes the geometric mean loss $\begin{array} { r } { \sqrt [ T ] { \prod _ { t = 1 } ^ { T } \ell _ { t } ( \mathcal { D } _ { t } ; \pmb { \theta } , \psi _ { t } ) } } \end{array}$ ; (iii) RLW [18], in which the task weights are sampled from the standard normal distribution; (iv) loss balancing methods including UW [20], DWA [21], IMTL-L [31], and IGBv2 [37]; (v) gradient balancing methods including MGDA [27], GradNorm [26], PCGrad [29], GradDrop [28], GradVac [13], IMTL-G [31], CAGrad [30], MTAdam [46], Nash-MTL [32], MetaBalance [47], MoCo [33], and Aligned-MTL [48]; and (vi) hybrid balancing method IMTL [31]. For comparison, we also include the single-task learning (STL) baseline, which learns each task separately.

All methods are implemented based on the open-source LibMTL library [49]. For all MTL methods, the hard-parameter sharing (HPS) pattern [50] is used, which consists of a task-sharing feature encoder and T task-specific heads. For the proposed DB-MTL, following MoCo [33], we perform grid search for $\beta$ over $\{ \bar { 0 . 1 } , 0 . 5 , 0 . 9 , \frac { 0 . 1 } { k ^ { 0 . 5 } } , \frac { \bar { 0 } . 5 } { k ^ { 0 . 5 } } , \frac { 0 . 9 } { k ^ { 0 . 5 } } \}$ for each dataset, where k is the number of iterations.

Implementation Details. Following RLW [18], we use the DeepLabV3+ network [51], which contains a ResNet-50 network with dilated convolutions pre-trained on the ImageNet dataset [52] as the shared encoder and the Atrous Spatial Pyramid Pooling [51] module as task-specific head. We train the model for 200 epochs by using the Adam optimizer [43] with learning rate $1 0 ^ { - 4 }$ and weight decay $1 0 ^ { - 5 }$ . The learning rate is halved to $5 \times 1 0 ^ { - 5 }$ after 100 epochs. The cross-entropy loss $\begin{array} { r } { \ell _ { s e g } = - \frac { 1 } { N \times H \times W } \sum _ { n = 1 } ^ { N } \sum _ { i = 1 } ^ { H \times W } \sum _ { c = 1 } ^ { C } y _ { n , i , c } \log ( \hat { y } _ { n , i , c } ) } \end{array}$ $L _ { 1 }$ loss $\begin{array} { r } { \ell _ { d e p t h } = \frac { 1 } { N \times H \times W } \sum _ { n = 1 } ^ { N } \sum _ { i = 1 } ^ { H \times W } | d _ { n , i } - \hat { d } _ { n , i } | } \end{array}$ , and cosine loss $\ell _ { n o r m a l } =$ $\begin{array} { r } { \frac { 1 } { N \times H \times W } \sum _ { n = 1 } ^ { N } \sum _ { i = 1 } ^ { H \times W } ( 1 - \frac { \mathbf { n } _ { n , i } \cdot \hat { \mathbf { n } } _ { n , i } } { | | \mathbf { n } _ { n , i } | | \cdot | | \mathbf { \hat { n } } _ { n , i } | | } ) } \end{array}$ are used as the loss functions of the semantic segmentation, depth estimation, and surface normal prediction tasks, respectively, where N is the batch size, H and W are the height and width of the image, $y _ { n , i , c }$ and $\hat { y } _ { n , i , c }$ are the ground truth label and predicted probability for pixel i in image n and class $c , d _ { n , i }$ and $\hat { d } _ { n , i }$ are the ground truth and predicted depth values for pixel i in image n, and ${ \bf n } _ { n , i }$ and $\hat { \mathbf { n } } _ { n , i }$ are the ground truth and predicted normal vectors for pixel i in image n. For NYUv2, the images are resized to 288 × 384, and the batch size is 8. For Cityscapes, the images are resized to $1 2 8 \times 2 5 6$ , and the batch size is 64. Each experiment is repeated three times.

Performance Evaluation. Following DWA [21] and RLW [18], we use (i) the mean intersection over union (mIoU) and class-wise pixel accuracy (PAcc) for semantic segmentation; (ii) relative error (RErr) and absolute error (AErr) for depth estimation; (iii) mean and median angle errors, and percentage of normals within $t ^ { \circ }$ (where $t = 1 1 . 2 5 , 2 2 . 5 , 3 0 )$ for surface normal prediction. Following [53, 4, 18], we report the relative performance improvement of an MTL method A over STL, averaged over all the metrics above, i.e.,

$$
\Delta_ {\mathrm{p}} (\mathcal {A}) = \frac {1}{T} \sum_ {t = 1} ^ {T} \Delta_ {\mathrm{p}, t} (\mathcal {A}), \tag {3}
$$

Table 1: Performance on NYUv2 with 3 tasks. ↑ (↓) means the higher (lower) the result, the better the performance. The best and second best results are marked in bold and underline, respectively.

<table><tr><td rowspan="3"></td><td colspan="2">Segmentation</td><td colspan="2">Depth Estimation</td><td colspan="5">Surface Normal Prediction</td><td rowspan="3"> $\Delta_{\mathrm{p}}\uparrow$ </td></tr><tr><td rowspan="2">mIoU↑</td><td rowspan="2">PAcc↑</td><td rowspan="2">AErr↓</td><td rowspan="2">RErr↓</td><td colspan="2">Angle Distance</td><td colspan="3">Within  $t^{\circ}$ </td></tr><tr><td>Mean↓</td><td>MED↓</td><td>11.25↑</td><td>22.5↑</td><td>30↑</td></tr><tr><td>STL</td><td>53.50</td><td>75.39</td><td>0.3926</td><td>0.1605</td><td>21.99</td><td>15.16</td><td>39.04</td><td>65.00</td><td>75.16</td><td>0.00</td></tr><tr><td>EW</td><td>53.93</td><td>75.53</td><td>0.3825</td><td>0.1577</td><td>23.57</td><td>17.01</td><td>35.04</td><td>60.99</td><td>72.05</td><td>-1.78±0.45</td></tr><tr><td>GLS</td><td>54.59</td><td>76.06</td><td>0.3785</td><td>0.1555</td><td>22.71</td><td>16.07</td><td>36.89</td><td>63.11</td><td>73.81</td><td>+0.30±0.30</td></tr><tr><td>RLW</td><td>54.04</td><td>75.58</td><td>0.3827</td><td>0.1588</td><td>23.07</td><td>16.49</td><td>36.12</td><td>62.08</td><td>72.94</td><td>-1.10±0.40</td></tr><tr><td>UW</td><td>54.29</td><td>75.64</td><td>0.3815</td><td>0.1583</td><td>23.48</td><td>16.92</td><td>35.26</td><td>61.17</td><td>72.21</td><td>-1.52±0.39</td></tr><tr><td>DWA</td><td>54.06</td><td>75.64</td><td>0.3820</td><td>0.1564</td><td>23.70</td><td>17.11</td><td>34.90</td><td>60.74</td><td>71.81</td><td>-1.71±0.25</td></tr><tr><td>IMTL-L</td><td>53.89</td><td>75.54</td><td>0.3834</td><td>0.1591</td><td>23.54</td><td>16.98</td><td>35.09</td><td>61.06</td><td>72.12</td><td>-1.92±0.25</td></tr><tr><td>IGBv2</td><td>54.61</td><td>76.00</td><td>0.3817</td><td>0.1576</td><td>22.68</td><td>15.98</td><td>37.14</td><td>63.25</td><td>73.87</td><td>+0.05±0.29</td></tr><tr><td>MGDA</td><td>53.52</td><td>74.76</td><td>0.3852</td><td>0.1566</td><td>22.74</td><td>16.00</td><td>37.12</td><td>63.22</td><td>73.84</td><td>-0.64±0.25</td></tr><tr><td>GradNorm</td><td>53.91</td><td>75.38</td><td>0.3842</td><td>0.1571</td><td>23.17</td><td>16.62</td><td>35.80</td><td>61.90</td><td>72.84</td><td>-1.24±0.15</td></tr><tr><td>PCGrad</td><td>53.94</td><td>75.62</td><td>0.3804</td><td>0.1578</td><td>23.52</td><td>16.93</td><td>35.19</td><td>61.17</td><td>72.19</td><td>-1.57±0.44</td></tr><tr><td>GradDrop</td><td>53.73</td><td>75.54</td><td>0.3837</td><td>0.1580</td><td>23.54</td><td>16.96</td><td>35.17</td><td>61.06</td><td>72.07</td><td>-1.85±0.39</td></tr><tr><td>GradVac</td><td>54.21</td><td>75.67</td><td>0.3859</td><td>0.1583</td><td>23.58</td><td>16.91</td><td>35.34</td><td>61.15</td><td>72.10</td><td>-1.75±0.39</td></tr><tr><td>IMTL-G</td><td>53.01</td><td>75.04</td><td>0.3888</td><td>0.1603</td><td>23.08</td><td>16.43</td><td>36.24</td><td>62.23</td><td>73.06</td><td>-1.89±0.54</td></tr><tr><td>CAGrad</td><td>53.97</td><td>75.54</td><td>0.3885</td><td>0.1588</td><td>22.47</td><td>15.71</td><td>37.77</td><td>63.82</td><td>74.30</td><td>-0.27±0.35</td></tr><tr><td>MTAdam</td><td>52.67</td><td>74.86</td><td>0.3873</td><td>0.1583</td><td>23.26</td><td>16.55</td><td>36.00</td><td>61.92</td><td>72.74</td><td>-1.97±0.23</td></tr><tr><td>Nash-MTL</td><td>53.41</td><td>74.95</td><td>0.3867</td><td>0.1612</td><td>22.57</td><td>15.94</td><td>37.30</td><td>63.40</td><td>74.09</td><td>-1.01±0.13</td></tr><tr><td>MetaBalance</td><td>53.92</td><td>75.57</td><td>0.3901</td><td>0.1594</td><td>22.85</td><td>16.16</td><td>36.72</td><td>62.91</td><td>73.62</td><td>-1.06±0.17</td></tr><tr><td>MoCo</td><td>52.25</td><td>74.56</td><td>0.3920</td><td>0.1622</td><td>22.82</td><td>16.24</td><td>36.58</td><td>62.72</td><td>73.49</td><td>-2.25±0.51</td></tr><tr><td>Aligned-MTL</td><td>52.94</td><td>75.00</td><td>0.3884</td><td>0.1570</td><td>22.65</td><td>16.07</td><td>36.88</td><td>63.18</td><td>73.94</td><td>-0.98±0.56</td></tr><tr><td>IMTL</td><td>53.63</td><td>75.44</td><td>0.3868</td><td>0.1592</td><td>22.58</td><td>15.85</td><td>37.44</td><td>63.52</td><td>74.09</td><td>-0.57±0.24</td></tr><tr><td>DB-MTL (ours)</td><td>53.92</td><td>75.60</td><td>0.3768</td><td>0.1557</td><td>21.97</td><td>15.37</td><td>38.43</td><td>64.81</td><td>75.24</td><td>+1.15±0.16</td></tr></table>

where $T$ is the number of tasks and

$$
\Delta_ {\mathrm{p}, t} (\mathcal {A}) = 100\% \times \frac {1}{N _ {t}} \sum_ {i = 1} ^ {N _ {t}} (- 1) ^ {s _ {t, i}} \frac {M _ {t , i} ^ {\mathcal {A}} - M _ {t , i} ^ {\mathrm{STL}}}{M _ {t , i} ^ {\mathrm{STL}}},
$$

where $N _ { t }$ is the number of metrics for task $t , M _ { t , i } ^ { A }$ is the ith metric value of method A on task $t ,$ and $s _ { t , i }$ is 0 if a larger value indicates better performance for the ith metric on task t, and 1 otherwise.

Performance Results. Table 1 shows the results on NYUv2. As can be seen, the proposed DB-MTL performs the best in terms of average $\Delta _ { \mathrm { p } }$ . Note that most of the MTL baselines perform better than STL on semantic segmentation and depth estimation, but have a large drop on the surface normal prediction task, sufering from the task balancing problem. Only the proposed DB-MTL has comparable performance with STL on the surface normal prediction task and maintains superiority on the other tasks.

Table 2: Performance on Cityscapes with 2 tasks. ↑ (↓) indicates that the higher (lower) the result, the better the performance. The best and second best results are highlighted in bold and underline, respectively.

<table><tr><td rowspan="2"></td><td colspan="2">Segmentation</td><td colspan="2">Depth Estimation</td><td rowspan="2"> $\Delta_p \uparrow$ </td></tr><tr><td>mIoU↑</td><td>PAcc↑</td><td>AErr↓</td><td>RErr↓</td></tr><tr><td>STL</td><td>69.06</td><td>91.54</td><td>0.01282</td><td>43.53</td><td>0.00</td></tr><tr><td>EW</td><td>68.93</td><td>91.58</td><td>0.01315</td><td>45.90</td><td>-2.05±0.56</td></tr><tr><td>GLS</td><td>68.69</td><td>91.45</td><td>0.01280</td><td>44.13</td><td>-0.39±1.06</td></tr><tr><td>RLW</td><td>69.03</td><td>91.57</td><td>0.01343</td><td>44.77</td><td>-1.91±0.21</td></tr><tr><td>UW</td><td>69.03</td><td>91.61</td><td>0.01338</td><td>45.89</td><td>-2.45±0.68</td></tr><tr><td>DWA</td><td>68.97</td><td>91.58</td><td>0.01350</td><td>45.10</td><td>-2.24±0.28</td></tr><tr><td>IMTL-L</td><td>68.98</td><td>91.59</td><td>0.01340</td><td>45.32</td><td>-2.15±0.88</td></tr><tr><td>IGBv2</td><td>68.44</td><td>91.31</td><td>0.01290</td><td>45.03</td><td>-1.31±0.61</td></tr><tr><td>MGDA</td><td>69.05</td><td>91.53</td><td>0.01280</td><td>44.07</td><td>-0.19±0.30</td></tr><tr><td>GradNorm</td><td>68.97</td><td>91.60</td><td>0.01320</td><td>44.88</td><td>-1.55±0.70</td></tr><tr><td>PCGrad</td><td>68.95</td><td>91.58</td><td>0.01342</td><td>45.54</td><td>-2.36±1.17</td></tr><tr><td>GradDrop</td><td>68.85</td><td>91.54</td><td>0.01354</td><td>44.49</td><td>-2.02±0.74</td></tr><tr><td>GradVac</td><td>68.98</td><td>91.58</td><td>0.01322</td><td>46.43</td><td>-2.45±0.54</td></tr><tr><td>IMTL-G</td><td>69.04</td><td>91.54</td><td>0.01280</td><td>44.30</td><td>-0.46±0.67</td></tr><tr><td>CAGrad</td><td>68.95</td><td>91.60</td><td>0.01281</td><td>45.04</td><td>-0.87±0.88</td></tr><tr><td>MTAdam</td><td>68.43</td><td>91.26</td><td>0.01340</td><td>45.62</td><td>-2.74±0.20</td></tr><tr><td>Nash-MTL</td><td>68.88</td><td>91.52</td><td>0.01265</td><td>45.92</td><td>-1.11±0.21</td></tr><tr><td>MetaBalance</td><td>69.02</td><td>91.56</td><td>0.01270</td><td>45.91</td><td>-1.18±0.58</td></tr><tr><td>MoCo</td><td>69.62</td><td>91.76</td><td>0.01360</td><td>45.50</td><td>-2.40±1.50</td></tr><tr><td>Aligned-MTL</td><td>69.00</td><td>91.59</td><td>0.01270</td><td>44.54</td><td>-0.43±0.44</td></tr><tr><td>IMTL</td><td>69.07</td><td>91.55</td><td>0.01280</td><td>44.06</td><td>-0.32±0.10</td></tr><tr><td>DB-MTL (ours)</td><td>69.17</td><td>91.56</td><td>0.01280</td><td>43.46</td><td>+0.20±0.40</td></tr></table>

Table 2 shows the results on Cityscapes. As can be seen, DB-MTL again achieves the best in terms of average $\Delta _ { \mathrm { p } } .$ . Note that all MTL baselines perform worse than STL in terms of average $\Delta _ { \mathrm { p } }$ and only the proposed DB-MTL outperforms STL on all tasks.

## 4.2. Evaluation on Molecular Property Prediction

Dataset. Following Nash-MTL [32], we use the QM9 [54] dataset, which is for molecular property prediction with 11 tasks. Each task performs regression on one property. We use the same split as in Nash-MTL [32]: 110, 000 for training, 10, 000 for validation, and 10, 000 for testing.

Table 3: Performance (MAE) on QM9 with 11 tasks. ↑ (↓) indicates that the higher (lower) the result, the better the performance. The best and second best results are highlighted in bold and underline, respectively.

<table><tr><td></td><td> $\mu$ </td><td> $\alpha$ </td><td> $\epsilon_{\text{HOMO}}$ </td><td> $\epsilon_{\text{LUMO}}$ </td><td> $\langle R^2\rangle$ </td><td>ZPVE</td><td> $U_0$ </td><td>U</td><td>H</td><td>G</td><td> $c_v$ </td><td> $\Delta_p \uparrow$ </td></tr><tr><td>STL</td><td>0.062</td><td>0.192</td><td>58.82</td><td>51.95</td><td>0.529</td><td>4.52</td><td>63.69</td><td>60.83</td><td>68.33</td><td>60.31</td><td>0.069</td><td>0.00</td></tr><tr><td>EW</td><td>0.096</td><td>0.286</td><td>67.46</td><td>82.80</td><td>4.655</td><td>12.4</td><td>128.3</td><td>128.8</td><td>129.2</td><td>125.6</td><td>0.116</td><td>-146.3±7.86</td></tr><tr><td>GLS</td><td>0.332</td><td>0.340</td><td>143.1</td><td>131.5</td><td>1.023</td><td>4.45</td><td>53.35</td><td>53.79</td><td>53.78</td><td>53.34</td><td>0.111</td><td>-81.16±15.5</td></tr><tr><td>RLW</td><td>0.112</td><td>0.331</td><td>74.59</td><td>90.48</td><td>6.015</td><td>15.6</td><td>156.0</td><td>156.8</td><td>157.3</td><td>151.6</td><td>0.133</td><td>-200.9±13.4</td></tr><tr><td>UW</td><td>0.336</td><td>0.382</td><td>155.1</td><td>144.3</td><td>0.965</td><td>4.58</td><td>61.41</td><td>61.79</td><td>61.83</td><td>61.40</td><td>0.116</td><td>-92.35±13.9</td></tr><tr><td>DWA</td><td>0.103</td><td>0.311</td><td>71.55</td><td>87.21</td><td>4.954</td><td>13.1</td><td>134.9</td><td>135.8</td><td>136.3</td><td>132.0</td><td>0.121</td><td>-160.9±16.7</td></tr><tr><td>IMTL-L</td><td>0.277</td><td>0.355</td><td>150.1</td><td>135.2</td><td>0.946</td><td>4.46</td><td>58.08</td><td>58.43</td><td>58.46</td><td>58.06</td><td>0.110</td><td>-77.06±11.1</td></tr><tr><td>IGBv2</td><td>0.235</td><td>0.377</td><td>132.3</td><td>139.9</td><td>2.214</td><td>5.90</td><td>64.55</td><td>65.06</td><td>65.12</td><td>64.28</td><td>0.121</td><td>-99.86±10.4</td></tr><tr><td>MGDA</td><td>0.181</td><td>0.325</td><td>118.6</td><td>92.45</td><td>2.411</td><td>5.55</td><td>103.7</td><td>104.2</td><td>104.4</td><td>103.7</td><td>0.110</td><td>-103.0±8.62</td></tr><tr><td>GradNorm</td><td>0.114</td><td>0.341</td><td>67.17</td><td>84.66</td><td>7.079</td><td>14.6</td><td>173.2</td><td>173.8</td><td>174.4</td><td>168.9</td><td>0.147</td><td>-227.5±1.85</td></tr><tr><td>PCGrad</td><td>0.104</td><td>0.293</td><td>75.29</td><td>88.99</td><td>3.695</td><td>8.67</td><td>115.6</td><td>116.0</td><td>116.2</td><td>113.8</td><td>0.109</td><td>-117.8±3.97</td></tr><tr><td>GradDrop</td><td>0.114</td><td>0.349</td><td>75.94</td><td>94.62</td><td>5.315</td><td>15.8</td><td>155.2</td><td>156.1</td><td>156.6</td><td>151.9</td><td>0.136</td><td>-191.4±9.62</td></tr><tr><td>GradVac</td><td>0.100</td><td>0.299</td><td>68.94</td><td>84.14</td><td>4.833</td><td>12.5</td><td>127.3</td><td>127.8</td><td>128.1</td><td>124.7</td><td>0.117</td><td>-150.7±7.41</td></tr><tr><td>IMTL-G</td><td>0.670</td><td>0.978</td><td>220.7</td><td>249.7</td><td>19.48</td><td>55.6</td><td>1109</td><td>1117</td><td>1123</td><td>1043</td><td>0.392</td><td>-1250±90.9</td></tr><tr><td>CAGrad</td><td>0.107</td><td>0.296</td><td>75.43</td><td>88.59</td><td>2.944</td><td>6.12</td><td>93.09</td><td>93.68</td><td>93.85</td><td>92.32</td><td>0.106</td><td>-87.25±1.51</td></tr><tr><td>MTAdam</td><td>0.593</td><td>1.352</td><td>232.3</td><td>419.0</td><td>24.31</td><td>69.7</td><td>1060</td><td>1067</td><td>1070</td><td>1007</td><td>0.627</td><td>-1403±203</td></tr><tr><td>Nash-MTL</td><td>0.115</td><td>0.263</td><td>85.54</td><td>86.62</td><td>2.549</td><td>5.85</td><td>83.49</td><td>83.88</td><td>84.05</td><td>82.96</td><td>0.097</td><td>-73.92±2.12</td></tr><tr><td>MetaBalance</td><td>0.090</td><td>0.277</td><td>70.50</td><td>78.43</td><td>4.192</td><td>11.2</td><td>113.7</td><td>114.2</td><td>114.5</td><td>111.7</td><td>0.110</td><td>-125.1±7.98</td></tr><tr><td>MoCo</td><td>0.489</td><td>1.096</td><td>189.5</td><td>247.3</td><td>34.33</td><td>64.5</td><td>754.6</td><td>760.1</td><td>761.6</td><td>720.3</td><td>0.522</td><td>-1314±65.2</td></tr><tr><td>Aligned-MTL</td><td>0.123</td><td>0.295</td><td>98.07</td><td>94.56</td><td>2.397</td><td>5.90</td><td>86.42</td><td>87.42</td><td>87.19</td><td>86.75</td><td>0.106</td><td>-80.58±4.18</td></tr><tr><td>IMTL</td><td>0.138</td><td>0.344</td><td>106.1</td><td>102.9</td><td>2.595</td><td>7.84</td><td>102.5</td><td>103.0</td><td>103.2</td><td>100.8</td><td>0.110</td><td>-104.3±11.7</td></tr><tr><td>DB-MTL (ours)</td><td>0.112</td><td>0.264</td><td>89.26</td><td>86.59</td><td>2.429</td><td>5.41</td><td>60.33</td><td>60.78</td><td>60.80</td><td>60.59</td><td>0.098</td><td>-58.10±3.89</td></tr></table>

Table 4: Classification accuracy (%) on Ofice-31 and Ofice-Home. ↑ indicates that the higher the result, the better the performance. The best and second best results are highlighted in bold and underline, respectively. Results of MoCo are from [33].

<table><tr><td rowspan="2"></td><td colspan="5">Office-31</td><td colspan="6">Office-Home</td></tr><tr><td>Amazon</td><td>DSLR</td><td>Webcam</td><td>Avg↑</td><td> $\Delta_p \uparrow$ </td><td>Artistic</td><td>Clipart</td><td>Product</td><td>Real</td><td>Avg↑</td><td> $\Delta_p \uparrow$ </td></tr><tr><td>STL</td><td>86.61</td><td>95.63</td><td>96.85</td><td>93.03</td><td>0.00</td><td>65.59</td><td>79.60</td><td>90.47</td><td>80.00</td><td>78.91</td><td>0.00</td></tr><tr><td>EW</td><td>83.53</td><td>97.27</td><td>96.85</td><td> $92.55_{\pm 0.62}$ </td><td> $-0.61_{\pm 0.67}$ </td><td>65.34</td><td>78.04</td><td>89.80</td><td>79.50</td><td> $78.17_{\pm 0.37}$ </td><td> $-0.92_{\pm 0.59}$ </td></tr><tr><td>GLS</td><td>82.84</td><td>95.62</td><td>96.29</td><td> $91.59_{\pm 0.58}$ </td><td> $-1.63_{\pm 0.61}$ </td><td>64.51</td><td>76.85</td><td>89.83</td><td>79.56</td><td> $77.69_{\pm 0.27}$ </td><td> $-1.58_{\pm 0.46}$ </td></tr><tr><td>RLW</td><td>83.82</td><td>96.99</td><td>96.85</td><td> $92.55_{\pm 0.89}$ </td><td> $-0.59_{\pm 0.95}$ </td><td>64.96</td><td>78.19</td><td>89.48</td><td>80.11</td><td> $78.18_{\pm 0.12}$ </td><td> $-0.92_{\pm 0.14}$ </td></tr><tr><td>UW</td><td>83.82</td><td>97.27</td><td>96.67</td><td> $92.58_{\pm 0.84}$ </td><td> $-0.56_{\pm 0.90}$ </td><td>65.97</td><td>77.65</td><td>89.41</td><td>79.28</td><td> $78.08_{\pm 0.30}$ </td><td> $-0.98_{\pm 0.46}$ </td></tr><tr><td>DWA</td><td>83.87</td><td>96.99</td><td>96.48</td><td> $92.43_{\pm 0.56}$ </td><td> $-0.70_{\pm 0.62}$ </td><td>65.27</td><td>77.64</td><td>89.05</td><td>79.56</td><td> $77.88_{\pm 0.28}$ </td><td> $-1.26_{\pm 0.49}$ </td></tr><tr><td>IMTL-L</td><td>84.04</td><td>96.99</td><td>96.48</td><td> $92.50_{\pm 0.52}$ </td><td> $-0.63_{\pm 0.58}$ </td><td>65.90</td><td>77.28</td><td>89.37</td><td>79.38</td><td> $77.98_{\pm 0.38}$ </td><td> $-1.10_{\pm 0.61}$ </td></tr><tr><td>IGBv2</td><td>84.52</td><td>98.36</td><td>98.05</td><td> $93.64_{\pm 0.26}$ </td><td> $+0.56_{\pm 0.25}$ </td><td>65.59</td><td>77.57</td><td>89.79</td><td>78.73</td><td> $77.92_{\pm 0.21}$ </td><td> $-1.21_{\pm 0.22}$ </td></tr><tr><td>MGDA</td><td>85.47</td><td>95.90</td><td>97.03</td><td> $92.80_{\pm 0.14}$ </td><td> $-0.27_{\pm 0.15}$ </td><td>64.19</td><td>77.60</td><td>89.58</td><td>79.31</td><td> $77.67_{\pm 0.20}$ </td><td> $-1.61_{\pm 0.34}$ </td></tr><tr><td>GradNorm</td><td>83.58</td><td>97.26</td><td>96.85</td><td> $92.56_{\pm 0.87}$ </td><td> $-0.59_{\pm 0.94}$ </td><td>66.28</td><td>77.86</td><td>88.66</td><td>79.60</td><td> $78.10_{\pm 0.63}$ </td><td> $-0.90_{\pm 0.93}$ </td></tr><tr><td>PCGrad</td><td>83.59</td><td>96.99</td><td>96.85</td><td> $92.48_{\pm 0.53}$ </td><td> $-0.68_{\pm 0.57}$ </td><td>66.35</td><td>77.18</td><td>88.95</td><td>79.50</td><td> $77.99_{\pm 0.19}$ </td><td> $-1.04_{\pm 0.32}$ </td></tr><tr><td>GradDrop</td><td>84.33</td><td>96.99</td><td>96.30</td><td> $92.54_{\pm 0.42}$ </td><td> $-0.59_{\pm 0.46}$ </td><td>63.57</td><td>77.86</td><td>89.23</td><td>79.35</td><td> $77.50_{\pm 0.23}$ </td><td> $-1.86_{\pm 0.24}$ </td></tr><tr><td>GradVac</td><td>83.76</td><td>97.27</td><td>96.67</td><td> $92.57_{\pm 0.73}$ </td><td> $-0.58_{\pm 0.78}$ </td><td>65.21</td><td>77.43</td><td>89.23</td><td>78.95</td><td> $77.71_{\pm 0.19}$ </td><td> $-1.49_{\pm 0.28}$ </td></tr><tr><td>IMTL-G</td><td>83.41</td><td>96.72</td><td>96.48</td><td> $92.20_{\pm 0.89}$ </td><td> $-0.97_{\pm 0.95}$ </td><td>64.70</td><td>77.17</td><td>89.61</td><td>79.45</td><td> $77.98_{\pm 0.38}$ </td><td> $-1.10_{\pm 0.61}$ </td></tr><tr><td>CAGrad</td><td>83.65</td><td>95.63</td><td>96.85</td><td> $92.04_{\pm 0.79}$ </td><td> $-1.14_{\pm 0.85}$ </td><td>64.01</td><td>77.50</td><td>89.65</td><td>79.53</td><td> $77.73_{\pm 0.16}$ </td><td> $-1.50_{\pm 0.29}$ </td></tr><tr><td>MTAdam</td><td>85.52</td><td>95.62</td><td>96.29</td><td> $92.48_{\pm 0.87}$ </td><td> $-0.60_{\pm 0.93}$ </td><td>62.23</td><td>77.86</td><td>88.73</td><td>77.94</td><td> $76.69_{\pm 0.65}$ </td><td> $-2.94_{\pm 0.85}$ </td></tr><tr><td>Nash-MTL</td><td>85.01</td><td>97.54</td><td>97.41</td><td> $93.32_{\pm 0.82}$ </td><td> $+0.24_{\pm 0.89}$ </td><td>66.29</td><td>78.76</td><td>90.04</td><td>80.11</td><td> $78.80_{\pm 0.52}$ </td><td> $-0.08_{\pm 0.69}$ </td></tr><tr><td>MetaBalance</td><td>84.21</td><td>95.90</td><td>97.40</td><td> $92.50_{\pm 0.28}$ </td><td> $-0.63_{\pm 0.30}$ </td><td>64.01</td><td>77.50</td><td>89.72</td><td>79.24</td><td> $77.61_{\pm 0.42}$ </td><td> $-1.70_{\pm 0.54}$ </td></tr><tr><td>MoCo</td><td>84.33</td><td>97.54</td><td>98.33</td><td>93.39</td><td>-</td><td>63.38</td><td>79.41</td><td>90.25</td><td>78.70</td><td>77.93</td><td>-</td></tr><tr><td>Aligned-MTL</td><td>83.36</td><td>96.45</td><td>97.04</td><td> $92.28_{\pm 0.46}$ </td><td> $-0.90_{\pm 0.48}$ </td><td>64.33</td><td>76.96</td><td>89.87</td><td>79.93</td><td> $77.77_{\pm 0.70}$ </td><td> $-1.50_{\pm 0.89}$ </td></tr><tr><td>IMTL</td><td>83.70</td><td>96.44</td><td>96.29</td><td> $92.14_{\pm 0.85}$ </td><td> $-1.02_{\pm 0.92}$ </td><td>64.07</td><td>76.85</td><td>89.65</td><td>79.81</td><td> $77.59_{\pm 0.29}$ </td><td> $-1.72_{\pm 0.45}$ </td></tr><tr><td>DB-MTL (ours)</td><td>85.12</td><td>98.63</td><td>98.51</td><td> $94.09_{\pm 0.19}$ </td><td> $+1.05_{\pm 0.20}$ </td><td>67.42</td><td>77.89</td><td>90.43</td><td>80.07</td><td> $78.95_{\pm 0.35}$ </td><td> $+0.17_{\pm 0.44}$ </td></tr></table>

Implementation Details. The experimental setups are the same with Nash-MTL [32]. Specifically, a graph neural network [55] is used as the shared encoder, and a linear layer is used as the task-specific head. The targets of each task are normalized to have zero mean and unit standard deviation. The batch size and training epoch are set to 128 and 300, respectively. The Adam optimizer [43] with the learning rate 0.001 is used for training, and the ReduceLROnPlateau scheduler [56] is used to reduce the learning rate once $\Delta _ { \mathrm { p } }$ on the validation dataset stops improving. The mean squared error (MSE) $\begin{array} { r } { \ell _ { m s e } = \frac { 1 } { N } \sum _ { n = 1 } ^ { N } ( p _ { n } - \hat { p } _ { n } ) ^ { 2 } } \end{array}$ is used as the loss function for each molecular property prediction task, where N is the batch size, $p _ { n }$ and ${ \hat { p } } _ { n }$ are the ground truth and predicted property values for sample n respectively. Mean absolute error (MAE) is used for performance evaluation. Each experiment is repeated three times.

Performance Results. Table 3 shows each task’s testing MAE and overall performance $\Delta _ { \mathrm { p } }$ (Eq. (3)) on QM9, using the same set of baselines as in Section 4.1. Note that QM9 is a challenging dataset in MTL and none of the MTL methods performs better than STL, as also observed in previous works [57, 32]. DB-MTL performs the best among all MTL methods and greatly improves over the second-best MTL method, Nash-MTL, in terms of average $\Delta _ { \mathrm { p } } .$

## 4.3. Evaluation on Image Classification

Datasets. Following RLW [18] and MoCo [33], two image classification datasets are used: (i) Ofice-31 [58], which contains 4, 110 images from three domains (tasks): Amazon, DSLR, and Webcam. Each task has 31 classes. (ii) Office-Home [59], which contains 15, 500 images from four domains (tasks): artistic images, clipart, product images, and real-world images. Each task has 65 object categories collected under ofice and home settings. We use the commonly-used data split as in RLW [18]: 60% for training, 20% for validation, and 20% for testing.

Implementation Details. Following RLW [18], a ResNet-18 [36] pre-trained on the ImageNet dataset [52] is used as a shared encoder, and a linear layer is used as a task-specific head. We resize the input image to 224 × 224. The batch size and number of training epochs are set to 64 and 100, respectively. The Adam optimizer [43] with learning rate $1 0 ^ { - 4 }$ and weight decay $1 0 ^ { - 5 }$ is used. For each image classification task, the cross-entropy loss $\ell _ { c l s } =$ $\begin{array} { r } { - \frac { 1 } { N } \sum _ { n = 1 } ^ { N } \sum _ { c = 1 } ^ { C } y _ { n , c } \log ( \hat { y } _ { n , c } ) } \end{array}$ is used as the loss function, where N is the batch size, $y _ { n , c }$ is the ground truth label and $\hat { y } _ { n , c }$ is the predicted probability for sample n and class c. Classification accuracy is used for evaluation. $\Delta _ { \mathrm { p } }$ in Eq. (3) is used as the overall performance metrics. Each experiment is repeated three times.

Table 5: Efects of each component in DB-MTL on diferent datasets in terms of $\Delta _ { \mathrm { p } }$ (Eq. (3)).

<table><tr><td>loss-scale balancing</td><td>gradient-magnitude balancing</td><td>NYUv2</td><td>Cityscapes</td><td>Office-31</td><td>Office-Home</td><td>QM9</td></tr><tr><td>✗</td><td>✗</td><td> $-1.78_{\pm 0.45}$ </td><td> $-2.05_{\pm 0.56}$ </td><td> $-0.61_{\pm 0.67}$ </td><td> $-0.92_{\pm 0.59}$ </td><td> $-146.3_{\pm 7.86}$ </td></tr><tr><td>✓</td><td>✗</td><td> $+0.06_{\pm 0.09}$ </td><td> $-0.38_{\pm 0.39}$ </td><td> $+0.93_{\pm 0.42}$ </td><td> $-0.73_{\pm 0.95}$ </td><td> $-74.40_{\pm 13.2}$ </td></tr><tr><td>✗</td><td>✓</td><td> $+0.76_{\pm 0.25}$ </td><td> $+0.12_{\pm 0.70}$ </td><td> $+0.01_{\pm 0.39}$ </td><td> $-0.78_{\pm 0.49}$ </td><td> $-65.73_{\pm 2.86}$ </td></tr><tr><td>✓</td><td>✓</td><td> $+1.15_{\pm 0.16}$ </td><td> $+0.20_{\pm 0.40}$ </td><td> $+1.05_{\pm 0.20}$ </td><td> $+0.17_{\pm 0.44}$ </td><td> $-58.10_{\pm 3.89}$ </td></tr></table>

![](images/74584ffe357f672729bedca0cb8c1f1e2c7106d2cb05d97e51a006038a2b6946.jpg)

<details>
<summary>bar</summary>

| Category | vanilla | w/ loss-scale balancing |
| --- | --- | --- |
| PCGrad | ~-1.5 | ~0.3 |
| GradVac | ~-1.7 | ~0.2 |
| IMTL-G | ~-1.9 | 0 |
| CAGrad | ~-0.3 | ~0.3 |
| Nash-MTL | -1 | ~-0.7 |
| Aligned-MTL | -1 | ~-0.1 |
</details>

Figure 1: Performance of existing gradient balancing methods with the loss-scale balancing method (i.e., logarithm transformation) on $N Y U v 2 .$ “vanilla” stands for the original method.

Performance Results. Table 4 shows the results on Ofice-31 and Ofice-Home, using the same set of baselines as in Section 4.1. On Ofice-31, DB-MTL achieves the top testing accuracy on the DSLR and Webcam tasks, and comparable performance on the Amazon task. On Ofice-Home, DB-MTL ranks top two on the Artistic, Product, and Real tasks. On both datasets, DB-MTL achieves the best average testing accuracy and $\Delta _ { \mathrm { p } } ,$ showing its efectiveness and demonstrating that balancing both loss scale and gradient magnitude is efective.

## 4.4. Efectiveness of Loss and Gradient Balancing Components

Ablation Study. DB-MTL has two components: loss-scale balancing (i.e., logarithm transformation) in Section 3.1 and gradient-magnitude balancing in Section 3.2. In this experiment, we perform an ablation study on the effectiveness of each component. We consider the four combinations: (i) use neither loss-scale nor gradient-magnitude balancing (i.e., the EW baseline); (ii) use only loss-scale balancing; (iii) use only gradient-magnitude balancing; (iv) use both loss-scale and gradient-magnitude balancing (i.e., the proposed DB-MTL).

Table 5 shows the $\Delta _ { \mathrm { p } }$ ’s of the four combinations on five datasets (NYUv2, Cityscapes, Ofice-31, Ofice-Home, and QM9 ). As can be seen, on all datasets, both components are beneficial to DB-MTL and combining them achieves the best performance.

Efectiveness of Logarithm Transformation. The logarithm transformation can also be used with other gradient balancing methods. We integrate it into PCGrad [29], GradVac [13], IMTL-G [31], CAGrad [30], Nash-MTL [32], and Aligned-MTL [48]. The experiment is performed on NYUv2 using the setup in Section 4.1. Figure 1 shows the $\Delta _ { \mathrm { p } } ~ ( \mathrm { E q . } ~ ( 3 ) )$ . As can be seen, logarithm transformation is consistently beneficial for these gradient balancing methods, showing the efectiveness of logarithm transformation. Moreover, DB-MTL still outperforms these gradient balancing baselines when they are combined with logarithm transformation, demonstrating the efectiveness of the proposed DB-MTL method.

Further to the discussion in Section 3.1, we compare the loss-scale balancing method (i.e., using logarithm transformation only) with IMTL-L [31] on four datasets (NYUv2, Cityscapes, Ofice-31, and Ofice-Home). As can be seen from Figure 2, the logarithm transformation consistently outperforms IMTL-L in terms of average $\Delta _ { \mathrm { p } }$ (Eq. (3)).

Efectiveness of Gradient-Magnitude Balancing. Further to the discussion in Section 3.2, we conduct a comparison between the proposed gradientmagnitude balancing method (i.e., DB-MTL without using logarithm transformation) and GradNorm [26] on four datasets: NYUv2, Cityscapes, Ofice-31, and Ofice-Home. As can be seen from Figure 3, the proposed method consistently achieves better performance than GradNorm in terms of average $\Delta _ { \mathrm { p } }$ on all datasets, demonstrating its efectiveness.

![](images/dbd0e489eb13f3a06a07401e57d2222492fd79136afe7182a5b313952541886b.jpg)

<details>
<summary>bar</summary>

| Category | IMTL-L | loss-scale balancing |
| --- | --- | --- |
| NYUv2 | ~-1.9 | ~0.1 |
| Cityscapes | ~-2.1 | ~-0.4 |
| Office-31 | ~-0.6 | ~0.9 |
| Office-Home | ~-1.1 | ~-0.7 |
</details>

Figure 2: Comparison of IMTL-L [31] and the loss-scale balancing method on four datasets.

![](images/d9123bea7e69146a1ec60a261e29635529197ff314db2740a47acc8e74abb8a2.jpg)

<details>
<summary>bar</summary>

| Category | GradNorm \((\Delta p)\) | gradient-magnitude balancing \((\Delta p)\) |
| --- | --- | --- |
| NYUv2 | ~-1.2 | ~0.8 |
| Cityscapes | ~-1.5 | ~0.1 |
| Office-31 | ~-0.6 | ~0.0 |
| Office-Home | ~-0.9 | ~-0.8 |
</details>

Figure 3: Comparison of GradNorm [26] and the gradient-magnitude balancing method on four datasets.

![](images/134be105890a78bfe313494369af9eba62b28404a0df725fd188c974341224cc.jpg)

<details>
<summary>bar</summary>

| Category | GLS | IGBv2 | DB-MTL |
| --- | --- | --- | --- |
| Segmentation | ~0.5 | ~1.3 | ~1.4 |
| Depth Estimation | ~3.9 | ~2.5 | ~4.3 |
| Normal Prediction | ~-1.9 | ~-3.6 | ~0.1 |
</details>

(a) Cross-stitch.

![](images/5dd24fc430762467ad78eda2c27ffa814df5f8894fc4f07cdf57d2d20939e9fa.jpg)

<details>
<summary>bar</summary>

| Category | GLS | IGBv2 | DB-MTL |
| --- | --- | --- | --- |
| Segmentation | ~2.1 | ~1.4 | ~0.1 |
| Depth Estimation | ~5.0 | ~2.5 | ~3.7 |
| Normal Prediction | ~-2.9 | ~-3.0 | ~-0.4 |
</details>

(b) MTAN.  
Figure 4: Performance on NYUv2 for Cross-stitch [60] and MTAN [21] architectures.

## 4.5. Sensitivity Analysis

Efect of MTL Architecture. The proposed DB-MTL is agnostic to the choice of MTL architectures. In this section, we demonstrate this by evaluating DB-MTL on NYUv2 using two more MTL architectures: Cross-stitch [60] and MTAN [21]. We compare with GLS [45] and IGBv2 [37], which perform well in Table 1. The implementation details are the same as in Section 4.1.

Figure 4 shows each task’s improvement performance $\Delta _ { \mathrm { p } , t }$ . For Crossstitch (Figure 4a), DB-MTL performs the best on all tasks. For MTAN (Figure 4b), all the MTL methods (GLS, IGBv2, and DB-MTL) perform better than STL on both semantic segmentation and depth estimation, but only DB-MTL achieves comparable performance as STL on the surface normal prediction task.

Table 6: Performance on the NYUv2 dataset with SegNet network. ↑ (↓) indicates that the higher (lower) the result, the better the performance. The best and second best results are highlighted in bold and underline, respectively. Superscripts ♯, §, ‡, and ∗ denote the results are from [30], [32], [33], and [48], respectively.

<table><tr><td rowspan="3"></td><td colspan="2">Segmentation</td><td colspan="2">Depth Estimation</td><td colspan="5">Surface Normal Prediction</td><td rowspan="3"> $\Delta_{p}\uparrow$ </td></tr><tr><td rowspan="2">mIoU↑</td><td rowspan="2">PAcc↑</td><td rowspan="2">AErr↓</td><td rowspan="2">RErr↓</td><td colspan="2">Angle Distance</td><td colspan="3">Within  $t^{\circ}$ </td></tr><tr><td>Mean↓</td><td>MED↓</td><td>11.25↑</td><td>22.5↑</td><td>30↑</td></tr><tr><td>STL§</td><td>38.30</td><td>63.76</td><td>0.6754</td><td>0.2780</td><td>25.01</td><td>19.21</td><td>30.14</td><td>57.20</td><td>69.15</td><td>0.00</td></tr><tr><td>EW§</td><td>39.29</td><td>65.33</td><td>0.5493</td><td>0.2263</td><td>28.15</td><td>23.96</td><td>22.09</td><td>47.50</td><td>61.08</td><td>+0.88</td></tr><tr><td>GLS</td><td>39.78</td><td>65.63</td><td>0.5318</td><td>0.2272</td><td>26.13</td><td>21.08</td><td>26.57</td><td>52.83</td><td>65.78</td><td>+5.15</td></tr><tr><td>RLW§</td><td>37.17</td><td>63.77</td><td>0.5759</td><td>0.2410</td><td>28.27</td><td>24.18</td><td>22.26</td><td>47.05</td><td>60.62</td><td>-2.16</td></tr><tr><td>UW§</td><td>36.87</td><td>63.17</td><td>0.5446</td><td>0.2260</td><td>27.04</td><td>22.61</td><td>23.54</td><td>49.05</td><td>63.65</td><td>+0.91</td></tr><tr><td>DWA§</td><td>39.11</td><td>65.31</td><td>0.5510</td><td>0.2285</td><td>27.61</td><td>23.18</td><td>24.17</td><td>50.18</td><td>62.39</td><td>+1.93</td></tr><tr><td>IMTL-L</td><td>39.78</td><td>65.27</td><td>0.5408</td><td>0.2347</td><td>26.26</td><td>20.99</td><td>26.42</td><td>53.03</td><td>65.94</td><td>+4.39</td></tr><tr><td>IGBv2</td><td>38.03</td><td>64.29</td><td>0.5489</td><td>0.2301</td><td>26.94</td><td>22.04</td><td>24.77</td><td>50.91</td><td>64.12</td><td>+2.11</td></tr><tr><td>MGDA§</td><td>30.47</td><td>59.90</td><td>0.6070</td><td>0.2555</td><td>24.88</td><td>19.45</td><td>29.18</td><td>56.88</td><td>69.36</td><td>-1.66</td></tr><tr><td>GradNorm*</td><td>20.09</td><td>52.06</td><td>0.7200</td><td>0.2800</td><td>24.83</td><td>18.86</td><td>30.81</td><td>57.94</td><td>69.73</td><td>-11.7</td></tr><tr><td>PCGrad§</td><td>38.06</td><td>64.64</td><td>0.5550</td><td>0.2325</td><td>27.41</td><td>22.80</td><td>23.86</td><td>49.83</td><td>63.14</td><td>+1.11</td></tr><tr><td>GradDrop§</td><td>39.39</td><td>65.12</td><td>0.5455</td><td>0.2279</td><td>27.48</td><td>22.96</td><td>23.38</td><td>49.44</td><td>62.87</td><td>+2.07</td></tr><tr><td>GradVac*</td><td>37.53</td><td>64.35</td><td>0.5600</td><td>0.2400</td><td>27.66</td><td>23.38</td><td>22.83</td><td>48.66</td><td>62.21</td><td>-0.49</td></tr><tr><td>IMTL-G§</td><td>39.35</td><td>65.60</td><td>0.5426</td><td>0.2256</td><td>26.02</td><td>21.19</td><td>26.20</td><td>53.13</td><td>66.24</td><td>+4.77</td></tr><tr><td>CAGrad#</td><td>39.18</td><td>64.97</td><td>0.5379</td><td>0.2229</td><td>25.42</td><td>20.47</td><td>27.37</td><td>54.73</td><td>67.73</td><td>+5.81</td></tr><tr><td>MTAdam</td><td>39.44</td><td>65.73</td><td>0.5326</td><td>0.2211</td><td>27.53</td><td>22.70</td><td>24.04</td><td>49.61</td><td>62.69</td><td>+3.21</td></tr><tr><td>Nash-MTL§</td><td>40.13</td><td>65.93</td><td>0.5261</td><td>0.2171</td><td>25.26</td><td>20.08</td><td>28.40</td><td>55.47</td><td>68.15</td><td>+7.65</td></tr><tr><td>MetaBalance</td><td>39.85</td><td>65.13</td><td>0.5445</td><td>0.2261</td><td>27.35</td><td>22.66</td><td>23.70</td><td>49.69</td><td>63.09</td><td>+2.67</td></tr><tr><td>MoCo†</td><td>40.30</td><td>66.07</td><td>0.5575</td><td>0.2135</td><td>26.67</td><td>21.83</td><td>25.61</td><td>51.78</td><td>64.85</td><td>+4.85</td></tr><tr><td>Aligned-MTL*</td><td>40.82</td><td>66.33</td><td>0.5300</td><td>0.2200</td><td>25.19</td><td>19.71</td><td>28.88</td><td>56.23</td><td>68.54</td><td>+8.16</td></tr><tr><td>IMTL</td><td>41.19</td><td>66.37</td><td>0.5323</td><td>0.2237</td><td>26.06</td><td>20.77</td><td>26.76</td><td>53.48</td><td>66.32</td><td>+6.45</td></tr><tr><td>DB-MTL (ours)</td><td>41.42</td><td>66.45</td><td>0.5251</td><td>0.2160</td><td>25.03</td><td>19.50</td><td>28.72</td><td>56.17</td><td>68.73</td><td>+8.91</td></tr></table>

Efect ofBackbone Network. We perform an experiment to evaluate DB-MTL on NYUv2 with the SegNet network [61] as the backbone. The implementation details are the same as in Section 4.1, except that the batch size is set to 2 and data augmentation is used (following CAGrad [30]). As can be seen from Table 6, DB-MTL again achieves the best performance in terms of average $\Delta _ { \mathrm { p } }$

Efect of EMA’s Forgetting Rate $\beta$ in Eq. (1). As mentioned in Section 4.1, we perform grid search for $\beta$ over $\{ 0 . 1 , \stackrel { - } { 0 . 5 } , \stackrel { - } { 0 . 9 } , \frac { 0 . 1 } { k ^ { 0 . 5 } } , \frac { 0 . 5 } { k ^ { 0 . 5 } } , \frac { 0 . 9 } { k ^ { 0 . 5 } } \}$ , where k is the number of iterations. In this experiment, we run DB-MTL on Ofice-31 with $\beta \in \{ 0 , 0 . 1 , 0 . 2 , \ldots , 0 . 9 , \frac { 0 . 1 } { k ^ { 0 . 5 } } , \frac { 0 . 2 } { k ^ { 0 . 5 } } , \ldots , \frac { 0 . 9 } { k ^ { 0 . 5 } } \}$ . The experimental setup is the same as in Section 4.3. As can be seen from Figure 5, the average $\Delta _ { \mathrm { p } }$ of DB-MTL is insensitive over a large range of $\beta ~ \big ( \{ \frac { 0 . 1 } { k ^ { 0 . 5 } } , \frac { 0 . 2 } { k ^ { 0 . 5 } } , \dots , \frac { 0 . 9 } { k ^ { 0 . 5 } } \big \} ,$ ), and performs better than DB-MTL without EMA $( \beta = 0 )$

![](images/e7bb5c1396c2f0f7c2b7415a385ff6f4e52ce1138115bddc5b77814f025c3e47.jpg)

<details>
<summary>boxplot</summary>

| Category | Q1 | Q2 (Median) | Q3 |
| --- | --- | --- | --- |
| 0 | ~0.8 | ~0.9 | ~0.95 |
| 0.1/k^0.5 | ~0.9 | ~1.05 | ~1.2 |
| 0.2/k^0.5 | ~0.7 | ~1.0 | ~1.3 |
| 0.3/k^0.5 | ~0.8 | ~1.1 | ~1.4 |
| 0.4/k^0.5 | ~1.05 | ~1.2 | ~1.3 |
| 0.5/k^0.5 | ~0.75 | ~1.2 | ~1.6 |
| 0.6/k^0.5 | ~1.5 | ~1.6 | ~1.7 |
| 0.7/k^0.5 | ~0.85 | ~1.0 | ~1.15 |
| 0.8/k^0.5 | ~0.85 | ~0.95 | ~1.05 |
| 0.9/k^0.5 | ~1.05 | ~1.1 | ~1.15 |
| 0.1 | ~0.6 | ~0.75 | ~0.9 |
| 0.2 | ~-0.25 | ~0.2 | ~0.55 |
| 0.3 | ~-0.4 | ~-0.2 | ~0.1 |
| 0.4 | ~-0.35 | ~0.15 | ~0.45 |
| 0.5 | ~-1.05 | ~-0.45 | ~-0.2 |
| 0.6 | ~-0.85 | ~-0.65 | ~-0.5 |
| 0.7 | ~-0.9 | ~-0.65 | ~-0.45 |
| 0.8 | ~-1.1 | ~-1.0 | ~-0.9 |
| 0.9 | ~-3.2 | ~-2.9 | ~-2.4 |
</details>

Figure 5: Efect of EMA’s Forgetting Rate $\beta$ in Eq. (1) on the Ofice-31 dataset. k denotes the number of iterations.

![](images/38feb2f4c974d6053378b28f02f70fa3a617dc1b505bf5a6617b2bb38ac3bc17.jpg)

<details>
<summary>boxplot</summary>

| Category | Q1 | Q2 (Median) | Q3 |
| --- | --- | --- | --- |
| 0.01 | ~-0.65 | ~-0.48 | ~-0.25 |
| 0.05 | ~0.38 | ~0.40 | ~0.42 |
| 0.1 | ~0.45 | ~0.48 | ~0.52 |
| 0.5 | ~0.25 | ~0.32 | ~0.45 |
| 1 | ~0.25 | ~0.32 | ~0.42 |
| 5 | ~0.18 | ~0.25 | ~0.35 |
| 10 | ~-0.42 | ~-0.35 | ~0.28 |
| 1/T | ~0.25 | ~0.32 | ~0.48 |
| min | ~0.28 | ~0.38 | ~0.48 |
| max | ~1.15 | ~1.25 | ~1.35 |
| mean | ~0.75 | ~0.82 | ~0.92 |
| median | ~0.48 | ~0.55 | ~0.62 |
</details>

Figure 6: $\Delta _ { \mathrm { p } }$ of diferent strategies for $\alpha _ { k }$ in $\operatorname { E q . }$ (2) on the NYUv2 dataset. $\mathrm { ^ { 6 } m i n } ^ { \mathrm { 7 } }$ $\mathrm { \ddot { \ ' } m a x ^ { 3 } , \mathrm { \ddot { \Omega } m e a n ^ { 3 } } }$ , and “median” denote the minimum, maximum, average, and median of $\| \hat { \mathbf { g } } _ { t , k } \| _ { 2 } ~ ( t = 1 , \ldots , T )$ , respectively. $T$ is the number of tasks.

![](images/7b57e8f52f596f4bdc40bb8b9c19621723abde8ada91715cdef93f9aca65645f.jpg)

<details>
<summary>bar</summary>

| Algorithm | Running Time (s/epoch) |
| :--- | :--- |
| EW | ~85 |
| GLS | ~85 |
| RLW | ~85 |
| UW | ~85 |
| DWA | ~85 |
| IMTL-L | ~85 |
| IGBv2 | ~85 |
| MGDA | ~240 |
| GradNorm | ~225 |
| PCGrad | ~245 |
| GradDrop | ~200 |
| GradVac | ~240 |
| IMTL-G | ~240 |
| CAGrad | ~240 |
| MTAdam | ~250 |
| Nash-MTL | ~300 |
| MetaBalance | ~240 |
| MoCo | ~245 |
| Aligned-MTL | ~250 |
| IMTL | ~250 |
| DB-MTL | ~245 |
</details>

Figure 7: The running time per epoch averaged 100 repetitions of diferent methods on NYUv2 dataset. Cyan, red, yellow, and blue denote loss balancing methods, gradient balancing methods, hybrid balancing methods, and others, respectively.

Efect of $\alpha _ { k }$ in $E q . \ ( 2 )$ . In this experiment, we use diferent settings of $\alpha _ { k }$ in Eq. (2), namely, (i) constant; (ii) minimum of $\{ \| \hat { \mathbf { g } } _ { t , k } \| _ { 2 } \} _ { t = 1 } ^ { T } ; \quad \mathrm { ( i i i ) }$ maximum of $\{ \| \hat { \mathbf { g } } _ { t , k } \| _ { 2 } \} _ { t = 1 } ^ { T } ;$ (iv) average of $\{ \| \hat { \mathbf { g } } _ { t , k } \| _ { 2 } \} _ { t = 1 } ^ { T } ;$ (v) median of $\{ \| \hat { \mathbf { g } } _ { t , k } \| _ { 2 } \} _ { t = 1 } ^ { T }$

![](images/841947e95bbc075ec7455419267d6c24e91ef65e93cd0360c4f284f6397dd6d5.jpg)  
Figure 8: Gradient norm curves of EW and DB-MTL on the NYUv2 dataset.

Figure 6 compares the results of these diferent DB-MTL variants on NYUv2. The experimental setup is the same as in Section 4.1. As can be seen, the maximum-norm strategy performs much better in terms of average $\Delta _ { \mathrm { p } } ,$ and thus it is used.

## 4.6. Analysis of Training Eficiency

Figure 7 shows the per-epoch running time of diferent MTL methods on NYUv2 dataset. All methods are run for 100 epochs on a single NVIDIA GeForce RTX 3090 GPU and the average running time per epoch is reported. As can be seen, DB-MTL has a similar running time as gradient balancing methods (i.e., MGDA [27], GradNorm [26], PCGrad [29], GradVac [13], IMTL-G [31], CAGrad [30], MTAdam [46], MetaBalance [47], MoCo [33], and Aligned-MTL [48]) and IMTL [31], but is larger than the loss balancing methods because each task’s gradient is computed in every iteration (i.e., step 6 in Algorithm 1). This is a common disadvantage for gradient balancing methods [13, 26, 32, 30, 27, 31, 29, 48, 46, 47]. Although DB-MTL is slower than loss balancing methods, it achieves better performance, as shown in Tables 1, 2, 3, 4, and 6.

## 4.7. Analysis of Training Stability

Figures 8 and 9 compare the gradient norms $\| \nabla _ { \pmb { \theta } _ { k } } \ell _ { t } ( B _ { t , k } ; \pmb { \theta } _ { k } , \pmb { \psi } _ { t , k } ) \| .$ and training losses of EW and DB-MTL on the NYUv2 dataset. As can be seen, for each task, the training loss of DB-MTL decreases smoothly and finally converges, and the gradient norm of DB-MTL is much more lower than EW. This indicates the logarithm transformation and maximum-norm strategy do not afect training stability.

![](images/86145959f574a841ddd6042749e2f9c79d82f454637a5a7742e12df93c89244d.jpg)

<details>
<summary>line</summary>

| Epochs | EW (Training Loss) | DB-MTL (Training Loss) |
| --- | --- | --- |
| 0 | 0.5 | 0.5 |
| ~10 | ~0.18 | ~0.15 |
| ~60 | ~0.37 | ~0.04 |
| ~95 | ~0.28 | ~0.03 |
| 200 | ~0.01 | ~0.01 |
</details>

(a) Segmentation task.

![](images/d54673a8f5dae7b5aec0e3f92e9fae1ab660d289db19e4eea3720e61cd7ae36c.jpg)

<details>
<summary>line</summary>

| Epochs | EW (Training Loss) | DB-MTL (Training Loss) |
| --- | --- | --- |
| 0 | ~0.8 | ~0.8 |
| 100 | ~0.15 | ~0.15 |
| 200 | ~0.08 | ~0.08 |
</details>

(b) Depth estimation task.

![](images/15b110ed13343905d5428808e0689b6e1653614c70ce9872cf447f11717cb0d3.jpg)

<details>
<summary>line</summary>

| Epochs | EW (Training Loss) | DB-MTL (Training Loss) |
| --- | --- | --- |
| 0 | ~0.3 | ~0.3 |
| 50 | ~0.06 | ~0.04 |
| 100 | ~0.04 | ~0.02 |
| 150 | ~0.03 | ~0.02 |
| 200 | ~0.02 | ~0.01 |
</details>

(c) Normal prediction task.

Figure 9: Training loss curves of EW and DB-MTL on the NYUv2 dataset.  
![](images/13fb0e1cb707c7fb5fc61d2da0be10328104422e9841676b39767f3da001572f.jpg)

<details>
<summary>line</summary>

| Epochs | EW | DB-MTL |
| --- | --- | --- |
| 0 | ~-0.02 | ~-0.015 |
| 50 | ~-0.003 | ~0.004 |
| 100 | ~-0.003 | ~0.002 |
</details>

(a) Amazon vs. DSLR.

![](images/e98eac95a078ab07ee510032e49f1282da02dffd540f410a88f60b742c9d1254.jpg)

<details>
<summary>line</summary>

| Epochs | EW | DB-MTL |
| --- | --- | --- |
| 0 | ~0.002 | ~0.002 |
| 5 | ~-0.018 | ~-0.012 |
| 10 | ~-0.008 | ~0.014 |
| 20 | ~-0.003 | ~0.008 |
| 30 | ~-0.002 | ~0.016 |
| 40 | ~-0.003 | ~0.022 |
| 50 | ~-0.004 | ~0.008 |
| 60 | ~-0.005 | ~0.016 |
| 70 | ~-0.004 | ~0.018 |
| 80 | ~-0.005 | ~0.009 |
| 90 | ~-0.004 | ~0.011 |
| 100 | ~-0.005 | ~0.008 |
</details>

(b) Amazon vs. Webcam.

![](images/cd4478b7f5cc67630a7d0031c829ba7aee4701fbd8199e9ac390efd165f4cbf4.jpg)

<details>
<summary>line</summary>

| Epochs | EW | DB-MTL |
| --- | --- | --- |
| 0 | ~-0.055 | ~-0.055 |
| 25 | ~-0.015 | ~0.015 |
| 50 | ~-0.008 | ~0.005 |
| 75 | ~-0.005 | ~0.005 |
| 100 | ~-0.005 | ~0.002 |
</details>

(c) DSLR vs. Webcam.  
Figure 10: Gradient cosine similarity of EW and DB-MTL on the Ofice-31 dataset.

## 4.8. Analysis of Gradient Conflict and Task Imbalance

Figures 10 shows the gradient cosine similarity of EW and DB-MTL on the Ofice-31 dataset, measuring the gradient conflict and task imbalance [29]. As can be seen, comapred to EW, the cosine similarity of DB-MTL increases faster and then keeps postive during the training process, indicating that DB-MTL can reduce the gradient conflict and improve the task balance.

## 5. Conclusion

In this paper, we alleviate the task-balancing problem in MTL by presenting Dual-Balancing Multi-Task Learning (DB-MTL), a novel approach that performs both loss-scale balancing (which makes all task losses have a similar scale via the logarithm transformation) and gradient-magnitude balancing (which rescales task gradients to comparable magnitudes using the maximum gradient norm). Extensive experiments on a number of benchmark datasets demonstrate that DB-MTL outperforms the current state-ofthe-art. Moreover, the logarithm transformation can also benefit existing gradient balancing methods. For future work, we will extend our approach to incorporate gradient variance in addition to magnitudes for more refined task weighting, and develop theoretical analysis to provide convergence guarantees and optimality conditions for our method.

## Acknowledgments

This work was supported in part by the National Natural Science Foundation of China under Grant No.92370204.

## References

[1] R. Caruana, Multitask learning, Machine Learning 28 (1997) 41–75.  
[2] Y. Zhang, Q. Yang, A survey on multi-task learning, IEEE Transactions on Knowledge and Data Engineering 34 (2022) 5586–5609.  
[3] W. Chen, X. Zhang, B. Lin, X. Lin, H. Zhao, Q. Zhang, J. T. Kwok, Gradient-based multi-objective deep learning: Algorithms, theories, applications, and beyond, arXiv preprint arXiv:2501.10945 (2025).  
[4] S. Vandenhende, S. Georgoulis, W. Van Gansbeke, M. Proesmans, D. Dai, L. Van Gool, Multi-task learning for dense prediction tasks: A survey, IEEE Transactions on Pattern Analysis and Machine Intelligence 44 (2021) 3614–3633.  
[5] H. Ye, D. Xu, Inverted pyramid multi-task transformer for dense scene understanding, in: European Conference on Computer Vision, 2022.  
[6] B. Lin, W. Jiang, P. Chen, Y. Zhang, S. Liu, Y.-C. Chen, MTMamba: Enhancing multi-task dense scene understanding by mamba-based decoders, in: European Conference on Computer Vision, 2024.  
[7] B. Lin, W. Jiang, P. Chen, S. Liu, Y.-C. Chen, MTMamba++: Enhancing multi-task dense scene understanding via mamba-based decoders, IEEE Transactions on Pattern Analysis and Machine Intelligence 47 (2025) 10633–10645.  
[8] H. Luo, W. Hu, Y. Wei, J. He, M. Yu, HirMTL: Hierarchical multi-task learning for dense scene understanding, Neural Networks 181 (2025) 106854.  
[9] P. Liu, X. Qiu, X.-J. Huang, Adversarial multi-task learning for text classification, in: Annual Meeting of the Association for Computational Linguistics, 2017.  
[10] X. Liu, P. He, W. Chen, J. Gao, Multi-task deep neural networks for natural language understanding, in: Annual Meeting of the Association for Computational Linguistics, 2019.  
[11] T. Sun, Y. Shao, X. Li, P. Liu, H. Yan, X. Qiu, X. Huang, Learning sparse sharing architectures for multiple tasks, in: AAAI Conference on Artificial Intelligence, 2020.  
[12] S. Chen, Y. Zhang, Q. Yang, Multi-task learning in natural language processing: An overview, ACM Computing Surveys 56 (2024) 1–32.  
[13] Z. Wang, Y. Tsvetkov, O. Firat, Y. Cao, Gradient vaccine: Investigating and improving multi-task optimization in massively multilingual models, in: International Conference on Learning Representations, 2021.  
[14] H. Tang, J. Liu, M. Zhao, X. Gong, Progressive layered extraction (PLE): A novel multi-task learning (MTL) model for personalized recommendations, in: ACM Conference on Recommender Systems, 2020.  
[15] H. Hazimeh, Z. Zhao, A. Chowdhery, M. Sathiamoorthy, Y. Chen, R. Mazumder, L. Hong, E. Chi, Dselect-k: Diferentiable selection in the mixture of experts with applications to multi-task learning, in: Neural Information Processing Systems, 2021.  
[16] Y. Wang, H. T. Lam, Y. Wong, Z. Liu, X. Zhao, Y. Wang, B. Chen, H. Guo, R. Tang, Multi-Task Deep Recommender Systems: A Survey, Preprint arXiv:2302.03525, 2023.  
[17] Q. Yi, L. Wu, J. Tang, Y. Zeng, Z. Song, Hybrid contrastive multiscenario learning for multi-task sequential-dependence recommendation, Neural Networks 183 (2025) 106953.  
[18] B. Lin, F. Ye, Y. Zhang, I. Tsang, Reasonable efectiveness of random weighting: A litmus test for multi-task learning, Transactions on Machine Learning Research (2022).  
[19] T. Standley, A. Zamir, D. Chen, L. Guibas, J. Malik, S. Savarese, Which tasks should be learned together in multi-task learning?, in: International Conference on Machine Learning, 2020.  
[20] A. Kendall, Y. Gal, R. Cipolla, Multi-task learning using uncertainty to weigh losses for scene geometry and semantics, in: IEEE Conference on Computer Vision and Pattern Recognition, 2018.  
[21] S. Liu, E. Johns, A. J. Davison, End-to-end multi-task learning with attention, in: IEEE Conference on Computer Vision and Pattern Recognition, 2019.  
[22] F. Ye, B. Lin, Z. Yue, P. Guo, Q. Xiao, Y. Zhang, Multi-objective meta learning, in: Neural Information Processing Systems, 2021.  
[23] F. Ye, B. Lin, Z. Yue, Y. Zhang, I. Tsang, Multi-objective meta-learning, Artificial Intelligence 335 (2024) 104184.  
[24] F. Ye, B. Lin, X. Cao, Y. Zhang, I. Tsang, A first-order multi-gradient algorithm for multi-objective bi-level optimization, in: European Conference on Artificial Intelligence, 2024.  
[25] S. Liu, S. James, A. Davison, E. Johns, Auto-Lambda: Disentangling dynamic task relationships, Transactions on Machine Learning Research (2022).  
[26] Z. Chen, V. Badrinarayanan, C.-Y. Lee, A. Rabinovich, GradNorm: Gradient normalization for adaptive loss balancing in deep multitask networks, in: International Conference on Machine Learning, 2018.  
[27] O. Sener, V. Koltun, Multi-task learning as multi-objective optimization, in: Neural Information Processing Systems, 2018.  
[28] Z. Chen, J. Ngiam, Y. Huang, T. Luong, H. Kretzschmar, Y. Chai, D. Anguelov, Just pick a sign: Optimizing deep multitask models with gradient sign dropout, in: Neural Information Processing Systems, 2020.  
[29] T. Yu, S. Kumar, A. Gupta, S. Levine, K. Hausman, C. Finn, Gradient surgery for multi-task learning, in: Neural Information Processing Systems, 2020.  
[30] B. Liu, X. Liu, X. Jin, P. Stone, Q. Liu, Conflict-averse gradient descent for multi-task learning, in: Neural Information Processing Systems, 2021.  
[31] L. Liu, Y. Li, Z. Kuang, J.-H. Xue, Y. Chen, W. Yang, Q. Liao, W. Zhang, Towards impartial multi-task learning, in: International Conference on Learning Representations, 2021.  
[32] A. Navon, A. Shamsian, I. Achituve, H. Maron, K. Kawaguchi, G. Chechik, E. Fetaya, Multi-task learning as a bargaining game, in: International Conference on Machine Learning, 2022.  
[33] H. D. Fernando, H. Shen, M. Liu, S. Chaudhury, K. Murugesan, T. Chen, Mitigating gradient bias in multi-objective learning: A provably convergent approach, in: International Conference on Learning Representations, 2023.  
[34] V. Kurin, A. De Palma, I. Kostrikov, S. Whiteson, M. P. Kumar, In defense of the unitary scalarization for deep multi-task learning, in: Neural Information Processing Systems, 2022.  
[35] D. Xin, B. Ghorbani, J. Gilmer, A. Garg, O. Firat, Do current multitask optimization methods in deep learning even help?, in: Neural Information Processing Systems, 2022.  
[36] K. He, X. Zhang, S. Ren, J. Sun, Deep residual learning for image recognition, in: IEEE Conference on Computer Vision and Pattern Recognition, 2016.  
[37] Y. Dai, N. Fei, Z. Lu, Improvable gap balancing for multi-task learning, in: Uncertainty in Artificial Intelligence, 2023.  
[38] J.-A. Désidéri, Multiple-gradient descent algorithm (MGDA) for multiobjective optimization, Comptes Rendus Mathematique 350 (2012) 313–318.  
[39] N. Silberman, D. Hoiem, P. Kohli, R. Fergus, Indoor segmentation and support inference from RGBD images, in: European Conference on Computer Vision, 2012.  
[40] S. Boyd, L. Vandenberghe, Convex optimization, Cambridge University Press, 2004.  
[41] T. Tieleman, G. Hinton, RMSProp: Neural networks for machine learning, Lecture 6.5, 2012.  
[42] M. D. Zeiler, AdaDelta: an adaptive learning rate method, Preprint arXiv:1212.5701, 2012.  
[43] D. P. Kingma, J. Ba, Adam: A method for stochastic optimization, in: International Conference on Learning Representations, 2015.  
[44] M. Cordts, M. Omran, S. Ramos, T. Rehfeld, M. Enzweiler, R. Benenson, U. Franke, S. Roth, B. Schiele, The cityscapes dataset for semantic urban scene understanding, in: IEEE Conference on Computer Vision and Pattern Recognition, 2016.  
[45] S. Chennupati, G. Sistu, S. Yogamani, S. A Rawashdeh, MultiNet++: Multi-stream feature aggregation and geometric loss strategy for multitask learning, in: IEEE Conference on Computer Vision and Pattern Recognition Workshops, 2019.  
[46] I. Malkiel, L. Wolf, MTAdam: Automatic balancing of multiple training loss terms, in: Conference on Empirical Methods in Natural Language Processing, 2021.  
[47] Y. He, X. Feng, C. Cheng, G. Ji, Y. Guo, J. Caverlee, MetaBalance: improving multi-task recommendations via adapting gradient magnitudes of auxiliary tasks, in: ACM Web Conference, 2022.  
[48] D. Senushkin, N. Patakin, A. Kuznetsov, A. Konushin, Independent component alignment for multi-task learning, in: IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023.  
[49] B. Lin, Y. Zhang, LibMTL: A Python library for multi-task learning, Journal of Machine Learning Research 24 (2023) 1–7.  
[50] R. Caruana, Multitask learning: A knowledge-based source of inductive bias, in: International Conference on Machine Learning, 1993.  
[51] L. Chen, Y. Zhu, G. Papandreou, F. Schrof, H. Adam, Encoder-decoder with atrous separable convolution for semantic image segmentation, in: European Conference on Computer Vision, 2018.  
[52] J. Deng, W. Dong, R. Socher, L.-J. Li, K. Li, L. Fei-Fei, ImageNet: A large-scale hierarchical image database, in: IEEE Conference on Computer Vision and Pattern Recognition, 2009.  
[53] K.-K. Maninis, I. Radosavovic, I. Kokkinos, Attentive single-tasking of multiple tasks, in: IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2019.  
[54] R. Ramakrishnan, P. O. Dral, M. Rupp, O. A. Von Lilienfeld, Quantum chemistry structures and properties of 134 kilo molecules, Scientific Data 1 (2014) 1–7.  
[55] J. Gilmer, S. S. Schoenholz, P. F. Riley, O. Vinyals, G. E. Dahl, Neural message passing for quantum chemistry, in: International Conference on Machine Learning, 2017.  
[56] A. Paszke, S. Gross, F. Massa, A. Lerer, J. Bradbury, G. Chanan, T. Killeen, Z. Lin, N. Gimelshein, L. Antiga, A. Desmaison, A. Köpf, E. Yang, Z. DeVito, M. Raison, A. Tejani, S. Chilamkurthy, B. Steiner, L. Fang, J. Bai, S. Chintala, PyTorch: An imperative style, highperformance deep learning library, in: Neural Information Processing Systems, 2019.  
[57] J. Gasteiger, J. Groß, S. Günnemann, Directional message passing for molecular graphs, in: International Conference on Learning Representations, 2020.  
[58] K. Saenko, B. Kulis, M. Fritz, T. Darrell, Adapting visual category models to new domains, in: European Conference on Computer Vision, 2010.  
[59] H. Venkateswara, J. Eusebio, S. Chakraborty, S. Panchanathan, Deep hashing network for unsupervised domain adaptation, in: IEEE Conference on Computer Vision and Pattern Recognition, 2017.  
[60] I. Misra, A. Shrivastava, A. Gupta, M. Hebert, Cross-stitch networks for multi-task learning, in: IEEE Conference on Computer Vision and Pattern Recognition, 2016.  
[61] V. Badrinarayanan, A. Kendall, R. Cipolla, SegNet: A deep convolutional encoder-decoder architecture for image segmentation, IEEE Transactions on Pattern Analysis and Machine Intelligence 39 (2017) 2481–2495.