# Reasonable Effectiveness of Random Weighting: A Litmus Test for Multi-Task Learning

Baijiong Lin<sup>1</sup>, Feiyang Ye<sup>1,2</sup>, Yu Zhang<sup>1,4,∗</sup>, Ivor W. Tsang<sup>3</sup>

<sup>1</sup> Department of Computer Science and Engineering,

Southern University of Science and Technology

<sup>2</sup> University of Technology Sydney

<sup>3</sup> Centre for Frontier AI Research (CFAR),

Agency for Science, Technology and Research (A<sup>∗</sup>STAR)

<sup>4</sup> Peng Cheng Laboratory

bj.lin.email@gmail.com,12060007@mail.sustech.edu.cn

yu.zhang.ust@gmail.com,ivor\_tsang@ihpc.a-star.edu.sg

## Abstract

Multi-Task Learning (MTL) has achieved success in various fields. However, how to balance different tasks to achieve good performance is a key problem. To achieve the task balancing, there are many works to carefully design dynamical loss/gradient weighting strategies but the basic random experiments are ignored to examine their effectiveness. In this paper, we propose the Random Weighting (RW) methods, including Random Loss Weighting (RLW) and Random Gradient Weighting (RGW), where an MTL model is trained with random loss/gradient weights sampled from a distribution. To show the effectiveness and necessity of RW methods, theoretically we analyze the convergence of RW and reveal that RW has a higher probability to escape local minima, resulting in better generalization ability. Empirically, we extensively evaluate the proposed RW methods to compare with twelve state-of-the-art methods on five image datasets and two multilingual problems from the XTREME benchmark to show RW methods can achieve comparable performance with state-of-the-art baselines. Therefore, we think that the RW methods are important baselines for MTL and should attract more attentions.

## 1 Introduction

Multi-Task Learning (MTL) [2, 38, 30] aims to jointly train several related tasks to improve their generalization performance by leveraging common knowledge among them. Since MTL could not only significantly reduce the model size as well as speed up the inference but also improve the performance, it has been successfully applied to various fields [38]. However, when all the tasks are not highly related, which may be reflected via conflicting gradients or dominating gradients [37], it is more difficult to train an MTL model than training them separately because some tasks dominantly influence model parameters, leading to unsatisfactory performance for other tasks. This phenomenon is related to the task balancing problem [30] in MTL.

Recently, several works focus on tackling this issue from an optimization perspective via dynamically weighting task losses or balancing task gradients in the training process, called loss balancing and gradient balancing methods, respectively. However, all of the existing works take Equal Weighting (EW) which uses the fixed and equal weights in the whole training process as a basic baseline to test the effectiveness of their methods. We think that this baseline is not sufficient and it is quite necessary to conduct random experiments, which is missing in existing works, as a baseline to test them.

Therefore, in this paper, we propose the Random Weighting (RW) methods including Random Loss Weighting (RLW) and Random Gradient Weighting (RGW) as more reasonable baselines to test loss and gradient balancing methods, respectively. Specifically, in each training iteration, we first sample loss/gradient weights from a distribution with some possible normalization and then minimize the aggregated loss/gradient weighted by the random loss/gradient weights. Although the RW methods seem unreasonable, they can not only converge but also achieve comparable performance with existing methods that use carefully tuned weights. Thus, we think the RW methods are important baselines for MTL and deserve more attention.

To better understand the effectiveness and necessity of RW methods, we provide both theoretical analyses and empirical evaluations. Theoretically, we show RW methods are the stochastic variants of EW. From this perspective, we give a convergence analysis for RW methods. Besides, we can show that RW methods have a higher probability to escape local minima than EW, resulting in better generalization performance. Empirically, we investigate lots of State-Of-The-Art (SOTA) task balancing approaches including four loss balancing methods and eight gradient balancing methods. On five Computer Vision (CV) datasets and two multilingual problems from the XTREME benchmark [11], we show that RW methods can consistently outperform EW and have competitive performance with existing SOTA methods.

In summary, the main contributions of this paper are three-fold.

• We propose the simple RW methods as novel baselines and litmus tests for MTL.  
• We provide the convergence guarantee and effectiveness analysis for RW methods.  
• Extensive experiments show that RW can outperform EW and achieve comparable performance with the SOTA methods.

## 2 An Overview of Task Balancing Methods

Notations. Suppose there are T tasks and task t has its corresponding dataset $\mathcal { D } _ { t }$ . An MTL model usually contains two parts of parameters: task-sharing parameters θ and task-specific parameters $\{ \psi _ { t } \} _ { t = 1 } ^ { T }$ . For example, in $\operatorname { C V } , \theta$ usually denotes parameters in the feature extractor shared by all the tasks and $\psi _ { t }$ represents parameters in the task-specific output module for task t. Let $\ell _ { t } ( \dot { \mathcal { D } _ { t } } ; \theta , \psi _ { t } )$ denotes the average loss on $\mathcal { D } _ { t }$ for task $t , \ \{ \lambda _ { t } \} _ { t = 1 } ^ { T }$ are task-specific loss weights with a constraint that $\lambda _ { t } ^ { l } \geq 0$ for all $t ' \mathbf { s } .$ Similarly, $\{ \lambda _ { t } ^ { g } \} _ { t = 1 } ^ { T }$ denote task-specific gradient weights.

Conventional Baseline with Fixed Weights. Since there are multiple losses in MTL, they usually are aggregated as a single one via loss weights as

$$
\mathcal {L} (\theta , \{\psi_ {t} \} _ {t = 1} ^ {T}) = \sum_ {t = 1} ^ {T} \lambda_ {t} ^ {l} \ell_ {t} (\mathcal {D} _ {t}; \theta , \psi_ {t}). \tag {1}
$$

Apparently, the most simple method for loss weighting is to assign the same weight to all the tasks in the whole training process, i.e., without loss of generality, $\begin{array} { r } { \lambda _ { t } ^ { l } = \frac { 1 } { T } } \end{array}$ for all t’s in every iteration. This approach is a common baseline in MTL and it is called EW in this paper.

Loss Balancing Methods. To achieve task balancing and improve the performance of MTL model, loss balancing methods aim to study how to generate appropriate loss weights $\{ \lambda _ { t } ^ { l } \} _ { t = 1 } ^ { T }$ in Eq. (1) in every iteration and some representative methods include Uncertainty Weights (UW) [13], Dynamic Weight Average (DWA) [19], IMTL-L [18] and Multi-Objective Meta Learning (MOML) [35]. These four methods focus on using higher loss weights for more difficult tasks measured by the uncertainty, learning speed, relative loss value, and validation performance, respectively. When minimizing Eq. (1), the learning rate of optimizing each task-specific parameter $\psi _ { t }$ will be affected by the corresponding loss weight $\ { \bar { \lambda } } _ { t } ^ { l } .$ , which is the major difference between loss balancing and gradient balancing methods.

Gradient Balancing Methods. This type of methods think that the task balancing problem is caused by conflicting task gradients and the inappropriate gradient to update task-sharing parameters, thus they solve it via generating appropriate gradient weights $\{ \lambda _ { t } ^ { g } \} _ { t = 1 } ^ { T }$ to balance the task gradients and make a better update of θ in every iteration as

$$
\theta = \theta - \eta \sum_ {t = 1} ^ {T} \lambda_ {t} ^ {g} \nabla_ {\theta} \ell_ {t} (\mathcal {D} _ {t}; \theta , \psi_ {t}). \tag {2}
$$

Noticeably, in such type methods, the gradient weights $\{ \lambda _ { t } ^ { g } \} _ { t = 1 } ^ { T }$ only affect the task-sharing parameter θ but not task-specific parameters $\{ \psi _ { t } \}$ , each of which is updated by the t-th task gradient $\nabla _ { \psi _ { t } } \ell _ { t } ( \mathcal { D } _ { t } ; \theta , \psi _ { t } )$

Some representative works include MGDA-UB [28], Gradient Normalization (GradNorm) [5], Projecting Conflicting Gradient (PCGrad) [37], Gradient sign Dropout (GradDrop) [6], Impartial Multi-Task Learning (IMTL-G) [18], Gradient Vaccine (GradVac) [32], Conflict-Averse Gradient (CAGrad) [17], and RotoGrad [12]. Those eight methods focus on finding an aggregated gradient by linearly combining all the task gradients under different constraints such as equal gradient magnitude in GradNorm and equal gradient projection in IMTL-G to eliminate the gradient conflicting.

Compared with the EW method, those two types of methods use a dynamic weighting process where loss/gradient weights vary over training iterations or epochs. Thus, it is natural to think how about training an MTL model with random weights. Inspired by this, we propose the RW methods by randomly sampling loss/gradient weights in each iteration as the random experiments for loss/gradient balancing methods, respectively. Besides, we think RW methods are more reasonable baselines than EW as the litmus tests for MTL methods.

## 3 The Random Weighting Methods

In this section, we introduce the RW methods, including the RLW and RGW methods.

We focus on the update of task-sharing parameter θ as it is the key problem in MTL. In the following, we mainly introduce the RLW method as the RGW method acts similarly to the RLW method. For notation simplicity, we do not distinguish between $\lambda _ { t } ^ { l }$ and $\lambda _ { t } ^ { g }$ and denote them by $\lambda _ { t } .$ Besides, we denote $\ell ( \theta ) \overset { \cdot } { = } ( \ell _ { \perp } ( \mathcal { D } _ { 1 } ; \theta , \psi _ { 1 } ) , \cdot \cdot \cdot , \ell _ { T } ( \mathcal { D } _ { T } ; \theta , \psi _ { T } ) )$ , where the datasets $\{ \mathcal { D } _ { t } \} _ { t = 1 } ^ { T }$ and the task-specific parameters $\{ \psi _ { t } \} _ { t = 1 } ^ { T }$ are omitted for brevity.

Different from those loss balancing methods, RLW considers the loss weights $\lambda = ( \lambda _ { 1 } , \cdots , \lambda _ { T } ) \in$ $\mathbb { R } ^ { T }$ as random variables and samples them from a random distribution in each iteration. To guarantee loss weights in λ to be non-negative, we can first sample $\tilde { \lambda } = ( \tilde { \lambda } _ { 1 } , \cdots , \tilde { \lambda } _ { T } )$ from any distribution $p ( \tilde { \lambda } )$ and then normalize $\tilde { \lambda }$ into λ via a mapping $f ,$ where $f : \mathbb { R } ^ { T }  \Delta ^ { T - 1 }$ is a normalization function such as the softmax function and $\Delta ^ { ^ { \bullet } T - 1 }$ denotes a simplex in $\mathbb { R } ^ { T }$ , i.e., $\lambda \in \Delta ^ { T - 1 }$ means $\sum _ { t = 1 } ^ { T } \lambda _ { t } = 1$ and $\lambda _ { t } \geq 0$ for all t. Note that $p ( \lambda )$ is different from $p ( \tilde { \lambda } )$ unless $f$ is an identity function. Finally, RLW updates the θ by computing the aggregated gradient $\nabla _ { \boldsymbol { \theta } } \lambda ^ { \top } \boldsymbol { \ell } ( \boldsymbol { \theta } )$

In this way, the RLW method uses dynamical loss weights in the training process, which is similar to existing loss balancing methods, but RLW uses random weights instead of carefully designed ones in the existing works. Therefore, RLW is a basic random experiment for those loss balancing methods to examine their effectiveness, which indicates RLW is a more reasonable baseline than the conventional EW.

Algorithm 1 A Training Iteration in RLW

<div class="mineru-algorithm" style="white-space: pre-wrap; font-family:monospace;">
Input: numbers of tasks $T$, learning rate $\eta$, dataset $\{\mathcal{D}_t\}_{t=1}^T$, weight distribution $p(\tilde{\boldsymbol{\lambda}})$, normalization function $f$
Output: task-sharing parameter $\theta'$, task-specific parameters $\{\psi'_t\}_{t=1}^T$
for $t = 1$ to $T$ do
    Compute loss $\ell_t(\mathcal{D}_t; \theta, \psi_t)$;
end for
Sample weights $\tilde{\boldsymbol{\lambda}}$ from $p(\tilde{\boldsymbol{\lambda}})$ and normalize it into $\boldsymbol{\lambda}$ via $f$; $\triangleright$ RLW Method
$\theta' = \theta - \eta \nabla_\theta \sum_{t=1}^T \lambda_t \ell_t(\mathcal{D}_t; \theta, \psi_t)$;
for $t = 1$ to $T$ do
    $\psi'_t = \psi_t - \eta \nabla_{\psi_t} \lambda_t \ell_t(\mathcal{D}_t; \theta, \psi_t)$;
end for
</div>

Algorithm 2 A Training Iteration in RGW

<div class="mineru-algorithm" style="white-space: pre-wrap; font-family:monospace;">
Input: numbers of tasks $T$, learning rate $\eta$, dataset $\{\mathcal{D}_t\}_{t=1}^T$, weight distribution $p(\tilde{\boldsymbol{\lambda}})$, normalization function $f$
Output: task-sharing parameter $\theta'$, task-specific parameters $\{\psi'_t\}_{t=1}^T$
for $t = 1$ to $T$ do
    Compute loss $\ell_t(\mathcal{D}_t; \theta, \psi_t)$;
    Compute gradient $g_t = \nabla_\theta \ell_t$ or $\nabla_z \ell_t$;
end for
Sample weights $\tilde{\boldsymbol{\lambda}}$ from $p(\tilde{\boldsymbol{\lambda}})$ and normalize it into $\boldsymbol{\lambda}$ via $f$; $\triangleright$ RGW Method
$\theta' = \theta - \eta \nabla_\theta \sum_{t=1}^T \lambda_t g_t$;
for $t = 1$ to $T$ do
    $\psi'_t = \psi_t - \eta \nabla_{\psi_t} \ell_t(\mathcal{D}_t; \theta, \psi_t)$;
end for
</div>

Noticeably, the loss weights λ are random variables and vary over training iterations, thus it is apparently that the gradient ∇<sub>θ</sub>λ<sup>></sup>\`(θ) of RLW is an unbiased estimation of the gradient E[λ]<sup>></sup>∇<sub>θ</sub>\`(θ), where $\mathbb { E } [ \lambda ]$ is the expectation of λ over the whole training process. This means that the RLW method is a stochastic variant of the loss balancing method with fixed weights E[λ]. In particular, if $\mathbb { E } [ \lambda ]$ is proportional to $\begin{array} { l } { \displaystyle { \left( \frac { 1 } { T } , \cdots , \frac { 1 } { T } \right) } } \end{array}$ , RLW is a stochastic variant of the conventional EW baseline. In Section 4, we theoretically show that RLW has a better generalization performance that EW because of the extra randomness from loss weight sampling, which indicates the RLW method is a more effective baseline than EW.

Similar to RLW, in each iteration, RGW first randomly samples gradient weights $\tilde { \lambda }$ from $p ( \tilde { \lambda } )$ , then normalizes it to obtain λ via $f ,$ and finally updates the task-sharing parameter θ by computing the aggregated gradient $\nabla _ { \boldsymbol { \theta } } \lambda ^ { \top } \boldsymbol { \ell } ( \boldsymbol { \theta } )$ . Following previous works [28, 6, 18, 12], we compute the gradient with respect to the final hidden feature representation z output from the shared parameter instead of the task-sharing parameter θ to reduce the computational cost. Thus, RGW is a random experiment for gradient balancing methods.

In this paper, we use the standard normal distribution for $p ( \tilde { \lambda } )$ and the softmax function for $f$ in both the RLW and RGW methods since it is easy to implement, has a more stable performance (as shown in experimental results in Section 5.3), and is as efficient as the EW strategy (as shown in experimental results in Section 5.4). Besides, $\mathbb { E } [ \lambda ]$ is proportional to $\begin{array} { l } { \displaystyle { \big ( \frac { 1 } { T } , \cdots , \frac { 1 } { T } \big ) } } \end{array}$ as proved in Appendix A, thus it is fair to compare with the EW strategy.

The training algorithms of both RW methods are summarized in Algorithm 1 and 2. The only difference between the RW methods and the existing works is the generation of loss/gradient weights (i.e., Line 6 in Algorithm 1 and Line 7 in Algorithm 2). Apparently, the sampling operation in the RW methods is very easy to implement and only bring negligibly additional computational costs when compared with the existing works. Note that random weights are involved in the update of task-specific parameters in the RLW method but not the RGW method (i.e., Line 9 in Algorithm 1 and Line 10 in Algorithm 2).

## 4 Analysis

In this section, we analyze how the extra randomness from the loss/gradient weight sampling affects the convergence and effectiveness of the RW methods compared with the EW strategy.

We focus on the update of task-sharing parameter θ and take RLW as an example for analysis, which can easily be extended to the RGW method. For notation simplicity, we simply use $\ell _ { t } ( \theta )$ instead of $\ell _ { t } ( \mathcal { D } _ { t } ; \theta , \psi _ { t } )$ to denote the loss function of task t in this section and Appendix B. For ease of analysis, we make the following assumption.

Assumption 1. $\mathbb { E } _ { \mathcal { D } _ { t } } [ \| \nabla \ell _ { t } ( \mathcal { D } _ { t } ; \boldsymbol { \theta } ) \| ^ { 2 } ]$ equals $\sigma _ { t } ^ { 2 }$ , the loss function $\ell _ { t } ( \theta )$ of task t is $L _ { t ^ { - } }$ Lipschitz continuous w.r.t. $\theta ,$ and λ satisfies $\dot { \mathbb { E } _ { \lambda } } [ \lambda ] = \mu$

In the following theorem, we analyze the convergence property of Algorithm 1 for the RLW method.

Theorem 1. Suppose the loss function $\ell _ { t } ( \theta )$ of task t is c<sub>t</sub>-strongly convex. We define $\theta _ { * } ~ =$ $\arg \operatorname* { m i n } _ { \theta } \lambda ^ { \top } \ell ( \theta )$ and denote by $\theta _ { k }$ the solution in the k-th iteration. $H \eta ,$ , the step size or equivalently the learning rate, satisfies $\eta \leq 1 / 2 c$ , where $c = \mathrm { m i n } _ { 1 \leq t \leq T } \{ c _ { t } \}$ , then under Assumption 1 we have

$$
\mathbb {E} [ \| \theta_ {k} - \theta_ {*} \| ^ {2} ] \leq (1 - 2 \eta c) ^ {k} \| \theta_ {0} - \theta_ {*} \| ^ {2} + \frac {\eta \kappa}{2 c}, \tag {3}
$$

where $\begin{array} { r } { \kappa = \sum _ { t = 1 } ^ { T } \sigma _ { t } ^ { 2 } } \end{array}$ . Then for any positive $\varepsilon , \mathbb { E } [ \| \theta _ { k } - \theta _ { * } \| ^ { 2 } ] \le \varepsilon$ can be achieved after $k =$ $\begin{array} { r } { \frac { \kappa } { 2 \varepsilon c ^ { 2 } } \log \left( \frac { \varepsilon _ { 0 } } { \varepsilon } \right) } \end{array}$ iterations with $\begin{array} { r } { \eta = \frac { \varepsilon c } { \kappa } } \end{array}$ , where $\varepsilon _ { 0 } = \mathbb { E } [ \| \theta _ { 0 } - \theta _ { * } \| ^ { 2 } ]$

Theorem 1 shows that the RLW method with a fixed step size has a linear convergence up to a radius around the optimal solution, which is similar to the EW strategy according to the property of the standard Stochastic Gradient Descent (SGD) method [23, 24]. Although the RLW method has a larger κ than the EW strategy, i.e., $\begin{array} { r } { \kappa _ { \mathrm { E W } } = \sum _ { t = 1 } ^ { T } \mu _ { t } ^ { 2 } \cdot \sum _ { t = 1 } ^ { T } \sigma _ { t } ^ { 2 } \le \kappa , } \end{array}$ , which may possibly require more iterations for the RLW method to reach the same accuracy as the EW strategy, experimental results in Section 5.4 show that empirically this does not cause much difference.

We next analyze the effectiveness of the RLW method from the perspective of stochastic optimization. It is observed that the SGD method can escape sharp local minima and converge to a better solution than Gradient Descent (GD) techniques under various settings with the help of noisy gradients [10, 16]. Inspired by those works, we prove Theorem 2 to show that the extra randomness in the RLW method can help RLW to better escape sharp local minima and achieve better generalization performance than the EW strategy.

Before presenting the theorem, for the ease of presentation, we introduction some notations. Here we consider the update step of these stochastic methods as $\theta _ { k + 1 } = \theta _ { k } - \eta ( \nabla \pmb { \mu } ^ { \top } \pmb { \ell } ( \theta _ { k } ) + \xi _ { k } )$ where $\xi _ { k }$ is a noise with $\begin{array} { r } { \mathbb { E } [ \xi _ { k } ] = 0 } \end{array}$ and $\| \xi _ { k } \| ^ { 2 } \leq r ,$ and r denotes the intensity of the noise. For the analysis, we construct an intermediate sequence $\varphi _ { k } = \theta _ { k } - \eta \nabla \pmb \mu ^ { \top } \pmb \ell ( \theta _ { k } )$ . Then we get $\mathbb { E } _ { \xi _ { k } } \left[ \varphi _ { k + 1 } \right] = \varphi _ { k } - \eta \nabla \mathbb { E } _ { \xi _ { k } } [ \pmb { \mu } ^ { \top } \pmb { \ell } ( \varphi _ { k } - \eta \xi _ { k } ) ]$ ]. Therefore, the sequence $\{ \varphi _ { k } \}$ can be regarded as an approximation of using GD to minimize the function $\mathbb { E } _ { \xi _ { k } } [ \pmb { \mu } ^ { \top } \pmb { \ell } ( \varphi - \eta \xi _ { k } ) ]$

Theorem 2. Suppose $\nabla \ell _ { t } ( \theta )$ is M -Lipschitz continuous and $\| \xi _ { k } \| ^ { 2 } \leq r .$ . Ifthe lossfunction $\ell _ { t } ( \theta )$ of task t is c<sub>t</sub>-one point strongly convex w.r.t. a local minimum θ<sub>∗</sub> after convolved with noise $\xi ,$ $i . e . , \langle \nabla \mathbb { E } _ { \xi } \ell _ { t } ( \varphi - \eta \xi ) , \varphi - \theta _ { * } \rangle \geq c _ { t } \Vert \varphi - \theta _ { * } \Vert ^ { 2 }$ , then under Assumption 1, after $\begin{array} { r } { K = \frac { 1 } { \rho } \log \left( \frac { \rho \varepsilon _ { 0 } } { \beta } \right) } \end{array}$ iterations with $\eta \ \leq \ \frac { c } { M ^ { 2 } }$ , with probability at least $1 - \delta ,$ , we have $\begin{array} { r } { \| \varphi _ { K } - \theta _ { * } \| ^ { 2 } \le \frac { 2 \beta } { \rho \delta } } \end{array}$ , where $\begin{array} { r } { \varepsilon _ { 0 } = \mathbb { E } [ \| \varphi _ { 0 } - \theta _ { * } \| ^ { 2 } ] , c = \operatorname* { m i n } _ { 1 \leq t \leq T } \{ c _ { t } \} , M = \operatorname* { m a x } _ { 1 \leq t \leq T } \{ M _ { t } \} , \rho = 2 \eta c - \eta ^ { 2 } M ^ { 2 } \kappa , } \end{array}$ , and $\beta =$ $\eta ^ { 2 } r ^ { 2 } ( 1 + \eta M ) ^ { 2 }$

Theorem 2 only requires that $\ell _ { t } ( \theta )$ is $c _ { t }$ -one point strongly convex w.r.t. θ<sub>∗</sub> after convolved with noise ξ, which can hold for deep neural networks [27]. It also implies that for both RLW and EW methods, their solutions have high probabilities to be close to a local minimum $\theta _ { * }$ depending on the noise ξ. Note that by adding extra noise, the sharp local minimum will disappear and only the flat local minimum with a large diameter will still exist [16]. On the other hand, those flat local minima could satisfy the one point strongly convexity assumption made in Theorem 2, thus the diameter of the converged flat local minimum is affected by the noise intensity.

Remark 1. Converging to flat local minima is important in neural network training because flat local minima may lead to better generalization [3, 14]. Due to the extra randomness from the sampling of loss weights, the RLW method can have a larger noise with a larger r than the EW strategy (refer to Appendix B.3). Hence, according to Theorem 2 and the above discussion, the RLW method can better escape sharp local minima and converge to a flatter local minimum than EW, resulting in better generalization performance.

## 5 Experiments

In this section, we empirically evaluate the proposed RLW and RGW methods on five computer vision datasets (i.e., NYUv2, CityScapes, CelebA, Office-31, and Office-Home) and two multilingual problems from the XTREME benchmark [11]. All the experiments are conducted on one single NVIDIA GeForce RTX 3090 GPU. Due to page limit, experimental results on the CityScapes, CelebA, Office-31, and Office-Home datasets are put in Appendix D.

Compared methods. The baseline methods in comparison include several SOTA task balancing methods as introduced in Section 2, including four loss balancing methods (i.e., UW, DWA, IMTL-L, and MOML) and eight gradient balancing methods (i.e., MGDA-UB, GradNorm, PCGrad, GradDrop, IMTL-G, GradVac, CAGrad, and RotoGrad). For all the baseline methods, we directly use the optimal hyperparameters used in their original papers.

Network architecture. The network architecture we used adopts the Hard-Parameter Sharing (HPS) pattern [1], which shares bottom layers of the network for all the tasks and uses separate top layers for each task. Other MTL architectures are studied in Section 5.6.

Evaluation metric. For homogeneous MTL problems (e.g., the XTREME benchmark and Office-31 dataset) which contain tasks of the same type such as the classification task, we directly use the average performance among tasks as the performance metric. For heterogeneous MTL problems (e.g., the NYUv2 dataset) that contain tasks of different types and may have multiple evaluation metrics for each task, by following [21, 30], we use the average of the relative improvement over the EW method on each metric of each task as the performance measure, which is formulated as

$$
\Delta_ {\mathrm{p}} = 100\% \times \frac {1}{T} \sum_ {t = 1} ^ {T} \frac {1}{N _ {t}} \sum_ {n = 1} ^ {N _ {t}} \frac {(- 1) ^ {p _ {t , n}} (M _ {t , n} - M _ {t , n} ^ {\mathrm{EW}})}{M _ {t , n} ^ {\mathrm{EW}}},
$$

where $N _ { t }$ denotes the number of metrics in task $t , M _ { t , n }$ denotes the performance of a task balancing method for the nth metric in task $t , M _ { t , n } ^ { \mathrm { E W } }$ is defined similarly for the EW method, and $p _ { t , n }$ is set to 1 if a higher value indicates better performance for the nth metric in task t and otherwise 0.

## 5.1 Results on the NYUv2 Dataset

Dataset. The NYUv2 dataset [29] is an indoor scene understanding dataset, which consists of video sequences recorded by the RGB and Depth cameras in the Microsoft Kinect. It contains 795 and 654 images for training and testing, respectively. This dataset includes three tasks: 13-class semantic segmentation, depth estimation, and surface normal prediction.

Implementation details. For the NYUv2 dataset, the DeepLabV3+ architecture [4] is used. Specifically, a ResNet-50 network pre-trained on the ImageNet dataset with dilated convolutions [36] is used as a shared encoder among tasks and the Atrous Spatial Pyramid Pooling (ASPP) [4] module is used as the task-specific head for each task. Input images are resized to $2 \bar { 8 8 } \times 3 8 4$ . The Adam optimizer [15] with the learning rate as $1 0 ^ { - 4 }$ and the weight decay as $1 0 ^ { - 5 }$ is used for training and the batch size is set to 8. We use the cross-entropy loss, $L _ { 1 }$ loss, and cosine loss as the loss function of the semantic segmentation, depth estimation, and surface normal prediction tasks, respectively.

Results. The results of different methods on the NYUv2 dataset are shown in Table 1. The top row shows the performance of the widely used EW strategy and we use it as a baseline to measure the relative improvement of different methods as shown in the definition of $\Delta _ { \mathrm { p } } .$ Rows 2-5 and 7-14 show the results of loss balancing and gradient balancing methods, respectively.

According to the results, we can see that both the RLW and RGW methods gain performance improvement over the EW strategy, which implies that training with extra randomness can have a better generalization ability. Besides, RLW has an improvement of 1.04% over the EW strategy and it is higher than all loss balancing methods. As for gradient balancing methods, half of those methods have negligible or even negative improvement over the EW strategy and RGW can outperform five of them. Compared with all baselines, RLW is even higher than all of them except the CAGrad and RotoGrad methods, which indicates the random weights can easily beat the carefully designed ones.

According to the above analysis, there are two important conclusions. Firstly, the conventional EW strategy is a weaker baseline than RLW and RGW for MTL. Secondly, RW methods are competitive to SOTA methods and even performs better than some of them.

Table 1: Performance on the NYUv2 dataset with three tasks: 13-class semantic segmentation, depth estimation, and surface normal prediction. The best results for each task on each measure over loss/gradient balancing methods are marked with superscript ${ * } / { \dagger }$ . The best results for each task on each measure over all methods are highlighted in bold. ↑ (↓) indicates that the higher (lower) the result, the better the performance.

<table><tr><td rowspan="3" colspan="2">Methods</td><td colspan="2">Segmentation</td><td colspan="2">Depth</td><td colspan="5">Surface Normal</td><td rowspan="3"> $\Delta_{\mathrm {p}} \uparrow$ </td></tr><tr><td rowspan="2">mIoU↑</td><td rowspan="2">Pix Acc↑</td><td rowspan="2">Abs Err↓</td><td rowspan="2">Rel Err↓</td><td colspan="2">Angle Distance</td><td colspan="3">Within  $t^{\circ}$ </td></tr><tr><td>Mean↓</td><td>Median↓</td><td>11.25↑</td><td>22.5↑</td><td>30↑</td></tr><tr><td></td><td>EW</td><td>53.77</td><td>75.45</td><td>0.3845</td><td>0.1605</td><td>23.5737</td><td>17.0438</td><td>35.04</td><td>60.93</td><td>72.07</td><td>+0.00%</td></tr><tr><td rowspan="5">Loss Bal.</td><td>UW</td><td>54.14</td><td>75.92</td><td>0.3833</td><td>0.1597</td><td>23.2989</td><td>16.8691</td><td>35.33</td><td>61.37</td><td>72.48</td><td>+0.64%</td></tr><tr><td>DWA</td><td>53.81</td><td>75.56</td><td>0.3792*</td><td>0.1565*</td><td>23.6111</td><td>17.0609</td><td>34.89</td><td>60.89</td><td>71.97</td><td>+0.63%</td></tr><tr><td>IMTL-L</td><td>53.50</td><td>75.18</td><td>0.3824</td><td>0.1596</td><td>23.3805</td><td>16.8088</td><td>35.44</td><td>61.43</td><td>72.43</td><td>+0.35%</td></tr><tr><td>MOML</td><td>54.98*</td><td>75.98*</td><td>0.3877</td><td>0.1618</td><td>23.2401*</td><td>16.7388</td><td>35.90*</td><td>61.81*</td><td>72.76*</td><td>+0.76%</td></tr><tr><td>RLW (ours)</td><td>54.11</td><td>75.77</td><td>0.3809</td><td>0.1575</td><td>23.3777</td><td>16.7385*</td><td>35.71</td><td>61.52</td><td>72.45</td><td>+1.04%*</td></tr><tr><td rowspan="9">Gradient Bal.</td><td>MGDA-UB</td><td>50.42</td><td>73.46</td><td>0.3834</td><td> $0.1555^†$ </td><td> $22.7827^†$ </td><td> $16.1432^†$ </td><td> $36.90^†$ </td><td>62.88</td><td>73.61</td><td>+0.38%</td></tr><tr><td>GradNorm</td><td>53.58</td><td>75.06</td><td>0.3931</td><td>0.1663</td><td>23.4360</td><td>16.9844</td><td>35.11</td><td>61.11</td><td>72.24</td><td>-0.99%</td></tr><tr><td>PCGrad</td><td>53.70</td><td>75.41</td><td>0.3903</td><td>0.1607</td><td>23.4281</td><td>16.9699</td><td>35.16</td><td>61.19</td><td>72.28</td><td>-0.16%</td></tr><tr><td>GradDrop</td><td>53.58</td><td>75.56</td><td>0.3855</td><td>0.1592</td><td>23.5518</td><td>17.0137</td><td>35.08</td><td>60.97</td><td>72.02</td><td>+0.08%</td></tr><tr><td>IMTL-G</td><td>53.54</td><td>75.45</td><td>0.3880</td><td>0.1589</td><td>23.0530</td><td>16.4328</td><td>36.21</td><td>62.31</td><td>73.15</td><td>+0.80%</td></tr><tr><td>GradVac</td><td> $54.89^†$ </td><td> $75.98^†$ </td><td>0.3828</td><td>0.1635</td><td>23.6865</td><td>17.1301</td><td>34.82</td><td>60.71</td><td>71.81</td><td>+0.07%</td></tr><tr><td>CAGrad</td><td>53.12</td><td>75.19</td><td>0.3871</td><td>0.1599</td><td>22.5257</td><td>15.8821</td><td>37.42</td><td> $63.50^†$ </td><td> $74.17^†$ </td><td> $+1.36\%^†$ </td></tr><tr><td>RotoGrad</td><td>53.90</td><td>75.46</td><td>0.3812</td><td>0.1596</td><td>23.0197</td><td>16.3714</td><td>36.37</td><td>62.28</td><td>73.05</td><td>+1.19%</td></tr><tr><td>RGW (ours)</td><td>53.85</td><td>75.87</td><td> $0.3772^†$ </td><td>0.1562</td><td>23.6725</td><td>17.2439</td><td>34.62</td><td>60.49</td><td>71.75</td><td>+0.62%</td></tr></table>

## 5.2 Results on the XTREME benchmark

Dataset. The XTREME benchmark [11] is a large-scale multilingual multi-task benchmark for cross-lingual generalization evaluation, which covers fifty languages and contains nine tasks. We conduct experiments on two tasks containing Paraphrase Identification (PI) and Part-Of-Speech (POS) tagging in this benchmark. The datasets used in the PI and POS tasks are the PAWS-X dataset [34] and Universal Dependency v2.5 treebanks [25], respectively. On each task, we construct a multilingual problem by choosing the four languages with largest numbers of data, i.e., English (en), Mandarin (zh), German (de) and Spanish (es), for the PI task and English, Mandarin, Telugu (te) and Vietnamese (vi) for the POS task. The statistics for each language are summarized in Table 5 in the Appendix. Different from the NYUv2 dataset where different tasks share the same input data, in those multilingual problems, each language/task has its own input data.

Implementation details. For each multilingual problem in the XTREME benchmark, a pre-trained multilingual BERT (mBERT) model [8] implemented via the open-source transformers library [33] is used as the shared encoder among languages and a fully connected layer is used as the languagespecific output layer for each language. The Adam optimizer with the learning rate as $2 \times 1 0 ^ { - 5 }$ and the weight decay as $1 0 ^ { - 8 }$ is used for training and the batch size is set to 32. The cross-entropy loss is used for the two multilingual problems.

Results. According to experimental results shown in Table 2, we can find some empirical observations, which are similar to those on the NYUv2 dataset. Firstly, both the RLW and RGW strategies outperform the EW method. Secondly, compared with the existing works, RLW and RGW can achieve comparable performance with existing loss/gradient balancing methods, respectively. Even, RLW or RGW methods could outperform all baseline methods. For example, RLW achieves the best performance (i.e., 90.25% average accuracy) on the PI problem and RGW achieves the best average F1 score of 91.16% on the POS problem. It is interesting to find that the performance of RLW and RGW are inconsistent in different datasets. There is because the random loss weights in RLW will affect the update of task-specific parameters while not in RGW, which has a different influence on the performance of different datasets.

Table 2: Performance on two multilingual problems, i.e., PI and POS from the XTREME benchmark. The best results for each language over loss/gradient balancing methods are marked with superscript ∗/†. The best results for each language over all methods are highlighted in bold.

<table><tr><td rowspan="2" colspan="2">Methods</td><td colspan="5">PI (Accuracy)</td><td colspan="5">POS (F1 Score)</td></tr><tr><td>en</td><td>zh</td><td>de</td><td>es</td><td>Avg</td><td>en</td><td>zh</td><td>te</td><td>vi</td><td>Avg</td></tr><tr><td rowspan="6">Loss Bal.</td><td>EW</td><td>94.29</td><td>84.99</td><td>89.79</td><td>90.94</td><td>90.00</td><td>95.06</td><td>89.01</td><td>91.41</td><td>86.65</td><td>90.53</td></tr><tr><td>UW</td><td>93.74</td><td>85.44*</td><td>90.24*</td><td>91.29</td><td>90.18</td><td>94.89</td><td>88.77</td><td>90.96</td><td>87.12</td><td>90.44</td></tr><tr><td>DWA</td><td>94.69*</td><td>84.99</td><td>89.49</td><td>91.44*</td><td>90.15</td><td>95.02</td><td>89.03</td><td>91.87</td><td>87.27*</td><td>90.80</td></tr><tr><td>IMTL-L</td><td>93.94</td><td>84.54</td><td>89.39</td><td>91.44*</td><td>89.82</td><td>95.57*</td><td>89.93*</td><td>91.77</td><td>86.11</td><td>90.84</td></tr><tr><td>MOML</td><td>93.89</td><td>83.74</td><td>89.94</td><td>90.99</td><td>89.64</td><td>95.15</td><td>89.11</td><td>92.41</td><td>87.24</td><td>90.98*</td></tr><tr><td>RLW (ours)</td><td>94.29</td><td>85.39</td><td>89.94</td><td>91.39</td><td>90.25*</td><td>95.01</td><td>88.87</td><td>92.86*</td><td>86.85</td><td>90.90</td></tr><tr><td rowspan="9">Gradient Bal.</td><td>MGDA-UB</td><td>94.09</td><td>84.14</td><td>89.14</td><td>90.59</td><td>89.49</td><td>94.89</td><td>88.43</td><td>91.01</td><td>86.04</td><td>90.01</td></tr><tr><td>GradNorm</td><td>94.19</td><td>83.59</td><td>88.89</td><td>91.24</td><td>89.47</td><td>94.88</td><td>88.80</td><td>91.78</td><td>86.96</td><td>90.61</td></tr><tr><td>PCGrad</td><td>94.19</td><td> $85.49^†$ </td><td>89.09</td><td>91.24</td><td>90.00</td><td>94.85</td><td>88.42</td><td>90.72</td><td>86.71</td><td>90.18</td></tr><tr><td>GradDrop</td><td>94.29</td><td>84.44</td><td>89.69</td><td>90.94</td><td>89.84</td><td>95.08</td><td>89.06</td><td>90.65</td><td>87.17</td><td>90.49</td></tr><tr><td>IMTL-G</td><td> $94.69^†$ </td><td>84.54</td><td>89.39</td><td>90.69</td><td>89.82</td><td>94.93</td><td>88.70</td><td>91.66</td><td>87.00</td><td>90.57</td></tr><tr><td>GradVac</td><td>94.29</td><td>84.94</td><td>89.19</td><td>90.89</td><td>89.83</td><td>94.87</td><td>88.41</td><td>90.62</td><td>86.47</td><td>90.09</td></tr><tr><td>CAGrad</td><td>94.34</td><td>84.59</td><td> $90.09^†$ </td><td>90.64</td><td>89.91</td><td>94.83</td><td>88.65</td><td>91.71</td><td>86.76</td><td>90.48</td></tr><tr><td>RotoGrad</td><td>93.99</td><td>83.89</td><td>89.29</td><td>90.94</td><td>89.52</td><td>95.44</td><td>89.79</td><td>91.42</td><td>86.33</td><td>90.74</td></tr><tr><td>RGW (ours)</td><td>94.55</td><td>84.99</td><td>89.29</td><td> $91.40^†$ </td><td> $90.06^†$ </td><td> $95.52^†$ </td><td> $90.13^†$ </td><td> $91.82^†$ </td><td> $87.18^†$ </td><td> $91.16^†$ </td></tr></table>

## 5.3 Robustness on Distribution

In this section, we evaluate the robustness of the proposed RW methods on the sampling distribution. Taking RLW as an example, we show its robustness by evaluating with five different sampling distri butions $( \mathrm { i } . \mathrm { e } . , p ( \tilde { \lambda } ) )$ for loss weights. The five distributions are uniform distribution between 0 and 1 (denoted by Uniform), standard normal distribution (denoted by Normal), Dirichlet distribution with α = 1 (denoted by Dirichlet), Bernoulli distribution with probability 1/2 (denoted by Bernoulli),

Bernoulli distribution with probability $1 / 2$ and a constraint $\textstyle \sum _ { t = 1 } ^ { T } \tilde { \lambda } _ { t } = 1$ (denoted by c-Bernoulli). We set $f$ as a function of $f ( \tilde { \lambda } ) = \tilde { \lambda } / ( \sum _ { t = 1 } ^ { T } \tilde { \lambda } _ { t } )$ for the Bernoulli distribution and the c-Bernoulli distribution, a softmax function for the Normal distribution and Uniform distribution, and an identity function for the Dirichlet distribution. We can prove that all the $\mathbb { E } [ \lambda ]$ ’s under these five distributions equal $\textstyle { \Bigl ( } { \frac { 1 } { T } } , \cdots , { \frac { 1 } { T } } { \Bigr ) }$ (refer to Appendix A), thus it is fair to compare among them.

Figure 1 shows the results of the RLW method with five sampling distributions on the NYUv2 dataset in terms of $\Delta _ { \mathrm { p } } ,$ where the experiment on each sampling distribution is repeated for 8 times. The results show that the RLW method with different distributions can always outperform the EW method, which shows the robustness of the RLW method with respect to the sampling distribution. In addition, compared with the uniform, Dirichlet, and Bernoulli distributions, RLW with the standard normal distribution achieves better and more stable performance. Although RLW with the c-Bernoulli distribution performs slightly better than the standard normal distribution, it is more unstable and may need a longer training time as shown in Section 5.4. Thus, in this paper, we use the standard normal distribution to sample loss weights.

## 5.4 Convergence Speed

Here we take RLW as an example to show the efficiency of RW methods. Figure 2 plots the performance curve on both NYUv2 and CelebA validation datasets to empirically compare the convergence speed of the EW and RLW methods.

On the NYUv2 dataset with three tasks, the performance curves of the RLW method with two sampling distributions are similar to that of the EW method, which indicates that the RLW method has a similar convergence property to the EW method on this dataset. As the number of tasks increases, i.e., on the CelebA dataset with 40 tasks, we find that the RLW method with the standard normal distribution still converges as fast as the EW method, while the RLW

![](images/8b1cbc2fb5f02a3f8931b3ddec4a96294c6e0bbbe789b2a8a1b2788136fa6895.jpg)

<details>
<summary>boxplot</summary>

| Category | Min | Q1 | Median (Q2) | Q3 | Max |
| --- | --- | --- | --- | --- | --- |
| Uniform | ~0.46 | ~0.64 | ~0.77 | ~1.01 | ~1.21 |
| Normal | ~0.61 | ~0.87 | ~1.01 | ~1.09 | ~1.29 |
| Dirichlet | ~0.10 | ~0.30 | ~0.54 | ~0.74 | ~0.90 |
| Bernoulli | ~0.53 | ~0.68 | ~0.78 | ~1.02 | ~1.30 |
| c-Bernoulli | ~0.37 | ~0.76 | ~1.08 | ~1.23 | ~1.56 |
</details>

Figure 1: Results of the RLW method with different sampling distributions in terms of $\Delta _ { \mathrm { p } }$

method with the c-Bernoulli distribution converges slower. One reason for this phenomenon is that only one task is used to update model parameters in each training iteration when using the c-Bernoulli distribution. Thus, in this paper, we use the standard normal distribution, which is as efficient as the EW method.

![](images/197c58381d1f1242ce8af05824b24020db5f81e7e4d397a317c30835708d91a6.jpg)  
Figure 2: Comparison on the convergence speed of the EW and RLW methods on the NYUv2 validation dataset (Left) and the CelebA validation dataset (Right).

## 5.5 Combination of Loss and Gradient Balancing Methods

The loss balancing methods are complementary with the gradient balancing methods. Following [18], we train an MTL model with different combinations of loss balancing and gradient balancing methods on the NYUv2 dataset to further improve the performance. We use the vanilla EW as the baseline to measure the relative improvement of the other different combinations as shown in the definition of $\Delta _ { \mathrm { p } }$

According to the results shown in Table $^ { 3 , }$ we can see that combined with the UW, DWA and IMTL-L methods, some gradient balancing methods performs better but others become worse. For example, $\Delta _ { \mathrm { p } }$ of the GradDrop method drops from 0.08% to −0.42% when combined with DWA. Noticeably, by combining with the proposed RLW method, all the gradient balancing methods can achieve better performance. Besides, on each gradient balancing method, the improvement induced by the RLW method is significantly larger than the other three loss balancing methods as well as the EW method. Moreover, RGW can also improve the performance of loss balancing methods except DWA. Thus, this experiment further demonstrates the effectiveness of the proposed RW methods.

Table 3: Results of different combinations of loss balancing and gradient balancing methods on the NYUv2 dataset in terms of $\Delta _ { \mathrm { p } } .$ . The best results in each row are highlighted in bold.

<table><tr><td></td><td>EW</td><td>UW</td><td>DWA</td><td>IMTL-L</td><td>RLW</td></tr><tr><td>Vanilla</td><td>+0.00%</td><td>+0.64%</td><td>+0.63%</td><td>+0.35%</td><td>+1.04%</td></tr><tr><td>MGDA-UB</td><td>+0.38%</td><td>+0.15%</td><td>+0.47%</td><td>-0.59%</td><td>+2.01%</td></tr><tr><td>GradNorm</td><td>-0.99%</td><td>+0.87%</td><td>-0.95%</td><td>+0.54%</td><td>+0.89%</td></tr><tr><td>PCGrad</td><td>-0.16%</td><td>+0.72%</td><td>+0.19%</td><td>+0.38%</td><td>+0.97%</td></tr><tr><td>GradDrop</td><td>+0.08%</td><td>+0.25%</td><td>-0.42%</td><td>+0.36%</td><td>+0.93%</td></tr><tr><td>IMTL-G</td><td>+0.80%</td><td>+0.45%</td><td>+1.20%</td><td>+0.18%</td><td>+1.50%</td></tr><tr><td>GradVac</td><td>+0.07%</td><td>-0.03%</td><td>+0.89%</td><td>+0.69%</td><td>+0.97%</td></tr><tr><td>CAGrad</td><td>+1.36%</td><td>+1.07%</td><td>+1.41%</td><td>+2.18%</td><td>+2.20%</td></tr><tr><td>RotoGrad</td><td>+1.19%</td><td>+1.03%</td><td>+0.75%</td><td>+1.40%</td><td>+1.45%</td></tr><tr><td>RGW</td><td>+0.62%</td><td>+0.82%</td><td>+0.41%</td><td>+0.78%</td><td>+1.46%</td></tr></table>

## 5.6 Effects of Different Architectures

The proposed RW methods can be seamlessly incorporated into all the MTL architectures. To see this, we take RLW as an example and combine it with three different MTL architectures, i.e., cross-stitch network [22], Multi-Task Attention Network (MTAN) [19], and NDDR-CNN [9]. We use the combination of EW and HPS as the baseline to measure the relative improvement of the other different combinations as shown in the definition of $\Delta _ { \mathrm { p } }$

According to the results on the NYUv2 dataset as shown in Table 4, we can see that the proposed RLW strategy outperforms the EW method under all the three architectures. When using the MTAN and NNDR-CNN architectures, RLW achieves better performance than the CAGrad method that performs best in the HPS architecture, which shows the potential of the proposed RLW method when choosing suitable MTL architectures. Moreover, combined with the RLW method, CAGrad can be further improved under the four architectures. For example, the combinations of RLW and CAGrad can achieve the best $\Delta _ { \mathrm { p } }$ of 3.53% under the NDDR-CNN architecture.

Table 4: Results of different combinations of task balancing methods and MTL architectures on the NYUv2 dataset in terms of $\Delta _ { \mathrm { p } } .$ . The best results for each architecture are highlighted in bold.

<table><tr><td></td><td>HPS</td><td>Cross-stitch</td><td>MTAN</td><td>NDDR-CNN</td></tr><tr><td>EW</td><td>+0.00%</td><td>+1.43%</td><td>+2.56%</td><td>+1.90%</td></tr><tr><td>CAGrad</td><td>+1.36%</td><td>+2.42%</td><td>+2.26%</td><td>+2.83%</td></tr><tr><td>RLW</td><td>+1.04%</td><td>+2.23%</td><td>+2.66%</td><td>+2.91%</td></tr><tr><td>RLW+CAGrad</td><td>+2.20%</td><td>+2.76%</td><td>+2.92%</td><td>+3.53%</td></tr></table>

## 6 Conclusions

In this paper, we propose the RW methods, an important yet ignored baselines for MTL, by training an MTL model with random loss/gradient weights. We analyze the convergence and effectiveness properties of the proposed RW method. Moreover, we provide a consistent and comparative compari son to show the RW methods can achieve comparable performance with state-of-the-art methods that use carefully designed weights, which indicates the random experiments could be used to examine the effectiveness of newly proposed MTL methods and RW methods should attract wide attention as the litmus tests. In our future work, we will apply the RW methods to more MTL applications.

## References

[1] Rich Caruana. Multitask learning: A knowledge-based source of inductive bias. In Proceedings ofthe 10th International Conference on Machine Learning, pages 41–48, 1993.  
[2] Rich Caruana. Multitask learning. Machine learning, 28(1):41–75, 1997.  
[3] Pratik Chaudhari, Anna Choromanska, Stefano Soatto, Yann LeCun, Carlo Baldassi, Christian Borgs, Jennifer Chayes, Levent Sagun, and Riccardo Zecchina. Entropy-sgd: Biasing gradient descent into wide valleys. Journal of Statistical Mechanics: Theory and Experiment, 2019(12):124018, 2019.  
[4] Liang-Chieh Chen, Yukun Zhu, George Papandreou, Florian Schroff, and Hartwig Adam. Encoder-decoder with atrous separable convolution for semantic image segmentation. In Proceedings ofthe 14th European Conference on Computer Vision, volume 11211, pages 833–851, 2018.  
[5] Zhao Chen, Vijay Badrinarayanan, Chen-Yu Lee, and Andrew Rabinovich. Gradnorm: Gradient normalization for adaptive loss balancing in deep multitask networks. In Proceedings ofthe International Conference on Machine Learning, pages 794–803. PMLR, 2018.  
[6] Zhao Chen, Jiquan Ngiam, Yanping Huang, Thang Luong, Henrik Kretzschmar, Yuning Chai, and Dragomir Anguelov. Just pick a sign: Optimizing deep multitask models with gradient sign dropout. In Proceedings ofthe 33rd Advances in Neural Information Processing Systems, 2020.  
[7] Marius Cordts, Mohamed Omran, Sebastian Ramos, Timo Rehfeld, Markus Enzweiler, Rodrigo Benenson, Uwe Franke, Stefan Roth, and Bernt Schiele. The cityscapes dataset for semantic urban scene understanding. In Proceedings ofIEEE Conference on Computer Vision and Pattern Recognition, pages 3213–3223, 2016.  
[8] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. BERT: pre-training of deep bidirectional transformers for language understanding. In Proceedings ofthe 2019 Conference ofthe North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pages 4171–4186, 2019.  
[9] Yuan Gao, Jiayi Ma, Mingbo Zhao, Wei Liu, and Alan L Yuille. Nddr-cnn: Layerwise feature fusing in multi-task cnns by neural discriminative dimensionality reduction. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 3205–3214, 2019.  
[10] Moritz Hardt, Ben Recht, and Yoram Singer. Train faster, generalize better: Stability of stochastic gradient descent. In Proceedings of the International Conference on Machine Learning, pages 1225–1234. PMLR, 2016.  
[11] Junjie Hu, Sebastian Ruder, Aditya Siddhant, Graham Neubig, Orhan Firat, and Melvin Johnson. XTREME: A massively multilingual multi-task benchmark for evaluating cross-lingual generalisation. In Proceedings ofthe 37th International Conference on Machine Learning, volume 119, pages 4411–4421. PMLR, 2020.  
[12] Adrián Javaloy and Isabel Valera. Rotograd: Gradient homogenization in multitask learning. In Proceedings ofthe 10th International Conference on Learning Representations, 2022.  
[13] Alex Kendall, Yarin Gal, and Roberto Cipolla. Multi-task learning using uncertainty to weigh losses for scene geometry and semantics. In Proceedings of IEEE Conference on Computer Vision and Pattern Recognition, pages 7482–7491, 2018.  
[14] Nitish Shirish Keskar, Jorge Nocedal, Ping Tak Peter Tang, Dheevatsa Mudigere, and Mikhail Smelyanskiy. On large-batch training for deep learning: Generalization gap and sharp minima. In Proceedings of the 5th International Conference on Learning Representations, 2017.  
[15] Diederik P. Kingma and Jimmy Ba. Adam: A method for stochastic optimization. In Proceedings ofthe 3rd International Conference on Learning Representations, 2015.  
[16] Bobby Kleinberg, Yuanzhi Li, and Yang Yuan. An alternative view: When does sgd escape local minima? In Proceedings ofthe International Conference on Machine Learning, pages 2698–2707. PMLR, 2018.  
[17] Bo Liu, Xingchao Liu, Xiaojie Jin, Peter Stone, and Qiang Liu. Conflict-averse gradient descent for multi-task learning. In Proceedings ofthe 35th Advances in Neural Information Processing Systems, 2021.  
[18] Liyang Liu, Yi Li, Zhanghui Kuang, Jing-Hao Xue, Yimin Chen, Wenming Yang, Qingmin Liao, and Wayne Zhang. Towards impartial multi-task learning. In Proceedings of the 9th International Conference on Learning Representations, 2021.  
[19] Shikun Liu, Edward Johns, and Andrew J. Davison. End-to-end multi-task learning with attention. In Proceedings of IEEE Conference on Computer Vision and Pattern Recognition, pages 1871–1880, 2019.  
[20] Ziwei Liu, Ping Luo, Xiaogang Wang, and Xiaoou Tang. Deep learning face attributes in the wild. In Proceedings ofInternational Conference on Computer Vision, 2015.  
[21] Kevis-Kokitsi Maninis, Ilija Radosavovic, and Iasonas Kokkinos. Attentive single-tasking of multiple tasks. In Proceedings ofIEEE Conference on Computer Vision and Pattern Recognition, pages 1851–1860, 2019.  
[22] Ishan Misra, Abhinav Shrivastava, Abhinav Gupta, and Martial Hebert. Cross-stitch networks for multi-task learning. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 3994–4003, 2016.  
[23] Eric Moulines and Francis Bach. Non-asymptotic analysis of stochastic approximation algorithms for machine learning. In Proceedings ofthe Advances in Neural Information Processing Systems, volume 24, pages 451–459, 2011.  
[24] Deanna Needell, Nathan Srebro, and Rachel Ward. Stochastic gradient descent, weighted sampling, and the randomized kaczmarz algorithm. Mathematical Programming, 155(1-2):549–573, 2016.  
[25] Joakim Nivre, Marie-Catherine de Marneffe, Filip Ginter, Jan Hajic, Christopher D. Manning, Sampo Pyysalo, Sebastian Schuster, Francis M. Tyers, and Daniel Zeman. Universal dependencies v2: An evergrowing multilingual treebank collection. In Proceedings of the 12th Language Resources and Evaluation Conference, pages 4034–4043, 2020.  
[26] Kate Saenko, Brian Kulis, Mario Fritz, and Trevor Darrell. Adapting visual category models to new domains. In Proceedings of the 6th European Conference on Computer Vision, pages 213–226. Springer, 2010.  
[27] Itay M Safran, Gilad Yehudai, and Ohad Shamir. The effects of mild over-parameterization on the optimization landscape of shallow relu neural networks. In Conference on Learning Theory, pages 3889–3934. PMLR, 2021.  
[28] Ozan Sener and Vladlen Koltun. Multi-task learning as multi-objective optimization. In Proceedings of the 31st Advances in Neural Information Processing Systems, pages 525–536, 2018.  
[29] Nathan Silberman, Derek Hoiem, Pushmeet Kohli, and Rob Fergus. Indoor segmentation and support inference from rgbd images. In Proceedings of the 8th European Conference on Computer Vision, pages 746–760, 2012.  
[30] Simon Vandenhende, Stamatios Georgoulis, Wouter Van Gansbeke, Marc Proesmans, Dengxin Dai, and Luc Van Gool. Multi-task learning for dense prediction tasks: A survey. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2021.  
[31] Hemanth Venkateswara, Jose Eusebio, Shayok Chakraborty, and Sethuraman Panchanathan. Deep hashing network for unsupervised domain adaptation. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 5018–5027, 2017.  
[32] Zirui Wang, Yulia Tsvetkov, Orhan Firat, and Yuan Cao. Gradient vaccine: Investigating and improving multi-task optimization in massively multilingual models. In Proceedings of the 9th International Conference on Learning Representations, 2021.  
[33] Thomas Wolf, Lysandre Debut, Victor Sanh, Julien Chaumond, Clement Delangue, Anthony Moi, Pierric Cistac, Tim Rault, Rémi Louf, Morgan Funtowicz, Joe Davison, Sam Shleifer, Patrick von Platen, Clara Ma, Yacine Jernite, Julien Plu, Canwen Xu, Teven Le Scao, Sylvain Gugger, Mariama Drame, Quentin Lhoest, and Alexander M. Rush. Transformers: State-of-the-art natural language processing. In Proceedings ofthe 2020 Conference on Empirical Methods in Natural Language Processing, pages 38–45, 2020.  
[34] Yinfei Yang, Yuan Zhang, Chris Tar, and Jason Baldridge. PAWS-X: A cross-lingual adversarial dataset for paraphrase identification. In Proceedings ofthe 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing, pages 3685–3690, 2019.  
[35] Feiyang Ye, Baijiong Lin, Zhixiong Yue, Pengxin Guo, Qiao Xiao, and Yu Zhang. Multi-objective meta learning. In Proceedings of the 35th Advances in Neural Information Processing Systems, 2021.  
[36] Fisher Yu, Vladlen Koltun, and Thomas A. Funkhouser. Dilated residual networks. In Proceedings ofthe IEEE Conference on Computer Vision and Pattern Recognition, pages 636–644, 2017.  
[37] Tianhe Yu, Saurabh Kumar, Abhishek Gupta, Sergey Levine, Karol Hausman, and Chelsea Finn. Gradient surgery for multi-task learning. In Proceedings of the 33rd Advances in Neural Information Processing Systems, 2020.  
[38] Yu Zhang and Qiang Yang. A survey on multi-task learning. IEEE Transactions on Knowledge and Data Engineering, 2021.

## Appendix

## A Proof of the Mean Value E(λ)

Suppose that $\tilde { \lambda } _ { t } ( t = 1 , \cdots , T )$ are independent and identically distributed (i.i.d.) random variables sampled from the Uniform or standard Normal distributions and f is the softmax function. Then we have $\begin{array} { r } { \lambda _ { t } = \frac { \exp ( \tilde { \lambda } _ { t } ) } { \sum _ { m = 1 } ^ { T } \exp ( \tilde { \lambda } _ { m } ) } } \end{array}$ and

$$
\mathbb {E} (\lambda_ {i}) = \mathbb {E} [ \exp (\tilde {\lambda} _ {i}) ] \mathbb {E} \left[ \frac {1}{\sum_ {m = 1} ^ {T} \exp (\tilde {\lambda} _ {m})} \right] + \operatorname{Cov} \left(\exp (\tilde {\lambda} _ {i}), \frac {1}{\sum_ {m = 1} ^ {T} \exp (\tilde {\lambda} _ {m})}\right),
$$

where $\operatorname { C o v } ( \cdot , \cdot )$ denotes the covariance between two random variables. Since $\{ \tilde { \lambda } _ { t } \} _ { t = 1 } ^ { T }$ are i.i.d random variables, we have $\mathbb { E } [ \exp ( \tilde { \lambda } _ { i } ) ] = \mathbb { E } [ \exp ( \tilde { \lambda } _ { j } ) ]$ and $\begin{array} { r l } { \mathrm { C o v } ( \exp ( \tilde { \lambda } _ { i } ) , 1 / \sum _ { m = 1 } ^ { T } \exp ( \tilde { \lambda } _ { m } ) ) } & { { } = } \end{array}$ $\mathrm { C o v } ( \exp ( \tilde { \lambda } _ { j } ) , 1 / \sum _ { m = 1 } ^ { T } \exp ( \tilde { \lambda } _ { m } ) )$ . Therefore, we obtain

$$
\mathbb {E} (\lambda_ {i}) = \mathbb {E} (\lambda_ {j}), \forall 1 \leq i, j \leq T.
$$

Moreover, we have

$$
\sum_ {t = 1} ^ {T} \mathbb {E} (\lambda_ {t}) = \sum_ {t = 1} ^ {T} \frac {\sum_ {k = 1} ^ {K} \lambda_ {t} ^ {k}}{K} = \frac {\sum_ {k = 1} ^ {K} \sum_ {t = 1} ^ {T} \lambda_ {t} ^ {k}}{K} = 1.
$$

Thus we have $\begin{array} { r } { \mathbb { E } ( \lambda ) = ( \frac { 1 } { T } , \cdot \cdot \cdot , \frac { 1 } { T } ) } \end{array}$ . Similarly, we can prove the same result for the Bernoulli and c-Bernoulli distributions with the normalization function f as $f ( \tilde { \lambda } ) = \tilde { \lambda } / ( \sum _ { t = 1 } ^ { T } \tilde { \lambda } _ { t } )$

## B Proof of Section 4

## B.1 Proof of Theorem 1

Suppose $\mathcal { L } _ { \mathrm { R L W } } ( \theta ) = \lambda ^ { \top } \ell ( \theta )$ , where λ is a random variable sampled from a random distribution in every training iteration.

Since $\ell _ { t }$ is c<sub>t</sub>-strongly convex w.r.t. $\theta ,$ for any two points $\theta _ { 1 }$ and $\theta _ { 2 }$ in $\mathbb { R } ^ { d }$ , we have

$$
\begin{array}{l} \left\langle \nabla \boldsymbol {\lambda} ^ {\top} \boldsymbol {\ell} (\theta_ {1}) - \nabla \boldsymbol {\lambda} ^ {\top} \boldsymbol {\ell} (\theta_ {2}), \theta_ {1} - \theta_ {2} \right\rangle = \sum_ {t = 1} ^ {T} \lambda_ {t} \left\langle \nabla \ell_ {t} (\theta_ {1}) - \nabla \ell_ {t} (\theta_ {2}), \theta_ {1} - \theta_ {2} \right\rangle \\ \geq \sum_ {t = 1} ^ {T} c _ {t} \lambda_ {t} \| \theta_ {1} - \theta_ {2} \| ^ {2}. \tag {4} \\ \end{array}
$$

Since $0 \leq \lambda _ { t } \leq 1$ , we have $\textstyle \sum _ { t = 1 } ^ { T } c _ { t } \lambda _ { t } \geq c ,$ , where $c = \mathrm { m i n } _ { 1 \leq t \leq T } \{ c _ { t } \}$ . Then for any $\lambda , { \mathcal { L } } _ { \mathrm { R L W } } ( \theta )$ is c-strongly convex.

With notations in Theorem 1, we have

$$
\begin{array}{l} \| \theta_ {k + 1} - \theta_ {*} \| ^ {2} = \| \theta_ {k} - \theta_ {*} - \eta \nabla \boldsymbol {\lambda} ^ {\top} \boldsymbol {\ell} (\tilde {\boldsymbol {\mathcal {D}}}; \theta_ {k}) \| ^ {2} \\ = \| \theta_ {k} - \theta_ {*} \| ^ {2} - 2 \eta \left\langle \theta_ {k} - \theta_ {*}, \nabla \boldsymbol {\lambda} ^ {\top} \boldsymbol {\ell} (\tilde {\boldsymbol {\mathcal {D}}}; \theta_ {k}) \right\rangle + \eta^ {2} \| \nabla \boldsymbol {\lambda} ^ {\top} \boldsymbol {\ell} (\tilde {\boldsymbol {\mathcal {D}}}; \theta_ {k}) \| ^ {2}. \\ \end{array}
$$

Note that $\mathbb { E } _ { \lambda } \left[ \mathbb { E } _ { \tilde { \pmb { \mathscr { D } } } } [ \nabla \lambda ^ { \top } \pmb { \ell } ( \tilde { \pmb { \mathscr { D } } } ; \theta _ { k } ) ] \right] = \nabla \pmb { \mu } ^ { \top } \pmb { \ell } ( \pmb { \mathscr { D } } ; \theta _ { k } )$ and

$$
\begin{array}{l} \mathbb {E} _ {\boldsymbol {\lambda}} \left[ \mathbb {E} _ {\tilde {\boldsymbol {\mathcal {D}}}} [ \| \nabla \boldsymbol {\lambda} ^ {\top} \boldsymbol {\ell} (\tilde {\boldsymbol {\mathcal {D}}}; \theta_ {k}) \| ^ {2} ] \right] \leq \mathbb {E} _ {\boldsymbol {\lambda}} \left[ \mathbb {E} _ {\tilde {\boldsymbol {\mathcal {D}}}} [ \| \boldsymbol {\lambda} ^ {\top} \| ^ {2} \| \nabla \boldsymbol {\ell} (\tilde {\boldsymbol {\mathcal {D}}}; \theta_ {k}) \| ^ {2} ] \right] \\ \leq \mathbb {E} _ {\boldsymbol {\lambda}} \left[ \sum_ {t = 1} ^ {T} \lambda_ {t} ^ {2} \right] \cdot \sum_ {t = 1} ^ {T} \sigma_ {t} ^ {2} \\ \leq \sum_ {t = 1} ^ {T} \sigma_ {t} ^ {2}, \\ \end{array}
$$

where the first inequality is due to the Cauchy-Schwarz inequality and the third inequality is due to $0 \leq \lambda _ { t } \leq 1$ and $\textstyle \sum _ { t } \lambda _ { t } = 1$ . Then, by defining $\begin{array} { r } { \kappa = \sum _ { t = 1 } ^ { T } \sigma _ { t } ^ { 2 } } \end{array}$ , we obtain

$$
\begin{array}{l} \mathbb {E} _ {\boldsymbol {\lambda}} \left[ \mathbb {E} _ {\tilde {\boldsymbol {\mathcal {D}}}} [ \| \theta_ {k + 1} - \theta_ {*} \| ^ {2} ] \right] \leq \| \theta_ {k} - \theta_ {*} \| ^ {2} - 2 \eta \left\langle \theta_ {k} - \theta_ {*}, \nabla \boldsymbol {\mu} ^ {\top} \boldsymbol {\ell} (\theta_ {k}) \right\rangle + \eta^ {2} \kappa \\ \leq (1 - 2 \eta c) \| \theta_ {k} - \theta_ {*} \| ^ {2} + \eta^ {2} \kappa . \tag {5} \\ \end{array}
$$

$\mathrm { I f ~ 1 - 2 } \eta c > 0$ , we recursively apply the inequality (5) over the first k iterations and we can obtain

$$
\begin{array}{l} \mathbb {E} [ \| \theta_ {k + 1} - \theta_ {*} \| ^ {2} ] \leq (1 - 2 \eta c) ^ {k} \| \theta_ {0} - \theta_ {*} \| ^ {2} + \sum_ {j = 0} ^ {k - 1} (1 - 2 \eta c) ^ {j} \eta^ {2} \kappa \\ \leq (1 - 2 \eta c) ^ {k} \| \theta_ {0} - \theta_ {*} \| ^ {2} + \frac {\eta \kappa}{2 c}. \\ \end{array}
$$

Thus the inequality (3) holds if $\begin{array} { r } { \eta \le \frac { 1 } { 2 c } } \end{array}$

According to inequality (5), the minimal value of a quadratic function $g _ { \varepsilon } ( \eta ) = ( 1 - 2 \eta c ) \varepsilon + \eta ^ { 2 } /$ κ is achieved at $\begin{array} { r } { \eta _ { * } = \frac { \varepsilon c } { \kappa } } \end{array}$ . By setting $\lVert \theta _ { 0 } - \theta _ { * } \rVert ^ { 2 } = \varepsilon _ { 0 }$ , we have

$$
\begin{array}{l} \mathbb {E} [ \| \theta_ {k + 1} - \theta_ {*} \| ^ {2} ] \leq g _ {\| \theta_ {k} - \theta_ {*} \| ^ {2}} (\eta_ {*}) \\ = (1 - \frac {2 \| \theta_ {k} - \theta_ {*} \| ^ {2} c ^ {2}}{\kappa}) \| \theta_ {k} - \theta_ {*} \| ^ {2} \\ \leq (1 - \frac {2 \varepsilon c ^ {2}}{\kappa}) \| \theta_ {k} - \theta_ {*} \| ^ {2} \\ \leq (1 - \frac {2 \varepsilon c ^ {2}}{\kappa}) ^ {k} \varepsilon_ {0}. \\ \end{array}
$$

Then if $\mathbb { E } [ \| \theta _ { k + 1 } - \theta _ { * } \| ^ { 2 } ] \ge \varepsilon .$ , we have $\begin{array} { r } { \varepsilon \le \big ( 1 - \frac { 2 \varepsilon c ^ { 2 } } { \kappa } \big ) ^ { k } \varepsilon _ { 0 } } \end{array}$ . Therefore, $\begin{array} { r } { k \leq \frac { \kappa } { 2 \varepsilon c ^ { 2 } } \log \left( \frac { \varepsilon _ { 0 } } { \varepsilon } \right) } \end{array}$

## B.2 Proof of Theorem 2

Since $\varphi _ { k } = \theta _ { k } - \eta \nabla \pmb { \mu } ^ { \top } \pmb { \ell } ( \theta _ { k } )$ and $\theta _ { k + 1 } = \theta _ { k } - \eta ( \nabla \mu ^ { \top } \ell ( \theta _ { k } ) + \xi _ { k } )$ , we have

$$
\varphi_ {k + 1} = \varphi_ {k} - \eta \xi_ {k} - \nabla \boldsymbol {\mu} ^ {\top} \boldsymbol {\ell} (\varphi_ {k} - \eta \xi_ {k}).
$$

Since the loss function $\ell _ { t } ( \theta )$ of task t is $c _ { t } \mathrm { - } \mathrm { 0 n e }$ point strongly convex w.r.t. a given point $\theta _ { * }$ after convolved with noise ξ, similar to inequality (4), we have

$$
\left\langle \nabla \mathbb {E} _ {\xi} [ \boldsymbol {\mu} ^ {\top} \boldsymbol {\ell} (\varphi - \eta \xi) ], \varphi - \theta_ {*} \right\rangle \geq c \| \varphi - \theta_ {*} \| ^ {2},
$$

where $c = \mathrm { m i n } _ { 1 \leq t \leq T } \{ c _ { t } \}$ . Since $\nabla \ell _ { t } ( \theta )$ is M<sub>t</sub>-Lipschitz continuous, for any two points $\theta _ { 1 }$ and $\theta _ { 2 }$ in $\mathbb { R } ^ { d }$ , we have

$$
\| \nabla \boldsymbol {\mu} ^ {\top} \boldsymbol {\ell} (\theta_ {1}) - \nabla \boldsymbol {\mu} ^ {\top} \boldsymbol {\ell} (\theta_ {2}) \| = \sum_ {t = 1} ^ {T} \mu_ {t} \| \nabla \ell_ {t} (\theta_ {1}) - \nabla \ell_ {t} (\theta_ {2}) \| \leq \sum_ {t = 1} ^ {T} M _ {t} \mu_ {t} \| \theta_ {1} - \theta_ {2} \|. \tag {6}
$$

Note that $\begin{array} { r } { \sum _ { t = 1 } ^ { T } M _ { t } \mu _ { t } \le M } \end{array}$ , where $M = \operatorname* { m a x } _ { 1 \leq t \leq T } \{ M _ { t } \}$ . Therefore, $\nabla \mu ^ { \top } \ell ( \theta )$ is M-Lipschitz continuous. Then we can get

$$
\begin{array}{l} \mathbb {E} [ \| \varphi_ {k + 1} - \theta_ {*} \| ^ {2} ] = \mathbb {E} [ \| \varphi_ {k} - \eta \xi_ {k} - \nabla \boldsymbol {\mu} ^ {\top} \boldsymbol {\ell} (\varphi_ {k} - \eta \xi_ {k}) - \theta_ {*} \| ^ {2} ] \\ \leq \mathbb {E} [ \| \varphi_ {k} - \theta_ {*} \| ^ {2} + \| \eta \xi_ {k} \| ^ {2} + \| \nabla \boldsymbol {\mu} ^ {\top} \boldsymbol {\ell} (\varphi_ {k} - \eta \xi_ {k}) \| ^ {2} - 2 \left\langle \varphi_ {k} - \theta_ {*}, \eta \xi_ {k} \right\rangle \\ \left. - 2 \left\langle \varphi_ {k} - \theta_ {*}, \nabla \boldsymbol {\mu} ^ {\top} \boldsymbol {\ell} \left(\varphi_ {k} - \eta \xi_ {k}\right) \right\rangle + 2 \left\langle \nabla \boldsymbol {\mu} ^ {\top} \boldsymbol {\ell} \left(\varphi_ {k} - \eta \xi_ {k}\right), \eta \xi_ {k} \right\rangle \right] \\ \leq \| \varphi_ {k} - \theta_ {*} \| ^ {2} + \eta^ {2} r ^ {2} + \mathbb {E} [ \| \nabla \boldsymbol {\mu} ^ {\top} \boldsymbol {\ell} (\varphi_ {k} - \eta \xi_ {k}) \| ^ {2} ] - 2 \eta c \| \varphi_ {k} - \theta_ {*} \| ^ {2} \\ + 2 \mathbb {E} [ \big \langle \nabla \boldsymbol {\mu} ^ {\top} \boldsymbol {\ell} (\varphi_ {k} - \eta \xi_ {k}) - \nabla \boldsymbol {\mu} ^ {\top} \boldsymbol {\ell} (\varphi_ {k}), \eta \xi_ {k} \big \rangle ] \\ \leq (1 - 2 \eta c) \| \varphi_ {k} - \theta_ {*} \| ^ {2} + \eta^ {2} r ^ {2} + \eta^ {2} \mathbb {E} [ \| M (\theta_ {*} - (\varphi_ {k} - \eta \xi_ {k})) \| ^ {2} ] + 2 \eta^ {3} r ^ {2} M \\ \leq (1 - 2 \eta c) \| \varphi_ {k} - \theta_ {*} \| ^ {2} + \eta^ {2} r ^ {2} + \eta^ {2} M ^ {2} \| \varphi_ {k} - \theta_ {*} \| ^ {2} + \mathbb {E} [ \langle \varphi_ {k} - \theta_ {*}, \eta \xi_ {k} \rangle ] \\ + \eta^ {2} M ^ {2} \mathbb {E} [ \| \eta \xi_ {k} \| ^ {2} ] + 2 \eta^ {3} r ^ {2} M \\ \leq (1 - 2 \eta c + \eta^ {2} M ^ {2}) \| \varphi_ {k} - \theta_ {*} \| ^ {2} + \eta^ {2} r ^ {2} (1 + \eta M) ^ {2}, \\ \end{array}
$$

where the second inequality is due to the convexity assumption and $\mathbb { Z } [ \xi _ { k } ] = 0 .$ , the third and forth inequalities are due to the Lipschitz continuity. We set $\rho = \dot { 2 } \eta c - \eta ^ { 2 } \dot { M } ^ { 2 }$ and $\beta = \eta ^ { 2 } r ^ { 2 } ( 1 + \eta M ) ^ { 2 }$ If $\dot { \rho } \geq 0$ , we have $\eta \leq \frac { c } { M ^ { 2 } }$ , then we get

$$
\begin{array}{l} \mathbb {E} [ \| \varphi_ {k + 1} - \theta_ {*} \| ^ {2} ] \leq (1 - \rho) \| \varphi_ {k} - \theta_ {*} \| ^ {2} + \beta \\ \leq (1 - \rho) ^ {k} \| \varphi_ {0} - \theta_ {*} \| ^ {2} + \sum_ {j = 0} ^ {k - 1} (1 - \rho) ^ {j} \beta \\ \leq (1 - \rho) ^ {k} \| \varphi_ {0} - \theta_ {*} \| ^ {2} + \frac {\beta}{\rho}. \\ \end{array}
$$

So if $\begin{array} { r } { K \le \frac { 1 } { \rho } \log \left( \frac { \rho \varepsilon _ { 0 } } { \beta } \right) } \end{array}$ , we have $\begin{array} { r } { \mathbb { E } [ \| \varphi _ { K + 1 } - \theta _ { * } \| ^ { 2 } ] \le \frac { 2 \beta } { \rho } } \end{array}$ . Then by the Markov inequality, with probability at least $\mathrm { i } - \delta$ , we have

$$
\| \varphi_ {K} - \theta_ {*} \| ^ {2} \leq \frac {2 \beta}{\rho \delta}.
$$

## B.3 Noise Upper Bound

Suppose the noise produced by the EW method is $\bar { \boldsymbol { \xi } } = \| \nabla \mu ^ { \top } \ell ( \tilde { \pmb { \mathcal { D } } } ; \theta ) - \nabla \mu ^ { \top } \ell ( \pmb { \mathcal { D } } ; \theta ) \|$ and $\| \bar { \xi } \| ^ { 2 } \leq R$ The noise produced by the RLW method is $\xi = \| \nabla \boldsymbol { \lambda } ^ { \top } \ell ( \tilde { \pmb { \mathscr { D } } } ; \theta ) - \nabla \pmb { \mu } ^ { \top } \ell ( \pmb { \mathscr { D } } ; \theta ) \|$ . We have

$$
\begin{array}{l} \| \xi \| ^ {2} = \| \nabla \boldsymbol {\lambda} ^ {\top} \boldsymbol {\ell} (\tilde {\boldsymbol {\mathcal {D}}}; \theta) - \nabla \boldsymbol {\mu} ^ {\top} \boldsymbol {\ell} (\tilde {\boldsymbol {\mathcal {D}}}; \theta) + \nabla \boldsymbol {\mu} ^ {\top} \boldsymbol {\ell} (\tilde {\boldsymbol {\mathcal {D}}}; \theta) - \nabla \boldsymbol {\mu} ^ {\top} \boldsymbol {\ell} (\boldsymbol {\mathcal {D}}; \theta) \| ^ {2} \\ = \| (\boldsymbol {\lambda} ^ {\top} - \boldsymbol {\mu} ^ {\top}) \nabla \ell (\tilde {\boldsymbol {\mathcal {D}}}; \theta) \| ^ {2} + 2 \left\langle (\boldsymbol {\lambda} ^ {\top} - \boldsymbol {\mu} ^ {\top}) \ell (\tilde {\boldsymbol {\mathcal {D}}}; \theta), \bar {\xi} \right\rangle + \| \bar {\xi} \| ^ {2}. \\ \end{array}
$$

Because the noise $\bar { \xi }$ can be any direction, there exists a constant $s > 0$ such that $\| \bar { \xi } \| ^ { 2 } = R$ and $\bar { \xi } = s ( \lambda ^ { \top } - { \pmb \mu } ^ { \top } ) \nabla \ell ( \tilde { \pmb D } ; \theta )$ . Then, we have $\| \xi \| ^ { 2 } \leq ( 1 + 2 s ) \| \lambda - \mu \| ^ { 2 } \| \nabla \ell ( \tilde { \pmb { \mathscr { D } } } ; \theta ) \| ^ { 2 } + R$ . Thus, the norm of the noise provided by the RLW method has a larger supremum than EW.

## C Additional Details about the XTREME Benchmark

Table 5: The numbers of training, validation, and test data for each language in PI and POS problems from the XTREME benchmark.

<table><tr><td></td><td>PI</td><td>POS</td></tr><tr><td>en</td><td>49.4K+2.0K+2.0K</td><td>6.9K+1.8K+3.2K</td></tr><tr><td>zh</td><td>49.4K+2.0K+2.0K</td><td>4.0K+0.5K+2.9K</td></tr><tr><td>de</td><td>49.4K+2.0K+2.0K</td><td>-</td></tr><tr><td>es</td><td>49.4K+2.0K+2.0K</td><td>-</td></tr><tr><td>te</td><td>-</td><td>1.0K+0.1K+0.1K</td></tr><tr><td>vi</td><td>-</td><td>1.4K+0.8K+0.8K</td></tr></table>

## D Additional Experimental Results

## D.1 Results on the CityScapes Dataset

Dataset. The CityScapes dataset [7] is a large-scale urban street scene understanding dataset and it is comprised of a diverse set of stereo video sequences recorded from 50 different cities in fine weather during the daytime. It contains 2,975 and 500 annotated images for training and test, respectively. This dataset includes two tasks: 7-class semantic segmentation and depth estimation.

Implementation details. For the CityScapes dataset, the network architecture and optimizer are the same as those in the NYUv2 dataset. We resize all the images to $1 2 8 \times 2 5 6$ and set the batch size to 64 for training. We use the cross-entropy loss and $L _ { 1 }$ loss for the semantic segmentation and depth estimation tasks, respectively.

Results. The results on the CityScapes dataset are shown in Table 6. The empirical observations are similar to those on the NYUv2 dataset in Table 1. Firstly, both the RLW and RGW strategies significantly outperform the EW method. Secondly, the RLW method can outperform most of the loss balancing baselines except the IMTL-L method. Moreover, the RGW method achieves 2.36% performance improvement and outperforms all of the baselines.

Table 6: Performance on the CityScapes dataset with two tasks: 7-class semantic segmentation and depth estimation. The best results for each task on each measure over loss/gradient balancing methods are marked with superscript ∗/†. The best results for each task on each measure are highlighted in bold. ↑ (↓) means the higher (lower) the result, the better the performance.

<table><tr><td rowspan="2" colspan="2">Methods</td><td colspan="2">Segmentation</td><td colspan="2">Depth</td><td rowspan="2"> $\Delta_p \uparrow$ </td></tr><tr><td>mIoU↑</td><td>Pix Acc↑</td><td>Abs Err↓</td><td>Rel Err↓</td></tr><tr><td></td><td>EW</td><td>68.71</td><td>91.50</td><td>0.0132</td><td>45.58</td><td>+0.00%</td></tr><tr><td rowspan="5">Loss Bal.</td><td>UW</td><td>68.84</td><td>91.53</td><td>0.0132</td><td>46.18</td><td>-0.09%</td></tr><tr><td>DWA</td><td>68.56</td><td>91.48</td><td>0.0135</td><td>44.49</td><td>+0.05%</td></tr><tr><td>IMTL-L</td><td>69.71*</td><td>91.77*</td><td>0.0128*</td><td>45.08</td><td>+1.58%*</td></tr><tr><td>MOML</td><td>69.34</td><td>91.65</td><td>0.0129</td><td>46.33</td><td>+0.59%</td></tr><tr><td>RLW (ours)</td><td>68.78</td><td>91.45</td><td>0.0134</td><td>43.68*</td><td>+0.69%</td></tr><tr><td rowspan="9">Gradient Bal.</td><td>MGDA-UB</td><td>68.41</td><td>91.13</td><td> $0.0124^†$ </td><td>46.85</td><td>+0.64%</td></tr><tr><td>GradNorm</td><td>68.60</td><td>91.48</td><td>0.0133</td><td>45.32</td><td>+0.01%</td></tr><tr><td>PCGrad</td><td>68.54</td><td>91.47</td><td>0.0135</td><td>44.82</td><td>-0.10%</td></tr><tr><td>GradDrop</td><td>68.62</td><td>91.45</td><td>0.0136</td><td>45.05</td><td>-0.42%</td></tr><tr><td>IMTL-G</td><td>68.62</td><td>91.48</td><td>0.0130</td><td>44.29</td><td>+1.09%</td></tr><tr><td>GradVac</td><td>68.60</td><td>91.47</td><td>0.0134</td><td>44.92</td><td>-0.06%</td></tr><tr><td>CAGrad</td><td>68.89</td><td>91.50</td><td>0.0128</td><td>44.72</td><td>+1.38%</td></tr><tr><td>RotoGrad</td><td>68.96</td><td>91.47</td><td>0.0127</td><td>43.85†</td><td>+2.13%</td></tr><tr><td>RGW (ours)</td><td> $69.68^†$ </td><td> $91.85^†$ </td><td>0.0127</td><td>43.91</td><td>+2.36%†</td></tr></table>

## D.2 Results on the CelebA Dataset

Dataset. The CelebA dataset [20] is a large-scale face attributes dataset with 202,599 face images, each of which has 40 attribute annotations. It is split into three parts: 162,770, 19,867, and 19,962 images for training, validation, and testing, respectively. Hence, this dataset contains 40 tasks and each task is a binary classification problem for one attribute.

Implementation details. We use the ResNet-18 network as a shared feature extractor and a fully connected layer with two output units as a task-specific head for each task. All the images are resized to 64 × 64. The Adam optimizer with the learning rate as $1 0 ^ { - 3 }$ is used for training and the batch size is set to 512. The cross-entropy loss is used for the 40 tasks.

Results. Since the number of tasks in the CelebA dataset is large, we only report the average classification accuracy on the forty tasks in Table 7. According to the results, the proposed RLW strategy slightly outperforms the EW method and performs comparably with loss balancing baseline methods. However, we can find that the RGW method and most of the gradient balancing methods are worse or achieve very limited improvement over the EW method, which indicates the gradient weighting is not suitable for the CelebA dataset.

## D.3 Results on the Office-31 and Office-Home Datasets

Datasets. The Office-31 dataset [26] consists of three domains: Amazon (A), DSLR (D), and Webcam (W), where each domain contains 31 object categories, and it contains 4,110 labeled images. We randomly split the whole dataset with 60% for training, 20% for validation, and the rest 20% for testing. The Office-Home dataset [31] has four domains: artistic images (Ar), clip art (Cl), product images (Pr), and real-world images (Rw). It has 15,500 labeled images in total and each domain contains 65 classes. We make the same split as the Office-31 dataset. For both datasets, we consider the multi-class classification problem on each domain as a task. Similar to multilingual problems from the XTREME benchmark, each task in both Office-31 and Office-Home datasets has its own input data.

Table 7: Average classification accuracy (%) of different methods on the CelebA dataset with forty tasks. The best results over loss/gradient balancing methods are marked with superscript ∗/†. The best results are highlighted in bold.

<table><tr><td></td><td>Methods</td><td>Avg Acc</td></tr><tr><td></td><td>EW</td><td>90.70</td></tr><tr><td rowspan="5">Loss Bal.</td><td>UW</td><td>90.84</td></tr><tr><td>DWA</td><td>90.77</td></tr><tr><td>IMTL-L</td><td>90.46</td></tr><tr><td>MOML</td><td>90.94*</td></tr><tr><td>RLW (ours)</td><td>90.73</td></tr><tr><td rowspan="9">Gradient Bal.</td><td>MGDA-UB</td><td>90.40</td></tr><tr><td>GradNorm</td><td>90.77</td></tr><tr><td>PCGrad</td><td> $90.85^†$ </td></tr><tr><td>GradDrop</td><td>90.71</td></tr><tr><td>IMTL-G</td><td>90.80</td></tr><tr><td>GradVac</td><td>90.75</td></tr><tr><td>CAGrad</td><td>90.72</td></tr><tr><td>RotoGrad</td><td>90.45</td></tr><tr><td>RGW (ours)</td><td>90.00</td></tr></table>

Implementation details. We use the same configuration for the Office-31 and Office-Home datasets. Specifically, the ResNet-18 network pre-trained on the ImageNet dataset is used as a shared backbone among tasks and a fully connected layer is applied as a task-specific output layer for each task. All the input images are resized to $2 2 4 \times 2 2 4$ . We use the Adam optimizer with the learning rate as $1 0 ^ { - 4 }$ and the weight decay as $1 0 ^ { - 5 }$ and set the batch size to 128 for training. The cross-entropy loss is used for all tasks in both datasets.

Table 8: Classification accuracy (%) of different methods on the Office-31 and Office-Home datasets. The best results for each domain over loss/gradient balancing methods are marked with superscript ∗/†. The best results for each task are highlighted in bold.

<table><tr><td rowspan="2" colspan="2">Methods</td><td colspan="4">Office-31</td><td colspan="5">Office-Home</td></tr><tr><td>A</td><td>D</td><td>W</td><td>Avg</td><td>Ar</td><td>Cl</td><td>Pr</td><td>Rw</td><td>Avg</td></tr><tr><td rowspan="6">Loss Bal.</td><td>EW</td><td>82.73</td><td>96.72</td><td>96.11</td><td>91.85</td><td>62.99</td><td>76.48</td><td>88.45</td><td>77.72</td><td>76.41</td></tr><tr><td>UW</td><td>82.73</td><td>96.72*</td><td>95.55</td><td>91.66</td><td>63.94</td><td>75.62</td><td>88.55</td><td>78.05</td><td>76.54</td></tr><tr><td>DWA</td><td>82.22</td><td>96.72*</td><td>96.11</td><td>91.68</td><td>63.37</td><td>76.05</td><td>89.08</td><td>77.62</td><td>76.53</td></tr><tr><td>IMTL-L</td><td>83.76</td><td>96.72*</td><td>95.55</td><td>92.01</td><td>65.46*</td><td>79.08*</td><td>88.45</td><td>78.81</td><td>77.95*</td></tr><tr><td>MOML</td><td>84.78*</td><td>95.08</td><td>96.67*</td><td>92.17</td><td>64.70</td><td>77.03</td><td>88.24</td><td>80.00</td><td>77.49</td></tr><tr><td>RLW (ours)</td><td>83.76</td><td>96.72*</td><td>96.67*</td><td>92.38*</td><td>62.80</td><td>76.48</td><td>90.57*</td><td>80.21*</td><td>77.52</td></tr><tr><td rowspan="9">Gradient Bal.</td><td>MGDA-UB</td><td>81.02</td><td>95.90</td><td> $97.77^†$ </td><td>91.56</td><td>64.32</td><td>75.29</td><td>89.72</td><td>79.35</td><td>77.17</td></tr><tr><td>GradNorm</td><td>83.93</td><td> $97.54^†$ </td><td>94.44</td><td>91.97</td><td> $65.46^†$ </td><td>75.29</td><td>88.66</td><td>78.91</td><td>77.08</td></tr><tr><td>PCGrad</td><td>82.22</td><td>96.72</td><td>95.55</td><td>91.49</td><td>63.94</td><td>76.05</td><td>88.87</td><td>78.27</td><td>76.78</td></tr><tr><td>GradDrop</td><td> $84.27^†$ </td><td>95.08</td><td>96.11</td><td>91.82</td><td>64.70</td><td>77.03</td><td>88.02</td><td>79.13</td><td>77.22</td></tr><tr><td>IMTL-G</td><td>82.22</td><td>95.90</td><td>96.11</td><td>91.41</td><td>63.37</td><td>76.05</td><td>89.19</td><td>79.24</td><td>76.96</td></tr><tr><td>GradVac</td><td>82.73</td><td> $97.54^†$ </td><td>95.55</td><td>91.94</td><td>63.18</td><td>76.48</td><td>88.66</td><td>77.83</td><td>76.53</td></tr><tr><td>CAGrad</td><td>82.22</td><td>96.72</td><td>96.67</td><td>91.87</td><td>63.75</td><td>75.94</td><td>89.08</td><td>78.27</td><td>76.75</td></tr><tr><td>RotoGrad</td><td>82.90</td><td>96.72</td><td>96.11</td><td>91.91</td><td>61.85</td><td>77.03</td><td> $90.36^†$ </td><td>78.59</td><td>76.95</td></tr><tr><td>RGW (ours)</td><td> $84.27^†$ </td><td>96.72</td><td>96.67</td><td> $92.55^†$ </td><td>65.08</td><td> $78.65^†$ </td><td>88.66</td><td> $79.89^†$ </td><td> $78.07^†$ </td></tr></table>

Results. According to the results shown in Table 8, we can see both the RLW and RGW strategies outperform the EW method on both two datasets in terms of the average classification accuracy over tasks, which implies the effectiveness of the RW methods. Moreover, the RGW method achieves the best performance (92.55% and 78.07% in term of the average accuracy) over all baselines on the Office-31 and Office-Home datasets, respectively.