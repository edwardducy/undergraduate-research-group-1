# Reasonable Effectiveness of Random Weighting:  
A Litmus Test for Multi-Task Learning

 Baijiong Lin Affiliation:  Department of Computer Science and EngineeringSouthern University of Science and Technology Email: [bj.lin.email@gmail.com](mailto:)    Feiyang Ye Affiliation:  Department of Computer Science and EngineeringSouthern University of Science and Technology Affiliation:  University of Technology Sydney Email: [12060007@mail.sustech.edu.cn](mailto:)    Yu Zhang ††thanks: Corresponding author: Yu Zhang Affiliation:  Department of Computer Science and EngineeringSouthern University of Science and Technology Affiliation:  Peng Cheng Laboratory Email: [yu.zhang.ust@gmail.com](mailto:)    Ivor W. Tsang Affiliation:  Centre for Frontier AI Research (CFAR)Agency for Science, Technology and Research (A\*STAR) Email: [ivortsang@ihpc.a-star.edu.sg](mailto:) 

###### Abstract

Multi-Task Learning (MTL) has achieved success in various fields. However, how to balance different tasks to achieve good performance is a key problem. To achieve the task balancing, there are many works to carefully design dynamical loss/gradient weighting strategies but the basic random experiments are ignored to examine their effectiveness. In this paper, we propose the Random Weighting (RW) methods, including Random Loss Weighting (RLW) and Random Gradient Weighting (RGW), where an MTL model is trained with random loss/gradient weights sampled from a distribution. To show the effectiveness and necessity of RW methods, theoretically we analyze the convergence of RW and reveal that RW has a higher probability to escape local minima, resulting in better generalization ability. Empirically, we extensively evaluate the proposed RW methods to compare with twelve state-of-the-art methods on five image datasets and two multilingual problems from the XTREME benchmark to show RW methods can achieve comparable performance with state-of-the-art baselines. Therefore, we think that the RW methods are important baselines for MTL and should attract more attentions.

## 1 Introduction

Multi-Task Learning (MTL) \[[2](#bib.bib2 ""), [38](#bib.bib38 ""), [30](#bib.bib30 "")\] aims to jointly train several related tasks to improve their generalization performance by leveraging common knowledge among them. Since MTL could not only significantly reduce the model size as well as speed up the inference but also improve the performance, it has been successfully applied to various fields \[[38](#bib.bib38 "")\]. However, when all the tasks are not highly related, which may be reflected via conflicting gradients or dominating gradients \[[37](#bib.bib37 "")\], it is more difficult to train an MTL model than training them separately because some tasks dominantly influence model parameters, leading to unsatisfactory performance for other tasks. This phenomenon is related to the task balancing problem \[[30](#bib.bib30 "")\] in MTL.

Recently, several works focus on tackling this issue from an optimization perspective via dynamically weighting task losses or balancing task gradients in the training process, called loss balancing and gradient balancing methods, respectively. However, all of the existing works take Equal Weighting (EW) which uses the fixed and equal weights in the whole training process as a basic baseline to test the effectiveness of their methods. We think that this baseline is not sufficient and it is quite necessary to conduct random experiments, which is missing in existing works, as a baseline to test them.

Therefore, in this paper, we propose the Random Weighting (RW) methods including Random Loss Weighting (RLW) and Random Gradient Weighting (RGW) as more reasonable baselines to test loss and gradient balancing methods, respectively. Specifically, in each training iteration, we first sample loss/gradient weights from a distribution with some possible normalization and then minimize the aggregated loss/gradient weighted by the random loss/gradient weights. Although the RW methods seem unreasonable, they can not only converge but also achieve comparable performance with existing methods that use carefully tuned weights. Thus, we think the RW methods are important baselines for MTL and deserve more attention.

To better understand the effectiveness and necessity of RW methods, we provide both theoretical analyses and empirical evaluations. Theoretically, we show RW methods are the stochastic variants of EW. From this perspective, we give a convergence analysis for RW methods. Besides, we can show that RW methods have a higher probability to escape local minima than EW, resulting in better generalization performance. Empirically, we investigate lots of State-Of-The-Art (SOTA) task balancing approaches including four loss balancing methods and eight gradient balancing methods. On five Computer Vision (CV) datasets and two multilingual problems from the XTREME benchmark \[[11](#bib.bib11 "")\], we show that RW methods can consistently outperform EW and have competitive performance with existing SOTA methods.

In summary, the main contributions of this paper are three-fold.

*   •

```
We propose the simple RW methods as novel baselines and litmus tests for MTL.
```
*   •

```
We provide the convergence guarantee and effectiveness analysis for RW methods.
```
*   •

```
Extensive experiments show that RW can outperform EW and achieve comparable performance with the SOTA methods.
```
## 2 An Overview of Task Balancing Methods

Notations. Suppose there are TT tasks and task tt has its corresponding dataset 𝒟t\\mathcal{D}\_{t}. An MTL model usually contains two parts of parameters: task-sharing parameters θ\\theta and task-specific parameters {ψt}t\=1T\\{\\psi\_{t}\\}\_{t=1}^{T}. For example, in CV, θ\\theta usually denotes parameters in the feature extractor shared by all the tasks and ψt\\psi\_{t} represents parameters in the task-specific output module for task tt. Let ℓt​(𝒟t,θ,ψt)\\ell\_{t}(\\mathcal{D}\_{t};\\theta,\\psi\_{t}) denotes the average loss on 𝒟t\\mathcal{D}\_{t} for task tt. {λt}t\=1T\\{\\lambda\_{t}\\}\_{t=1}^{T} are task-specific loss weights with a constraint that λtl≥0\\lambda\_{t}^{l}\\geq 0 for all tt’s. Similarly, {λtg}t\=1T\\{\\lambda\_{t}^{g}\\}\_{t=1}^{T} denote task-specific gradient weights.

Conventional Baseline with Fixed Weights. Since there are multiple losses in MTL, they usually are aggregated as a single one via loss weights as

ℒ⁡(θ,{ψt}t\=1T)\=∑t\=1Tλtl​ℓt​(𝒟t,θ,ψt).\\mathcal{L}(\\theta,\\{\\psi\_{t}\\}\_{t=1}^{T})=\\sum\_{t=1}^{T}\\lambda\_{t}^{l}\\ell\_{t}(\\mathcal{D}\_{t};\\theta,\\psi\_{t}).

(1)

Apparently, the most simple method for loss weighting is to assign the same weight to all the tasks in the whole training process, i.e., without loss of generality, λtl\=1T\\lambda\_{t}^{l}=\\frac{1}{T} for all tt’s in every iteration. This approach is a common baseline in MTL and it is called EW in this paper.

Loss Balancing Methods. To achieve task balancing and improve the performance of MTL model, loss balancing methods aim to study how to generate appropriate loss weights {λtl}t\=1T\\{\\lambda\_{t}^{l}\\}\_{t=1}^{T} in Eq. ([1](#S2.E1 "In 2 An Overview of Task Balancing Methods ‣ Reasonable Effectiveness of Random Weighting:A Litmus Test for Multi-Task Learning")) in every iteration and some representative methods include Uncertainty Weights (UW) \[[13](#bib.bib13 "")\], Dynamic Weight Average (DWA) \[[19](#bib.bib19 "")\], IMTL-L \[[18](#bib.bib18 "")\] and Multi-Objective Meta Learning (MOML) \[[35](#bib.bib35 "")\]. These four methods focus on using higher loss weights for more difficult tasks measured by the uncertainty, learning speed, relative loss value, and validation performance, respectively. When minimizing Eq. ([1](#S2.E1 "In 2 An Overview of Task Balancing Methods ‣ Reasonable Effectiveness of Random Weighting:A Litmus Test for Multi-Task Learning")), the learning rate of optimizing each task-specific parameter ψt\\psi\_{t} will be affected by the corresponding loss weight λtl\\lambda\_{t}^{l}, which is the major difference between loss balancing and gradient balancing methods.

Gradient Balancing Methods. This type of methods think that the task balancing problem is caused by conflicting task gradients and the inappropriate gradient to update task-sharing parameters, thus they solve it via generating appropriate gradient weights {λtg}t\=1T\\{\\lambda\_{t}^{g}\\}\_{t=1}^{T} to balance the task gradients and make a better update of θ\\theta in every iteration as

θ\=θ−η​∑t\=1Tλtg​∇θℓt​(𝒟t,θ,ψt).\\theta=\\theta-\\eta\\sum\_{t=1}^{T}\\lambda\_{t}^{g}\\nabla\_{\\theta}\\ell\_{t}(\\mathcal{D}\_{t};\\theta,\\psi\_{t}).

(2)

Noticeably, in such type methods, the gradient weights {λtg}t\=1T\\{\\lambda\_{t}^{g}\\}\_{t=1}^{T} only affect the task-sharing parameter θ\\theta but not task-specific parameters {ψt}\\{\\psi\_{t}\\}, each of which is updated by the tt-th task gradient ∇ψtℓt​(𝒟t,θ,ψt)\\nabla\_{\\psi\_{t}}\\ell\_{t}(\\mathcal{D}\_{t};\\theta,\\psi\_{t}).

Some representative works include MGDA-UB \[[28](#bib.bib28 "")\], Gradient Normalization (GradNorm) \[[5](#bib.bib5 "")\], Projecting Conflicting Gradient (PCGrad) \[[37](#bib.bib37 "")\], Gradient sign Dropout (GradDrop) \[[6](#bib.bib6 "")\], Impartial Multi-Task Learning (IMTL-G) \[[18](#bib.bib18 "")\], Gradient Vaccine (GradVac) \[[32](#bib.bib32 "")\], Conflict-Averse Gradient (CAGrad) \[[17](#bib.bib17 "")\], and RotoGrad \[[12](#bib.bib12 "")\]. Those eight methods focus on finding an aggregated gradient by linearly combining all the task gradients under different constraints such as equal gradient magnitude in GradNorm and equal gradient projection in IMTL-G to eliminate the gradient conflicting.

Compared with the EW method, those two types of methods use a dynamic weighting process where loss/gradient weights vary over training iterations or epochs. Thus, it is natural to think how about training an MTL model with random weights. Inspired by this, we propose the RW methods by randomly sampling loss/gradient weights in each iteration as the random experiments for loss/gradient balancing methods, respectively. Besides, we think RW methods are more reasonable baselines than EW as the litmus tests for MTL methods.

## 3 The Random Weighting Methods

In this section, we introduce the RW methods, including the RLW and RGW methods.

We focus on the update of task-sharing parameter θ\\theta as it is the key problem in MTL. In the following, we mainly introduce the RLW method as the RGW method acts similarly to the RLW method. For notation simplicity, we do not distinguish between λtl{\\lambda}^{l}\_{t} and λtg{\\lambda}^{g}\_{t} and denote them by λt{\\lambda\_{t}}. Besides, we denote ℓ⁡(θ)\=(ℓ1​(𝒟1,θ,ψ1),⋯,ℓT​(𝒟T,θ,ψT))\\bm{\\ell}(\\theta)=(\\ell\_{1}(\\mathcal{D}\_{1};\\theta,\\psi\_{1}),\\cdots,\\ell\_{T}(\\mathcal{D}\_{T};\\theta,\\psi\_{T})), where the datasets {𝒟t}t\=1T\\{\\mathcal{D}\_{t}\\}\_{t=1}^{T} and the task-specific parameters {ψt}t\=1T\\{\\psi\_{t}\\}\_{t=1}^{T} are omitted for brevity.

Different from those loss balancing methods, RLW considers the loss weights 𝝀\=(λ1,⋯,λT)∈ℝT\\bm{\\lambda}=(\\lambda\_{1},\\cdots,\\lambda\_{T})\\in\\mathbb{R}^{T} as random variables and samples them from a random distribution in each iteration. To guarantee loss weights in 𝝀\\bm{\\lambda} to be non-negative, we can first sample 𝝀\~\=(λ\~1,⋯,λ\~T)\\tilde{\\bm{\\lambda}}=(\\tilde{\\lambda}\_{1},\\cdots,\\tilde{\\lambda}\_{T}) from any distribution p⁡(𝝀\~)p(\\tilde{\\bm{\\lambda}}) and then normalize 𝝀\~\\tilde{\\bm{\\lambda}} into 𝝀\\bm{\\lambda} via a mapping ff, where f:ℝT→ΔT−1f:\\mathbb{R}^{T}\\rightarrow\\Delta^{T-1} is a normalization function such as the softmax function and ΔT−1\\Delta^{T-1} denotes a simplex in ℝT\\mathbb{R}^{T}, i.e., 𝝀∈ΔT−1\\bm{\\lambda}\\in\\Delta^{T-1} means ∑t\=1Tλt\=1\\sum\_{t=1}^{T}\\lambda\_{t}=1 and λt≥0\\lambda\_{t}\\geq 0 for all tt. Note that p⁡(𝝀)p(\\bm{\\lambda}) is different from p⁡(𝝀\~)p(\\tilde{\\bm{\\lambda}}) unless ff is an identity function. Finally, RLW updates the θ\\theta by computing the aggregated gradient ∇θ𝝀⊤​ℓ​(θ)\\nabla\_{\\theta}\\bm{{\\lambda}}^{\\top}\\bm{\\ell}(\\theta).

In this way, the RLW method uses dynamical loss weights in the training process, which is similar to existing loss balancing methods, but RLW uses random weights instead of carefully designed ones in the existing works. Therefore, RLW is a basic random experiment for those loss balancing methods to examine their effectiveness, which indicates RLW is a more reasonable baseline than the conventional EW.

Algorithm 1 A Training Iteration in RLW

1: Input: numbers of tasks TT, learning rate η\\eta, dataset {𝒟t}t\=1T\\{\\mathcal{D}\_{t}\\}\_{t=1}^{T}, weight distribution p⁡(𝝀\~)p(\\tilde{\\bm{\\lambda}}), normalization function ff 

2: Output: task-sharing parameter θ′\\theta^{\\prime}, task-specific parameters {ψt′}t\=1T\\{\\psi\_{t}^{\\prime}\\}\_{t=1}^{T} 

3: for t\=1t=1 to TT do 

4:   Compute loss ℓt​(𝒟t,θ,ψt)\\ell\_{t}({\\mathcal{D}}\_{t};\\theta,\\psi\_{t}); 

5: end for 

6: Sample weights 𝝀\~\\tilde{\\bm{\\lambda}} from p⁡(𝝀\~)p(\\tilde{\\bm{\\lambda}}) and normalize it into 𝝀{\\bm{\\lambda}} via ff;⊳\\triangleright RLW Method 

7: θ′\=θ−η​∇θ​∑t\=1Tλt​ℓt​(𝒟t,θ,ψt)\\theta^{\\prime}=\\theta-\\eta\\nabla\_{\\theta}\\sum\_{t=1}^{T}{\\lambda}\_{t}\\ell\_{t}({\\mathcal{D}}\_{t};\\theta,\\psi\_{t}); 

8: for t\=1t=1 to TT do 

9:   ψt′\=ψt−η​∇ψtλt​ℓt​(𝒟t,θ,ψt)\\psi\_{t}^{\\prime}=\\psi\_{t}-\\eta\\nabla\_{\\psi\_{t}}{\\lambda}\_{t}\\ell\_{t}({\\mathcal{D}}\_{t};\\theta,\\psi\_{t}); 

10: end for 

Algorithm 2 A Training Iteration in RGW

1: Input: numbers of tasks TT, learning rate η\\eta, dataset {𝒟t}t\=1T\\{\\mathcal{D}\_{t}\\}\_{t=1}^{T}, weight distribution p⁡(𝝀\~)p(\\tilde{\\bm{\\lambda}}), normalization function ff 

2: Output: task-sharing parameter θ′\\theta^{\\prime}, task-specific parameters {ψt′}t\=1T\\{\\psi\_{t}^{\\prime}\\}\_{t=1}^{T} 

3: for t\=1t=1 to TT do 

4:   Compute loss ℓt​(𝒟t,θ,ψt)\\ell\_{t}({\\mathcal{D}}\_{t};\\theta,\\psi\_{t}); 

5:   Compute gradient gt\=∇θℓtg\_{t}=\\nabla\_{\\theta}\\ell\_{t} or ∇zℓt\\nabla\_{z}\\ell\_{t}; 

6: end for 

7: Sample weights 𝝀\~\\tilde{\\bm{\\lambda}} from p⁡(𝝀\~)p(\\tilde{\\bm{\\lambda}}) and normalize it into 𝝀{\\bm{\\lambda}} via ff;⊳\\triangleright RGW Method 

8: θ′\=θ−η​∇θ​∑t\=1Tλt​gt\\theta^{\\prime}=\\theta-\\eta\\nabla\_{\\theta}\\sum\_{t=1}^{T}{\\lambda}\_{t}g\_{t}; 

9: for t\=1t=1 to TT do 

10:   ψt′\=ψt−η​∇ψtℓt​(𝒟t,θ,ψt)\\psi\_{t}^{\\prime}=\\psi\_{t}-\\eta\\nabla\_{\\psi\_{t}}\\ell\_{t}({\\mathcal{D}}\_{t};\\theta,\\psi\_{t}); 

11: end for 

Noticeably, the loss weights 𝝀\\bm{\\lambda} are random variables and vary over training iterations, thus it is apparently that the gradient ∇θ𝝀⊤​ℓ​(θ)\\nabla\_{\\theta}\\bm{{\\lambda}}^{\\top}\\bm{\\ell}(\\theta) of RLW is an unbiased estimation of the gradient 𝔼​\[𝝀\]⊤​∇θℓ​(θ)\\mathbb{E}\[\\bm{\\lambda}\]^{\\top}\\nabla\_{\\theta}\\bm{\\ell}(\\theta), where 𝔼⁡\[𝝀\]\\mathbb{E}\[\\bm{\\lambda}\] is the expectation of 𝝀\\bm{\\lambda} over the whole training process. This means that the RLW method is a stochastic variant of the loss balancing method with fixed weights 𝔼⁡\[𝝀\]\\mathbb{E}\[\\bm{\\lambda}\]. In particular, if 𝔼⁡\[𝝀\]\\mathbb{E}\[\\bm{\\lambda}\] is proportional to (1T,⋯,1T)(\\frac{1}{T},\\cdots,\\frac{1}{T}), RLW is a stochastic variant of the conventional EW baseline. In Section [4](#S4 "4 Analysis ‣ Reasonable Effectiveness of Random Weighting:A Litmus Test for Multi-Task Learning"), we theoretically show that RLW has a better generalization performance that EW because of the extra randomness from loss weight sampling, which indicates the RLW method is a more effective baseline than EW.

Similar to RLW, in each iteration, RGW first randomly samples gradient weights 𝝀\~\\tilde{\\bm{\\lambda}} from p⁡(𝝀\~)p(\\tilde{\\bm{\\lambda}}), then normalizes it to obtain 𝝀\\bm{\\lambda} via ff, and finally updates the task-sharing parameter θ\\theta by computing the aggregated gradient ∇θ𝝀⊤​ℓ​(θ)\\nabla\_{\\theta}\\bm{{\\lambda}}^{\\top}\\bm{\\ell}(\\theta). Following previous works \[[28](#bib.bib28 ""), [6](#bib.bib6 ""), [18](#bib.bib18 ""), [12](#bib.bib12 "")\], we compute the gradient with respect to the final hidden feature representation zz output from the shared parameter instead of the task-sharing parameter θ\\theta to reduce the computational cost. Thus, RGW is a random experiment for gradient balancing methods.

In this paper, we use the standard normal distribution for p⁡(𝝀\~)p(\\tilde{\\bm{\\lambda}}) and the softmax function for ff in both the RLW and RGW methods since it is easy to implement, has a more stable performance (as shown in experimental results in Section [5.3](#S5.SS3 "5.3 Robustness on Distribution ‣ 5 Experiments ‣ Reasonable Effectiveness of Random Weighting:A Litmus Test for Multi-Task Learning")), and is as efficient as the EW strategy (as shown in experimental results in Section [5.4](#S5.SS4 "5.4 Convergence Speed ‣ 5 Experiments ‣ Reasonable Effectiveness of Random Weighting:A Litmus Test for Multi-Task Learning")). Besides, 𝔼⁡\[𝝀\]\\mathbb{E}\[\\bm{\\lambda}\] is proportional to (1T,⋯,1T)(\\frac{1}{T},\\cdots,\\frac{1}{T}) as proved in Appendix [A](#A1 "Appendix A Proof of the Mean Value 𝔼(𝜆) ‣ Reasonable Effectiveness of Random Weighting:A Litmus Test for Multi-Task Learning"), thus it is fair to compare with the EW strategy.

The training algorithms of both RW methods are summarized in Algorithm [1](#alg1 "Algorithm 1 ‣ 3 The Random Weighting Methods ‣ Reasonable Effectiveness of Random Weighting:A Litmus Test for Multi-Task Learning") and [2](#alg2 "Algorithm 2 ‣ 3 The Random Weighting Methods ‣ Reasonable Effectiveness of Random Weighting:A Litmus Test for Multi-Task Learning"). The only difference between the RW methods and the existing works is the generation of loss/gradient weights (i.e., Line 6 in Algorithm [1](#alg1 "Algorithm 1 ‣ 3 The Random Weighting Methods ‣ Reasonable Effectiveness of Random Weighting:A Litmus Test for Multi-Task Learning") and Line 7 in Algorithm [2](#alg2 "Algorithm 2 ‣ 3 The Random Weighting Methods ‣ Reasonable Effectiveness of Random Weighting:A Litmus Test for Multi-Task Learning")). Apparently, the sampling operation in the RW methods is very easy to implement and only bring negligibly additional computational costs when compared with the existing works. Note that random weights are involved in the update of task-specific parameters in the RLW method but not the RGW method (i.e., Line 9 in Algorithm [1](#alg1 "Algorithm 1 ‣ 3 The Random Weighting Methods ‣ Reasonable Effectiveness of Random Weighting:A Litmus Test for Multi-Task Learning") and Line 10 in Algorithm [2](#alg2 "Algorithm 2 ‣ 3 The Random Weighting Methods ‣ Reasonable Effectiveness of Random Weighting:A Litmus Test for Multi-Task Learning")).

## 4 Analysis

In this section, we analyze how the extra randomness from the loss/gradient weight sampling affects the convergence and effectiveness of the RW methods compared with the EW strategy.

We focus on the update of task-sharing parameter θ\\theta and take RLW as an example for analysis, which can easily be extended to the RGW method. For notation simplicity, we simply use ℓt​(θ)\\ell\_{t}(\\theta) instead of ℓt​(𝒟t,θ,ψt)\\ell\_{t}(\\mathcal{D}\_{t};\\theta,\\psi\_{t}) to denote the loss function of task tt in this section and Appendix [B](#A2 "Appendix B Proof of Section ‣ Reasonable Effectiveness of Random Weighting:A Litmus Test for Multi-Task Learning"). For ease of analysis, we make the following assumption.

###### Assumption 1.

𝔼𝒟t​\[‖∇ℓt​(𝒟t,θ)‖2\]\\mathbb{E}\_{\\mathcal{D}\_{t}}\[\\|\\nabla\\ell\_{t}(\\mathcal{D}\_{t};\\theta)\\|^{2}\] equals σt2\\sigma\_{t}^{2}, the loss function ℓt​(θ)\\ell\_{t}(\\theta) of task tt is LtL\_{t}-Lipschitz continuous w.r.t. θ\\theta, and 𝝀\\bm{\\lambda} satisfies 𝔼𝝀​\[𝝀\]\=𝝁\\mathbb{E}\_{\\bm{\\lambda}}\[\\bm{\\lambda}\]=\\bm{\\mu}.

In the following theorem, we analyze the convergence property of Algorithm [1](#alg1 "Algorithm 1 ‣ 3 The Random Weighting Methods ‣ Reasonable Effectiveness of Random Weighting:A Litmus Test for Multi-Task Learning") for the RLW method.

###### Theorem 1.

Suppose the loss function ℓt​(θ)\\ell\_{t}(\\theta) of task tt is ctc\_{t}-strongly convex. We define θ∗\=arg⁡minθ​𝛌⊤​ℓ​(θ)\\theta\_{\*}=\\arg\\min\_{\\theta}\\bm{\\lambda}^{\\top}\\bm{\\ell}(\\theta) and denote by θk\\theta\_{k} the solution in the kk-th iteration. If η\\eta, the step size or equivalently the learning rate, satisfies η≤1/2​c\\eta\\leq 1/2c, where c\=min1≤t≤T⁡{ct}c=\\min\_{1\\leq t\\leq T}\\{c\_{t}\\}, then under Assumption [1](#Thmassumption1 "Assumption 1. ‣ 4 Analysis ‣ Reasonable Effectiveness of Random Weighting:A Litmus Test for Multi-Task Learning") we have

𝔼⁡\[‖θk−θ∗‖2\]≤(1−2​η​c)k​‖θ0−θ∗‖2+η​κ2​c,\\mathbb{E}\[\\|\\theta\_{k}-\\theta\_{\*}\\|^{2}\]\\leq(1-2\\eta c)^{k}\\|\\theta\_{0}-\\theta\_{\*}\\|^{2}+\\frac{\\eta\\kappa}{2c},

(3)

where κ\=∑t\=1Tσt2\\kappa=\\sum\_{t=1}^{T}\\sigma\_{t}^{2}. Then for any positive ε\\varepsilon, 𝔼⁡\[‖θk−θ∗‖2\]≤ε\\mathbb{E}\[\\|\\theta\_{k}-\\theta\_{\*}\\|^{2}\]\\leq\\varepsilon can be achieved after k\=κ2​ε​c2​log⁡(ε0ε)k=\\frac{\\kappa}{2\\varepsilon c^{2}}\\log\\left(\\frac{\\varepsilon\_{0}}{\\varepsilon}\\right) iterations with η\=ε​cκ\\eta=\\frac{\\varepsilon c}{\\kappa}, where ε0\=𝔼⁡\[‖θ0−θ∗‖2\]\\varepsilon\_{0}=\\mathbb{E}\[\\|\\theta\_{0}-\\theta\_{\*}\\|^{2}\].

Theorem [1](#Thmtheorem1 "Theorem 1. ‣ 4 Analysis ‣ Reasonable Effectiveness of Random Weighting:A Litmus Test for Multi-Task Learning") shows that the RLW method with a fixed step size has a linear convergence up to a radius around the optimal solution, which is similar to the EW strategy according to the property of the standard Stochastic Gradient Descent (SGD) method \[[23](#bib.bib23 ""), [24](#bib.bib24 "")\]. Although the RLW method has a larger κ\\kappa than the EW strategy, i.e., κEW\=∑t\=1Tμt2⋅∑t\=1Tσt2≤κ\\kappa\_{\\mathrm{EW}}=\\sum\_{t=1}^{T}\\mu\_{t}^{2}\\cdot\\sum\_{t=1}^{T}\\sigma\_{t}^{2}\\leq\\kappa, which may possibly require more iterations for the RLW method to reach the same accuracy as the EW strategy, experimental results in Section [5.4](#S5.SS4 "5.4 Convergence Speed ‣ 5 Experiments ‣ Reasonable Effectiveness of Random Weighting:A Litmus Test for Multi-Task Learning") show that empirically this does not cause much difference.

We next analyze the effectiveness of the RLW method from the perspective of stochastic optimization. It is observed that the SGD method can escape sharp local minima and converge to a better solution than Gradient Descent (GD) techniques under various settings with the help of noisy gradients \[[10](#bib.bib10 ""), [16](#bib.bib16 "")\]. Inspired by those works, we prove Theorem [2](#Thmtheorem2 "Theorem 2. ‣ 4 Analysis ‣ Reasonable Effectiveness of Random Weighting:A Litmus Test for Multi-Task Learning") to show that the extra randomness in the RLW method can help RLW to better escape sharp local minima and achieve better generalization performance than the EW strategy.

Before presenting the theorem, for the ease of presentation, we introduction some notations. Here we consider the update step of these stochastic methods as θk+1\=θk−η(∇𝝁⊤ℓ(θk)+ξk)\\theta\_{k+1}=\\theta\_{k}-\\eta(\\nabla\\bm{\\mu}^{\\top}\\bm{\\ell}(\\theta\_{k})+\\xi\_{k}), where ξk\\xi\_{k} is a noise with 𝔼⁡\[ξk\]\=0\\mathbb{E}\[\\xi\_{k}\]=0 and ‖ξk‖2≤r\\|\\xi\_{k}\\|^{2}\\leq r, and rr denotes the intensity of the noise. For the analysis, we construct an intermediate sequence φk\=θk−η∇𝝁⊤ℓ(θk)\\varphi\_{k}=\\theta\_{k}-\\eta\\nabla\\bm{\\mu}^{\\top}\\bm{\\ell}(\\theta\_{k}). Then we get 𝔼ξk\[φk+1\]\=φk−η∇𝔼ξk\[𝝁⊤ℓ(φk−ηξk)\]\\mathbb{E}\_{\\xi\_{k}}\[\\varphi\_{k+1}\]=\\varphi\_{k}-\\eta\\nabla\\mathbb{E}\_{\\xi\_{k}}\[\\bm{\\mu}^{\\top}\\bm{\\ell}(\\varphi\_{k}-\\eta\\xi\_{k})\]. Therefore, the sequence {φk}\\{\\varphi\_{k}\\} can be regarded as an approximation of using GD to minimize the function 𝔼ξk​\[𝝁⊤​ℓ​(φ−η​ξk)\]\\mathbb{E}\_{\\xi\_{k}}\[\\bm{\\mu}^{\\top}\\bm{\\ell}(\\varphi-\\eta\\xi\_{k})\].

###### Theorem 2.

Suppose ∇ℓt​(θ)\\nabla\\ell\_{t}(\\theta) is MtM\_{t}-Lipschitz continuous and ‖ξk‖2≤r\\|\\xi\_{k}\\|^{2}\\leq r. If the loss function ℓt​(θ)\\ell\_{t}(\\theta) of task tt is ctc\_{t}-one point strongly convex w.r.t. a local minimum θ∗\\theta\_{\*} after convolved with noise ξ\\xi, i.e., ⟨∇𝔼ξℓt(φ−ηξ),φ−θ∗⟩≥ct∥φ−θ∗∥2\\left<\\nabla\\mathbb{E}\_{\\xi}\\ell\_{t}(\\varphi-\\eta\\xi),\\varphi-\\theta\_{\*}\\right>\\geq c\_{t}\\|\\varphi-\\theta\_{\*}\\|^{2}, then under Assumption [1](#Thmassumption1 "Assumption 1. ‣ 4 Analysis ‣ Reasonable Effectiveness of Random Weighting:A Litmus Test for Multi-Task Learning"), after K\=1ρ​log⁡(ρ​ε0β)K=\\frac{1}{\\rho}\\log\\left(\\frac{\\rho\\varepsilon\_{0}}{\\beta}\\right) iterations with η≤cM2\\eta\\leq\\frac{c}{M^{2}}, with probability at least 1−δ1-\\delta, we have ‖φK−θ∗‖2≤2​βρ​δ,\\|\\varphi\_{K}-\\theta\_{\*}\\|^{2}\\leq\\frac{2\\beta}{\\rho\\delta}, where ε0\=𝔼⁡\[‖φ0−θ∗‖2\]\\varepsilon\_{0}=\\mathbb{E}\[\\|\\varphi\_{0}-\\theta\_{\*}\\|^{2}\], c\=min1≤t≤T⁡{ct}c=\\min\_{1\\leq t\\leq T}\\{c\_{t}\\}, M\=max1≤t≤T⁡{Mt}M=\\max\_{1\\leq t\\leq T}\\{M\_{t}\\}, ρ\=2​η​c−η2​M2\\rho=2\\eta c-\\eta^{2}M^{2}, and β\=η2​r2​(1+η​M)2\\beta=\\eta^{2}r^{2}(1+\\eta M)^{2}.

Theorem [2](#Thmtheorem2 "Theorem 2. ‣ 4 Analysis ‣ Reasonable Effectiveness of Random Weighting:A Litmus Test for Multi-Task Learning") only requires that ℓt​(θ)\\ell\_{t}(\\theta) is ctc\_{t}-one point strongly convex w.r.t. θ∗\\theta\_{\*} after convolved with noise ξ\\xi, which can hold for deep neural networks \[[27](#bib.bib27 "")\]. It also implies that for both RLW and EW methods, their solutions have high probabilities to be close to a local minimum θ∗\\theta\_{\*} depending on the noise ξ\\xi. Note that by adding extra noise, the sharp local minimum will disappear and only the flat local minimum with a large diameter will still exist \[[16](#bib.bib16 "")\]. On the other hand, those flat local minima could satisfy the one point strongly convexity assumption made in Theorem [2](#Thmtheorem2 "Theorem 2. ‣ 4 Analysis ‣ Reasonable Effectiveness of Random Weighting:A Litmus Test for Multi-Task Learning"), thus the diameter of the converged flat local minimum is affected by the noise intensity.

###### Remark 1.

Converging to flat local minima is important in neural network training because flat local minima may lead to better generalization \[[3](#bib.bib3 ""), [14](#bib.bib14 "")\]. Due to the extra randomness from the sampling of loss weights, the RLW method can have a larger noise with a larger rr than the EW strategy (refer to Appendix [B.3](#A2.SS3 "B.3 Noise Upper Bound ‣ Appendix B Proof of Section ‣ Reasonable Effectiveness of Random Weighting:A Litmus Test for Multi-Task Learning")). Hence, according to Theorem [2](#Thmtheorem2 "Theorem 2. ‣ 4 Analysis ‣ Reasonable Effectiveness of Random Weighting:A Litmus Test for Multi-Task Learning") and the above discussion, the RLW method can better escape sharp local minima and converge to a flatter local minimum than EW, resulting in better generalization performance.

## 5 Experiments

In this section, we empirically evaluate the proposed RLW and RGW methods on five computer vision datasets (i.e., NYUv2, CityScapes, CelebA, Office-31, and Office-Home) and two multilingual problems from the XTREME benchmark \[[11](#bib.bib11 "")\]. All the experiments are conducted on one single NVIDIA GeForce RTX 3090 GPU. Due to page limit, experimental results on the CityScapes, CelebA, Office-31, and Office-Home datasets are put in Appendix [D](#A4 "Appendix D Additional Experimental Results ‣ Reasonable Effectiveness of Random Weighting:A Litmus Test for Multi-Task Learning").

Compared methods. The baseline methods in comparison include several SOTA task balancing methods as introduced in Section [2](#S2 "2 An Overview of Task Balancing Methods ‣ Reasonable Effectiveness of Random Weighting:A Litmus Test for Multi-Task Learning"), including four loss balancing methods (i.e., UW, DWA, IMTL-L, and MOML) and eight gradient balancing methods (i.e., MGDA-UB, GradNorm, PCGrad, GradDrop, IMTL-G, GradVac, CAGrad, and RotoGrad). For all the baseline methods, we directly use the optimal hyperparameters used in their original papers.

Network architecture. The network architecture we used adopts the Hard-Parameter Sharing (HPS) pattern \[[1](#bib.bib1 "")\], which shares bottom layers of the network for all the tasks and uses separate top layers for each task. Other MTL architectures are studied in Section [5.6](#S5.SS6 "5.6 Effects of Different Architectures ‣ 5 Experiments ‣ Reasonable Effectiveness of Random Weighting:A Litmus Test for Multi-Task Learning").

Evaluation metric. For homogeneous MTL problems (e.g., the XTREME benchmark and Office-31 dataset) which contain tasks of the same type such as the classification task, we directly use the average performance among tasks as the performance metric. For heterogeneous MTL problems (e.g., the NYUv2 dataset) that contain tasks of different types and may have multiple evaluation metrics for each task, by following \[[21](#bib.bib21 ""), [30](#bib.bib30 "")\], we use the average of the relative improvement over the EW method on each metric of each task as the performance measure, which is formulated as

Δp\=100%×1T∑t\=1T1Nt∑n\=1Nt(−1)pt,n​(Mt,n−Mt,nEW)Mt,nEW,\\Delta\_{\\mathrm{p}}=100\\%\\times\\frac{1}{T}\\sum\_{t=1}^{T}\\frac{1}{N\_{t}}\\sum\_{n=1}^{N\_{t}}\\frac{(-1)^{p\_{t,n}}(M\_{t,n}-M^{\\mathrm{EW}}\_{t,n})}{M^{\\mathrm{EW}}\_{t,n}}, where NtN\_{t} denotes the number of metrics in task tt, Mt,nM\_{t,n} denotes the performance of a task balancing method for the nnth metric in task tt, Mt,nEWM^{\\mathrm{EW}}\_{t,n} is defined similarly for the EW method, and pt,np\_{t,n} is set to 11 if a higher value indicates better performance for the nnth metric in task tt and otherwise 00.

### 5.1 Results on the NYUv2 Dataset

Dataset. The NYUv2 dataset \[[29](#bib.bib29 "")\] is an indoor scene understanding dataset, which consists of video sequences recorded by the RGB and Depth cameras in the Microsoft Kinect. It contains 795 and 654 images for training and testing, respectively. This dataset includes three tasks: 13-class semantic segmentation, depth estimation, and surface normal prediction.

Implementation details. For the NYUv2 dataset, the DeepLabV3+ architecture \[[4](#bib.bib4 "")\] is used. Specifically, a ResNet-50 network pre-trained on the ImageNet dataset with dilated convolutions \[[36](#bib.bib36 "")\] is used as a shared encoder among tasks and the Atrous Spatial Pyramid Pooling (ASPP) \[[4](#bib.bib4 "")\] module is used as the task-specific head for each task. Input images are resized to 288×384288\\times 384. The Adam optimizer \[[15](#bib.bib15 "")\] with the learning rate as 10−410^{-4} and the weight decay as 10−510^{-5} is used for training and the batch size is set to 8. We use the cross-entropy loss, L1L\_{1} loss, and cosine loss as the loss function of the semantic segmentation, depth estimation, and surface normal prediction tasks, respectively.

Results. The results of different methods on the NYUv2 dataset are shown in Table [1](#S5.T1 "Table 1 ‣ 5.1 Results on the NYUv2 Dataset ‣ 5 Experiments ‣ Reasonable Effectiveness of Random Weighting:A Litmus Test for Multi-Task Learning"). The top row shows the performance of the widely used EW strategy and we use it as a baseline to measure the relative improvement of different methods as shown in the definition of Δp\\Delta\_{\\mathrm{p}}. Rows 2-5 and 7-14 show the results of loss balancing and gradient balancing methods, respectively.

According to the results, we can see that both the RLW and RGW methods gain performance improvement over the EW strategy, which implies that training with extra randomness can have a better generalization ability. Besides, RLW has an improvement of 1.04%1.04\\% over the EW strategy and it is higher than all loss balancing methods. As for gradient balancing methods, half of those methods have negligible or even negative improvement over the EW strategy and RGW can outperform five of them. Compared with all baselines, RLW is even higher than all of them except the CAGrad and RotoGrad methods, which indicates the random weights can easily beat the carefully designed ones.

According to the above analysis, there are two important conclusions. Firstly, the conventional EW strategy is a weaker baseline than RLW and RGW for MTL. Secondly, RW methods are competitive to SOTA methods and even performs better than some of them.

Table 1: Performance on the NYUv2 dataset with three tasks: 13-class semantic segmentation, depth estimation, and surface normal prediction. The best results for each task on each measure over loss/gradient balancing methods are marked with superscript ∗\*/†{\\dagger}. The best results for each task on each measure over all methods are highlighted in bold. ↑\\uparrow (↓\\downarrow) indicates that the higher (lower) the result, the better the performance.

Methods

Segmentation

Depth

Surface Normal

Δp\\Delta\_{\\mathrm{p}}↑{\\uparrow}

mIoU↑{\\uparrow}

Pix Acc↑\\uparrow

Abs Err↓\\downarrow

Rel Err↓\\downarrow

Angle Distance

Within t∘t^{\\circ}

Mean↓\\downarrow

Median↓\\downarrow

11.25↑\\uparrow

22.5↑\\uparrow

30↑\\uparrow

EW

53.77

75.45

0.3845

0.1605

23.5737

17.0438

35.04

60.93

72.07

+0.00%

 Loss Bal. 

UW

54.14

75.92

0.3833

0.1597

23.2989

16.8691

35.33

61.37

72.48

+0.64%

DWA

53.81

75.56

0.3792∗

0.1565∗

23.6111

17.0609

34.89

60.89

71.97

+0.63%

IMTL-L

53.50

75.18

0.3824

0.1596

23.3805

16.8088

35.44

61.43

72.43

+0.35%

MOML

54.98∗

75.98∗

0.3877

0.1618

23.2401∗

16.7388

35.90∗

61.81∗

72.76∗

+0.76%

RLW (ours)

54.11

75.77

0.3809

0.1575

23.3777

16.7385∗

35.71

61.52

72.45

+1.04%∗

 Gradient Bal. 

MGDA-UB

50.42

73.46

0.3834

0.1555†

22.7827†

16.1432†

36.90†

62.88

73.61

+0.38%

GradNorm

53.58

75.06

0.3931

0.1663

23.4360

16.9844

35.11

61.11

72.24

-0.99%

PCGrad

53.70

75.41

0.3903

0.1607

23.4281

16.9699

35.16

61.19

72.28

-0.16%

GradDrop

53.58

75.56

0.3855

0.1592

23.5518

17.0137

35.08

60.97

72.02

+0.08%

IMTL-G

53.54

75.45

0.3880

0.1589

23.0530

16.4328

36.21

62.31

73.15

+0.80%

GradVac

54.89†

75.98†

0.3828

0.1635

23.6865

17.1301

34.82

60.71

71.81

+0.07%

CAGrad

53.12

75.19

0.3871

0.1599

22.5257

15.8821

37.42

63.50†

74.17†

+1.36%†

RotoGrad

53.90

75.46

0.3812

0.1596

23.0197

16.3714

36.37

62.28

73.05

+1.19%

RGW (ours)

53.85

75.87

0.3772†

0.1562

23.6725

17.2439

34.62

60.49

71.75

+0.62%

### 5.2 Results on the XTREME benchmark

Dataset. The XTREME benchmark \[[11](#bib.bib11 "")\] is a large-scale multilingual multi-task benchmark for cross-lingual generalization evaluation, which covers fifty languages and contains nine tasks. We conduct experiments on two tasks containing Paraphrase Identification (PI) and Part-Of-Speech (POS) tagging in this benchmark. The datasets used in the PI and POS tasks are the PAWS-X dataset \[[34](#bib.bib34 "")\] and Universal Dependency v2.5 treebanks \[[25](#bib.bib25 "")\], respectively. On each task, we construct a multilingual problem by choosing the four languages with largest numbers of data, i.e., English (en), Mandarin (zh), German (de) and Spanish (es), for the PI task and English, Mandarin, Telugu (te) and Vietnamese (vi) for the POS task. The statistics for each language are summarized in Table [5](#A3.T5 "Table 5 ‣ Appendix C Additional Details about the XTREME Benchmark ‣ Reasonable Effectiveness of Random Weighting:A Litmus Test for Multi-Task Learning") in the Appendix. Different from the NYUv2 dataset where different tasks share the same input data, in those multilingual problems, each language/task has its own input data.

Implementation details. For each multilingual problem in the XTREME benchmark, a pre-trained multilingual BERT (mBERT) model \[[8](#bib.bib8 "")\] implemented via the open-source transformers library \[[33](#bib.bib33 "")\] is used as the shared encoder among languages and a fully connected layer is used as the language-specific output layer for each language. The Adam optimizer with the learning rate as 2×10−52\\times 10^{-5} and the weight decay as 10−810^{-8} is used for training and the batch size is set to 32. The cross-entropy loss is used for the two multilingual problems.

Results. According to experimental results shown in Table [2](#S5.T2 "Table 2 ‣ 5.2 Results on the XTREME benchmark ‣ 5 Experiments ‣ Reasonable Effectiveness of Random Weighting:A Litmus Test for Multi-Task Learning"), we can find some empirical observations, which are similar to those on the NYUv2 dataset. Firstly, both the RLW and RGW strategies outperform the EW method. Secondly, compared with the existing works, RLW and RGW can achieve comparable performance with existing loss/gradient balancing methods, respectively. Even, RLW or RGW methods could outperform all baseline methods. For example, RLW achieves the best performance (i.e., 90.25% average accuracy) on the PI problem and RGW achieves the best average F1 score of 91.16% on the POS problem. It is interesting to find that the performance of RLW and RGW are inconsistent in different datasets. There is because the random loss weights in RLW will affect the update of task-specific parameters while not in RGW, which has a different influence on the performance of different datasets.

Table 2: Performance on two multilingual problems, i.e., PI and POS from the XTREME benchmark. The best results for each language over loss/gradient balancing methods are marked with superscript ∗\*/†{\\dagger}. The best results for each language over all methods are highlighted in bold.

| Methods |        |        |        |        |        |        |        |        |        |
| ------- | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ |
| en      | zh     | de     | es     | Avg    | en     | zh     | te     | vi     | Avg    |
| 94.29   | 84.99  | 89.79  | 90.94  | 90.00  | 95.06  | 89.01  | 91.41  | 86.65  | 90.53  |
| 93.74   | 85.44∗ | 90.24∗ | 91.29  | 90.18  | 94.89  | 88.77  | 90.96  | 87.12  | 90.44  |
| 94.69∗  | 84.99  | 89.49  | 91.44∗ | 90.15  | 95.02  | 89.03  | 91.87  | 87.27∗ | 90.80  |
| 93.94   | 84.54  | 89.39  | 91.44∗ | 89.82  | 95.57∗ | 89.93∗ | 91.77  | 86.11  | 90.84  |
| 93.89   | 83.74  | 89.94  | 90.99  | 89.64  | 95.15  | 89.11  | 92.41  | 87.24  | 90.98∗ |
| 94.29   | 85.39  | 89.94  | 91.39  | 90.25∗ | 95.01  | 88.87  | 92.86∗ | 86.85  | 90.90  |
| 94.09   | 84.14  | 89.14  | 90.59  | 89.49  | 94.89  | 88.43  | 91.01  | 86.04  | 90.01  |
| 94.19   | 83.59  | 88.89  | 91.24  | 89.47  | 94.88  | 88.80  | 91.78  | 86.96  | 90.61  |
| 94.19   | 85.49† | 89.09  | 91.24  | 90.00  | 94.85  | 88.42  | 90.72  | 86.71  | 90.18  |
| 94.29   | 84.44  | 89.69  | 90.94  | 89.84  | 95.08  | 89.06  | 90.65  | 87.17  | 90.49  |
| 94.69†  | 84.54  | 89.39  | 90.69  | 89.82  | 94.93  | 88.70  | 91.66  | 87.00  | 90.57  |
| 94.29   | 84.94  | 89.19  | 90.89  | 89.83  | 94.87  | 88.41  | 90.62  | 86.47  | 90.09  |
| 94.34   | 84.59  | 90.09† | 90.64  | 89.91  | 94.83  | 88.65  | 91.71  | 86.76  | 90.48  |
| 93.99   | 83.89  | 89.29  | 90.94  | 89.52  | 95.44  | 89.79  | 91.42  | 86.33  | 90.74  |
| 94.55   | 84.99  | 89.29  | 91.40† | 90.06† | 95.52† | 90.13† | 91.82† | 87.18† | 91.16† |

### 5.3 Robustness on Distribution

In this section, we evaluate the robustness of the proposed RW methods on the sampling distribution. Taking RLW as an example, we show its robustness by evaluating with five different sampling distributions (i.e., p⁡(𝝀\~)p(\\tilde{\\bm{\\lambda}})) for loss weights. The five distributions are uniform distribution between 00 and 11 (denoted by Uniform), standard normal distribution (denoted by Normal), Dirichlet distribution with α\=1\\alpha=1 (denoted by Dirichlet), Bernoulli distribution with probability 1/21/2 (denoted by Bernoulli), Bernoulli distribution with probability 1/21/2 and a constraint ∑t\=1Tλ\~t\=1\\sum\_{t=1}^{T}\\tilde{\\lambda}\_{t}=1 (denoted by c-Bernoulli). We set ff as a function of f⁡(𝝀\~)\=𝝀\~/(∑t\=1Tλ\~t)f(\\tilde{\\bm{\\lambda}})=\\tilde{\\bm{\\lambda}}/(\\sum\_{t=1}^{T}\\tilde{\\lambda}\_{t}) for the Bernoulli distribution and the c-Bernoulli distribution, a softmax function for the Normal distribution and Uniform distribution, and an identity function for the Dirichlet distribution. We can prove that all the 𝔼⁡\[𝝀\]\\mathbb{E}\[\\bm{\\lambda}\]’s under these five distributions equal (1T,⋯,1T)(\\frac{1}{T},\\cdots,\\frac{1}{T}) (refer to Appendix [A](#A1 "Appendix A Proof of the Mean Value 𝔼(𝜆) ‣ Reasonable Effectiveness of Random Weighting:A Litmus Test for Multi-Task Learning")), thus it is fair to compare among them.

Figure [1](#S5.F1 "Figure 1 ‣ 5.3 Robustness on Distribution ‣ 5 Experiments ‣ Reasonable Effectiveness of Random Weighting:A Litmus Test for Multi-Task Learning") shows the results of the RLW method with five sampling distributions on the NYUv2 dataset in terms of Δp\\Delta\_{\\mathrm{p}}, where the experiment on each sampling distribution is repeated for 8 times. The results show that the RLW method with different distributions can always outperform the EW method, which shows the robustness of the RLW method with respect to the sampling distribution. In addition, compared with the uniform, Dirichlet, and Bernoulli distributions, RLW with the standard normal distribution achieves better and more stable performance. Although RLW with the c-Bernoulli distribution performs slightly better than the standard normal distribution, it is more unstable and may need a longer training time as shown in Section [5.4](#S5.SS4 "5.4 Convergence Speed ‣ 5 Experiments ‣ Reasonable Effectiveness of Random Weighting:A Litmus Test for Multi-Task Learning"). Thus, in this paper, we use the standard normal distribution to sample loss weights.

Figure 1: Results of the RLW method with different sampling distributions in terms of Δp\\Delta\_{\\mathrm{p}}.

### 5.4 Convergence Speed

Here we take RLW as an example to show the efficiency of RW methods. Figure [2](#S5.F2 "Figure 2 ‣ 5.4 Convergence Speed ‣ 5 Experiments ‣ Reasonable Effectiveness of Random Weighting:A Litmus Test for Multi-Task Learning") plots the performance curve on both NYUv2 and CelebA validation datasets to empirically compare the convergence speed of the EW and RLW methods.

On the NYUv2 dataset with three tasks, the performance curves of the RLW method with two sampling distributions are similar to that of the EW method, which indicates that the RLW method has a similar convergence property to the EW method on this dataset. As the number of tasks increases, i.e., on the CelebA dataset with 40 tasks, we find that the RLW method with the standard normal distribution still converges as fast as the EW method, while the RLW method with the c-Bernoulli distribution converges slower. One reason for this phenomenon is that only one task is used to update model parameters in each training iteration when using the c-Bernoulli distribution. Thus, in this paper, we use the standard normal distribution, which is as efficient as the EW method.

Figure 2: Comparison on the convergence speed of the EW and RLW methods on the NYUv2 validation dataset (Left) and the CelebA validation dataset (Right).

### 5.5 Combination of Loss and Gradient Balancing Methods

The loss balancing methods are complementary with the gradient balancing methods. Following \[[18](#bib.bib18 "")\], we train an MTL model with different combinations of loss balancing and gradient balancing methods on the NYUv2 dataset to further improve the performance. We use the vanilla EW as the baseline to measure the relative improvement of the other different combinations as shown in the definition of Δp\\Delta\_{\\mathrm{p}}.

According to the results shown in Table [3](#S5.T3 "Table 3 ‣ 5.5 Combination of Loss and Gradient Balancing Methods ‣ 5 Experiments ‣ Reasonable Effectiveness of Random Weighting:A Litmus Test for Multi-Task Learning"), we can see that combined with the UW, DWA and IMTL-L methods, some gradient balancing methods performs better but others become worse. For example, Δp\\Delta\_{\\mathrm{p}} of the GradDrop method drops from 0.08%0.08\\% to −0.42%-0.42\\% when combined with DWA. Noticeably, by combining with the proposed RLW method, all the gradient balancing methods can achieve better performance. Besides, on each gradient balancing method, the improvement induced by the RLW method is significantly larger than the other three loss balancing methods as well as the EW method. Moreover, RGW can also improve the performance of loss balancing methods except DWA. Thus, this experiment further demonstrates the effectiveness of the proposed RW methods.

Table 3: Results of different combinations of loss balancing and gradient balancing methods on the NYUv2 dataset in terms of Δp\\Delta\_{\\mathrm{p}}. The best results in each row are highlighted in bold.

| EW       | UW     | DWA    | IMTL-L | RLW    |        |
| -------- | ------ | ------ | ------ | ------ | ------ |
| Vanilla  | +0.00% | +0.64% | +0.63% | +0.35% | +1.04% |
| MGDA-UB  | +0.38% | +0.15% | +0.47% | -0.59% | +2.01% |
| GradNorm | -0.99% | +0.87% | -0.95% | +0.54% | +0.89% |
| PCGrad   | -0.16% | +0.72% | +0.19% | +0.38% | +0.97% |
| GradDrop | +0.08% | +0.25% | -0.42% | +0.36% | +0.93% |
| IMTL-G   | +0.80% | +0.45% | +1.20% | +0.18% | +1.50% |
| GradVac  | +0.07% | -0.03% | +0.89% | +0.69% | +0.97% |
| CAGrad   | +1.36% | +1.07% | +1.41% | +2.18% | +2.20% |
| RotoGrad | +1.19% | +1.03% | +0.75% | +1.40% | +1.45% |
| RGW      | +0.62% | +0.82% | +0.41% | +0.78% | +1.46% |

### 5.6 Effects of Different Architectures

The proposed RW methods can be seamlessly incorporated into all the MTL architectures. To see this, we take RLW as an example and combine it with three different MTL architectures, i.e., cross-stitch network \[[22](#bib.bib22 "")\], Multi-Task Attention Network (MTAN) \[[19](#bib.bib19 "")\], and NDDR-CNN \[[9](#bib.bib9 "")\]. We use the combination of EW and HPS as the baseline to measure the relative improvement of the other different combinations as shown in the definition of Δp\\Delta\_{\\mathrm{p}}.

According to the results on the NYUv2 dataset as shown in Table [4](#S5.T4 "Table 4 ‣ 5.6 Effects of Different Architectures ‣ 5 Experiments ‣ Reasonable Effectiveness of Random Weighting:A Litmus Test for Multi-Task Learning"), we can see that the proposed RLW strategy outperforms the EW method under all the three architectures. When using the MTAN and NNDR-CNN architectures, RLW achieves better performance than the CAGrad method that performs best in the HPS architecture, which shows the potential of the proposed RLW method when choosing suitable MTL architectures. Moreover, combined with the RLW method, CAGrad can be further improved under the four architectures. For example, the combinations of RLW and CAGrad can achieve the best Δp\\Delta\_{\\mathrm{p}} of 3.53%3.53\\% under the NDDR-CNN architecture.

Table 4: Results of different combinations of task balancing methods and MTL architectures on the NYUv2 dataset in terms of Δp\\Delta\_{\\mathrm{p}}. The best results for each architecture are highlighted in bold.

|        | HPS    | Cross-stitch | MTAN   | NDDR-CNN |
| ------ | ------ | ------------ | ------ | -------- |
| +0.00% | +1.43% | +2.56%       | +1.90% |          |
| +1.36% | +2.42% | +2.26%       | +2.83% |          |
| +1.04% | +2.23% | +2.66%       | +2.91% |          |
| +2.20% | +2.76% | +2.92%       | +3.53% |          |

## 6 Conclusions

In this paper, we propose the RW methods, an important yet ignored baselines for MTL, by training an MTL model with random loss/gradient weights. We analyze the convergence and effectiveness properties of the proposed RW method. Moreover, we provide a consistent and comparative comparison to show the RW methods can achieve comparable performance with state-of-the-art methods that use carefully designed weights, which indicates the random experiments could be used to examine the effectiveness of newly proposed MTL methods and RW methods should attract wide attention as the litmus tests. In our future work, we will apply the RW methods to more MTL applications.

## References

*   \[1\] Rich Caruana. Multitask learning: A knowledge-based source of inductive bias. In Proceedings of the 10th International Conference on Machine Learning, pages 41–48, 1993.
*   \[2\] Rich Caruana. Multitask learning. Machine learning, 28(1):41–75, 1997.
*   \[3\] Pratik Chaudhari, Anna Choromanska, Stefano Soatto, Yann LeCun, Carlo Baldassi, Christian Borgs, Jennifer Chayes, Levent Sagun, and Riccardo Zecchina. Entropy-sgd: Biasing gradient descent into wide valleys. Journal of Statistical Mechanics: Theory and Experiment, 2019(12):124018, 2019.
*   \[4\] Liang-Chieh Chen, Yukun Zhu, George Papandreou, Florian Schroff, and Hartwig Adam. Encoder-decoder with atrous separable convolution for semantic image segmentation. In Proceedings of the 14th European Conference on Computer Vision, volume 11211, pages 833–851, 2018.
*   \[5\] Zhao Chen, Vijay Badrinarayanan, Chen-Yu Lee, and Andrew Rabinovich. Gradnorm: Gradient normalization for adaptive loss balancing in deep multitask networks. In Proceedings of the International Conference on Machine Learning, pages 794–803. PMLR, 2018.
*   \[6\] Zhao Chen, Jiquan Ngiam, Yanping Huang, Thang Luong, Henrik Kretzschmar, Yuning Chai, and Dragomir Anguelov. Just pick a sign: Optimizing deep multitask models with gradient sign dropout. In Proceedings of the 33rd Advances in Neural Information Processing Systems, 2020.
*   \[7\] Marius Cordts, Mohamed Omran, Sebastian Ramos, Timo Rehfeld, Markus Enzweiler, Rodrigo Benenson, Uwe Franke, Stefan Roth, and Bernt Schiele. The cityscapes dataset for semantic urban scene understanding. In Proceedings of IEEE Conference on Computer Vision and Pattern Recognition, pages 3213–3223, 2016.
*   \[8\] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. BERT: pre-training of deep bidirectional transformers for language understanding. In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pages 4171–4186, 2019.
*   \[9\] Yuan Gao, Jiayi Ma, Mingbo Zhao, Wei Liu, and Alan L Yuille. Nddr-cnn: Layerwise feature fusing in multi-task cnns by neural discriminative dimensionality reduction. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 3205–3214, 2019.
*   \[10\] Moritz Hardt, Ben Recht, and Yoram Singer. Train faster, generalize better: Stability of stochastic gradient descent. In Proceedings of the International Conference on Machine Learning, pages 1225–1234. PMLR, 2016.
*   \[11\] Junjie Hu, Sebastian Ruder, Aditya Siddhant, Graham Neubig, Orhan Firat, and Melvin Johnson. XTREME: A massively multilingual multi-task benchmark for evaluating cross-lingual generalisation. In Proceedings of the 37th International Conference on Machine Learning, volume 119, pages 4411–4421. PMLR, 2020.
*   \[12\] Adrián Javaloy and Isabel Valera. Rotograd: Gradient homogenization in multitask learning. In Proceedings of the 10th International Conference on Learning Representations, 2022.
*   \[13\] Alex Kendall, Yarin Gal, and Roberto Cipolla. Multi-task learning using uncertainty to weigh losses for scene geometry and semantics. In Proceedings of IEEE Conference on Computer Vision and Pattern Recognition, pages 7482–7491, 2018.
*   \[14\] Nitish Shirish Keskar, Jorge Nocedal, Ping Tak Peter Tang, Dheevatsa Mudigere, and Mikhail Smelyanskiy. On large-batch training for deep learning: Generalization gap and sharp minima. In Proceedings of the 5th International Conference on Learning Representations, 2017.
*   \[15\] Diederik P. Kingma and Jimmy Ba. Adam: A method for stochastic optimization. In Proceedings of the 3rd International Conference on Learning Representations, 2015.
*   \[16\] Bobby Kleinberg, Yuanzhi Li, and Yang Yuan. An alternative view: When does sgd escape local minima? In Proceedings of the International Conference on Machine Learning, pages 2698–2707. PMLR, 2018.
*   \[17\] Bo Liu, Xingchao Liu, Xiaojie Jin, Peter Stone, and Qiang Liu. Conflict-averse gradient descent for multi-task learning. In Proceedings of the 35th Advances in Neural Information Processing Systems, 2021.
*   \[18\] Liyang Liu, Yi Li, Zhanghui Kuang, Jing-Hao Xue, Yimin Chen, Wenming Yang, Qingmin Liao, and Wayne Zhang. Towards impartial multi-task learning. In Proceedings of the 9th International Conference on Learning Representations, 2021.
*   \[19\] Shikun Liu, Edward Johns, and Andrew J. Davison. End-to-end multi-task learning with attention. In Proceedings of IEEE Conference on Computer Vision and Pattern Recognition, pages 1871–1880, 2019.
*   \[20\] Ziwei Liu, Ping Luo, Xiaogang Wang, and Xiaoou Tang. Deep learning face attributes in the wild. In Proceedings of International Conference on Computer Vision, 2015.
*   \[21\] Kevis-Kokitsi Maninis, Ilija Radosavovic, and Iasonas Kokkinos. Attentive single-tasking of multiple tasks. In Proceedings of IEEE Conference on Computer Vision and Pattern Recognition, pages 1851–1860, 2019.
*   \[22\] Ishan Misra, Abhinav Shrivastava, Abhinav Gupta, and Martial Hebert. Cross-stitch networks for multi-task learning. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 3994–4003, 2016.
*   \[23\] Eric Moulines and Francis Bach. Non-asymptotic analysis of stochastic approximation algorithms for machine learning. In Proceedings of the Advances in Neural Information Processing Systems, volume 24, pages 451–459, 2011.
*   \[24\] Deanna Needell, Nathan Srebro, and Rachel Ward. Stochastic gradient descent, weighted sampling, and the randomized kaczmarz algorithm. Mathematical Programming, 155(1-2):549–573, 2016.
*   \[25\] Joakim Nivre, Marie-Catherine de Marneffe, Filip Ginter, Jan Hajic, Christopher D. Manning, Sampo Pyysalo, Sebastian Schuster, Francis M. Tyers, and Daniel Zeman. Universal dependencies v2: An evergrowing multilingual treebank collection. In Proceedings of the 12th Language Resources and Evaluation Conference, pages 4034–4043, 2020.
*   \[26\] Kate Saenko, Brian Kulis, Mario Fritz, and Trevor Darrell. Adapting visual category models to new domains. In Proceedings of the 6th European Conference on Computer Vision, pages 213–226. Springer, 2010.
*   \[27\] Itay M Safran, Gilad Yehudai, and Ohad Shamir. The effects of mild over-parameterization on the optimization landscape of shallow relu neural networks. In Conference on Learning Theory, pages 3889–3934. PMLR, 2021.
*   \[28\] Ozan Sener and Vladlen Koltun. Multi-task learning as multi-objective optimization. In Proceedings of the 31st Advances in Neural Information Processing Systems, pages 525–536, 2018.
*   \[29\] Nathan Silberman, Derek Hoiem, Pushmeet Kohli, and Rob Fergus. Indoor segmentation and support inference from rgbd images. In Proceedings of the 8th European Conference on Computer Vision, pages 746–760, 2012.
*   \[30\] Simon Vandenhende, Stamatios Georgoulis, Wouter Van Gansbeke, Marc Proesmans, Dengxin Dai, and Luc Van Gool. Multi-task learning for dense prediction tasks: A survey. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2021.
*   \[31\] Hemanth Venkateswara, Jose Eusebio, Shayok Chakraborty, and Sethuraman Panchanathan. Deep hashing network for unsupervised domain adaptation. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 5018–5027, 2017.
*   \[32\] Zirui Wang, Yulia Tsvetkov, Orhan Firat, and Yuan Cao. Gradient vaccine: Investigating and improving multi-task optimization in massively multilingual models. In Proceedings of the 9th International Conference on Learning Representations, 2021.
*   \[33\] Thomas Wolf, Lysandre Debut, Victor Sanh, Julien Chaumond, Clement Delangue, Anthony Moi, Pierric Cistac, Tim Rault, Rémi Louf, Morgan Funtowicz, Joe Davison, Sam Shleifer, Patrick von Platen, Clara Ma, Yacine Jernite, Julien Plu, Canwen Xu, Teven Le Scao, Sylvain Gugger, Mariama Drame, Quentin Lhoest, and Alexander M. Rush. Transformers: State-of-the-art natural language processing. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing, pages 38–45, 2020.
*   \[34\] Yinfei Yang, Yuan Zhang, Chris Tar, and Jason Baldridge. PAWS-X: A cross-lingual adversarial dataset for paraphrase identification. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing, pages 3685–3690, 2019.
*   \[35\] Feiyang Ye, Baijiong Lin, Zhixiong Yue, Pengxin Guo, Qiao Xiao, and Yu Zhang. Multi-objective meta learning. In Proceedings of the 35th Advances in Neural Information Processing Systems, 2021.
*   \[36\] Fisher Yu, Vladlen Koltun, and Thomas A. Funkhouser. Dilated residual networks. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 636–644, 2017.
*   \[37\] Tianhe Yu, Saurabh Kumar, Abhishek Gupta, Sergey Levine, Karol Hausman, and Chelsea Finn. Gradient surgery for multi-task learning. In Proceedings of the 33rd Advances in Neural Information Processing Systems, 2020.
*   \[38\] Yu Zhang and Qiang Yang. A survey on multi-task learning. IEEE Transactions on Knowledge and Data Engineering, 2021.

## Appendix

## Appendix A Proof of the Mean Value 𝔼⁡(λ)\\mathbb{E}({\\mathbf{\\lambda}})

Suppose that λ\~t​(t\=1,⋯,T)\\tilde{{\\lambda}}\_{t}(t=1,\\cdots,T) are independent and identically distributed (i.i.d.) random variables sampled from the Uniform or standard Normal distributions and ff is the softmax function. Then we have λt\=exp⁡(λ\~t)∑m\=1Texp⁡(λ\~m){\\lambda}\_{t}=\\frac{\\exp(\\tilde{{\\lambda}}\_{t})}{\\sum\_{m=1}^{T}\\exp(\\tilde{{\\lambda}}\_{m})} and

𝔼⁡(λi)\=𝔼⁡\[exp⁡(λ\~i)\]​𝔼​\[1∑m\=1Texp⁡(λ\~m)\]+Cov⁡(exp⁡(λ\~i),1∑m\=1Texp⁡(λ\~m)),\\mathbb{E}({\\lambda\_{i}})=\\mathbb{E}\[\\exp(\\tilde{{\\lambda}}\_{i})\]\\mathbb{E}\\bigg\[\\frac{1}{\\sum\_{m=1}^{T}\\exp(\\tilde{{\\lambda}}\_{m})}\\bigg\]+\\mathrm{Cov}\\left(\\exp(\\tilde{{\\lambda}}\_{i}),\\frac{1}{\\sum\_{m=1}^{T}\\exp(\\tilde{{\\lambda}}\_{m})}\\right), where Cov⁡(⋅,⋅)\\mathrm{Cov}(\\cdot,\\cdot) denotes the covariance between two random variables. Since {λ\~t}t\=1T\\{\\tilde{{\\lambda}}\_{t}\\}\_{t=1}^{T} are i.i.d random variables, we have 𝔼⁡\[exp⁡(λ\~i)\]\=𝔼⁡\[exp⁡(λ\~j)\]\\mathbb{E}\[\\exp(\\tilde{{\\lambda}}\_{i})\]=\\mathbb{E}\[\\exp(\\tilde{{\\lambda}}\_{j})\] and Cov⁡(exp⁡(λ\~i),1/∑m\=1Texp⁡(λ\~m))\=Cov⁡(exp⁡(λ\~j),1/∑m\=1Texp⁡(λ\~m))\\mathrm{Cov}(\\exp(\\tilde{{\\lambda}}\_{i}),1/\\sum\_{m=1}^{T}\\exp(\\tilde{{\\lambda}}\_{m}))=\\mathrm{Cov}(\\exp(\\tilde{{\\lambda}}\_{j}),1/\\sum\_{m=1}^{T}\\exp(\\tilde{{\\lambda}}\_{m})). Therefore, we obtain

𝔼⁡(λi)\=𝔼⁡(λj),∀1≤i,j≤T.\\mathbb{E}({\\lambda}\_{i})=\\mathbb{E}({\\lambda}\_{j}),\\forall 1\\leq i,j\\leq T.

Moreover, we have

∑t\=1T𝔼⁡(λt)\=∑t\=1T∑k\=1KλtkK\=∑k\=1K∑t\=1TλtkK\=1.\\sum\_{t=1}^{T}\\mathbb{E}({\\lambda}\_{t})=\\sum\_{t=1}^{T}\\frac{\\sum\_{k=1}^{K}{\\lambda}\_{t}^{k}}{K}=\\frac{\\sum\_{k=1}^{K}\\sum\_{t=1}^{T}{\\lambda}\_{t}^{k}}{K}=1.

Thus we have 𝔼⁡(𝝀)\=(1T,⋯,1T).\\mathbb{E}(\\bm{\\lambda})=(\\frac{1}{T},\\cdots,\\frac{1}{T}). Similarly, we can prove the same result for the Bernoulli and c-Bernoulli distributions with the normalization function ff as f⁡(𝝀\~)\=𝝀\~/(∑t\=1Tλ\~t)f(\\tilde{\\bm{\\lambda}})=\\tilde{\\bm{\\lambda}}/(\\sum\_{t=1}^{T}\\tilde{\\lambda}\_{t}).

## Appendix B Proof of Section [4](#S4 "4 Analysis ‣ Reasonable Effectiveness of Random Weighting:A Litmus Test for Multi-Task Learning")

### B.1 Proof of Theorem [1](#Thmtheorem1 "Theorem 1. ‣ 4 Analysis ‣ Reasonable Effectiveness of Random Weighting:A Litmus Test for Multi-Task Learning")

Suppose ℒRLW​(θ)\=𝝀⊤​ℓ​(θ)\\mathcal{L}\_{\\mathrm{RLW}}(\\theta)=\\bm{\\lambda}^{\\top}\\bm{\\ell}(\\theta), where 𝝀\\bm{\\lambda} is a random variable sampled from a random distribution in every training iteration.

Since ℓt\\ell\_{t} is ctc\_{t}-strongly convex w.r.t. θ\\theta, for any two points θ1\\theta\_{1} and θ2\\theta\_{2} in ℝd\\mathbb{R}^{d}, we have

⟨∇𝝀⊤ℓ(θ1)−∇𝝀⊤ℓ(θ2),θ1−θ2⟩\\displaystyle\\left<\\nabla\\bm{\\lambda}^{\\top}\\bm{\\ell}(\\theta\_{1})-\\nabla\\bm{\\lambda}^{\\top}\\bm{\\ell}(\\theta\_{2}),\\theta\_{1}-\\theta\_{2}\\right>

\=∑t\=1Tλt​⟨∇ℓt​(θ1)−∇ℓt​(θ2),θ1−θ2⟩\\displaystyle=\\sum\_{t=1}^{T}\\lambda\_{t}\\left<\\nabla\\ell\_{t}(\\theta\_{1})-\\nabla\\ell\_{t}(\\theta\_{2}),\\theta\_{1}-\\theta\_{2}\\right>

≥∑t\=1Tct​λt​‖θ1−θ2‖2.\\displaystyle\\geq\\sum\_{t=1}^{T}c\_{t}\\lambda\_{t}\\|\\theta\_{1}-\\theta\_{2}\\|^{2}.

(4)

Since 0≤λt≤10\\leq\\lambda\_{t}\\leq 1, we have ∑t\=1Tct​λt≥c\\sum\_{t=1}^{T}c\_{t}\\lambda\_{t}\\geq c, where c\=min1≤t≤T⁡{ct}c=\\min\_{1\\leq t\\leq T}\\{c\_{t}\\}. Then for any 𝝀\\bm{\\lambda}, ℒRLW​(θ)\\mathcal{L}\_{\\mathrm{RLW}}(\\theta) is cc-strongly convex.

With notations in Theorem [1](#Thmtheorem1 "Theorem 1. ‣ 4 Analysis ‣ Reasonable Effectiveness of Random Weighting:A Litmus Test for Multi-Task Learning"), we have

‖θk+1−θ∗‖2\\displaystyle\\|\\theta\_{k+1}-\\theta\_{\*}\\|^{2}

\=∥θk−θ∗−η∇𝝀⊤ℓ(𝓓\~;θk)∥2\\displaystyle=\\|\\theta\_{k}-\\theta\_{\*}-\\eta\\nabla\\bm{{\\lambda}}^{\\top}\\bm{\\ell}(\\tilde{\\bm{\\mathcal{D}}};\\theta\_{k})\\|^{2}

\=∥θk−θ∗∥2−2η⟨θk−θ∗,∇𝝀⊤ℓ(𝓓\~;θk)⟩+η2∥∇𝝀⊤ℓ(𝓓\~;θk)∥2.\\displaystyle=\\|\\theta\_{k}-\\theta\_{\*}\\|^{2}-2\\eta\\left<\\theta\_{k}-\\theta\_{\*},\\nabla\\bm{{\\lambda}}^{\\top}\\bm{\\ell}(\\tilde{\\bm{\\mathcal{D}}};\\theta\_{k})\\right>+\\eta^{2}\\|\\nabla\\bm{{\\lambda}}^{\\top}\\bm{\\ell}(\\tilde{\\bm{\\mathcal{D}}};\\theta\_{k})\\|^{2}.

Note that 𝔼𝝀\[𝔼𝓓\~\[∇𝝀⊤ℓ(𝓓\~;θk)\]\]\=∇𝝁⊤ℓ(𝓓;θk)\\mathbb{E}\_{\\bm{{\\lambda}}}\\left\[\\mathbb{E}\_{\\tilde{\\bm{\\mathcal{D}}}}\[\\nabla\\bm{{\\lambda}}^{\\top}\\bm{\\ell}(\\tilde{\\bm{\\mathcal{D}}};\\theta\_{k})\]\\right\]=\\nabla\\bm{\\mu}^{\\top}\\bm{\\ell}(\\bm{\\mathcal{D}};\\theta\_{k}) and

𝔼𝝀\[𝔼𝓓\~\[∥∇𝝀⊤ℓ(𝓓\~;θk)∥2\]\]\\displaystyle\\mathbb{E}\_{\\bm{{\\lambda}}}\\left\[\\mathbb{E}\_{\\tilde{\\bm{\\mathcal{D}}}}\[\\|\\nabla\\bm{{\\lambda}}^{\\top}\\bm{\\ell}(\\tilde{\\bm{\\mathcal{D}}};\\theta\_{k})\\|^{2}\]\\right\]

≤𝔼𝝀​\[𝔼𝓓\~​\[‖𝝀⊤‖2​‖∇ℓ​(𝓓\~,θk)‖2\]\]\\displaystyle\\leq\\mathbb{E}\_{\\bm{{\\lambda}}}\\left\[\\mathbb{E}\_{\\tilde{\\bm{\\mathcal{D}}}}\[\\|\\bm{{\\lambda}}^{\\top}\\|^{2}\\|\\nabla\\bm{\\ell}(\\tilde{\\bm{\\mathcal{D}}};\\theta\_{k})\\|^{2}\]\\right\]

≤𝔼𝝀​\[∑t\=1Tλt2\]⋅∑t\=1Tσt2\\displaystyle\\leq\\mathbb{E}\_{\\bm{{\\lambda}}}\\bigg\[\\sum\_{t=1}^{T}\\lambda\_{t}^{2}\\bigg\]\\cdot\\sum\_{t=1}^{T}\\sigma\_{t}^{2}

≤∑t\=1Tσt2,\\displaystyle\\leq\\sum\_{t=1}^{T}\\sigma\_{t}^{2}, where the first inequality is due to the Cauchy-Schwarz inequality and the third inequality is due to 0≤λt≤10\\leq\\lambda\_{t}\\leq 1 and ∑tλt\=1\\sum\_{t}\\lambda\_{t}=1. Then, by defining κ\=∑t\=1Tσt2\\kappa=\\sum\_{t=1}^{T}\\sigma\_{t}^{2}, we obtain

𝔼𝝀​\[𝔼𝓓\~​\[‖θk+1−θ∗‖2\]\]\\displaystyle\\mathbb{E}\_{\\bm{{\\lambda}}}\\left\[\\mathbb{E}\_{\\tilde{\\bm{\\mathcal{D}}}}\[\\|\\theta\_{k+1}-\\theta\_{\*}\\|^{2}\]\\right\]

≤∥θk−θ∗∥2−2η⟨θk−θ∗,∇𝝁⊤ℓ(θk)⟩+η2κ\\displaystyle\\leq\\|\\theta\_{k}-\\theta\_{\*}\\|^{2}-2\\eta\\left<\\theta\_{k}-\\theta\_{\*},\\nabla\\bm{\\mu}^{\\top}\\bm{\\ell}(\\theta\_{k})\\right>+\\eta^{2}\\kappa

≤(1−2​η​c)​‖θk−θ∗‖2+η2​κ.\\displaystyle\\leq(1-2\\eta c)\\|\\theta\_{k}-\\theta\_{\*}\\|^{2}+\\eta^{2}\\kappa.

(5)

If 1−2​η​c\>01-2\\eta c>0, we recursively apply the inequality ([5](#A2.Ex11 "In B.1 Proof of Theorem ‣ Appendix B Proof of Section ‣ Reasonable Effectiveness of Random Weighting:A Litmus Test for Multi-Task Learning")) over the first kk iterations and we can obtain

𝔼⁡\[‖θk+1−θ∗‖2\]\\displaystyle\\mathbb{E}\[\\|\\theta\_{k+1}-\\theta\_{\*}\\|^{2}\]

≤(1−2​η​c)k​‖θ0−θ∗‖2+∑j\=0k−1(1−2​η​c)j​η2​κ\\displaystyle\\leq(1-2\\eta c)^{k}\\|\\theta\_{0}-\\theta\_{\*}\\|^{2}+\\sum\_{j=0}^{k-1}(1-2\\eta c)^{j}\\eta^{2}\\kappa

≤(1−2​η​c)k​‖θ0−θ∗‖2+η​κ2​c.\\displaystyle\\leq(1-2\\eta c)^{k}\\|\\theta\_{0}-\\theta\_{\*}\\|^{2}+\\frac{\\eta\\kappa}{2c}.

Thus the inequality ([3](#S4.E3 "In Theorem 1. ‣ 4 Analysis ‣ Reasonable Effectiveness of Random Weighting:A Litmus Test for Multi-Task Learning")) holds if η≤12​c\\eta\\leq\\frac{1}{2c}.

According to inequality ([5](#A2.Ex11 "In B.1 Proof of Theorem ‣ Appendix B Proof of Section ‣ Reasonable Effectiveness of Random Weighting:A Litmus Test for Multi-Task Learning")), the minimal value of a quadratic function gε​(η)\=(1−2​η​c)​ε+η2​κg\_{\\varepsilon}(\\eta)=(1-2\\eta c)\\varepsilon+\\eta^{2}\\kappa is achieved at η∗\=ε​cκ\\eta\_{\*}=\\frac{\\varepsilon c}{\\kappa}. By setting ‖θ0−θ∗‖2\=ε0\\|\\theta\_{0}-\\theta\_{\*}\\|^{2}=\\varepsilon\_{0}, we have

𝔼⁡\[‖θk+1−θ∗‖2\]\\displaystyle\\mathbb{E}\[\\|\\theta\_{k+1}-\\theta\_{\*}\\|^{2}\]

≤g‖θk−θ∗‖2​(η∗)\\displaystyle\\leq g\_{\\|\\theta\_{k}-\\theta\_{\*}\\|^{2}}(\\eta\_{\*})

\=(1−2​‖θk−θ∗‖2​c2κ)​‖θk−θ∗‖2\\displaystyle=(1-\\frac{2\\|\\theta\_{k}-\\theta\_{\*}\\|^{2}c^{2}}{\\kappa})\\|\\theta\_{k}-\\theta\_{\*}\\|^{2}

≤(1−2​ε​c2κ)​‖θk−θ∗‖2\\displaystyle\\leq(1-\\frac{2\\varepsilon c^{2}}{\\kappa})\\|\\theta\_{k}-\\theta\_{\*}\\|^{2}

≤(1−2​ε​c2κ)k​ε0.\\displaystyle\\leq(1-\\frac{2\\varepsilon c^{2}}{\\kappa})^{k}\\varepsilon\_{0}.

Then if 𝔼⁡\[‖θk+1−θ∗‖2\]≥ε\\mathbb{E}\[\\|\\theta\_{k+1}-\\theta\_{\*}\\|^{2}\]\\geq\\varepsilon, we have ε≤(1−2​ε​c2κ)k​ε0.\\varepsilon\\leq(1-\\frac{2\\varepsilon c^{2}}{\\kappa})^{k}\\varepsilon\_{0}. Therefore, k≤κ2​ε​c2​log⁡(ε0ε).k\\leq\\frac{\\kappa}{2\\varepsilon c^{2}}\\log\\left(\\frac{\\varepsilon\_{0}}{\\varepsilon}\\right).

### B.2 Proof of Theorem [2](#Thmtheorem2 "Theorem 2. ‣ 4 Analysis ‣ Reasonable Effectiveness of Random Weighting:A Litmus Test for Multi-Task Learning")

Since φk\=θk−η∇𝝁⊤ℓ(θk)\\varphi\_{k}=\\theta\_{k}-\\eta\\nabla\\bm{\\mu}^{\\top}\\bm{\\ell}(\\theta\_{k}) and θk+1\=θk−η(∇𝝁⊤ℓ(θk)+ξk)\\theta\_{k+1}=\\theta\_{k}-\\eta(\\nabla\\bm{\\mu}^{\\top}\\bm{\\ell}(\\theta\_{k})+\\xi\_{k}), we have

φk+1\=φk−ηξk−∇𝝁⊤ℓ(φk−ηξk).\\varphi\_{k+1}=\\varphi\_{k}-\\eta\\xi\_{k}-\\nabla\\bm{\\mu}^{\\top}\\bm{\\ell}(\\varphi\_{k}-\\eta\\xi\_{k}).

Since the loss function ℓt​(θ)\\ell\_{t}(\\theta) of task tt is ctc\_{t}-one point strongly convex w.r.t. a given point θ∗\\theta\_{\*} after convolved with noise ξ\\xi, similar to inequality ([4](#A2.Ex5 "In B.1 Proof of Theorem ‣ Appendix B Proof of Section ‣ Reasonable Effectiveness of Random Weighting:A Litmus Test for Multi-Task Learning")), we have

⟨∇𝔼ξ​\[𝝁⊤​ℓ​(φ−η​ξ)\],φ−θ∗⟩≥c​‖φ−θ∗‖2,\\left<\\nabla\\mathbb{E}\_{\\xi}\[\\bm{\\mu}^{\\top}\\bm{\\ell}(\\varphi-\\eta\\xi)\],\\varphi-\\theta\_{\*}\\right>\\geq c\\|\\varphi-\\theta\_{\*}\\|^{2}, where c\=min1≤t≤T⁡{ct}c=\\min\_{1\\leq t\\leq T}\\{c\_{t}\\}. Since ∇ℓt​(θ)\\nabla\\ell\_{t}(\\theta) is MtM\_{t}-Lipschitz continuous, for any two points θ1\\theta\_{1} and θ2\\theta\_{2} in ℝd\\mathbb{R}^{d}, we have

∥∇𝝁⊤ℓ(θ1)−∇𝝁⊤ℓ(θ2)∥\=∑t\=1Tμt∥∇ℓt(θ1)−∇ℓt(θ2)∥≤∑t\=1TMtμt∥θ1−θ2∥.\\|\\nabla\\bm{\\mu}^{\\top}\\bm{\\ell}(\\theta\_{1})-\\nabla\\bm{\\mu}^{\\top}\\bm{\\ell}(\\theta\_{2})\\|=\\sum\_{t=1}^{T}\\mu\_{t}\\|\\nabla\\ell\_{t}(\\theta\_{1})-\\nabla\\ell\_{t}(\\theta\_{2})\\|\\leq\\sum\_{t=1}^{T}M\_{t}\\mu\_{t}\\|\\theta\_{1}-\\theta\_{2}\\|.

(6)

Note that ∑t\=1TMt​μt≤M\\sum\_{t=1}^{T}M\_{t}\\mu\_{t}\\leq M, where M\=max1≤t≤T⁡{Mt}M=\\max\_{1\\leq t\\leq T}\\{M\_{t}\\}. Therefore, ∇𝝁⊤ℓ(θ)\\nabla\\bm{\\mu}^{\\top}\\bm{\\ell}(\\theta) is MM-Lipschitz continuous. Then we can get

𝔼⁡\[‖φk+1−θ∗‖2\]\\displaystyle\\mathbb{E}\[\\|\\varphi\_{k+1}-\\theta\_{\*}\\|^{2}\]

\=𝔼\[∥φk−ηξk−∇𝝁⊤ℓ(φk−ηξk)−θ∗∥2\]\\displaystyle=\\mathbb{E}\[\\|\\varphi\_{k}-\\eta\\xi\_{k}-\\nabla\\bm{\\mu}^{\\top}\\bm{\\ell}(\\varphi\_{k}-\\eta\\xi\_{k})-\\theta\_{\*}\\|^{2}\]

≤𝔼\[∥φk−θ∗∥2+∥ηξk∥2+∥∇𝝁⊤ℓ(φk−ηξk)∥2−2⟨φk−θ∗,ηξk⟩\\displaystyle\\leq\\mathbb{E}\[\\|\\varphi\_{k}-\\theta\_{\*}\\|^{2}+\\|\\eta\\xi\_{k}\\|^{2}+\\|\\nabla\\bm{\\mu}^{\\top}\\bm{\\ell}(\\varphi\_{k}-\\eta\\xi\_{k})\\|^{2}-2\\left<\\varphi\_{k}-\\theta\_{\*},\\eta\\xi\_{k}\\right>

−2⟨φk−θ∗,∇𝝁⊤ℓ(φk−ηξk)⟩+2⟨∇𝝁⊤ℓ(φk−ηξk),ηξk⟩\]\\displaystyle\\ ~~~~~~-2\\left<\\varphi\_{k}-\\theta\_{\*},\\nabla\\bm{\\mu}^{\\top}\\bm{\\ell}(\\varphi\_{k}-\\eta\\xi\_{k})\\right>+2\\left<\\nabla\\bm{\\mu}^{\\top}\\bm{\\ell}(\\varphi\_{k}-\\eta\\xi\_{k}),\\eta\\xi\_{k}\\right>\]

≤∥φk−θ∗∥2+η2r2+𝔼\[∥∇𝝁⊤ℓ(φk−ηξk)∥2\]−2ηc∥φk−θ∗∥2\\displaystyle\\leq\\|\\varphi\_{k}-\\theta\_{\*}\\|^{2}+\\eta^{2}r^{2}+\\mathbb{E}\[\\|\\nabla\\bm{\\mu}^{\\top}\\bm{\\ell}(\\varphi\_{k}-\\eta\\xi\_{k})\\|^{2}\]-2\\eta c\\|\\varphi\_{k}-\\theta\_{\*}\\|^{2}

+2𝔼\[⟨∇𝝁⊤ℓ(φk−ηξk)−∇𝝁⊤ℓ(φk),ηξk⟩\]\\displaystyle\\ ~~~~~~+2\\mathbb{E}\[\\left<\\nabla\\bm{\\mu}^{\\top}\\bm{\\ell}(\\varphi\_{k}-\\eta\\xi\_{k})-\\nabla\\bm{\\mu}^{\\top}\\bm{\\ell}(\\varphi\_{k}),\\eta\\xi\_{k}\\right>\]

≤(1−2​η​c)​‖φk−θ∗‖2+η2​r2+η2​𝔼​\[‖M⁡(θ∗−(φk−η​ξk))‖2\]+2​η3​r2​M\\displaystyle\\leq(1-2\\eta c)\\|\\varphi\_{k}-\\theta\_{\*}\\|^{2}+\\eta^{2}r^{2}+\\eta^{2}\\mathbb{E}\[\\|M(\\theta\_{\*}-(\\varphi\_{k}-\\eta\\xi\_{k}))\\|^{2}\]+2\\eta^{3}r^{2}M

≤(1−2​η​c)​‖φk−θ∗‖2+η2​r2+η2​M2​‖φk−θ∗‖2+𝔼⁡\[⟨φk−θ∗,η​ξk⟩\]\\displaystyle\\leq(1-2\\eta c)\\|\\varphi\_{k}-\\theta\_{\*}\\|^{2}+\\eta^{2}r^{2}+\\eta^{2}M^{2}\\|\\varphi\_{k}-\\theta\_{\*}\\|^{2}+\\mathbb{E}\[\\left<\\varphi\_{k}-\\theta\_{\*},\\eta\\xi\_{k}\\right>\]

+η2​M2​𝔼​\[‖η​ξk‖2\]+2​η3​r2​M\\displaystyle\\ ~~~~~~+\\eta^{2}M^{2}\\mathbb{E}\[\\|\\eta\\xi\_{k}\\|^{2}\]+2\\eta^{3}r^{2}M

≤(1−2​η​c+η2​M2)​‖φk−θ∗‖2+η2​r2​(1+η​M)2,\\displaystyle\\leq(1-2\\eta c+\\eta^{2}M^{2})\\|\\varphi\_{k}-\\theta\_{\*}\\|^{2}+\\eta^{2}r^{2}(1+\\eta M)^{2}, where the second inequality is due to the convexity assumption and 𝔼⁡\[ξk\]\=0\\mathbb{E}\[\\xi\_{k}\]=0, the third and forth inequalities are due to the Lipschitz continuity. We set ρ\=2​η​c−η2​M2\\rho=2\\eta c-\\eta^{2}M^{2} and β\=η2​r2​(1+η​M)2\\beta=\\eta^{2}r^{2}(1+\\eta M)^{2}. If ρ≥0\\rho\\geq 0, we have η≤cM2\\eta\\leq\\frac{c}{M^{2}}, then we get

𝔼⁡\[‖φk+1−θ∗‖2\]\\displaystyle\\mathbb{E}\[\\|\\varphi\_{k+1}-\\theta\_{\*}\\|^{2}\]

≤(1−ρ)​‖φk−θ∗‖2+β\\displaystyle\\leq(1-\\rho)\\|\\varphi\_{k}-\\theta\_{\*}\\|^{2}+\\beta

≤(1−ρ)k​‖φ0−θ∗‖2+∑j\=0k−1(1−ρ)j​β\\displaystyle\\leq(1-\\rho)^{k}\\|\\varphi\_{0}-\\theta\_{\*}\\|^{2}+\\sum\_{j=0}^{k-1}(1-\\rho)^{j}\\beta

≤(1−ρ)k​‖φ0−θ∗‖2+βρ.\\displaystyle\\leq(1-\\rho)^{k}\\|\\varphi\_{0}-\\theta\_{\*}\\|^{2}+\\frac{\\beta}{\\rho}.

So if K≤1ρ​log⁡(ρ​ε0β)K\\leq\\frac{1}{\\rho}\\log\\left(\\frac{\\rho\\varepsilon\_{0}}{\\beta}\\right), we have 𝔼⁡\[‖φK+1−θ∗‖2\]≤2​βρ.\\mathbb{E}\[\\|\\varphi\_{K+1}-\\theta\_{\*}\\|^{2}\]\\leq\\frac{2\\beta}{\\rho}. Then by the Markov inequality, with probability at least 1−δ1-\\delta, we have

‖φK−θ∗‖2≤2​βρ​δ.\\|\\varphi\_{K}-\\theta\_{\*}\\|^{2}\\leq\\frac{2\\beta}{\\rho\\delta}.

### B.3 Noise Upper Bound

Suppose the noise produced by the EW method is ξ¯\=∥∇𝝁⊤ℓ(𝓓\~;θ)−∇𝝁⊤ℓ(𝓓;θ)∥\\bar{\\xi}=\\|\\nabla\\bm{{\\mu}}^{\\top}\\bm{\\ell}(\\tilde{\\bm{\\mathcal{D}}};\\theta)-\\nabla\\bm{{\\mu}}^{\\top}\\bm{\\ell}(\\bm{\\mathcal{D}};\\theta)\\| and ‖ξ¯‖2≤R\\|\\bar{\\xi}\\|^{2}\\leq R. The noise produced by the RLW method is ξ\=∥∇𝝀⊤ℓ(𝓓\~;θ)−∇𝝁⊤ℓ(𝓓;θ)∥\\xi=\\|\\nabla\\bm{{\\lambda}}^{\\top}\\bm{\\ell}(\\tilde{\\bm{\\mathcal{D}}};\\theta)-\\nabla\\bm{{\\mu}}^{\\top}\\bm{\\ell}(\\bm{\\mathcal{D}};\\theta)\\|. We have

‖ξ‖2\\displaystyle\\|\\xi\\|^{2}

\=∥∇𝝀⊤ℓ(𝓓\~;θ)−∇𝝁⊤ℓ(𝓓\~;θ)+∇𝝁⊤ℓ(𝓓\~;θ)−∇𝝁⊤ℓ(𝓓;θ)∥2\\displaystyle=\\|\\nabla\\bm{{\\lambda}}^{\\top}\\bm{\\ell}(\\tilde{\\bm{\\mathcal{D}}};\\theta)-\\nabla\\bm{{\\mu}}^{\\top}\\bm{\\ell}(\\tilde{\\bm{\\mathcal{D}}};\\theta)+\\nabla\\bm{{\\mu}}^{\\top}\\bm{\\ell}(\\tilde{\\bm{\\mathcal{D}}};\\theta)-\\nabla\\bm{{\\mu}}^{\\top}\\bm{\\ell}(\\bm{\\mathcal{D}};\\theta)\\|^{2}

\=∥(𝝀⊤−𝝁⊤)∇ℓ(𝓓\~;θ)∥2+2⟨(𝝀⊤−𝝁⊤)ℓ(𝓓\~;θ),ξ¯⟩+∥ξ¯∥2.\\displaystyle=\\|(\\bm{{\\lambda}}^{\\top}-\\bm{{\\mu}}^{\\top})\\nabla\\bm{\\ell}(\\tilde{\\bm{\\mathcal{D}}};\\theta)\\|^{2}+2\\left<(\\bm{{\\lambda}}^{\\top}-\\bm{{\\mu}}^{\\top})\\bm{\\ell}(\\tilde{\\bm{\\mathcal{D}}};\\theta),\\bar{\\xi}\\right>+\\|\\bar{\\xi}\\|^{2}.

Because the noise ξ¯\\bar{\\xi} can be any direction, there exists a constant s\>0s>0 such that ‖ξ¯‖2\=R\\|\\bar{\\xi}\\|^{2}=R and ξ¯\=s(𝝀⊤−𝝁⊤)∇ℓ(𝓓\~;θ)\\bar{\\xi}=s(\\bm{{\\lambda}}^{\\top}-\\bm{{\\mu}}^{\\top})\\nabla\\bm{\\ell}(\\tilde{\\bm{\\mathcal{D}}};\\theta). Then, we have ‖ξ‖2≤(1+2​s)​‖𝝀−𝝁‖2​‖∇ℓ​(𝓓\~,θ)‖2+R\\|\\xi\\|^{2}\\leq(1+2s)\\|\\bm{{\\lambda}}-\\bm{{\\mu}}\\|^{2}\\|\\nabla\\bm{\\ell}(\\tilde{\\bm{\\mathcal{D}}};\\theta)\\|^{2}+R. Thus, the norm of the noise provided by the RLW method has a larger supremum than EW.

## Appendix C Additional Details about the XTREME Benchmark

Table 5: The numbers of training, validation, and test data for each language in PI and POS problems from the XTREME benchmark.

| PI | POS             |                |
| -- | --------------- | -------------- |
| en | 49.4K+2.0K+2.0K | 6.9K+1.8K+3.2K |
| zh | 49.4K+2.0K+2.0K | 4.0K+0.5K+2.9K |
| de | 49.4K+2.0K+2.0K | -              |
| es | 49.4K+2.0K+2.0K | -              |
| te | -               | 1.0K+0.1K+0.1K |
| vi | -               | 1.4K+0.8K+0.8K |

## Appendix D Additional Experimental Results

### D.1 Results on the CityScapes Dataset

#### Dataset.

The CityScapes dataset \[[7](#bib.bib7 "")\] is a large-scale urban street scene understanding dataset and it is comprised of a diverse set of stereo video sequences recorded from 50 different cities in fine weather during the daytime. It contains 2,975 and 500 annotated images for training and test, respectively. This dataset includes two tasks: 7-class semantic segmentation and depth estimation.

#### Implementation details.

For the CityScapes dataset, the network architecture and optimizer are the same as those in the NYUv2 dataset. We resize all the images to 128×256128\\times 256 and set the batch size to 64 for training. We use the cross-entropy loss and L1L\_{1} loss for the semantic segmentation and depth estimation tasks, respectively.

#### Results.

The results on the CityScapes dataset are shown in Table [6](#A4.T6 "Table 6 ‣ Results. ‣ D.1 Results on the CityScapes Dataset ‣ Appendix D Additional Experimental Results ‣ Reasonable Effectiveness of Random Weighting:A Litmus Test for Multi-Task Learning"). The empirical observations are similar to those on the NYUv2 dataset in Table [1](#S5.T1 "Table 1 ‣ 5.1 Results on the NYUv2 Dataset ‣ 5 Experiments ‣ Reasonable Effectiveness of Random Weighting:A Litmus Test for Multi-Task Learning"). Firstly, both the RLW and RGW strategies significantly outperform the EW method. Secondly, the RLW method can outperform most of the loss balancing baselines except the IMTL-L method. Moreover, the RGW method achieves 2.36% performance improvement and outperforms all of the baselines.

Table 6: Performance on the CityScapes dataset with two tasks: 7-class semantic segmentation and depth estimation. The best results for each task on each measure over loss/gradient balancing methods are marked with superscript ∗\*/†{\\dagger}. The best results for each task on each measure are highlighted in bold. ↑\\uparrow (↓\\downarrow) means the higher (lower) the result, the better the performance.

Methods

Segmentation

Depth

Δp\\Delta\_{\\mathrm{p}}↑{\\uparrow}

mIoU↑{\\uparrow}

Pix Acc↑\\uparrow

Abs Err↓\\downarrow

Rel Err↓\\downarrow

EW

68.71

91.50

0.0132

45.58

+0.00%

 Loss Bal. 

UW

68.84

91.53

0.0132

46.18

-0.09%

DWA

68.56

91.48

0.0135

44.49

+0.05%

IMTL-L

69.71∗

91.77∗

0.0128∗

45.08

+1.58%∗

MOML

69.34

91.65

0.0129

46.33

+0.59%

RLW (ours)

68.78

91.45

0.0134

43.68∗

+0.69%

 Gradient Bal. 

MGDA-UB

68.41

91.13

0.0124†

46.85

+0.64%

GradNorm

68.60

91.48

0.0133

45.32

+0.01%

PCGrad

68.54

91.47

0.0135

44.82

-0.10%

GradDrop

68.62

91.45

0.0136

45.05

-0.42%

IMTL-G

68.62

91.48

0.0130

44.29

+1.09%

GradVac

68.60

91.47

0.0134

44.92

-0.06%

CAGrad

68.89

91.50

0.0128

44.72

+1.38%

RotoGrad

68.96

91.47

0.0127

43.85†

+2.13%

RGW (ours)

69.68†

91.85†

0.0127

43.91

+2.36%†

### D.2 Results on the CelebA Dataset

#### Dataset.

The CelebA dataset \[[20](#bib.bib20 "")\] is a large-scale face attributes dataset with 202,599 face images, each of which has 40 attribute annotations. It is split into three parts: 162,770, 19,867, and 19,962 images for training, validation, and testing, respectively. Hence, this dataset contains 40 tasks and each task is a binary classification problem for one attribute.

Table 7: Average classification accuracy (%) of different methods on the CelebA dataset with forty tasks. The best results over loss/gradient balancing methods are marked with superscript ∗\*/†{\\dagger}. The best results are highlighted in bold. 

Methods

Avg Acc

EW

90.70

 Loss Bal. 

UW

90.84

DWA

90.77

IMTL-L

90.46

MOML

90.94∗

RLW (ours)

90.73

 Gradient Bal. 

MGDA-UB

90.40

GradNorm

90.77

PCGrad

90.85†

GradDrop

90.71

IMTL-G

90.80

GradVac

90.75

CAGrad

90.72

RotoGrad

90.45

RGW (ours)

90.00

#### Implementation details.

We use the ResNet-18 network as a shared feature extractor and a fully connected layer with two output units as a task-specific head for each task. All the images are resized to 64×6464\\times 64. The Adam optimizer with the learning rate as 10−310^{-3} is used for training and the batch size is set to 512. The cross-entropy loss is used for the 40 tasks.

#### Results.

Since the number of tasks in the CelebA dataset is large, we only report the average classification accuracy on the forty tasks in Table [7](#A4.T7 "Table 7 ‣ Dataset. ‣ D.2 Results on the CelebA Dataset ‣ Appendix D Additional Experimental Results ‣ Reasonable Effectiveness of Random Weighting:A Litmus Test for Multi-Task Learning"). According to the results, the proposed RLW strategy slightly outperforms the EW method and performs comparably with loss balancing baseline methods. However, we can find that the RGW method and most of the gradient balancing methods are worse or achieve very limited improvement over the EW method, which indicates the gradient weighting is not suitable for the CelebA dataset.

### D.3 Results on the Office-31 and Office-Home Datasets

#### Datasets.

The Office-31 dataset \[[26](#bib.bib26 "")\] consists of three domains: Amazon (A), DSLR (D), and Webcam (W), where each domain contains 31 object categories, and it contains 4,110 labeled images. We randomly split the whole dataset with 60% for training, 20% for validation, and the rest 20% for testing. The Office-Home dataset \[[31](#bib.bib31 "")\] has four domains: artistic images (Ar), clip art (Cl), product images (Pr), and real-world images (Rw). It has 15,500 labeled images in total and each domain contains 65 classes. We make the same split as the Office-31 dataset. For both datasets, we consider the multi-class classification problem on each domain as a task. Similar to multilingual problems from the XTREME benchmark, each task in both Office-31 and Office-Home datasets has its own input data.

#### Implementation details.

We use the same configuration for the Office-31 and Office-Home datasets. Specifically, the ResNet-18 network pre-trained on the ImageNet dataset is used as a shared backbone among tasks and a fully connected layer is applied as a task-specific output layer for each task. All the input images are resized to 224×224224\\times 224. We use the Adam optimizer with the learning rate as 10−410^{-4} and the weight decay as 10−510^{-5} and set the batch size to 128 for training. The cross-entropy loss is used for all tasks in both datasets.

Table 8: Classification accuracy (%) of different methods on the Office-31 and Office-Home datasets. The best results for each domain over loss/gradient balancing methods are marked with superscript ∗\*/†{\\dagger}. The best results for each task are highlighted in bold. 

Methods

Office-31

Office-Home

A

D

W

Avg

Ar

Cl

Pr

Rw

Avg

EW

82.73

96.72

96.11

91.85

62.99

76.48

88.45

77.72

76.41

 Loss Bal. 

UW

82.73

96.72∗

95.55

91.66

63.94

75.62

88.55

78.05

76.54

DWA

82.22

96.72∗

96.11

91.68

63.37

76.05

89.08

77.62

76.53

IMTL-L

83.76

96.72∗

95.55

92.01

65.46∗

79.08∗

88.45

78.81

77.95∗

MOML

84.78∗

95.08

96.67∗

92.17

64.70

77.03

88.24

80.00

77.49

RLW (ours)

83.76

96.72∗

96.67∗

92.38∗

62.80

76.48

90.57∗

80.21∗

77.52

 Gradient Bal. 

MGDA-UB

81.02

95.90

97.77†

91.56

64.32

75.29

89.72

79.35

77.17

GradNorm

83.93

97.54†

94.44

91.97

65.46†

75.29

88.66

78.91

77.08

PCGrad

82.22

96.72

95.55

91.49

63.94

76.05

88.87

78.27

76.78

GradDrop

84.27†

95.08

96.11

91.82

64.70

77.03

88.02

79.13

77.22

IMTL-G

82.22

95.90

96.11

91.41

63.37

76.05

89.19

79.24

76.96

GradVac

82.73

97.54†

95.55

91.94

63.18

76.48

88.66

77.83

76.53

CAGrad

82.22

96.72

96.67

91.87

63.75

75.94

89.08

78.27

76.75

RotoGrad

82.90

96.72

96.11

91.91

61.85

77.03

90.36†

78.59

76.95

RGW (ours)

84.27†

96.72

96.67

92.55†

65.08

78.65†

88.66

79.89†

78.07†

#### Results.

According to the results shown in Table [8](#A4.T8 "Table 8 ‣ Implementation details. ‣ D.3 Results on the Office-31 and Office-Home Datasets ‣ Appendix D Additional Experimental Results ‣ Reasonable Effectiveness of Random Weighting:A Litmus Test for Multi-Task Learning"), we can see both the RLW and RGW strategies outperform the EW method on both two datasets in terms of the average classification accuracy over tasks, which implies the effectiveness of the RW methods. Moreover, the RGW method achieves the best performance (92.55% and 78.07% in term of the average accuracy) over all baselines on the Office-31 and Office-Home datasets, respectively.

[](javascript:toggleReadingMode\(\); "Disable reading mode, show header and footer")