# Careful with that Scalpel: Improving Gradient Surgery with an EMA

 Yu-Guan Hsieh Affiliation: Apple    James Thornton Affiliation: Apple    Eugene Ndiaye Affiliation: Apple    Michal Klein Affiliation: Apple    Marco Cuturi Affiliation: Apple    Pierre Ablin Affiliation: Apple Correspondence to: [p\_ablin@apple.com](mailto:p_ablin@apple.com) 

###### Abstract

Beyond minimizing a single training loss, many deep learning estimation pipelines rely on an auxiliary objective to quantify and encourage desirable properties of the model (e.g. performance on another dataset, robustness, agreement with a prior). Although the simplest approach to incorporating an auxiliary loss is to sum it with the training loss as a regularizer, recent works have shown that one can improve performance by blending the gradients beyond a simple sum; this is known as gradient surgery. We cast the problem as a constrained minimization problem where the auxiliary objective is minimized among the set of minimizers of the training loss. To solve this bilevel problem, we follow a parameter update direction that combines the training loss gradient and the orthogonal projection of the auxiliary gradient to the training gradient. In a setting where gradients come from mini-batches, we explain how, using a moving average of the training loss gradients, we can carefully maintain this critical orthogonality property. We demonstrate that our method, Bloop, can lead to much better performances on NLP and vision experiments than other gradient surgery methods without EMA.

###### Keywords: 

Machine Learning, ICML 

## 1 Introduction

Overparameterized neural networks trained on large datasets admit multiple solutions with the same optimal training loss ([Cooper, 2018](#bib.bib6 ""); [Li et al., 2018](#bib.bib23 "")). Although these parameters may seem equivalent when viewed through their training loss, they result in different functions, which may exhibit starkly different behaviors on unseen data points. Practitioners are usually interested in generalization — one would rather use the network with lower test loss between two networks — but there are countless other metrics of interest, such as performance on another dataset, robustness, or model calibration. In all of these cases, one aims to train the neural network by minimizing a training loss LmainL\_{{\\mathrm{main}}} while keeping an eye on an auxiliary metric or loss LauxL\_{\\mathrm{aux}}.

Optimization trade-offs. Our focus in this paper is on methods that achieve the best possible trade-off between training and auxiliary losses, using a hyper-parameter λ≥0\\lambda\\geq 0 to control that trade-off: λ\=0\\lambda=0 corresponds to training on LmainL\_{{\\mathrm{main}}} exclusively, while increasing λ\\lambda usually decreases LauxL\_{\\mathrm{aux}} at the expense of LmainL\_{{\\mathrm{main}}}. Using the auxiliary loss as a *regularizer* results in the mixed training method, arguably the simplest approach to control that trade-off:

minθ⁡Lmain​(θ)+λ​Laux​(θ).\\min\_{\\theta}L\_{{\\mathrm{main}}}(\\theta)+\\lambda L\_{\\mathrm{aux}}(\\theta).

(1)

Mixed training, however, runs into optimization issues if the directions of the largest curvature of the training loss and that of the auxiliary loss are not aligned — see [Section 3.3](#S3.SS3 "3.3 Conditioning compared to regularization method ‣ 3 Theoretical Analysis ‣ Careful with that Scalpel: Improving Gradient Surgery with an EMA") for an example.

The Simple Bilevel Approach. Provided that modern deep neural networks are inherently overparameterized, leading to multiple minimizers, an ideal solution would be to find the minimizer of LmainL\_{{\\mathrm{main}}} that achieves the smallest auxiliary loss. This corresponds to solving [Equation 1](#S1.E1 "Equation 1 ‣ 1 Introduction ‣ Careful with that Scalpel: Improving Gradient Surgery with an EMA") in the limit where λ→0\\lambda\\to 0, and can also be expressed as the following *simple bilevel* problem ([Dempe et al., 2010](#bib.bib8 "")):

min⁡Laux​(θ)​ s.t. ​θ∈arg⁡min⁡Lmain​(θ).\\displaystyle\\begin{split}\\min L\_{\\mathrm{aux}}(\\theta)~\\text{ s.t. }~\\theta\\in\\arg\\min L\_{{\\mathrm{main}}}(\\theta).\\end{split}

(2)

Problem ([2](#S1.E2 "Equation 2 ‣ 1 Introduction ‣ Careful with that Scalpel: Improving Gradient Surgery with an EMA")) is a constrained optimization problem on the set of minimizers of LmainL\_{{\\mathrm{main}}}, a high-dimensional set with no clear structure, except when LmainL\_{{\\mathrm{main}}} is convex, in which case several provably convergent approaches have been proposed ([Sabach & Shtern, 2017](#bib.bib28 ""); [Gong & Liu, 2021](#bib.bib12 ""); [Cao et al., 2023](#bib.bib2 "")). However, to the best of our knowledge, these methods have not been applied to training neural networks, where these convergence guarantees do not hold.

Connections to Multi-Task Learning. The problem of simultaneously optimizing the main and auxiliary loss is also a special case of *multi-task* learning ([Caruana, 1997](#bib.bib3 "")) involving only two tasks. Many of the approaches proposed to tackle this problem more efficiently rely on the idea of *gradient surgery*, which stitches together and possibly modify the gradients of both losses when they disagree ([Yu et al., 2020](#bib.bib35 "")). While multi-task methods tend to treat the two losses equally, we are interested in our work in cases where there is a clear hierarchy between the two.

Two types of auxiliary losses. Auxiliary objectives largely fall into two categories. The first consists of objectives that guide optimization of the main loss but are not intrinsically meaningful; also known as inductive biases, they are only useful to reach a lower test loss. Weight decay, Laux\=12∥⋅∥2L\_{\\mathrm{aux}}=\\frac{1}{2}\\|\\cdot\\|^{2}, fits this description: using it improves generalization, but practitioners rarely care about the final norm of their parameters. The second category of auxiliary losses quantify instead a desirable property: Trading off an increase in the main loss for a decrease in the auxiliary loss might be relevant to applications. For instance, the main objective might a loss on a large dataset, whereas the auxiliary objective may be a loss on a smaller, specialized dataset. Ideally, one wishes to achieve a model with high accuracy on both, and hope that the auxiliary loss might also help generalization on the large training set, but both objectives remain meaningful on their own. Another example is in training neural networks that are also smooth, i.e., with a small Lipschitz constant. This is beneficial for the networks’ robustness ([Cisse et al., 2017](#bib.bib5 "")). To enforce this during training, one can use a proxy for the Lipschitz constant of the neural network as an auxiliary loss ([Tsuzuku et al., 2018](#bib.bib32 ""); [Terjék, 2019](#bib.bib31 "")).

Contributions. To handle the optimization tradeoff between main and auxiliary losses, we introduce in [Section 2](#S2 "2 The Bloop Algorithm ‣ Careful with that Scalpel: Improving Gradient Surgery with an EMA") the Bloop (BiLevel Optimization with Orthogonal Projection) method. Our method is inspired by the simple bilevel problem, but similar to the regularization approach, has a tunable hyperparameter, λ\\lambda, to control the trade-off between losses. At the heart of the method is a projection of the auxiliary gradient to be orthogonal to the primary loss gradient. We first provide a theoretical justification for this approach in the full-batch case. In the stochastic setting, we rely on an exponential moving average (EMA) of the training gradient to estimate the projection direction, and retain most of the full-batch theoretical properties.In [Section 3](#S3 "3 Theoretical Analysis ‣ Careful with that Scalpel: Improving Gradient Surgery with an EMA"), we analyze Bloop’s stationary points, and show that they are first-order stationary points of the simple bilevel problem. We demonstrate the convergence of the iterates towards the stationary points of the training loss, under appropriate hypothesis on the step size and the EMA accumulation factor, highlighting the importance of the EMA. In [Section 4](#S4 "4 Related Works ‣ Careful with that Scalpel: Improving Gradient Surgery with an EMA"), we discuss related methods that perform variants of gradient surgery. In [Section 5](#S5 "5 Experiments ‣ Careful with that Scalpel: Improving Gradient Surgery with an EMA"), we explore the applicability of our method to a variety of tasks: training network parameters with an explicit bias; multi-task learning; training language models to perform well on a large generic dataset and a small specific dataset. In our experiments, Bloop exhibits a better Pareto front than both the mixed method and multi-task methods that do not use an EMA.

## 2 The Bloop Algorithm

In this section, we introduce Bloop, a simple and intuitive iterative algorithm to optimize two losses simultaneously. We then discuss how the method can be extended to address stochasticity in the gradients, and multi-level optimization.

### 2.1 Full-batch setting and main intuition

At each step, Bloop builds a parameter update direction d∈ℝpd\\in\\mathbb{R}^{p} which is then fed to an optimizer (e.g. Adam ([Kingma & Ba, 2014](#bib.bib19 ""))) in order to converge to the solution of [Equation 2](#S1.E2 "Equation 2 ‣ 1 Introduction ‣ Careful with that Scalpel: Improving Gradient Surgery with an EMA"). For instance, the gradient descent optimizer would iterate θ←θ−η​d\\theta\\leftarrow\\theta-\\eta d. At the current iterate θ\\theta, we let gmain\=∇Lmain​(θ)g\_{{\\mathrm{main}}}=\\nabla L\_{{\\mathrm{main}}}(\\theta) and gaux\=∇Laux​(θ)g\_{\\mathrm{aux}}=\\nabla L\_{\\mathrm{aux}}(\\theta).

We design our direction from first principles. We seek a direction in the span of these two gradients, d\=ω​gmain+λ​gauxd=\\omega g\_{\\mathrm{main}}+\\lambda g\_{\\mathrm{aux}} with ω\\omega and λ\\lambda two scalars. Our primary goal is to make progress on the main loss at the same speed as gradient descent; hence we target Lmain​(θ−η​d)≃Lmain​(θ−η​gmain).L\_{{\\mathrm{main}}}(\\theta-\\eta d)\\simeq L\_{{\\mathrm{main}}}(\\theta-\\eta g\_{\\mathrm{main}}).

At the first order in the step-size η\\eta, we see that the component of the direction in the direction gmaing\_{\\mathrm{main}} should be the same as that of gmaing\_{\\mathrm{main}}, i.e., we want ⟨d,gmain⟩\=‖gmain‖2\\langle d,g\_{\\mathrm{main}}\\rangle=\\|g\_{\\mathrm{main}}\\|^{2}. This gives the equation (1−ω)​‖gmain‖2\=λ⁡⟨gmain,gaux⟩(1-\\omega)\\|g\_{\\mathrm{main}}\\|^{2}=\\lambda\\langle g\_{\\mathrm{main}},g\_{\\mathrm{aux}}\\rangle. Our secondary goal is the optimization of the auxiliary loss, hence we impose that the coefficient in front of gauxg\_{\\mathrm{aux}} is positive, i.e. that λ\>0\\lambda>0. These two conditions alone give us our update rule: we find that such a direction is necessarily

Hyperparameter λ≥0\\lambda\\geq 0 trades-off the two objectives, and π⁡(gaux,gmain)\\pi(g\_{\\mathrm{aux}};g\_{\\mathrm{main}}) is the projection of gauxg\_{\\mathrm{aux}} orthogonal to gmaing\_{\\mathrm{main}}. This direction admits an intuitive explanation: since we primarily want to optimize the main loss, we follow gmaing\_{\\mathrm{main}}; the projection part is aligned with gauxg\_{\\mathrm{aux}}, and does not interfere with gmaing\_{\\mathrm{main}} thanks to the orthogonality condition. Moreover, the fact that ⟨d,gmain⟩\=‖gmain‖2\\langle d,g\_{\\mathrm{main}}\\rangle=\\|g\_{\\mathrm{main}}\\|^{2} means that following this direction does not change the optimization with respect to LmainL\_{{\\mathrm{main}}} when step-sizes are small. Specifically, we write down the Taylor expansion at the first order

Lmain​(θ−η​d)≃Lmain​(θ)−η⁡⟨gmain,d⟩,(Orthogonality) ≃Lmain​(θ)−η​‖gmain‖2.\\displaystyle\\begin{split}L\_{{\\mathrm{main}}}(\\theta-\\eta d)&\\simeq L\_{{\\mathrm{main}}}(\\theta)-\\eta\\langle g\_{\\mathrm{main}},d\\rangle,\\\\ \\text{\\color\[rgb\]{0.5,0.5,0.5}(Orthogonality)\\hskip 5.0pt}&\\simeq L\_{{\\mathrm{main}}}(\\theta)-\\eta\\|g\_{\\mathrm{main}}\\|^{2}.\\end{split}

(4)

This is the same as standard gradient descent where d\=gmaind=g\_{\\mathrm{main}}. [Figure 1](#S2.F1 "Figure 1 ‣ 2.1 Full-batch setting and main intuition ‣ 2 The Bloop Algorithm ‣ Careful with that Scalpel: Improving Gradient Surgery with an EMA") illustrates the geometric principle of Bloop.

Figure 1: Principle of the Bloop method: the direction we follow is the sum of the gradient of the main loss gmaing\_{\\mathrm{main}}, and of the projection of the gradient of the auxiliary loss, orthogonal to gmaing\_{\\mathrm{main}}. This enforces that, at the first order, following this direction yields the same decrease in LmainL\_{{\\mathrm{main}}} as following gmaing\_{\\mathrm{main}}.

### 2.2 Stochastic extension for large-scale problems

When dealing with neural networks trained over large datasets, the losses are written as sums over many samples:

Lmain​(θ)\=1n​∑i\=1nLmaini​(θ),Laux​(θ)\=1m​∑j\=1mLauxj​(θ).L\_{{\\mathrm{main}}}(\\theta)=\\frac{1}{n}\\sum\_{i=1}^{n}L\_{{\\mathrm{main}}}^{i}(\\theta),\\kern 5.0ptL\_{\\mathrm{aux}}(\\theta)=\\frac{1}{m}\\sum\_{j=1}^{m}L\_{\\mathrm{aux}}^{j}(\\theta).

In practice, we can only use a mini-batch of gradients to make progress on the problem, as the computation of the full-batch gradient of these losses is out of the question. Concretely, we assume that we have computed the two mini-batch gradients gmainbatchg^{\\mathrm{batch}}\_{\\mathrm{main}}, gauxbatchg^{\\mathrm{batch}}\_{\\mathrm{aux}}, which are by design unbiased estimators of the full-batch gradients:

𝔼⁡\[gmainbatch\]\=gmain​ and ​𝔼​\[gauxbatch\]\=gaux.\\mathbb{E}\[g^{\\mathrm{batch}}\_{\\mathrm{main}}\]=g\_{\\mathrm{main}}~\\text{ and }~\\mathbb{E}\[g^{\\mathrm{batch}}\_{\\mathrm{aux}}\]=g\_{\\mathrm{aux}}.

In the above, the expectation is taken over the randomness of the mini-batch choice. Extending the direction dd to this *stochastic* setting is not straightforward, and careful design makes a big difference in the final performance. A key insight behind standard, single-level, stochastic gradient descent on LmainL\_{{\\mathrm{main}}} is that, for small step sizes, it has on average the same decrease as gradient descent:

𝔼⁡\[Lmain​(θ−η​gmainbatch)\]≃Lmain​(θ)−η​𝔼​\[⟨gmain,gmainbatch⟩\](Linearity of dot) ≃Lmain​(θ)−η⁡⟨gmain,𝔼⁡\[gmainbatch\]⟩(Unbiased gradient) ≃Lmain​(θ)−η​‖gmain‖2\\displaystyle\\begin{split}\\mathbb{E}\[L\_{{\\mathrm{main}}}(\\theta-\\eta g^{\\mathrm{batch}}\_{\\mathrm{main}})\]&\\simeq L\_{{\\mathrm{main}}}(\\theta)-\\eta\\mathbb{E}\[\\langle g\_{\\mathrm{main}},g^{\\mathrm{batch}}\_{\\mathrm{main}}\\rangle\]\\\\ \\text{\\color\[rgb\]{0.5,0.5,0.5}(Linearity of dot)\\hskip 20.00003pt}&\\simeq L\_{{\\mathrm{main}}}(\\theta)-\\eta\\langle g\_{\\mathrm{main}},\\mathbb{E}\[g^{\\mathrm{batch}}\_{\\mathrm{main}}\]\\rangle\\\\ \\text{\\color\[rgb\]{0.5,0.5,0.5}(Unbiased gradient)\\hskip 10.00002pt}&\\simeq L\_{{\\mathrm{main}}}(\\theta)-\\eta\\|g\_{\\mathrm{main}}\\|^{2}\\\\ \\end{split}

We want to preserve this behavior as much as possible. A first idea is simply to plug the mini-batch gradients in [Equation 3](#S2.E3 "Equation 3 ‣ 2.1 Full-batch setting and main intuition ‣ 2 The Bloop Algorithm ‣ Careful with that Scalpel: Improving Gradient Surgery with an EMA"), i.e. consider

dsimplebatch\=gmainbatch+λ​π​(gauxbatch,gmainbatch).d^{\\mathrm{batch}}\_{\\mathrm{simple}}=g^{\\mathrm{batch}}\_{\\mathrm{main}}+\\lambda\\pi(g^{\\mathrm{batch}}\_{\\mathrm{aux}};g^{\\mathrm{batch}}\_{\\mathrm{main}}).

The pitfall of projecting on stochastic gradients. The main issue with the above method is that the projection is nonlinear with respect to its second argument: in general, 𝔼⁡\[π⁡(gauxbatch,gmainbatch)\]≠π⁡(gaux,gmain)\\mathbb{E}\[\\pi(g^{\\mathrm{batch}}\_{\\mathrm{aux}};g^{\\mathrm{batch}}\_{\\mathrm{main}})\]\\neq\\pi(g\_{\\mathrm{aux}};g\_{\\mathrm{main}}). As a consequence, it is not true anymore that ⟨dsimplebatch,gmain⟩\=‖gmain‖2\\langle d^{\\mathrm{batch}}\_{\\mathrm{simple}},g\_{\\mathrm{main}}\\rangle=\\|g\_{\\mathrm{main}}\\|^{2}, even in expectation, which in turn leads to a behavior starkly different from SGD on LmainL\_{{\\mathrm{main}}}. We can improve this intuition using a simplified model of the training dynamics. Assume that gmainbatch\=gmain+σ​εg^{\\mathrm{batch}}\_{\\mathrm{main}}=g\_{\\mathrm{main}}+\\sigma\\varepsilon, where ε∼𝒩⁡(0,I)\\varepsilon\\sim\\mathcal{N}(0,I) is the random gradient noise, and σ\>0\\sigma>0 is the noise variance. In the limit where σ\\sigma is large in front of ‖gmain‖\\|g\_{\\mathrm{main}}\\|, we get that on average 𝔼ε​\[π⁡(gauxbatch,gmainbatch)\]\=(1−1p)​gauxbatch\\mathbb{E}\_{\\varepsilon}\[\\pi(g\_{\\mathrm{aux}}^{\\mathrm{batch}};g\_{\\mathrm{main}}^{\\mathrm{batch}})\]=(1-\\frac{1}{p})g\_{\\mathrm{aux}}^{\\mathrm{batch}} with pp the parameter’s dimension. Therefore, the simple direction is on average dsimplebatch\=gmainbatch+λ⁡(1−1p)​gauxbatchd^{\\mathrm{batch}}\_{\\mathrm{simple}}=g^{\\mathrm{batch}}\_{\\mathrm{main}}+\\lambda(1-\\frac{1}{p})g\_{\\mathrm{aux}}^{\\mathrm{batch}}. We recover the same direction as that of the mixed training method, with a new λ′\=λ⁡(1−1p)\\lambda^{\\prime}=\\lambda(1-\\frac{1}{p}), and the orthogonalization becomes useless.

In order to illustrate this intuition, we conduct a synthetic experiment, explained in [Figure 2](#S2.F2 "Figure 2 ‣ 2.2 Stochastic extension for large-scale problems ‣ 2 The Bloop Algorithm ‣ Careful with that Scalpel: Improving Gradient Surgery with an EMA").

Figure 2: Effect of randomness on the projection: We fix the dimension of the parameter space to p\=100p=100, and draw both gmaing\_{\\text{main}} and gauxg\_{\\text{aux}} from the Gaussian distribution 𝒩⁡(𝟎,I)\\mathcal{N}(\\mathbf{0},I). These two vectors are fixed in the remainder of the experiment. We draw gmainbatch∼gmain+σ​𝒩​(𝟎,I)g\_{\\text{main}}^{\\text{batch}}\\sim g\_{\\text{main}}+\\sigma\\mathcal{N}(\\mathbf{0},I) and use Monte-Carlo simulation to estimate 𝔼⁡\[dsimplebatch\]\=gmain+𝔼⁡\[π⁡(gaux,gmainbatch)\]\\mathbb{E}\[d\_{\\text{simple}}^{\\text{batch}}\]=g\_{\\text{main}}+\\mathbb{E}\[\\pi(g\_{\\text{aux}};g\_{\\text{main}}^{\\text{batch}})\]. We compare its value against dbloop\=gmain+π⁡(gaux,gmain)d\_{\\text{bloop}}=g\_{\\text{main}}+\\pi(g\_{\\text{aux}};g\_{\\text{main}}), its theoretical value when σ\=0\\sigma=0 (the target direction), and dmixed\=gmain+(1−1/100)​gauxd\_{\\text{mixed}}=g\_{\\text{main}}+(1-1/100)g\_{\\text{aux}}, its theoretical value when σ\\sigma tends to infinity. We see that the 𝔼⁡\[dsimplebatch\]\\mathbb{E}\[d^{\\text{batch}}\_{\\text{simple}}\] becomes closer to the gradient of the mixed method when the noise starts to dominate.

The EMA solution. The previous analysis indicates that we need a better estimate of gmaing\_{\\mathrm{main}} than the mini-batch gradient. A simple solution to this is to use an Exponential Moving Average (EMA) of the previous batch gradients, gmainEMAg\_{\\mathrm{main}}^{\\mathrm{EMA}}, which is updated at each iteration by doing gmainEMA←(1−ρ)​gmainEMA+ρ​gmainbatchg\_{\\mathrm{main}}^{\\mathrm{EMA}}\\leftarrow(1-\\rho)g\_{\\mathrm{main}}^{\\mathrm{EMA}}+\\rho g\_{\\mathrm{main}}^{\\mathrm{batch}}, with ρ∈\[0,1\]\\rho\\in\[0,1\] a parameter that controls the speed of the EMA. This can be a much better estimator of gmaing\_{\\mathrm{main}} than gmainbatchg\_{\\mathrm{main}}^{\\mathrm{batch}}, because it averages gradients over the optimization trajectory, drastically reducing the variance. Intuitively, we need to accumulate the EMA faster than the speed of the optimization algorithm that updates the parameters. Hence, ρ\\rho should be greater than the step-size η\\eta. We use this gradient EMA solely in the projection, and propose the direction

We do not replace the first gmainbatchg^{\\mathrm{batch}}\_{\\mathrm{main}} in the formula by the EMA, because dbatchd^{\\mathrm{batch}} is an optimization *direction*, that is then plugged into any optimizer like Adam, which will use a smart adaptive step to reach the solution quickly. Since the EMA does not depend on the current batch, and the projection is linear with respect to its first argument, we have that 𝔼⁡\[dbatch\]\=gmain+λ​π​(gaux,gmainEMA)\\mathbb{E}\[d^{\\mathrm{batch}}\]=g\_{\\mathrm{main}}+\\lambda\\pi(g\_{\\mathrm{aux}};g^{\\mathrm{EMA}}\_{\\mathrm{main}}), and as a consequence, the expected decrease on LmainL\_{{\\mathrm{main}}} following this direction is 𝔼⁡\[Lmain​(θ−η​dbatch)\]≃Lmain​(θ)−η​‖gmain‖2+η​λ​⟨π⁡(gaux,gmainEMA),gmain⟩\\mathbb{E}\[L\_{{\\mathrm{main}}}(\\theta-\\eta d^{\\mathrm{batch}})\]\\simeq L\_{{\\mathrm{main}}}(\\theta)-\\eta\\|g\_{\\mathrm{main}}\\|^{2}+\\eta\\lambda\\langle\\pi(g\_{\\mathrm{aux}};g^{\\mathrm{EMA}}\_{\\mathrm{main}}),g\_{\\mathrm{main}}\\rangle. When the EMA accumulation gmainEMAg^{\\mathrm{EMA}}\_{\\mathrm{main}} is close to gmaing\_{\\mathrm{main}}, the last term becomes small because the two vectors are approximately orthogonal. Thus,

𝔼⁡\[Lmain​(θ−η​dbatch)\]≃Lmain​(θ)−η​‖gmain‖2,\\mathbb{E}\[L\_{{\\mathrm{main}}}(\\theta-\\eta d^{\\mathrm{batch}})\]\\simeq L\_{{\\mathrm{main}}}(\\theta)-\\eta\\|g\_{\\mathrm{main}}\\|^{2}, and we recover the same behavior as SGD on LmainL\_{{\\mathrm{main}}}. The new direction is no longer in the span of (gmainbatchCLOSE(g^{\\mathrm{batch}}\_{\\mathrm{main}}, OPENgauxbatch)g^{\\mathrm{batch}}\_{\\mathrm{aux}}) because it also has a component in the direction of gmainEMAg^{\\mathrm{EMA}}\_{\\mathrm{main}}.

The theory presented in the next section clearly highlights the importance of this EMA, and in our experiments, we find that this simple EMA modification drastically improves the performance of the algorithm on a variety of tasks. In fact, we found that in many cases, standard multi-task methods without EMA have very similar performances to the mixed training method.

[Algorithm 1](#alg1 "Algorithm 1 ‣ 2.2 Stochastic extension for large-scale problems ‣ 2 The Bloop Algorithm ‣ Careful with that Scalpel: Improving Gradient Surgery with an EMA")gives the full pseudo-code of the Bloop method. We use optax-like notations ([DeepMind et al., 2020](#bib.bib7 "")) for the optimizer, which is abstracted as a method that, given a direction dd, current parameters θ\\theta and a state ss containing all its hyper-parameters like learning rate and internal state like EMAs for adaptive methods, returns the updated parameters θ\\theta and updated state ss.

 Input: Hyperparameter λ\\lambda, EMA parameter ρ\\rho, initial parameters θ\\theta, optimizer optim, optimizer state ss, initial EMA gmainEMAg\_{\\mathrm{main}}^{\\mathrm{EMA}} 

 for t\=0,…,T−1t=0,\\dots,T-1 do 

  Sample gradients gmainbatchg\_{\\mathrm{main}}^{\\mathrm{batch}}, gauxbatchg\_{\\mathrm{aux}}^{\\mathrm{batch}} Compute the Bloop direction dbatchd^{\\mathrm{batch}} using [Equation 5](#S2.E5 "Equation 5 ‣ 2.2 Stochastic extension for large-scale problems ‣ 2 The Bloop Algorithm ‣ Careful with that Scalpel: Improving Gradient Surgery with an EMA") Update θ,s←optim​(dbatch,θ,s)\\theta,s\\leftarrow\\texttt{optim}(d^{\\mathrm{batch}},\\theta,s) Update EMA: gmainEMA←(1−ρ)​gmainEMA+ρ​gmainbatchg\_{\\mathrm{main}}^{\\mathrm{EMA}}\\leftarrow(1-\\rho)g\_{\\mathrm{main}}^{\\mathrm{EMA}}+\\rho g\_{\\mathrm{main}}^{\\mathrm{batch}} 

 end for 

Algorithm 1 The Bloop algorithm

### 2.3 Extension to multi-level hierarchical optimization

Our algorithm can be extended to multi-level optimization, where we have more than two losses and they have a *hierarchy*. For simplicity, we present here the case with 33 losses: LmainL\_{{\\mathrm{main}}}, Laux1L\_{\\mathrm{aux}}^{1} and Laux2L\_{\\mathrm{aux}}^{2}. The hierarchy means that we minimize LmainL\_{{\\mathrm{main}}}, and then, among this set of minimizers, we minimize Laux1L\_{\\mathrm{aux}}^{1}. Finally, we minimize Laux2L\_{\\mathrm{aux}}^{2} among this new set. This gives the trilevel optimization problem:

min⁡Laux2​(θ)​ s.t. θ∈(arg​min⁡Laux1​(θ)​ s.t. ​θ∈arg​min⁡Lmain​(θ))\\displaystyle\\begin{split}&\\min L\_{\\mathrm{aux}}^{2}(\\theta)\\text{ s.t. }\\\\ \\theta&\\in\\left(\\argmin L\_{\\mathrm{aux}}^{1}(\\theta)\\text{ s.t. }\\theta\\in\\argmin L\_{{\\mathrm{main}}}(\\theta)\\right)\\end{split}

(6)

Our algorithm can be straightforwardly extended to this case by following a Gram-Schmidt like orthogonalization process: letting gmaing\_{\\mathrm{main}}, gaux1g\_{\\mathrm{aux}}^{1} and gaux2g\_{\\mathrm{aux}}^{2} the gradients of the three losses, we go in the direction

d\=gmain+λ1​π​(gaux1,gmain)+λ2​π​(gaux2,(gmain,gaux1))d=g\_{\\mathrm{main}}+\\lambda^{1}\\pi(g\_{\\mathrm{aux}}^{1};g\_{\\mathrm{main}})+\\lambda^{2}\\pi(g\_{\\mathrm{aux}}^{2};(g\_{\\mathrm{main}},g\_{\\mathrm{aux}}^{1}))

where π⁡(gaux2,(gmain,gaux1))\\pi(g\_{\\mathrm{aux}}^{2};(g\_{\\mathrm{main}},g\_{\\mathrm{aux}}^{1})) is the projection of gaux2g\_{\\mathrm{aux}}^{2} on the orthogonal of the span of (gmain,gaux1)(g\_{\\mathrm{main}},g\_{\\mathrm{aux}}^{1}). Thanks to orthogonality, this direction satisfies ⟨d,gmain⟩\=‖gmain‖2\\langle d,g\_{\\mathrm{main}}\\rangle=\\|g\_{\\mathrm{main}}\\|^{2}; hence in terms of optimization with respect to LmainL\_{{\\mathrm{main}}}, the direction behaves just like gmaing\_{\\mathrm{main}}, and ⟨d,gaux1⟩\=⟨gmain+λ1​π​(gaux1,gmain),gaux1⟩\\langle d,g\_{\\mathrm{aux}}^{1}\\rangle=\\langle g\_{\\mathrm{main}}+\\lambda^{1}\\pi(g\_{\\mathrm{aux}}^{1};g\_{\\mathrm{main}}),g\_{\\mathrm{aux}}^{1}\\rangle; hence in terms of optimization with respect to Laux1L\_{\\mathrm{aux}}^{1}, the direction behaves just like the bilevel direction dd introduced in [Equation 3](#S2.E3 "Equation 3 ‣ 2.1 Full-batch setting and main intuition ‣ 2 The Bloop Algorithm ‣ Careful with that Scalpel: Improving Gradient Surgery with an EMA").

## 3 Theoretical Analysis

This section aims at understanding the theoretical properties of the proposed direction in the full-batch and the mini-batch settings by linking it with the simple bilevel problem ([Equation 2](#S1.E2 "Equation 2 ‣ 1 Introduction ‣ Careful with that Scalpel: Improving Gradient Surgery with an EMA")). All the proofs are deferred to [Appendix A](#A1 "Appendix A Convergence analysis ‣ Careful with that Scalpel: Improving Gradient Surgery with an EMA").

### 3.1 Approximate stationary points of Bloop

At a solution to the simple bilevel problem, we have ∇Lmain​(θ)\=0\\nabla L\_{{\\mathrm{main}}}(\\theta)=0, hence the solutions to the bilevel problem are also solutions of

minLaux(θ) s.t. ∇Lmain(θ)\=0.\\min L\_{\\mathrm{aux}}(\\theta)\\text{ s.t. }\\nabla L\_{{\\mathrm{main}}}(\\theta)=0.

The Lagrangian for this equation is ℒ⁡(θ,v)\=Laux​(θ)−⟨v,∇Lmain​(θ)⟩\\mathcal{L}(\\theta,v)=L\_{\\mathrm{aux}}(\\theta)-\\langle v,\\nabla L\_{{\\mathrm{main}}}(\\theta)\\rangle with v∈ℝpv\\in\\mathbb{R}^{p} the Lagrange multiplier. Accordingly, the first-order optimality conditions are gmain\=0g\_{\\mathrm{main}}=0 and that there exists vv such that gaux\=∇2Lmain​(θ)​vg\_{\\mathrm{aux}}=\\nabla^{2}L\_{\\mathrm{main}}(\\theta)v. A first natural question to ask is whether the direction that we propose in [Equation 3](#S2.E3 "Equation 3 ‣ 2.1 Full-batch setting and main intuition ‣ 2 The Bloop Algorithm ‣ Careful with that Scalpel: Improving Gradient Surgery with an EMA") cancels at these points. However, the projection is ill-defined when gmain\=0g\_{{\\mathrm{main}}}=0. We thus assume that ‖gmain‖\\|g\_{{\\mathrm{main}}}\\| is positive hereinafter and focus on the case where dd is small but non-zero.11 1 Although we can simply set d\=λ​gauxd=\\lambda g\_{\\mathrm{aux}} when gmain\=0g\_{\\mathrm{main}}=0, the study of this particular case is straightforward and gives little insight on the general case. We therefore omit it here. To analyze this, we introduce the following assumption.

###### Assumption 1 (Local Error Bound [Luo & Tseng, 1993](#bib.bib25 "")).

There exists c\>0c>0 such that for ε\\varepsilon small enough and for any θ\\theta satisfying ‖gmain​(θ)‖≤ε\\|g\_{\\mathrm{main}}(\\theta)\\|\\leq\\varepsilon, we have

Dist⁡(θ,∇Lmain−1​({0}))≤c​‖gmain‖.\\mathrm{Dist}(\\theta,\\nabla L\_{{\\mathrm{main}}}^{-1}(\\{0\\}))\\leq c\\|g\_{\\mathrm{main}}\\|.

This local error bound condition is implied by a local Polyak-Lojasiewicz inequality, which is verified, for instance, for overparameterized least-squares and some neural network loss functions ([Liu et al., 2022](#bib.bib24 "")). With this in hand, we are now ready to present our result regarding the approximate first-order stationary points of the full-batch Bloop method.

###### Proposition 1 (Stationary points).

If dd in [Equation 3](#S2.E3 "Equation 3 ‣ 2.1 Full-batch setting and main intuition ‣ 2 The Bloop Algorithm ‣ Careful with that Scalpel: Improving Gradient Surgery with an EMA") is such that ‖d‖≤ε\\|d\\|\\leq\\varepsilon, then we have ‖gmain‖≤ε\\|g\_{\\mathrm{main}}\\|\\leq\\varepsilon. Moreover if [Assumption 1](#Thmtheorem1 "Assumption 1 (Local Error Bound , ). ‣ 3.1 Approximate stationary points of Bloop ‣ 3 Theoretical Analysis ‣ Careful with that Scalpel: Improving Gradient Surgery with an EMA") holds, the Hessian of LmainL\_{{\\mathrm{main}}} is M−M-Lipschitz, and ε\\varepsilon is small enough, then there exists v∈ℝpv\\in\\mathbb{R}^{p} such that

‖gaux−∇2Lmain​(θ)​v‖≤(λ−1+M​c2​‖gaux‖/2)​ε.\\|g\_{\\mathrm{aux}}-\\nabla^{2}L\_{\\mathrm{main}}(\\theta)v\\|\\leq(\\lambda^{-1}+Mc^{2}\\|g\_{\\mathrm{aux}}\\|/2)\\varepsilon.

Conversely, given a point θ∗\\theta^{\*} that satisfies the first order optimality conditions of [Equation 2](#S1.E2 "Equation 2 ‣ 1 Introduction ‣ Careful with that Scalpel: Improving Gradient Surgery with an EMA"), we have that limε→0d⁡(θ∗+ε​v)\=0\\lim\_{\\varepsilon\\to 0}d(\\theta^{\*}+\\varepsilon v)=0 where vv is the Lagrange multiplier.

In short, [Proposition 1](#Thmproposition1 "Proposition 1 (Stationary points). ‣ 3.1 Approximate stationary points of Bloop ‣ 3 Theoretical Analysis ‣ Careful with that Scalpel: Improving Gradient Surgery with an EMA") relates the (approximate) stationary points of Bloop to the (approximate) stationary points of the bilevel problem. Moreover, as an immediate consequence of the proposition, we see that we additionally assume LauxL\_{\\mathrm{aux}} to be Lipschitz continuous, the limit points of Bloop must be stationary points of the simple bilevel problem.

### 3.2 Convergence of stochastic Bloop

Our main theorem is a convergence result of the *stochastic version* of Bloop. It clearly highlights the role of the EMA: without EMA, obtaining such results would be impossible.

###### Theorem 2 (Convergence of Bloop).

Consider the Bloop method in the stochastic setting with the SGD optimizer. Let ρ\\rho be the EMA parameter and η\\eta be the step-size of the algorithm. Assume that (i) LmainL\_{{\\mathrm{main}}} is LL-smooth, (ii) the stochastic directions are uniformly bounded, i.e., ‖dt‖≤D\\|d^{t}\\|\\leq D for all tt, (iii) the variance of the gradients of LmainL\_{{\\mathrm{main}}} is bounded with 𝔼i\[∥∇Lmaini(θ)−∇Lmain(θ)∥2≤C2\\mathbb{E}\_{i}\[\\|\\nabla L\_{{\\mathrm{main}}}^{i}(\\theta)-\\nabla L\_{{\\mathrm{main}}}(\\theta)\\|^{2}\\leq C^{2}, and (iv) the auxiliary gradients are bounded as ‖∇Laux​(θ)‖≤B\\|\\nabla L\_{\\mathrm{aux}}(\\theta)\\|\\leq B. Then, for a number of iterations TT, taking a step size η≃T−34\\eta\\simeq T^{-\\frac{3}{4}} and an EMA parameter ρ≃η23\\rho\\simeq\\eta^{\\frac{2}{3}} gives

1T​∑t\=0T−1𝔼⁡\[‖∇Lmain​(θt)‖2\]\=O⁡(T−14)\\frac{1}{T}\\sum\_{t=0}^{T-1}\\mathbb{E}\[\\|\\nabla L\_{{\\mathrm{main}}}(\\theta^{t})\\|^{2}\]=O(T^{-\\frac{1}{4}})

If LmainL\_{{\\mathrm{main}}} is additionally μ\\mu-PL ([Karimi et al., 2016](#bib.bib18 "")), we have

𝔼⁡\[Lmain​(θT)−min⁡Lmain\]≤(1−2​η​μ)T​Lmain​(θ0)+O⁡(η13).\\mathbb{E}\[L\_{{\\mathrm{main}}}(\\theta^{T})-\\min L\_{{\\mathrm{main}}}\]\\leq(1-2\\eta\\mu)^{T}L\_{{\\mathrm{main}}}(\\theta^{0})+O(\\eta^{\\frac{1}{3}}).

[Theorem 2](#Thmtheorem2 "Theorem 2 (Convergence of Bloop). ‣ 3.2 Convergence of stochastic Bloop ‣ 3 Theoretical Analysis ‣ Careful with that Scalpel: Improving Gradient Surgery with an EMA") demonstrates the convergence of stochastic Bloop either in terms of the expected gradient norm or the expected optimiality gap. In spirit, this suggests that the Bloop iterate would end up being arbitrarily close to the stationary points of LmainL\_{{\\mathrm{main}}}. The theorem also instructs us on the role of the EMA coefficient ρ\\rho compared to the learning rate η\\eta. We see that we should take ρ\\rho to be slightly larger than η\\eta: in this regime, the gradient EMA gtrainEMAg\_{\\mathrm{train}}^{\\mathrm{EMA}} is a good approximation of gtraing\_{{\\mathrm{train}}}.

Also note that this result differs significantly from those obtained in the multi-task learning literature, which show convergence of the algorithms to points where either both losses are minimized or where their gradients are opposed ([Yu et al., 2020](#bib.bib35 "")). Here, even in the extreme case where losses are the exact opposite (Laux\=−LmainL\_{\\mathrm{aux}}=-L\_{{\\mathrm{main}}}), full-batch Bloop provably converges to the minimizers of LmainL\_{{\\mathrm{main}}} under PL condition. This is not a surprise since in that case, the projection π⁡(gaux,gmain)\\pi(g\_{\\mathrm{aux}},g\_{\\mathrm{main}}) cancels and the iterates of Bloop are that of gradient descent on LmainL\_{{\\mathrm{main}}}.

Unlike [Gong & Liu (2021)](#bib.bib12 ""), we do not demonstrate the convergence of our algorithm to the KKT points of the simple bilevel problem. Our results are thus weaker in that regard, albeit in a different setting since [Gong & Liu (2021)](#bib.bib12 "") are not in the stochastic case.

### 3.3 Conditioning compared to regularization method

We illustrate below that the regularization method can lead to poorly conditioned problems, resulting in hard optimization problems, while our method alleviates this. For this, we take the following simple 22D example, where θ\=(a,b)\\theta=(a,b):

Lmain​(θ)\=12​a2,Laux​(θ)\=12​((a−1)2+b2).L\_{{\\mathrm{main}}}(\\theta)=\\frac{1}{2}a^{2}\\,,~~L\_{\\mathrm{aux}}(\\theta)=\\frac{1}{2}((a-1)^{2}+b^{2}).

The solution to the bilevel problem is θ∗\=0\\theta^{\*}=0, while the solution to the regularized problem is θ\=(λ/(1+λ),0)\\theta=(\\lambda/(1+\\lambda),0). We recover the same solution in the limit λ→0\\lambda\\to 0. However, the Hessian of the regularized problem is 𝐝𝐢𝐚𝐠⁡(1+λ,λ)\\diag(1+\\lambda,\\lambda); hence the conditioning of the regularized problem is 1+1/λ1+1/\\lambda which goes to infinity as λ→0\\lambda\\to 0. In view of this, the regularized method either converges to a point far from the solution (λ\\lambda large) or converges slowly (λ\\lambda small). On the contrary, the projection method goes in the direction d\=(a,λ​b)d=(a,\\lambda b). This is equivalent to gradient descent on a quadratic loss with the correct θ∗\\theta^{\*} minimizer — *regardless of λ\\lambda* — and Hessian equal to 𝐝𝐢𝐚𝐠⁡(1,λ)\\diag(1,\\lambda), which is well conditioned when λ\\lambda is not too far from 11.

Table 1: Comparison of similar gradient surgery methods for the two tasks setting. For brevity, we write gm:=gmaing\_{\\mathrm{m}}:=g\_{\\mathrm{main}} and ϕ:=cos⁡(gm,gaux)\=⟨gm,gaux⟩‖gm‖​‖gaux‖\\phi:=\\cos(g\_{\\mathrm{m}},g\_{\\mathrm{aux}})=\\frac{\\langle g\_{\\mathrm{m}},g\_{\\mathrm{aux}}\\rangle}{\\|g\_{\\mathrm{m}}\\|\\|g\_{\\mathrm{aux}}\\|}. (⋅)¯\\bar{(\\cdot)} indicates that EMA has been applied, and ψ\\psi is a dynamic barrier function described in ([Gong & Liu, 2021](#bib.bib12 "")).

Method

Modified Direction

Bloop (ours)

gm+λ⁡(gaux−⟨gaux,g¯m⟩‖g¯m‖2​g¯m)g\_{\\mathrm{m}}+\\lambda\\left(g\_{\\mathrm{aux}}-\\frac{\\langle g\_{\\mathrm{aux}},\\bar{g}\_{\\mathrm{m}}\\rangle}{\\|\\bar{g}\_{\\mathrm{m}}\\|^{2}}\\bar{g}\_{\\mathrm{m}}\\right)

Mixed (Regularized)

gm+λ​gauxg\_{\\mathrm{m}}+\\lambda g\_{\\mathrm{aux}}

A-GEM

[Chaudhry et al. (2018)](#bib.bib4 "")

gm−min⁡(0,⟨gm,gaux⟩)‖gaux‖2​gauxg\_{\\mathrm{m}}-\\frac{\\min(0,\\langle g\_{\\mathrm{m}},g\_{\\mathrm{aux}}\\rangle)}{\\|g\_{\\mathrm{aux}}\\|^{2}}g\_{\\mathrm{aux}}

Dynamic Barrier

[Gong & Liu (2021)](#bib.bib12 "")

gaux+max⁡(0,ψ⁡(θ)−⟨gm,gaux⟩‖gm‖2)​gmg\_{\\mathrm{aux}}+\\max(0,\\frac{\\psi(\\theta)-\\langle g\_{\\mathrm{m}},g\_{\\mathrm{aux}}\\rangle}{\\|g\_{\\mathrm{m}}\\|^{2}})g\_{\\mathrm{m}}

MTL-MOO

[Sener & Koltun (2018)](#bib.bib29 "")

⟨gm−gaux,gaux⟩‖gm−gaux‖2​gm+\\frac{\\langle g\_{\\mathrm{m}}-g\_{\\mathrm{aux}},g\_{\\mathrm{aux}}\\rangle}{\\|g\_{\\mathrm{m}}-g\_{\\mathrm{aux}}\\|^{2}}g\_{\\mathrm{m}}+

(1−⟨gm−gaux,gaux⟩‖gm−gaux‖2)​gaux(1-\\frac{\\langle g\_{\\mathrm{m}}-g\_{\\mathrm{aux}},g\_{\\mathrm{aux}}\\rangle}{\\|g\_{\\mathrm{m}}-g\_{\\mathrm{aux}}\\|^{2}})g\_{\\mathrm{aux}}

Cosine Similarity

[Du et al. (2018)](#bib.bib10 "")

gm+gaux​max⁡(0,ϕ)g\_{\\mathrm{m}}+g\_{\\mathrm{aux}}\\max(0,\\phi)

GradVac

[Wang & Tsvetkov (2021)](#bib.bib33 "")

gm+‖gm‖​(ϕ¯​1−ϕ2−ϕ​1−ϕ¯2)‖gaux​1−ϕ¯2‖g\_{\\mathrm{m}}+\\frac{\\|g\_{\\mathrm{m}}\\|\\left(\\bar{\\phi}\\sqrt{1-\\phi^{2}}-\\phi\\sqrt{1-\\bar{\\phi}^{2}}\\right)}{\\|g\_{\\mathrm{aux}}\\sqrt{1-\\bar{\\phi}^{2}}\\|}

PCGrad

[Yu et al. (2020)](#bib.bib35 "")

gm−min⁡(0,⟨gaux,gm⟩)​gm‖gm‖2g\_{\\mathrm{m}}-\\min(0,\\langle g\_{\\mathrm{aux}},g\_{\\mathrm{m}}\\rangle)\\frac{g\_{\\mathrm{m}}}{\\|g\_{\\mathrm{m}}\\|^{2}}

+gaux−min⁡(0,⟨gaux,gm⟩)​gaux‖gaux‖2+g\_{\\mathrm{aux}}-\\min(0,\\langle g\_{\\mathrm{aux}},g\_{\\mathrm{m}}\\rangle)\\frac{g\_{\\mathrm{aux}}}{\\|g\_{\\mathrm{aux}}\\|^{2}}

Meta-Balance

[He et al. (2022)](#bib.bib15 "")

gm+‖gm‖‖gaux‖​gauxg\_{\\mathrm{m}}+\\frac{\\|g\_{\\mathrm{m}}\\|}{\\|g\_{\\mathrm{aux}}\\|}g\_{\\mathrm{aux}}

## 4 Related Works

Our work sits at the intersection of two fields of machine learning: the solution of the simple bilevel problem and multi-task learning. There are however a number of differences between the two. In particular, in the multi-task learning problem each task is considered jointly whereas in the bilevel setting there is a hierarchy to the primary and auxiliary objectives. Another key difference is in the notion of task versus auxiliary objective. A task typically requires a dataset as input, whereas an auxiliary objective is more general and can incorporate losses without the need for data, such as the L2L^{2} norm in weight decay.

Given the similarity, a number of gradient surgery methods that have been proposed in multi-task literature can be used to minimize both the main and the auxiliary objectives. We summarize the most relevant ones in [Table 1](#S3.T1 "In 3.3 Conditioning compared to regularization method ‣ 3 Theoretical Analysis ‣ Careful with that Scalpel: Improving Gradient Surgery with an EMA"). Some works try to leverage the auxiliary loss to obtain improvements on the main loss only ([Du et al., 2018](#bib.bib10 ""); [Dery et al., 2021](#bib.bib9 "")).

The Dynamic Barrier (DB) algorithm of [Gong & Liu (2021)](#bib.bib12 ""), as detailed in [Table 1](#S3.T1 "In 3.3 Conditioning compared to regularization method ‣ 3 Theoretical Analysis ‣ Careful with that Scalpel: Improving Gradient Surgery with an EMA"), uses a similar orthogonal projection as in our proposal. It provably solves the bilevel problem. However, DB includes an additional barrier function, ϕ\\phi e.g. ϕ\=‖gaux‖2\\phi=\\|g\_{\\mathrm{aux}}\\|^{2}, to control the trade-off between objectives, whereas we use a scalar, λ\\lambda, similar to regularization methods, for this purpose. The other main differences between our proposal and the DB method are that we always use the projection, rather than conditioning on ⟨gm,gaux⟩\\langle g\_{\\mathrm{m}},g\_{\\mathrm{aux}}\\rangle, and most importantly, we use an EMA of main gradients to compute the projection, rather than the stochastic gradient. With ϕ\=‖gaux‖2\\phi=\\|g\_{\\mathrm{aux}}\\|^{2} and without the conditional update or EMA, the approaches would be the same. [Gong & Liu (2021)](#bib.bib12 "") do not discuss stochastic extensions of the method, which is of key importance to practitioners.

[Yu et al. (2020)](#bib.bib35 "") propose PCGrad, which, as shown in [Table 1](#S3.T1 "In 3.3 Conditioning compared to regularization method ‣ 3 Theoretical Analysis ‣ Careful with that Scalpel: Improving Gradient Surgery with an EMA"), can be regarded as a symmetrized version of our method. Unlike our method, the projection is again conditioned. Concretely, the parameters are updated in the direction of the combined gradient gmain+gauxg\_{\\mathrm{main}}+g\_{\\mathrm{aux}} when they are aligned, and projections are performed when this is not the case. The gradient alignment condition and the symmetry between the gradients implies that the algorithm does not solve the bilevel problem; instead ([Yu et al., 2020](#bib.bib35 ""), Thm.1) show that it minimizes the sum of the two losses or finds a point where gauxg\_{\\mathrm{aux}} and gmaing\_{\\mathrm{main}} go in opposite directions. Similarly to the DB method, no EMA is used in the projection.

## 5 Experiments

(a) Training an MLP on MNIST with an auxiliary loss that is a proxy for its Lipschitz constant.

(b) Training a ResNet50 on Imagenet with squared L2 norm as the auxiliary loss.

Figure 3: Trade-offs between the main and the auxiliary objectives in problems where the auxiliary loss is used to impose an explicit bias on the neural network. The symbols correspond to the parameters reached at the end of training and form a Pareto front, the transparent curves are the training trajectories. Bloop achieves a better trade-off than the other methods, which all perform similarly here. 

In this section we demonstrate the effectiveness of Bloop via numerical experiments on problems of three distinct categories: the use of auxiliary loss for imposing an explicit bias, multi-task learning, and joint dataset training. For each of these experiments, we use an optimizer with hyperparameters that work well for the minimization of solely the main loss, and never change these hyperparameters. As for the EMA parameter of Bloop, we take it as ρ\=0.01\\rho=0.01 in all experiments unless otherwise stated. Further experimental details can be found in [Appendix B](#A2 "Appendix B Experimental Details ‣ Careful with that Scalpel: Improving Gradient Surgery with an EMA").

Note that Bloop incurs a negligible training cost compared to the standard regularized training, as it only requires two additional dot products in the parameter space.

The code for the Bloop method is available at [https://github.com/apple/ml-bloop](https://github.com/apple/ml-bloop "").

### 5.1 Baselines and evaluation

We compare Bloop ([Algorithm 1](#alg1 "In 2.2 Stochastic extension for large-scale problems ‣ 2 The Bloop Algorithm ‣ Careful with that Scalpel: Improving Gradient Surgery with an EMA")) to other popular gradient surgery methods that follow a similar design. We focus on the stochastic setup where we only have access to the gradients over a mini-batch of samples at each iteration.

Mixed. This method minimizes the regularized objective Lmain+λ​LauxL\_{{\\mathrm{main}}}+\\lambda L\_{\\mathrm{aux}} with the direction d\=gmainbatch+λ​gauxbatchd=g^{\\mathrm{batch}}\_{\\mathrm{main}}+\\lambda g^{\\mathrm{batch}}\_{\\mathrm{aux}}.

Dynamic Barrier (DB). The original formulation of the DB method requires both an estimate of a lower bound on LmainL\_{{\\mathrm{main}}}, as well as an estimate of Lmain​(θ)L\_{{\\mathrm{main}}}(\\theta), which are cumbersome to estimate in deep learning setups. We therefore forgo this part of the algorithm and instead incorporate the scaling factor λ\\lambda to control the trade-off. We also replace the gradients in the original method by stochastic gradients. This results in the update direction d\=μ​gmainbatch+λ​gauxbatchd=\\mu g^{\\mathrm{batch}}\_{\\mathrm{main}}+\\lambda g^{\\mathrm{batch}}\_{\\mathrm{aux}} where μ\=max⁡(1−λ​⟨gmainbatch,gauxbatch⟩‖gmainbatch‖2,0).\\mu=\\max\\left(1-\\lambda\\frac{\\langle g^{\\mathrm{batch}}\_{\\mathrm{main}},g^{\\mathrm{batch}}\_{\\mathrm{aux}}\\rangle}{\\|g^{\\mathrm{batch}}\_{\\mathrm{main}}\\|^{2}},0\\right).

PCGrad. Being motivated from a multi-task perspective, the original formulation of PCGrad does not use the scaling factor λ\\lambda. By incorporating this factor, the update direction becomes d\=gmainbatch+λ​gauxbatchd=g^{\\mathrm{batch}}\_{\\mathrm{main}}+\\lambda g^{\\mathrm{batch}}\_{\\mathrm{aux}} if ⟨gmainbatch,gauxbatch⟩\>0\\langle g^{\\mathrm{batch}}\_{\\mathrm{main}},g^{\\mathrm{batch}}\_{\\mathrm{aux}}\\rangle>0, and d\=π⁡(gmainbatch,gauxbatch)+λ​π​(gauxbatch,gmainbatch)d=\\pi(g^{\\mathrm{batch}}\_{\\mathrm{main}},g^{\\mathrm{batch}}\_{\\mathrm{aux}})+\\lambda\\pi(g^{\\mathrm{batch}}\_{\\mathrm{aux}},g^{\\mathrm{batch}}\_{\\mathrm{main}}) otherwise.

Evaluation of the algorithms. To provide a comprehensive insight into how the algorithm design affects the training dynamics, we report the metrics on both the training and the test sets. Moreover, we trace the evolution of these metrics along training.

Pareto fronts. All algorithms that we consider here have thus a parameter λ\\lambda that trades-off between the train and the auxiliary losses. After a fixed number of iterations, the algorithm algo finds a final parameter θalgo​(λ)\\theta^{\\texttt{algo}}(\\lambda) that explicitly depends on λ\\lambda. Generally, Lmain​(θalgo​(λ))L\_{{\\mathrm{main}}}(\\theta^{\\texttt{algo}}(\\lambda)) is a decreasing function of λ\\lambda while Laux​(θalgo​(λ))L\_{\\mathrm{aux}}(\\theta^{\\texttt{algo}}(\\lambda)) is increasing with λ\\lambda. We can then vary λ\\lambda to get the set of pairs 𝒫(algo)\={(Lmain(θalgo(λ)),Laux((θalgo(λ)))|λ≥0}\\mathcal{P}(\\texttt{algo})=\\{(L\_{{\\mathrm{main}}}(\\theta^{\\texttt{algo}}(\\lambda)),L\_{\\mathrm{aux}}((\\theta^{\\texttt{algo}}(\\lambda)))|\\kern 5.0pt\\lambda\\geq 0\\}, called the Pareto front of algo.

### 5.2 Imposing an explicit bias during training

To begin with, we first investigate the situation where the auxiliary objective is used to enforce a certain desirable property (bias) on the neural network.

Training smooth neural networks. Following our discussion in [Section 1](#S1 "1 Introduction ‣ Careful with that Scalpel: Improving Gradient Surgery with an EMA"), we explore the potential of Bloop in training smooth neural networks. For this, we use the MNIST dataset ([LeCun et al., 2010](#bib.bib21 "")) and an MLP of two hidden layers. With this minimal architecture, a simple induction argument shows that the Lipschitz constant of the network is upper-bounded by ∏l\=1L‖Wl‖2\\prod\_{l=1}^{L}\\|W\_{l}\\|\_{2}, where WlW\_{l} is the weight matrix of the l−l-th linear layer, ∥⋅∥2\\|\\cdot\\|\_{2} is the spectral norm, and L\=3L=3 is the number of layers. We thus define the auxiliary loss as Laux\=log⁡(∏l\=1L‖Wl‖2)L\_{\\mathrm{aux}}=\\log(\\prod\_{l=1}^{L}\\|W\_{l}\\|\_{2}). The use of logarithm here makes training easier. On the other hand, we use the standard cross-entropy loss as the main loss.

Training networks with small weights. For this experiment, we train a ResNet50 using standard cross-entropy loss on Imagenet, and try to simultaneously achieve a low ℓ2\\ell\_{2} norm of the parameters of the network. The auxiliary loss is therefore Laux​(θ)\=12​‖θ‖2L\_{\\mathrm{aux}}(\\theta)=\\frac{1}{2}\\|\\theta\\|^{2}. In that case, the mixed method is similar to training with a weight decay λ\\lambda.

Results. The results are reported in [Figure 3](#S5.F3 "In 5 Experiments ‣ Careful with that Scalpel: Improving Gradient Surgery with an EMA"). We see Bloop induces training trajectories that are fundamentally different from all other methods, and leads to better Pareto fronts when trading off the main and the auxiliary training losses. In both experiments, we observe that Bloop leads to a significantly better Pareto front when looking at the training loss ([3(a)](#S5.F3.sf1 "Figure 3(a) ‣ Figure 3 ‣ 5 Experiments ‣ Careful with that Scalpel: Improving Gradient Surgery with an EMA"), left and [3(b)](#S5.F3.sf2 "Figure 3(b) ‣ Figure 3 ‣ 5 Experiments ‣ Careful with that Scalpel: Improving Gradient Surgery with an EMA"), left). Whether this translates or not to a better Pareto front in terms of test loss is problem dependent: in [3(a)](#S5.F3.sf1 "Figure 3(a) ‣ Figure 3 ‣ 5 Experiments ‣ Careful with that Scalpel: Improving Gradient Surgery with an EMA"), right, the Pareto front of Bloop is only slightly better than that of the other methods, while in [3(b)](#S5.F3.sf2 "Figure 3(b) ‣ Figure 3 ‣ 5 Experiments ‣ Careful with that Scalpel: Improving Gradient Surgery with an EMA"), right, it is significantly better.

Figure 4: Trade-off between the performances in the Cifar10Mnist multi-task learning problem. Bloop gives a better Pareto front.

(a) Results on the language modeling task. The main, pre-training loss is the next-token-prediction loss over the large c4 dataset, while the auxiliary, specialization loss is the next-token-prediction loss over the small RCV-1 dataset.

(b) Results on the translation task. The main pre-training loss is the translation loss over the large paracrawl dataset, while the auxiliary specialization loss is the translation loss over the small WMT dataset.

Figure 5: Trade-offs between the main and the auxiliary objectives in problems in natural language processing experiments with transformer models, where the main loss is the loss over a large dataset and the auxiliary loss is a loss over a small dataset that can be overfitted easily. We observe that Bloop gets a significantly better Pareto front than all other methods, which perform similarly to the mixed method. Bloop gains in terms of optimization on the training losses transfer to the evaluation losses. 

### 5.3 Multi-task learning

As discussed in [Section 1](#S1 "1 Introduction ‣ Careful with that Scalpel: Improving Gradient Surgery with an EMA"), multi-task learning represents another typical scenario in which such auxiliary objectives emerge. Following [Hotegni et al. (2023)](#bib.bib17 ""), we construct a Cifar10Mnist dataset by overlapping digits from MNIST on images from CIFAR-10 ([Krizhevsky et al., 2009](#bib.bib20 "")) — see [Figure 9](#A2.F9 "In Appendix B Experimental Details ‣ Careful with that Scalpel: Improving Gradient Surgery with an EMA") in [Appendix B](#A2 "Appendix B Experimental Details ‣ Careful with that Scalpel: Improving Gradient Surgery with an EMA") for an illustration. The main and the auxiliary tasks correpond respectively to identifying the label for the background CIFAR-10 image and for the MNIST digit. There is a natural hierachy between the two tasks here because identifying the CIFAR-10 label is more difficult than identifying the MNIST one. For this dataset, we train a ResNet18 with two classification heads to minimize the two cross-entropy losses. In this experiment, we found that taking ρ\=0.001\\rho=0.001 for Bloop gave better results.

Results. As shown in [Figure 4](#S5.F4 "In 5.2 Imposing an explicit bias during training ‣ 5 Experiments ‣ Careful with that Scalpel: Improving Gradient Surgery with an EMA"), the trajectories of Bloop are again much more different than those of the other methods, which share quite similar behaviors. Moreover, Bloop gets a slightly improved Pareto front over those methods.

### 5.4 Joint training on two datasets

Figure 6: A different look at the results in [5(a)](#S5.F5.sf1 "Figure 5(a) ‣ Figure 5 ‣ 5.2 Imposing an explicit bias during training ‣ 5 Experiments ‣ Careful with that Scalpel: Improving Gradient Surgery with an EMA"). We display the value of the final mixed loss (1−t)​Lmain+t​Laux(1-t)L\_{\\mathrm{main}}+tL\_{\\mathrm{aux}} for the different values of λ\\lambda in the algorithms we used. Bold lines correspond to evaluation loss, while dotted lines correspond to train loss. We see that Bloop allows to get to a lower mixed loss when tt is small. This is a striking phenomenon, since the mixed method directly minimizes that loss.

With the advent of large foundation models, it becomes increasingly common to train a model on multiple data sources ([Gunasekar et al., 2023](#bib.bib14 ""); [Sun et al., 2023](#bib.bib30 ""); [Xu et al., 2023](#bib.bib34 ""); [Oquab et al., 2024](#bib.bib26 "")). Yet, these datasets could have intrinsically different characteristics, and it may be natural to prioritize one over another, for instance when one dataset has far more samples than another. We explore the benefit of Bloop in such multi-dataset setting. Our experimental setup is similar to that of [Grangier et al. (2023)](#bib.bib13 "").

Transformer pre-training. We consider the problem of performing next-token-prediction with a decoder-only transformer on text data. The network is a transformer with 12 decoder layers, 8 attention heads, a residual dimension of 256, and a feed-forward latent dimension of 1024. The main loss corresponds to the prediction loss over a large pre-training dataset, while the auxiliary loss corresponds to that on a smaller but higher-quality dataset. Due to the lack of data, training only on the small high-quality dataset leads to severe overfitting and poor performance; hence, we resort to training on both datasets, using the proposed baselines or Bloop. For the training set, we use 3030M examples from the c4 dataset ([Raffel et al., 2020](#bib.bib27 "")), while the auxiliary loss corresponds to 2020K examples from the RCV-1 dataset ([Lewis et al., 2004](#bib.bib22 "")).

Translation. In this experiment, we train a network to translate English into German. The network is a transformer with 6 encoder layers and 6 decoder layers, 16 attention heads, a residual dimension of 1,024, and a feed-forward latent dimension of 4,096. Like in the pre-training experiment, we have a large generic dataset, the Paracrawl dataset ([Bañón et al., 2020](#bib.bib1 "")), with 36m sentence pairs, which defines the main loss. The auxiliary loss is the loss over a smaller but higher quality dataset, the 2009-2019 WMT dataset, yielding 10k sentence pairs ([Farhad et al., 2021](#bib.bib11 "")). We use the 2020 WMT dataset (2k pairs) as an evaluation set.

Results. [Figure 5](#S5.F5 "Figure 5 ‣ 5.2 Imposing an explicit bias during training ‣ 5 Experiments ‣ Careful with that Scalpel: Improving Gradient Surgery with an EMA") displays the results. We observe siginificantly improved results for Bloop, which has once again a better Pareto front, and achieves smaller pre-training loss. These gains are kept when looking at the evaluation losses. [Figure 6](#S5.F6 "Figure 6 ‣ 5.4 Joint training on two datasets ‣ 5 Experiments ‣ Careful with that Scalpel: Improving Gradient Surgery with an EMA") gives a different perspective on those results.

Figure 7: Effect of the EMA parameter ρ\\rho on Bloop’s performance. We use the same next-token prediction losses as in [5(a)](#S5.F5.sf1 "Figure 5(a) ‣ Figure 5 ‣ 5.2 Imposing an explicit bias during training ‣ 5 Experiments ‣ Careful with that Scalpel: Improving Gradient Surgery with an EMA"), and display the training curves for a fixed λ\=0.2\\lambda=0.2.

### 5.5 Role of the EMA

We investigate the importance of the EMA parameter ρ\\rho in Bloop. As already seen in [Section 3](#S3 "3 Theoretical Analysis ‣ Careful with that Scalpel: Improving Gradient Surgery with an EMA"), it is critical from a theoretical point-of-view for the algorithm’s convergence. We further illustrate this via the transformer pre-training experiment with a fixed λ\=0.2\\lambda=0.2.

[Figure 7](#S5.F7 "Figure 7 ‣ 5.4 Joint training on two datasets ‣ 5 Experiments ‣ Careful with that Scalpel: Improving Gradient Surgery with an EMA")displays the results. We see that when the EMA is too small (ρ\=0.001\\rho=0.001), the value of gmainEMAg\_{\\mathrm{main}}^{\\mathrm{EMA}} is outdated compared to the current value of the gradient gmaing\_{\\mathrm{main}}, and therefore, the performance on both the main and auxiliary losses is bad. On the contrary, taking a too-large EMA (ρ\=0.9\\rho=0.9) means that gmainEMAg\_{\\mathrm{main}}^{\\mathrm{EMA}} has a high variance, and we recover a trajectory extremely similar to that of the mixed method. Choices between these two extremes (ρ\=0.01\\rho=0.01, or ρ\=0.1\\rho=0.1) lead to a tradeoff between main and auxiliary loss.

## Discussion

A striking phenomenon that we observe in all our experiments is that PCGrad and DB work very similarly to the mixed method. We posit that this observation is due to the high gradient variance coming from the main loss, which is also what our theory predicts. Adding an EMA to reduce this variance leads to the Bloop method, which here has a different behavior to the other methods, often leading to improved Pareto fronts.

In the Appendix [C](#A3 "Appendix C Additional Experiment ‣ Careful with that Scalpel: Improving Gradient Surgery with an EMA"), we describe an experiment where Bloop does not work better than the other methods. We attempted to train a ResNet to have a good performance on Imagenet and Cifar10, with a shared trunk and two classification heads. We found that all methods performed equally well; in that case, Bloop leads to the same Pareto front as the other method. Yet, once again, PCGrad and DB have the same practical performance as the mixed method.

Overall, adding an EMA to reduce variance in the projection direction is a simple idea that can have a big impact on gradient surgery methods.

## Acknowledgements

The authors thank Alaa El Nouby, David Grangier, Miguel Sarabia del Castillo, Arno Blaas, Jason Ramapuram, Dan Busbridge, Adam Golinski, Luca Zappella and Federico Danielli for fruitful discussions. The authors are indebted to David Grangier and Awni Hannun for their help with the codebase.

## Impact Statement

This paper presents work whose goal is to advance the field of Machine Learning. There are many potential societal consequences of our work, none which we feel must be specifically highlighted here.

## References

*   Bañón et al. (2020) Bañón, M., Chen, P., Haddow, B., Heafield, K., Hoang, H., Esplà-Gomis, M., Forcada, M., Kamran, A., Kirefu, F., Koehn, P., et al. Paracrawl: Web-scale acquisition of parallel corpora. Association for Computational Linguistics (ACL), 2020.
*   Cao et al. (2023) Cao, J., Jiang, R., Abolfazli, N., Hamedani, E. Y., and Mokhtari, A. Projection-free methods for stochastic simple bilevel optimization with convex lower-level problem. *arXiv preprint arXiv:2308.07536*, 2023.
*   Caruana (1997) Caruana, R. Multitask learning. *Machine learning*, 28:41–75, 1997.
*   Chaudhry et al. (2018) Chaudhry, A., Ranzato, M., Rohrbach, M., and Elhoseiny, M. Efficient lifelong learning with a-gem. In *International Conference on Learning Representations*, 2018.
*   Cisse et al. (2017) Cisse, M., Bojanowski, P., Grave, E., Dauphin, Y., and Usunier, N. Parseval networks: Improving robustness to adversarial examples. In *International conference on machine learning*, pp. 854–863. PMLR, 2017.
*   Cooper (2018) Cooper, Y. The loss landscape of overparameterized neural networks. *arXiv preprint arXiv:1804.10200*, 2018.
*   DeepMind et al. (2020) DeepMind, Babuschkin, I., Baumli, K., Bell, A., Bhupatiraju, S., Bruce, J., Buchlovsky, P., Budden, D., Cai, T., Clark, A., Danihelka, I., Dedieu, A., Fantacci, C., Godwin, J., Jones, C., Hemsley, R., Hennigan, T., Hessel, M., Hou, S., Kapturowski, S., Keck, T., Kemaev, I., King, M., Kunesch, M., Martens, L., Merzic, H., Mikulik, V., Norman, T., Papamakarios, G., Quan, J., Ring, R., Ruiz, F., Sanchez, A., Sartran, L., Schneider, R., Sezener, E., Spencer, S., Srinivasan, S., Stanojević, M., Stokowiec, W., Wang, L., Zhou, G., and Viola, F. The DeepMind JAX Ecosystem, 2020. URL [http://github.com/google-deepmind](http://github.com/google-deepmind "").
*   Dempe et al. (2010) Dempe, S., Dinh, N., and Dutta, J. Optimality conditions for a simple convex bilevel programming problem. *Variational Analysis and Generalized Differentiation in Optimization and Control: In Honor of Boris S. Mordukhovich*, pp. 149–161, 2010.
*   Dery et al. (2021) Dery, L. M., Dauphin, Y., and Grangier, D. Auxiliary task update decomposition: The good, the bad and the neutral. *arXiv preprint arXiv:2108.11346*, 2021.
*   Du et al. (2018) Du, Y., Czarnecki, W. M., Jayakumar, S. M., Farajtabar, M., Pascanu, R., and Lakshminarayanan, B. Adapting auxiliary losses using gradient similarity. *arXiv preprint arXiv:1812.02224*, 2018.
*   Farhad et al. (2021) Farhad, A., Arkady, A., Magdalena, B., Ondřej, B., Rajen, C., Vishrav, C., Costa-jussa, M. R., Cristina, E.-B., Angela, F., Christian, F., et al. Findings of the 2021 conference on machine translation (wmt21). In *Proceedings of the Sixth Conference on Machine Translation*, pp. 1–88. Association for Computational Linguistics, 2021.
*   Gong & Liu (2021) Gong, C. and Liu, X. Bi-objective trade-off with dynamic barrier gradient descent. *NeurIPS 2021*, 2021.
*   Grangier et al. (2023) Grangier, D., Ablin, P., and Hannun, A. Adaptive training distributions with scalable online bilevel optimization. *arXiv preprint arXiv:2311.11973*, 2023.
*   Gunasekar et al. (2023) Gunasekar, S., Zhang, Y., Aneja, J., Mendes, C. C. T., Del Giorno, A., Gopi, S., Javaheripi, M., Kauffmann, P., de Rosa, G., Saarikivi, O., et al. Textbooks are all you need. *arXiv preprint arXiv:2306.11644*, 2023.
*   He et al. (2022) He, Y., Feng, X., Cheng, C., Ji, G., Guo, Y., and Caverlee, J. Metabalance: improving multi-task recommendations via adapting gradient magnitudes of auxiliary tasks. In *Proceedings of the ACM Web Conference 2022*, pp. 2205–2215, 2022.
*   (16) Heek, J., Levskaya, A., Oliver, A., Ritter, M., Rondepierre, B., Steiner, A., and van Zee, M. Flax: A neural network library and ecosystem for jax, 2020. *URL http://github. com/google/flax*, 1.
*   Hotegni et al. (2023) Hotegni, S. S., Berkemeier, M., and Peitz, S. Multi-objective optimization for sparse deep multi-task learning. *arXiv preprint arXiv:2308.12243*, 2023.
*   Karimi et al. (2016) Karimi, H., Nutini, J., and Schmidt, M. Linear convergence of gradient and proximal-gradient methods under the polyak-łojasiewicz condition. In *Machine Learning and Knowledge Discovery in Databases: European Conference, ECML PKDD 2016, Riva del Garda, Italy, September 19-23, 2016, Proceedings, Part I 16*, pp. 795–811. Springer, 2016.
*   Kingma & Ba (2014) Kingma, D. P. and Ba, J. Adam: A method for stochastic optimization. *arXiv preprint arXiv:1412.6980*, 2014.
*   Krizhevsky et al. (2009) Krizhevsky, A., Hinton, G., et al. Learning multiple layers of features from tiny images. 2009.
*   LeCun et al. (2010) LeCun, Y., Cortes, C., and Burges, C. Mnist handwritten digit database. *ATT Labs \[Online\]. Available: http://yann.lecun.com/exdb/mnist*, 2, 2010.
*   Lewis et al. (2004) Lewis, D. D., Yang, Y., Russell-Rose, T., and Li, F. Rcv1: A new benchmark collection for text categorization research. *Journal of machine learning research*, 5(Apr):361–397, 2004.
*   Li et al. (2018) Li, H., Xu, Z., Taylor, G., Studer, C., and Goldstein, T. Visualizing the loss landscape of neural nets. *Advances in neural information processing systems*, 31, 2018.
*   Liu et al. (2022) Liu, C., Zhu, L., and Belkin, M. Loss landscapes and optimization in over-parameterized non-linear systems and neural networks. *Applied and Computational Harmonic Analysis*, 59:85–116, 2022.
*   Luo & Tseng (1993) Luo, Z.-Q. and Tseng, P. Error bounds and convergence analysis of feasible descent methods: a general approach. *Annals of Operations Research*, 46(1):157–178, 1993.
*   Oquab et al. (2024) Oquab, M., Darcet, T., Moutakanni, T., Vo, H. V., Szafraniec, M., Khalidov, V., Fernandez, P., HAZIZA, D., Massa, F., El-Nouby, A., Assran, M., Ballas, N., Galuba, W., Howes, R., Huang, P.-Y., Li, S.-W., Misra, I., Rabbat, M., Sharma, V., Synnaeve, G., Xu, H., Jegou, H., Mairal, J., Labatut, P., Joulin, A., and Bojanowski, P. DINOv2: Learning robust visual features without supervision. *Transactions on Machine Learning Research*, 2024. ISSN 2835-8856. URL [https://openreview.net/forum?id=a68SUt6zFt](https://openreview.net/forum?id=a68SUt6zFt "").
*   Raffel et al. (2020) Raffel, C., Shazeer, N., Roberts, A., Lee, K., Narang, S., Matena, M., Zhou, Y., Li, W., and Liu, P. J. Exploring the limits of transfer learning with a unified text-to-text transformer. *The Journal of Machine Learning Research*, 21(1):5485–5551, 2020.
*   Sabach & Shtern (2017) Sabach, S. and Shtern, S. A first order method for solving convex bilevel optimization problems. *SIAM Journal on Optimization*, 27(2):640–660, 2017.
*   Sener & Koltun (2018) Sener, O. and Koltun, V. Multi-task learning as multi-objective optimization. *Advances in neural information processing systems*, 31, 2018.
*   Sun et al. (2023) Sun, Q., Cui, Y., Zhang, X., Zhang, F., Yu, Q., Luo, Z., Wang, Y., Rao, Y., Liu, J., Huang, T., et al. Generative multimodal models are in-context learners. *arXiv preprint arXiv:2312.13286*, 2023.
*   Terjék (2019) Terjék, D. Adversarial lipschitz regularization. *arXiv preprint arXiv:1907.05681*, 2019.
*   Tsuzuku et al. (2018) Tsuzuku, Y., Sato, I., and Sugiyama, M. Lipschitz-margin training: Scalable certification of perturbation invariance for deep neural networks. *Advances in neural information processing systems*, 31, 2018.
*   Wang & Tsvetkov (2021) Wang, Z. and Tsvetkov, Y. Gradient vaccine: Investigating and improving multi-task optimization in massively multilingual models. In *Proceedings of the International Conference on Learning Representations (ICLR)*, 2021.
*   Xu et al. (2023) Xu, H., Xie, S., Tan, X. E., Huang, P.-Y., Howes, R., Sharma, V., Li, S.-W., Ghosh, G., Zettlemoyer, L., and Feichtenhofer, C. Demystifying clip data. *arXiv preprint arXiv:2309.16671*, 2023.
*   Yu et al. (2020) Yu, T., Kumar, S., Gupta, A., Levine, S., Hausman, K., and Finn, C. Gradient surgery for multi-task learning. *Advances in Neural Information Processing Systems*, 33:5824–5836, 2020.

 

Appendix

  

## Appendix A Convergence analysis

In this appendix we provide proofs for the theoretical results of [Section 3](#S3 "3 Theoretical Analysis ‣ Careful with that Scalpel: Improving Gradient Surgery with an EMA").

### A.1 Proof of [Proposition 1](#Thmproposition1 "Proposition 1 (Stationary points). ‣ 3.1 Approximate stationary points of Bloop ‣ 3 Theoretical Analysis ‣ Careful with that Scalpel: Improving Gradient Surgery with an EMA")

In the following, we will prove the two implications in the proposition separately.

#### Small Bloop Vector →\\rightarrow Near-Stationary Point.

By orthogonality, we have ‖d‖2\=‖gmain‖2+λ2​‖π⁡(gaux,gmain)‖2\\|d\\|^{2}=\\|g\_{\\mathrm{main}}\\|^{2}+\\lambda^{2}\\|\\pi(g\_{\\mathrm{aux}};g\_{\\mathrm{main}})\\|^{2}. This implies immediately ‖gmain‖≤ε\\|g\_{\\mathrm{main}}\\|\\leq\\varepsilon and ‖π⁡(gaux,gmain)‖≤ε​λ−1\\|\\pi(g\_{\\mathrm{aux}};g\_{\\mathrm{main}})\\|\\leq\\varepsilon\\lambda^{-1} provided that ‖d‖≤ε\\|d\\|\\leq\\varepsilon.

Let us next consider the case where [Assumption 1](#Thmtheorem1 "Assumption 1 (Local Error Bound , ). ‣ 3.1 Approximate stationary points of Bloop ‣ 3 Theoretical Analysis ‣ Careful with that Scalpel: Improving Gradient Surgery with an EMA") holds and that the Hessian of LmainL\_{{\\mathrm{main}}} is M-Lipschitz continuous. With the local error bound, i.e., [Assumption 1](#Thmtheorem1 "Assumption 1 (Local Error Bound , ). ‣ 3.1 Approximate stationary points of Bloop ‣ 3 Theoretical Analysis ‣ Careful with that Scalpel: Improving Gradient Surgery with an EMA"), we know there exists θ∗\\theta^{\*} such that ∇Lmain​(θ∗)\=0\\nabla L\_{{\\mathrm{main}}}(\\theta^{\*})=0 and ‖θ−θ∗‖≤c​‖gmain‖\\|\\theta-\\theta^{\*}\\|\\leq c\\|g\_{\\mathrm{main}}\\|. Performing a Taylor expansion with Lagrange form of the remainder of order 2, we obtain

∇Lmain​(θ)\\displaystyle\\nabla L\_{\\text{main}}(\\theta)

\=∇Lmain​(θ∗)+∇2Lmain​(θ∗)​(θ−θ∗)+12​∇3Lmain​(θ′)​\[θ−θ∗,θ−θ∗\]\\displaystyle=\\nabla L\_{{\\mathrm{main}}}(\\theta^{\\ast})+\\nabla^{2}L\_{\\text{main}}(\\theta^{\\ast})(\\theta-\\theta^{\\ast})+\\frac{1}{2}\\nabla^{3}L\_{\\text{main}}(\\theta^{\\prime})\[\\theta-\\theta^{\\ast},\\theta-\\theta^{\\ast}\]

(7)

\=∇2Lmain​(θ∗)​(θ−θ∗)+12​∇3Lmain​(θ′)​\[θ−θ∗,θ−θ∗\],\\displaystyle=\\nabla^{2}L\_{\\text{main}}(\\theta^{\\ast})(\\theta-\\theta^{\\ast})+\\frac{1}{2}\\nabla^{3}L\_{\\text{main}}(\\theta^{\\prime})\[\\theta-\\theta^{\\ast},\\theta-\\theta^{\\ast}\], for some θ′\\theta^{\\prime} that lies on the line that connects θ\\theta and θ∗\\theta^{\\ast}. Using the M-Lipschitzness of ∇2Lmain\\nabla^{2}L\_{{\\mathrm{main}}}, the norm of r\=∇Lmain​(θ)−∇2Lmain​(θ)​(θ−θ∗)r=\\nabla L\_{{\\mathrm{main}}}(\\theta)-\\nabla^{2}L\_{{\\mathrm{main}}}(\\theta)(\\theta-\\theta^{\*}) can then be bounded by

‖r‖\=12​∇3Lmain​(θ′)​\[θ−θ∗,θ−θ∗\]≤M2​‖θ−θ∗‖2≤M​c22​‖gmain‖2.\\|r\\|=\\frac{1}{2}\\nabla^{3}L\_{\\text{main}}(\\theta^{\\prime})\[\\theta-\\theta^{\\ast},\\theta-\\theta^{\\ast}\]\\leq\\frac{M}{2}\\|\\theta-\\theta^{\*}\\|^{2}\\leq\\frac{Mc^{2}}{2}\\|g\_{\\mathrm{main}}\\|^{2}\\kern 5.0pt.

We now claim that the desired inequality holds true with

v\=⟨gaux,gmain⟩‖gmain‖2​(θ−θ∗).v=\\frac{\\langle g\_{\\mathrm{aux}},g\_{\\mathrm{main}}\\rangle}{\\|g\_{\\mathrm{main}}\\|^{2}}(\\theta-\\theta^{\*}).

For this, we decompose

⟨gaux,gmain⟩‖gmain‖2​gmain\=∇2Lmain​(θ)​v+⟨gaux,gmain⟩‖gmain‖2​r\\frac{\\langle g\_{\\mathrm{aux}},g\_{\\mathrm{main}}\\rangle}{\\|g\_{\\mathrm{main}}\\|^{2}}g\_{\\mathrm{main}}=\\nabla^{2}L\_{{\\mathrm{main}}}(\\theta)v+\\frac{\\langle g\_{\\mathrm{aux}},g\_{\\mathrm{main}}\\rangle}{\\|g\_{\\mathrm{main}}\\|^{2}}r

Subsequently, ‖gaux−∇2Lmain​(θ)​v‖\\displaystyle\\|g\_{\\mathrm{aux}}-\\nabla^{2}L\_{\\mathrm{main}}(\\theta)v\\|

≤‖gaux−⟨gaux,gmain⟩‖gmain‖2​gmain‖+‖⟨gaux,gmain⟩‖gmain‖2​r‖\\displaystyle\\leq\\left\\|g\_{\\mathrm{aux}}-\\frac{\\langle g\_{\\mathrm{aux}},g\_{\\mathrm{main}}\\rangle}{\\|g\_{\\mathrm{main}}\\|^{2}}g\_{\\mathrm{main}}\\right\\|+\\left\\|\\frac{\\langle g\_{\\mathrm{aux}},g\_{\\mathrm{main}}\\rangle}{\\|g\_{\\mathrm{main}}\\|^{2}}r\\right\\|

≤‖π⁡(gaux,gmain)‖+M​c2​‖gaux‖​‖gmain‖2\\displaystyle\\leq\\|\\pi(g\_{\\mathrm{aux}};g\_{\\mathrm{main}})\\|+\\frac{Mc^{2}\\|g\_{\\mathrm{aux}}\\|\\|g\_{\\mathrm{main}}\\|}{2}

≤(λ−1+M​c22​‖gaux‖)​ε.\\displaystyle\\leq\\left(\\lambda^{-1}+\\frac{Mc^{2}}{2}\\|g\_{\\mathrm{aux}}\\|\\right)\\varepsilon.

#### Near-Stationary Point →\\rightarrow Small Bloop Vector.

Reciprocally, by plugging θ\=θ∗+ε​v\\theta=\\theta^{\*}+\\varepsilon v into ([7](#A1.E7 "Equation 7 ‣ Small Bloop Vector → Near-Stationary Point. ‣ A.1 Proof of ‣ Appendix A Convergence analysis ‣ Careful with that Scalpel: Improving Gradient Surgery with an EMA")), we get

gmain\=∇Lmain​(θ)\=ε​∇2Lmain​(θ∗)​v+ε22​∇3Lmain​(θ′)​\[v,v\]\=ε​∇2Lmain​(θ∗)​v+o⁡(ε).g\_{\\mathrm{main}}=\\nabla L\_{{\\mathrm{main}}}(\\theta)=\\varepsilon\\nabla^{2}L\_{\\text{main}}(\\theta^{\\ast})v+\\frac{\\varepsilon^{2}}{2}\\nabla^{3}L\_{\\text{main}}(\\theta^{\\prime})\[v,v\]=\\varepsilon\\nabla^{2}L\_{\\text{main}}(\\theta^{\\ast})v+o(\\varepsilon).

Moreover, by continuity of ∇Laux\\nabla L\_{\\mathrm{aux}} and the optimality condition ∇Laux​(θ∗)\=∇2Lmain​(θ)​v\\nabla L\_{\\mathrm{aux}}(\\theta^{\\ast})=\\nabla^{2}L\_{{\\mathrm{main}}}(\\theta)v, we have gaux\=∇2Lmain​(θ∗)​v+o⁡(1)g\_{\\mathrm{aux}}=\\nabla^{2}L\_{{\\mathrm{main}}}(\\theta^{\\ast})v+o(1) when ε\\varepsilon goes to 00.

From here, we will prove limε→0π⁡(gaux,gmain)\=0\\lim\_{\\varepsilon\\rightarrow 0}\\pi(g\_{\\mathrm{aux}};g\_{\\mathrm{main}})=0 by distinguishing between two cases:

Case 1: Laux​(θ∗)\=𝟎L\_{\\text{aux}}(\\theta^{\\ast})=\\mathbf{0}.  In other words, ∇2Lmain​(θ∗)​v\=𝟎\\nabla^{2}L\_{\\text{main}}(\\theta^{\\ast})v=\\mathbf{0}. Thus gaux\=o⁡(1)g\_{\\text{aux}}=o(1). π⁡(gaux,gmain)\\pi(g\_{\\text{aux}};g\_{\\text{main}}) being a projection of gauxg\_{\\text{aux}}, we have ‖π⁡(gaux,gmain)‖≤‖gaux‖\\|\\pi(g\_{\\text{aux}};g\_{\\text{main}})\\|\\leq\\|g\_{\\text{aux}}\\|. This show π⁡(gaux,gmain)\=o⁡(1)\\pi(g\_{\\text{aux}};g\_{\\text{main}})=o(1).

Case 2: ∇Laux​(θ∗)≠𝟎\\nabla L\_{\\text{aux}}(\\theta^{\\ast})\\neq\\mathbf{0}. This indicates ∇2Lmain​(θ∗)​v≠𝟎\\nabla^{2}L\_{\\text{main}}(\\theta^{\*})v\\neq\\mathbf{0}. We use the formula

π⁡(gaux,gmain)\=gaux−⟨gaux,gmain⟩‖gmain‖2​gmain.\\pi(g\_{\\text{aux}};g\_{\\text{main}})=g\_{\\text{aux}}-\\frac{\\langle g\_{\\text{aux}},g\_{\\text{main}}\\rangle}{\\|g\_{\\text{main}}\\|^{2}}g\_{\\text{main}}.

(8)

With ⟨gaux,gmain⟩\=ε​‖∇2Lmain​(θ∗)​v‖2+o⁡(ε)\\langle g\_{\\text{aux}},g\_{\\text{main}}\\rangle=\\varepsilon\\|\\nabla^{2}L\_{\\text{main}}(\\theta^{\\ast})v\\|^{2}+o(\\varepsilon) and ‖gmain‖2\=ε2​‖∇2Lmain​(θ∗)​v‖2+o⁡(ε2)\\|g\_{\\text{main}}\\|^{2}=\\varepsilon^{2}\\|\\nabla^{2}L\_{\\text{main}}(\\theta^{\\ast})v\\|^{2}+o(\\varepsilon^{2}), we have

ε⁡⟨gaux,gmain⟩‖gmain‖2\=‖∇2Lmain​(θ∗)​v‖2+o⁡(1)‖∇2Lmain​(θ∗)​v‖2+o⁡(1)\=1+o⁡(1),\\frac{\\varepsilon\\langle g\_{\\text{aux}},g\_{\\text{main}}\\rangle}{\\|g\_{\\text{main}}\\|^{2}}=\\frac{\\|\\nabla^{2}L\_{\\text{main}}(\\theta^{\\ast})v\\|^{2}+o(1)}{\\|\\nabla^{2}L\_{\\text{main}}(\\theta^{\*})v\\|^{2}+o(1)}=1+o(1), where the last equality holds since ‖∇2Lmain​(θ∗)​v‖2≠0\\|\\nabla^{2}L\_{\\text{main}}(\\theta^{\\ast})v\\|^{2}\\neq 0.

On the other hand, gmainε\=∇2Lmain​(θ∗)​v+o⁡(1).\\frac{g\_{\\text{main}}}{\\varepsilon}=\\nabla^{2}L\_{\\text{main}}(\\theta^{\\ast})v+o(1).

We have thus

⟨gaux,gmain⟩‖gmain‖2​gmain\=∇2Lmain​(θ∗)​v+o⁡(1).\\frac{\\langle g\_{\\text{aux}},g\_{\\text{main}}\\rangle}{\\|g\_{\\text{main}}\\|^{2}}g\_{\\text{main}}=\\nabla^{2}L\_{\\text{main}}(\\theta^{\\ast})v+o(1).

Using ([8](#A1.E8 "Equation 8 ‣ Near-Stationary Point → Small Bloop Vector. ‣ A.1 Proof of ‣ Appendix A Convergence analysis ‣ Careful with that Scalpel: Improving Gradient Surgery with an EMA")) and gaux\=∇2Lmain​(θ∗)​v+o⁡(1)g\_{\\mathrm{aux}}=\\nabla^{2}L\_{{\\mathrm{main}}}(\\theta^{\\ast})v+o(1), we deduce π⁡(gaux,gmain)\=o⁡(1)\\pi(g\_{\\text{aux}};g\_{\\text{main}})=o(1).

Conclude. In the two cases, we have shown limε→0π⁡(gaux,gmain)\=0\\lim\_{\\varepsilon\\rightarrow 0}\\pi(g\_{\\text{aux}};g\_{\\text{main}})=0. Moreover, we also have limε→0gmain\=0\\lim\_{\\varepsilon\\rightarrow 0}g\_{\\mathrm{main}}=0. Adding the two we get exactly limε→0d⁡(θ∗+ε​v)\=0\\lim\_{\\varepsilon\\to 0}d(\\theta^{\*}+\\varepsilon v)=0.

### A.2 Proof of [Theorem 2](#Thmtheorem2 "Theorem 2 (Convergence of Bloop). ‣ 3.2 Convergence of stochastic Bloop ‣ 3 Theoretical Analysis ‣ Careful with that Scalpel: Improving Gradient Surgery with an EMA")

Here, LmainL\_{{\\mathrm{main}}} and LauxL\_{\\mathrm{aux}} are the empirical risks

Lmain​(θ)\=1n​∑i\=1nLi​(θ)​ and ​Laux​(θ)\=1m​∑j\=1mLj′​(θ).L\_{{\\mathrm{main}}}(\\theta)=\\frac{1}{n}\\sum\_{i=1}^{n}L\_{i}(\\theta)~~\\text{ and }~~L\_{\\mathrm{aux}}(\\theta)=\\frac{1}{m}\\sum\_{j=1}^{m}L^{\\prime}\_{j}(\\theta).

We consider the Bloop method with SGD, which has an EMA gEMAtg\_{\\mathrm{EMA}}^{t} and parameters θt\\theta^{t} which are updated following

Sample ​i,j\\displaystyle\\text{Sample }i,j

∼ Uniform\\displaystyle\\sim\\text{ Uniform}

gEMAt+1\\displaystyle g\_{\\mathrm{EMA}}^{t+1}

\=(1−ρ)gEMAt+ρ∇Li(θt)\\displaystyle=(1-\\rho)g\_{\\mathrm{EMA}}^{t}+\\rho\\nabla L\_{i}(\\theta^{t})

dt\\displaystyle d^{t}

\=∇Li​(θt)+λ​π​(∇Lj′​(θt),gEMAt)\\displaystyle=\\nabla L\_{i}(\\theta^{t})+\\lambda\\pi(\\nabla L^{\\prime}\_{j}(\\theta^{t});g\_{\\mathrm{EMA}}^{t})

θt+1\\displaystyle\\theta^{t+1}

\=θt−η​dt\\displaystyle=\\theta^{t}-\\eta d^{t}

Our analysis works by controlling two quantities: the distance from the EMA to the full-batch train gradient

ϕ1t\=𝔼⁡\[‖gEMAt+1−∇Lmain​(θt)‖2\]\\phi\_{1}^{t}=\\mathbb{E}\\left\[\\|g\_{\\mathrm{EMA}}^{t+1}-\\nabla L\_{{\\mathrm{main}}}(\\theta^{t})\\|^{2}\\right\]

and the train loss

ϕ2t\=𝔼⁡\[Lmain​(θt)\].\\phi\_{2}^{t}=\\mathbb{E}\\left\[L\_{{\\mathrm{main}}}(\\theta^{t})\\right\].

Control of the EMA. For the EMA, we get by expanding

ϕ1t+1\\displaystyle\\phi\_{1}^{t+1}

\=𝔼⁡\[‖gEMAt−ρ⁡(gEMAt−∇Li​(θt))−∇Lmain​(θt)‖2\]\\displaystyle=\\mathbb{E}\\left\[\\|g^{t}\_{\\mathrm{EMA}}-\\rho(g^{t}\_{\\mathrm{EMA}}-\\nabla L\_{i}(\\theta^{t}))-\\nabla L\_{{\\mathrm{main}}}(\\theta^{t})\\|^{2}\\right\]

\=(1−ρ)2​𝔼​\[‖gEMAt−∇Lmain​(θt)‖2\]+ρ2​𝔼​\[‖∇Li​(θt)−∇Lmain​(θt)‖2\]\\displaystyle=(1-\\rho)^{2}\\mathbb{E}\\left\[\\|g^{t}\_{\\mathrm{EMA}}-\\nabla L\_{{\\mathrm{main}}}(\\theta^{t})\\|^{2}\\right\]+\\rho^{2}\\mathbb{E}\\left\[\\|\\nabla L\_{i}(\\theta^{t})-\\nabla L\_{{\\mathrm{main}}}(\\theta^{t})\\|^{2}\\right\]

≤(1−ρ)​𝔼​\[‖gEMAt−∇Lmain​(θt)‖2\]+ρ2​C2\\displaystyle\\leq(1-\\rho)\\mathbb{E}\\left\[\\|g^{t}\_{\\mathrm{EMA}}-\\nabla L\_{{\\mathrm{main}}}(\\theta^{t})\\|^{2}\\right\]+\\rho^{2}C^{2}

where C2C^{2} upper bounds the train gradients variance and where ρ<1\\rho<1. Let a\=gEMAt−∇Lmain​(θt−1)a=g^{t}\_{\\mathrm{EMA}}-\\nabla L\_{{\\mathrm{main}}}(\\theta^{t-1}) and b\=∇Lmain​(θt−1)−∇Lmain​(θt)b=\\nabla L\_{{\\mathrm{main}}}(\\theta^{t-1})-\\nabla L\_{{\\mathrm{main}}}(\\theta^{t}). Since the inequality ‖a+b‖2≤(1+δ)​‖a‖2+(1+δ−1)​‖b‖2\\|a+b\\|^{2}\\leq(1+\\delta)\\|a\\|^{2}+(1+\\delta^{-1})\\|b\\|^{2} holds true for all δ\\delta, we have specifically that

‖gEMAt−∇Lmain​(θt−1)‖2≤(1+δ)​ϕ1t+(1+δ−1)​L2​η2​‖dt−1‖2\\|g^{t}\_{\\mathrm{EMA}}-\\nabla L\_{{\\mathrm{main}}}(\\theta^{t-1})\\|^{2}\\leq(1+\\delta)\\phi\_{1}^{t}+(1+\\delta^{-1})L^{2}\\eta^{2}\\|d^{t-1}\\|^{2}

for δ\=ρ2\\delta=\\frac{\\rho}{2}. Using (1−ρ)​(1+ρ2)≤1−ρ2(1-\\rho)(1+\\frac{\\rho}{2})\\leq 1-\\frac{\\rho}{2} then gives the descent lemma on the EMA:

ϕ1t+1≤(1−ρ2)​ϕ1t+ρ2​C2+2​L2​η2ρ​‖dt−1‖2.\\phi\_{1}^{t+1}\\leq\\left(1-\\frac{\\rho}{2}\\right)\\phi\_{1}^{t}+\\rho^{2}C^{2}+\\frac{2L^{2}\\eta^{2}}{\\rho}\\|d^{t-1}\\|^{2}.

Next, we bound crudely ‖dt−1‖≤D\\|d^{t-1}\\|\\leq D, and equalize the last two terms, i.e. take ρ\=(2​L2​D2C2)13​η23\\rho=\\left(\\frac{2L^{2}D^{2}}{C^{2}}\\right)^{\\frac{1}{3}}\\eta^{\\frac{2}{3}}, so that the descent on the EMA becomes

ϕ1t+1≤(1−ρ2)​ϕ1t+2​ρ2​C2\\phi\_{1}^{t+1}\\leq\\left(1-\\frac{\\rho}{2}\\right)\\phi\_{1}^{t}+2\\rho^{2}C^{2}

which in turn implies that

ϕ1t≤4​ρ​C2.\\phi\_{1}^{t}\\leq 4\\rho C^{2}.

#### Control of the loss.

The LL-smoothness of LmainL\_{{\\mathrm{main}}} and the fact that 𝔼i,j​\[dt\]\=∇Lmain​(θt)+λ​π​(∇Laux,gEMAt)\\mathbb{E}\_{i,j}\[d^{t}\]=\\nabla L\_{{\\mathrm{main}}}(\\theta^{t})+\\lambda\\pi(\\nabla L\_{\\mathrm{aux}};g\_{\\mathrm{EMA}}^{t}) gives:

ϕ2t+1≤ϕ2t−η​‖∇Lmain​(θt)‖2−η​λ​⟨π⁡(∇Laux,gEMAt),∇Lmain​(θt)⟩+L​η22​‖dt‖2.\\phi\_{2}^{t+1}\\leq\\phi\_{2}^{t}-\\eta\\|\\nabla L\_{{\\mathrm{main}}}(\\theta^{t})\\|^{2}-\\eta\\lambda\\langle\\pi(\\nabla L\_{\\mathrm{aux}};g\_{\\mathrm{EMA}}^{t}),\\nabla L\_{{\\mathrm{main}}}(\\theta^{t})\\rangle+\\frac{L\\eta^{2}}{2}\\|d^{t}\\|^{2}.

We omit expectation from the above formula for the ease of presentation, and we will continue doing so for this part of the proof. The annoying middle term is controlled by

−η​λ​⟨π⁡(∇Laux,gEMAt),∇Lmain​(θt)⟩\\displaystyle-\\eta\\lambda\\langle\\pi(\\nabla L\_{\\mathrm{aux}};g\_{\\mathrm{EMA}}^{t}),\\nabla L\_{{\\mathrm{main}}}(\\theta^{t})\\rangle

\=−η​λ​⟨π⁡(∇Laux,gEMAt),∇Lmain​(θt)−gEMAt⟩\\displaystyle=-\\eta\\lambda\\langle\\pi(\\nabla L\_{\\mathrm{aux}};g\_{\\mathrm{EMA}}^{t}),\\nabla L\_{{\\mathrm{main}}}(\\theta^{t})-g\_{\\mathrm{EMA}}^{t}\\rangle

≤η​λ​B​‖∇Lmain​(θt)−∇Lmain​(θt+1)‖+η​λ​B​‖∇Lmain​(θt+1)−gEMAt‖\\displaystyle\\leq\\eta\\lambda B\\|\\nabla L\_{{\\mathrm{main}}}(\\theta^{t})-\\nabla L\_{{\\mathrm{main}}}(\\theta^{t+1})\\|+\\eta\\lambda B\\|\\nabla L\_{{\\mathrm{main}}}(\\theta^{t+1})-g\_{\\mathrm{EMA}}^{t}\\|

≤η2​λ​L​B​‖dt‖+η​λ​B​‖∇Lmain​(θt+1)−gEMAt‖\\displaystyle\\leq\\eta^{2}\\lambda LB\\|d^{t}\\|+\\eta\\lambda B\\|\\nabla L\_{{\\mathrm{main}}}(\\theta^{t+1})-g\_{\\mathrm{EMA}}^{t}\\|

where BB upper bounds ‖∇Laux‖\\|\\nabla L\_{\\mathrm{aux}}\\|. The last 𝔼⁡\[‖dt‖2\]\\mathbb{E}\[\\|d^{t}\\|^{2}\] is simply bounded by D2D^{2}. Hence we get the descent lemma on the train loss:

ϕ2t+1≤ϕ2t−η​‖∇Lmain​(θt)‖2+η​λ​B​ϕ1t+η2​(L​D22+λ​L​B​D).\\phi\_{2}^{t+1}\\leq\\phi\_{2}^{t}-\\eta\\|\\nabla L\_{{\\mathrm{main}}}(\\theta^{t})\\|^{2}+\\eta\\lambda B\\sqrt{\\phi\_{1}^{t}}+\\eta^{2}\\left(\\frac{LD^{2}}{2}+\\lambda LBD\\right).

Plugging the rate for ϕ1t\\phi\_{1}^{t}, we finally get

ϕ2t+1≤ϕ2t−η​‖∇Lmain​(θt)‖2+η43​C1+η2​C2\\phi\_{2}^{t+1}\\leq\\phi\_{2}^{t}-\\eta\\|\\nabla L\_{{\\mathrm{main}}}(\\theta^{t})\\|^{2}+\\eta^{\\frac{4}{3}}C\_{1}+\\eta^{2}C\_{2}

for some constants C1,C2≥0C\_{1},C\_{2}\\geq 0. In the above we have also used that

𝔼​\[‖∇Lmain​(θt+1)−gEMAt‖\]2≤𝔼⁡\[‖∇Lmain​(θt+1)−gEMAt‖2\].\\mathbb{E}\[\\|\\nabla L\_{{\\mathrm{main}}}(\\theta^{t+1})-g\_{\\mathrm{EMA}}^{t}\\|\]^{2}\\leq\\mathbb{E}\[\\|\\nabla L\_{{\\mathrm{main}}}(\\theta^{t+1})-g\_{\\mathrm{EMA}}^{t}\\|^{2}\].

Taking η≤(C1C2)32\\eta\\leq\\left(\\frac{C\_{1}}{C\_{2}}\\right)^{\\frac{3}{2}} ensures that the last term is smaller than the previous, yielding the simple inequality:

ϕ2t+1≤ϕ2t−η​‖∇Lmain​(θt)‖2+2​η43​C1.\\phi\_{2}^{t+1}\\leq\\phi\_{2}^{t}-\\eta\\|\\nabla L\_{{\\mathrm{main}}}(\\theta^{t})\\|^{2}+2\\eta^{\\frac{4}{3}}C\_{1}.

We now have two kinds of results depending on the context:

#### Non-convex result.

Without further assumption, summing the previous inequalities for t\=0​…​T−1t=0\\dots T-1 gives

1T​∑t\=0T−1𝔼⁡\[‖∇Lmain​(θt)‖2\]≤Lmain​(θ0)η​T+2​η13​C1.\\frac{1}{T}\\sum\_{t=0}^{T-1}\\mathbb{E}\[\\|\\nabla L\_{{\\mathrm{main}}}(\\theta^{t})\\|^{2}\]\\leq\\frac{L\_{{\\mathrm{main}}}(\\theta^{0})}{\\eta T}+2\\eta^{\\frac{1}{3}}C\_{1}.

Hence, taking η≃T−34\\eta\\simeq T^{-\\frac{3}{4}} gives a O⁡(T−14)O(T^{-\\frac{1}{4}}) rate.

#### PL-result.

We here assume that LmainL\_{{\\mathrm{main}}} verifies the PL inequality 12​‖∇Lmain​(θ)‖2≥μ​Lmain​(θ)\\frac{1}{2}\\|\\nabla L\_{{\\mathrm{main}}}(\\theta)\\|^{2}\\geq\\mu L\_{{\\mathrm{main}}}(\\theta), where we posit min⁡Lmain\=0\\min L\_{{\\mathrm{main}}}=0 without loss of generalitiy. The descent lemma gives

ϕ2t+1≤(1−2​η​μ)​ϕ2t+2​η43​C1.\\phi\_{2}^{t+1}\\leq(1-2\\eta\\mu)\\phi\_{2}^{t}+2\\eta^{\\frac{4}{3}}C\_{1}.

By unrolling it we obtain

𝔼⁡\[Lmain​(θT)\]≤(1−2​η​μ)T​Lmain​(θ0)+(1−(1−2​η​μ)T)​η13​C1μ.\\mathbb{E}\[L\_{{\\mathrm{main}}}(\\theta^{T})\]\\leq(1-2\\eta\\mu)^{T}L\_{{\\mathrm{main}}}(\\theta^{0})+(1-(1-2\\eta\\mu)^{T})\\frac{\\eta^{\\frac{1}{3}}C\_{1}}{\\mu}.

This shows a linear convergence to a radius proportional to η13\\eta^{\\frac{1}{3}}.

## Appendix B Experimental Details

In this appendix we report the missing details from [Section 5](#S5 "5 Experiments ‣ Careful with that Scalpel: Improving Gradient Surgery with an EMA").

Training smooth networks. For this experiment we use an MLP with ReLU activations. The features are of size 728→256→128→10728\\rightarrow 256\\rightarrow 128\\rightarrow 10. All the methods are trained with Adam optimizer at learning rate of 3×10−43\\times 10^{-4} for 100 epochs and a cosine learning rate schedule. For consistency with the other classification experiments we also include 55 epochs of warm-up. The batch size is fixed at 256256, and we take a grid of λ\\lambda with log10⁡(λ)\=−4,−3.5,…,−0.5,0\\log\_{10}(\\lambda)=-4,-3.5,\\dots,-0.5,0.

Imagenet training with L2 regularization. For ImageNet training, we employ SGD with a batch size of 20482048, Nesterov momentum of 0.90.9, and a learning rate of 0.80.8. This learning rate is derived by scaling the base rate of 0.10.1 by a factor of 88, corresponding to the ratio 2048/2562048/256. Additionally, we apply a cosine learning rate schedule with 55 warm-up epochs and utilize random cropping and flipping for data augmentation during training. The network is trained for 100100 epochs. This configuration is known to work well for the ResNet50 architecture that we are using here. The grid of λ\\lambda is 14 uniform values in log scale between 10−610^{-6} and 10−210^{-2}, and 00. We display results for all methods in [Figure 8](#A2.F8 "Figure 8 ‣ Appendix B Experimental Details ‣ Careful with that Scalpel: Improving Gradient Surgery with an EMA") with a slightly smaller grid of λ\\lambda’s.

Figure 8: Results of all methods on the imagenet + L2 problem. PCGrad and DB have similar performance to the mixed method.

Multi-task learning with Cifar10Mnist. The overall setup for this problem is similar to that for Imagenet training, with the exceptions that we use a smaller architecture—ResNet18 instead of ResNet50, and a smaller batch size—256256 instead of 20482048. We also scale down the learning rate to 0.10.1 to account for the smaller batch size. The values of the trade-off parameter λ\\lambda goes from 10−310^{-3} to 10310^{3} and are split equally on log scale. Unlike Adam, SGD does not adjust the learning rate scale automatically. This causes unstable training when λ\\lambda is too large. We thus futher scale the learning rate 0.10.1 by 1/(1+λ)1/(1+\\lambda) for each independent run.

![Refer to caption](2402.02998v2/figures/cifar10mnist.png)

Figure 9: Sample images from the Cifar10Mnist dataset.

Next token prediction. Our model is a byte-level decoder-only transformer. It has 12 layers, 8 attention heads, a residual dimension of 256, and a feed-forward dimension 1024. We use a batch-size of 128 for both datasets, the optimizer is Adam with a learning rate of 0.0020.002. We train the model for 300​K300K iterations. The grid of λ\\lambda consists of 1616 values evenly spaced in log-space between 10−410^{-4} and 1010, as well as 00.

Translation. Our model is an encore-decoder transformer. It has 6 encoder and decoder layers, 16 attention heads, a residual dimension of 1024, and a feed-forward dimension 4096. We use a batch-size of 256 for both datasets, the optimizer is Adam with a learning rate of 0.00020.0002. We train the model for 500​K500K iterations. Our implementation is derived from the flax example ([Heek et al.,](#bib.bib16 "") ). The grid of λ\\lambda consists of 1616 values evenly spaced in log-space between 10−410^{-4} and 1010, as well as 00.

## Appendix C Additional Experiment

We present the results of another experiment, where all methods, including Bloop, gave similar Pareto fronts. Here, we aim to perform classification on both the Imagenet and the CIFAR-10 datasets. The network is a ResNet50 with with two separate classification heads. This problem sits in the middle ground between the multi-task learning and the joint dataset training problem that we describe in [Section 5](#S5 "5 Experiments ‣ Careful with that Scalpel: Improving Gradient Surgery with an EMA"): we have two separate datasets for the two distinct tasks. Similar to before, the main loss is the training loss on the larger dataset, i.e., Imagenet, and the auxiliary loss is the training loss on the smaller dataset, i.e. Cifar10. We choose λ\\lambda to be equally split on log scale from 10−310^{-3} to 1010. The remaining configurations follow the experiment of Imagenet training with L2 regularization, except that we also scale the learning rate by 1/(1+λ)1/(1+\\lambda) to avoid instability as in the multi-task experiment.

The results are shown in [Figure 10](#A3.F10 "Figure 10 ‣ Appendix C Additional Experiment ‣ Careful with that Scalpel: Improving Gradient Surgery with an EMA"). Unlike the experiments of [Section 5](#S5 "5 Experiments ‣ Careful with that Scalpel: Improving Gradient Surgery with an EMA"), there is little trade-off between the two tasks. We can increase accuracy on CIFAR-10 without sacrificing performance on Imagenet. For this reason, there are only very few points at the Pareto front and all methods perform similarly at these points. We posit that here, the two losses are not conflicting enough to see the gradient surgery methods have an edge.

Figure 10: Results on the Imagenet / Cifar10 experiment. All algorithms perform generally similarly except for very high values of λ\\lambda, which leads to worse performance for all algorithms.

[](javascript:toggleReadingMode\(\); "Disable reading mode, show header and footer")