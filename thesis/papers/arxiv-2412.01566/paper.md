# Multi-objective Deep Learning: Taxonomy and Survey of the State of the Art

 Sebastian Peitz    Sedjro Salomon Hotegni ††thanks: Both authors are with the Department of Computer Science, TU Dortmund, Dortmund, Germany, and with the Lamarr Institute for Machine Learning and Artificial Intelligence, e-mail: {sebastian.peitz, salomon.hotegni}@tu-dortmund.de.

###### Abstract

Simultaneously considering multiple objectives in machine learning has been a popular approach for several decades, with various benefits for multi-task learning, the consideration of secondary goals such as sparsity, or multicriteria hyperparameter tuning. However—as multi-objective optimization is significantly more costly than single-objective optimization—the recent focus on deep learning architectures poses considerable additional challenges due to the very large number of parameters, strong nonlinearities and stochasticity. This survey covers recent advancements in the area of multi-objective deep learning. We introduce a taxonomy of existing methods—based on the type of training algorithm as well as the decision maker’s needs—before listing recent advancements, and also successful applications. All three main learning paradigms supervised learning, unsupervised learning and reinforcement learning are covered, and we also address the recently very popular area of generative modeling.

###### Index Terms:

Machine learning, deep learning, reinforcement learning, unsupervised learning, multi-objective optimization 

## I Introduction

Conflicting design or decision criteria are everywhere, and the challenge of identifying suitable decisions has been around for a very long time. To address this problem, we are looking for the set of optimal compromises—also called the *Pareto set* after the Italian economist Vilfredo Pareto. In simple terms, a decision is Pareto optimal if we are unable to further improve all objectives at the same time, but instead have to accept a trade-off if we want to further improve one of the criteria. In mathematical terms, this can be cast as a *multi-objective optimization problem*, see \[[120](#bib.bib120 ""), [43](#bib.bib43 "")\] for detailed introductions. Problems of this type have been studied in a large number of fields, examples being political decision-making \[[175](#bib.bib175 "")\], medical therapy planning \[[94](#bib.bib94 "")\], the design of numerical algorithms \[[185](#bib.bib185 "")\], or the control of complex dynamical systems \[[136](#bib.bib136 "")\].

Very naturally, multiple criteria also arise in the area of machine learning, for instance if we want to solve multiple tasks with a single model, which results in several performance measures. Alternatively, we might be interested in secondary objectives such as sparsity/efficiency, robustness or interpretability. Specifically in the area of reinforcement learning, researchers have made strong arguments to intensify multi-objective research \[[146](#bib.bib146 ""), [184](#bib.bib184 "")\]. However, since both multi-objective optimization and deep learning can be computationally expensive, their combination gives rise to particular challenges in terms of algorithmic efficiency and decision-making.

The goal of this survey paper is to provide a detailed overview of the state of the art in multi-objective deep learning. A large number of articles has appeared in recent years, and progress has been rapid. To keep an overview and ease the entry for researchers interested in—but until now less familiar with—multi-objective learning, we provide a taxonomy of the various approaches (Section [III](#S3 "III A taxonomy of multi-objective deep learning ‣ Multi-objective Deep Learning: Taxonomy and Survey of the State of the Art")), before surveying the current state of the art in Section [IV](#S4 "IV Multi-objective deep learning survey ‣ Multi-objective Deep Learning: Taxonomy and Survey of the State of the Art"). We consider the three main learning paradigms supervised learning ([IV-A](#S4.SS1 "IV-A Supervised learning ‣ IV Multi-objective deep learning survey ‣ Multi-objective Deep Learning: Taxonomy and Survey of the State of the Art")), unsupervised learning ([IV-B](#S4.SS2 "IV-B Unsupervised and self-supervised learning ‣ IV Multi-objective deep learning survey ‣ Multi-objective Deep Learning: Taxonomy and Survey of the State of the Art")) and reinforcement learning ([V](#S5 "V Deep multi-objective reinforcement learning ‣ Multi-objective Deep Learning: Taxonomy and Survey of the State of the Art")—since this is a special case, we will discuss it separately), and we also provide an overview of generative modeling ([IV-C](#S4.SS3 "IV-C Generative modeling ‣ IV Multi-objective deep learning survey ‣ Multi-objective Deep Learning: Taxonomy and Survey of the State of the Art")), neural architecture search ([IV-D](#S4.SS4 "IV-D Neural architecture search ‣ IV Multi-objective deep learning survey ‣ Multi-objective Deep Learning: Taxonomy and Survey of the State of the Art")) and successful applications of multi-objective deep learning ([IV-E](#S4.SS5 "IV-E Applications ‣ IV Multi-objective deep learning survey ‣ Multi-objective Deep Learning: Taxonomy and Survey of the State of the Art")). Finally, we briefly comment on the vice-versa combination in Section [VI](#S6 "VI Deep learning for multi-objective optimization ‣ Multi-objective Deep Learning: Taxonomy and Survey of the State of the Art"), i.e., the usage of deep learning to enhance the solution of multi-objective optimization problems.

## II Preliminaries

Before introducing our taxonomy and survey to multi-objective deep learning we here give brief introductions to deep learning ([II-A](#S2.SS1 "II-A Deep learning ‣ II Preliminaries ‣ Multi-objective Deep Learning: Taxonomy and Survey of the State of the Art")) and multi-objective optimization ([II-B](#S2.SS2 "II-B Multi-objective optimization ‣ II Preliminaries ‣ Multi-objective Deep Learning: Taxonomy and Survey of the State of the Art")), respectively and highlight the specifics of multi-objective machine learning in Section [II-C](#S2.SS3 "II-C Multi-objective machine learning ‣ II Preliminaries ‣ Multi-objective Deep Learning: Taxonomy and Survey of the State of the Art").

### II-A Deep learning

This section’s main purpose is to introduce notation and to highlight the similarity between various areas of machine learning when it comes to training and optimization. We briefly cover supervised ([II-A1](#S2.SS1.SSS1 "II-A1 Supervised learning ‣ II-A Deep learning ‣ II Preliminaries ‣ Multi-objective Deep Learning: Taxonomy and Survey of the State of the Art")) and unsupervised learning ([II-A2](#S2.SS1.SSS2 "II-A2 Unsupervised and self-supervised learning ‣ II-A Deep learning ‣ II Preliminaries ‣ Multi-objective Deep Learning: Taxonomy and Survey of the State of the Art")) as well as generative modeling ([II-A3](#S2.SS1.SSS3 "II-A3 Generative modeling ‣ II-A Deep learning ‣ II Preliminaries ‣ Multi-objective Deep Learning: Taxonomy and Survey of the State of the Art")). For much more detailed introductions, the reader is referred to one of the excellent text books \[[19](#bib.bib19 ""), [69](#bib.bib69 ""), [62](#bib.bib62 ""), [20](#bib.bib20 "")\].

#### II-A1 Supervised learning

The overarching theme in supervised learning is function approximation from examples. In other words, we want to approximate an unknown input-to-output mapping y\=f⁡(x)y=f(x), f:ℝn→ℝmf:\\mathbb{R}^{n}\\rightarrow\\mathbb{R}^{m}, by a parametrized function fθf\_{\\theta}. Here, θ∈ℝq\\theta\\in\\mathbb{R}^{q} represents the (usually high-dimensional) vector of trainable parameters. There exist many options how to construct fθf\_{\\theta}, such as polynomials, Fourier series, or Gaussian processes. The specialty in deep learning \[[96](#bib.bib96 "")\] is that fθf\_{\\theta} represents a deep neural network with ℓ\\ell layers, in which affine transformations (W,b)(W,b) alternate with nonlinear activation functions σ\\sigma:

z(0)\\displaystyle z^{(0)}

\=x,\\displaystyle=x, z(j)\\displaystyle z^{(j)}

\=σ(j)(W(j)z(j−1)+b(j)),j\=1,…,ℓ,\\displaystyle=\\sigma^{(j)}\\left(W^{(j)}z^{(j-1)}+b^{(j)}\\right),\\quad j=1,\\ldots,\\ell, y\\displaystyle y

\=z(ℓ).\\displaystyle=z^{(\\ell)}.

All trainable weights are collected in θ\={(W(j),b(j))}j\=1ℓ\\theta=\\left\\{\\left(W^{(j)},b^{(j)}\\right)\\right\\}\_{j=1}^{\\ell}.

###### Remark 1.

Besides the just-mentioned classical *feed-forward* architecture, there are numerous other deep learning models used for function approximation such as *convolutional networks*, *residual networks*, or *recurrent networks* with feedback loops \[[62](#bib.bib62 ""), [20](#bib.bib20 "")\]. However, the goal of this article is not to cover the details of deep learning models, but to shed light on the training with multiple criteria, which is why we are not going into more detail here.

To fit fθf\_{\\theta} to ff, we use a dataset 𝒟\={(xi,yi)}i\=1N\\mathcal{D}=\\{(x\_{i},y\_{i})\\}\_{i=1}^{N} of NN samples and minimize the *empirical loss* LL:

θ∗\=arg⁡minθ∈ℝq⁡L⁡(θ).\\displaystyle\\theta^{\*}=\\arg\\min\_{\\theta\\in\\mathbb{R}^{q}}L(\\theta).

(1)

A common choice for LL is the mean squared error, i.e., L⁡(θ)\=1N​∑i\=1N‖yi−fθ​(xi)‖22,L(\\theta)=\\frac{1}{N}\\sum\_{i=1}^{N}\\left\\|y\_{i}-f\_{\\theta}(x\_{i})\\right\\|\_{2}^{2}, but there are numerous alternatives as well as additional terms (e.g., for regularization or sparsity \[[97](#bib.bib97 "")\]). Regardless of the specific choice, Problem ([1](#S2.E1 "In II-A1 Supervised learning ‣ II-A Deep learning ‣ II Preliminaries ‣ Multi-objective Deep Learning: Taxonomy and Survey of the State of the Art")) is a high-dimensional, nonlinear optimization problem. The typical approach to find θ∗\\theta^{\*} (or at least a θ\\theta that yields a satisfactory performance) is gradient-based optimization using backpropagation, often in combination with stochasticity and momentum, as in the popular Adam algorithm \[[90](#bib.bib90 "")\] or adaptations thereof (e.g., \[[25](#bib.bib25 "")\]). Consequently, the gradient ∇L​(θ)\\nabla L(\\theta) plays a central role in training.

#### II-A2 Unsupervised and self-supervised learning

Unlike supervised learning, which depends on labeled data to model input-output relationships, unsupervised and self-supervised learning focus on extracting meaningful representations or patterns from unlabeled data \[[156](#bib.bib156 ""), [164](#bib.bib164 "")\]. While unsupervised learning relies on directly identifying latent structures in the data \[[29](#bib.bib29 ""), [40](#bib.bib40 ""), [194](#bib.bib194 "")\], self-supervised learning uses surrogate tasks, derived from the data itself, to create labels and train models that can generalize to downstream tasks \[[64](#bib.bib64 ""), [87](#bib.bib87 "")\].

The objectives in unsupervised learning often involve clustering \[[27](#bib.bib27 "")\], dimensionality reduction \[[154](#bib.bib154 "")\], or density estimation \[[193](#bib.bib193 "")\], formulated as optimization problems. Clustering focuses on partitioning a dataset 𝒟\={(xi)}i\=1N\\mathcal{D}=\\{(x\_{i})\\}\_{i=1}^{N} into NCN\_{C} clusters. A popular objective is to minimize intra-cluster variance while maximizing inter-cluster separation:

minC1,…,CK∑k\=1NC∑xi∈Ck∥xi−μk∥2,μk\=1|Ck|∑xi∈Ckxi,\\min\_{C\_{1},\\dots,C\_{K}}\\sum\_{k=1}^{N\_{C}}\\sum\_{x\_{i}\\in C\_{k}}\\|x\_{i}-\\mu\_{k}\\|^{2},\\quad\\mu\_{k}=\\frac{1}{|C\_{k}|}\\sum\_{x\_{i}\\in C\_{k}}x\_{i}, where CkC\_{k} is the set of data points assigned to the kthk^{\\text{th}} cluster, and μk\\mu\_{k} is its centroid. Techniques like K-means and Gaussian Mixture Models (GMMs) \[[143](#bib.bib143 "")\] address this optimization. Modern neural clustering approaches embed data into latent spaces, enhancing the flexibility and scalability of clustering methods \[[202](#bib.bib202 ""), [101](#bib.bib101 "")\]. For dimensionality reduction, the goal is to find a lower-dimensional representation for high-dimensional data, retaining as much information as possible \[[79](#bib.bib79 "")\]. Autoencoders, in particular, leverage neural networks to learn compressed representations by minimizing a reconstruction loss \[[117](#bib.bib117 ""), [14](#bib.bib14 "")\], e.g., min⁡∑i\=1Nθ,ϕ⁡‖xi−gϕ​(fθ​(xi))‖22,\\min\_{\\theta,\\phi}\\sum\_{i=1}^{N}\\left\\|x\_{i}-g\_{\\phi}(f\_{\\theta}(x\_{i}))\\right\\|\_{2}^{2}, where fθ​(x)\=zf\_{\\theta}(x)=z encodes the input into a latent variable z∈ℝmz\\in\\mathbb{R}^{m}, and gϕ​(z)\=x\~g\_{\\phi}(z)=\\tilde{x} decodes the latent variable back to an output x\~\\tilde{x} that is equal in size to the input. If we choose a small latent space dimension mm and successfully optimize for a small loss (i.e., gϕ​(fθ​(xi))\=x\~i≈xig\_{\\phi}(f\_{\\theta}(x\_{i}))=\\tilde{x}\_{i}\\approx x\_{i}), then we have successfully found an intrinsic, low-dimensional structure that encodes the information of the data set. In density estimation, probabilistic methods like Kernel Density Estimation (KDE) \[[30](#bib.bib30 "")\] and Variational Autoencoders (VAEs) \[[91](#bib.bib91 ""), [92](#bib.bib92 "")\] model the underlying distribution of data. These methods provide insights into data regularities and outliers, making them useful for anomaly detection and data generation \[[128](#bib.bib128 ""), [26](#bib.bib26 "")\] (see also the next Section [II-A3](#S2.SS1.SSS3 "II-A3 Generative modeling ‣ II-A Deep learning ‣ II Preliminaries ‣ Multi-objective Deep Learning: Taxonomy and Survey of the State of the Art") for details).

Self-supervised learning trains models by creating surrogate tasks, transforming unlabeled data into structured problems resembling supervised learning. These tasks encourage the model to learn representations that generalize well to downstream tasks. Common pretext tasks include contrastive learning and generative models \[[111](#bib.bib111 ""), [95](#bib.bib95 "")\]. Self-supervised learning employs predictive tasks as well. Notable examples include predicting the next frame in a video \[[78](#bib.bib78 "")\], determining the rotation angle of an image \[[83](#bib.bib83 "")\], and reconstructing masked portions of input data \[[72](#bib.bib72 "")\]. Masked Autoencoders (MAEs) \[[72](#bib.bib72 "")\], for example, reconstruct masked inputs xx by minimizing:

LMAE\=‖xmasked−gθ​(fϕ​(xvisible))‖22,{L}\_{\\text{MAE}}=\\|x\_{\\text{masked}}-g\_{\\theta}(f\_{\\phi}(x\_{\\text{visible}}))\\|\_{2}^{2}, where fϕf\_{\\phi} is the encoder that processes the visible patches xvisiblex\_{\\text{visible}} to produce latent representations, and gθg\_{\\theta} is the decoder that reconstructs the masked input xmaskedx\_{\\text{masked}} from these representations. Problems in both unsupervised and self-supervised deep learning are commonly addressed through gradient-based optimization methods, where a loss function quantifying the task objective (e.g., reconstruction error, contrastive loss) is minimized using backpropagation and stochastic gradient descent or its variants.

#### II-A3 Generative modeling

The central goal of generative modeling (cf. \[[151](#bib.bib151 "")\] for a very clear and concise introduction) is to learn an unknown probability distribution 𝒳\\mathcal{X} using training samples x∼𝒳x\\sim\\mathcal{X}, the task is thus closely related to the self-supervised learning framework described in the previous section. To this end, one constructs a generative model g:ℝp→ℝng:\\mathbb{R}^{p}\\rightarrow\\mathbb{R}^{n} that maps points z∈ℝpz\\in\\mathbb{R}^{p}—drawn from a lower-dimensional, more tractable probability distribution 𝒵\\mathcal{Z}—to xx in such a manner that g⁡(z)∼𝒳g(z)\\sim\\mathcal{X}. As an example, consider a generator gg that maps points zz drawn from a multivariate Gaussian distribution to images xx in such a way that the generated images follow the same probability distribution as the training data. In other words, the set of generated images is statistically indistinguishable from the set of real images, since p⁡(g⁡(z))\=p⁡(x)p(g(z))=p(x). Since the introduction of Generative Adversarial Networks (GANs) in 2014 by Goodfellow et al. \[[63](#bib.bib63 "")\] and of variational autoencoders by Kingma and Welling in 2013 \[[91](#bib.bib91 "")\] (see also the previous section [II-A2](#S2.SS1.SSS2 "II-A2 Unsupervised and self-supervised learning ‣ II-A Deep learning ‣ II Preliminaries ‣ Multi-objective Deep Learning: Taxonomy and Survey of the State of the Art")), this area of machine learning has gained massive attention, with a large number of new architectures such as diffusion models \[[165](#bib.bib165 ""), [148](#bib.bib148 "")\], and culminating in the recent wave of transformer-based architectures \[[188](#bib.bib188 "")\] and large language models (LLMs).

To draw the connection to gradient-based learning, let’s briefly consider the GAN architecture, which consists of a generator gθ:ℝp→ℝng\_{\\theta}:\\mathbb{R}^{p}\\rightarrow\\mathbb{R}^{n} and a discriminator fϕ:ℝn→\[0,1\]f\_{\\phi}:\\mathbb{R}^{n}\\rightarrow\[0,1\], both of which can be realized by deep neural networks. The task of the generator is to transform inputs z∼𝒵z\\sim\\mathcal{Z} into outputs xx, while the discriminator’s job is to decide whether an input xx has been drawn from a real dataset, or was created by the generator. It is thus a standard binary classifier predicting the probability that xx is real. As a consequence, the two networks share the same loss function

L𝖦𝖠𝖭​(θ,ϕ)\=𝔼x∼𝒳​\[log⁡(fϕ​(x))\]+𝔼z∼𝒵​\[log⁡(1−fϕ​(gθ​(z)))\].L\_{\\mathsf{GAN}}(\\theta,\\phi)=\\mathbb{E}\_{x\\sim\\mathcal{X}}\\left\[\\log(f\_{\\phi}(x))\\right\]+\\mathbb{E}\_{z\\sim\\mathcal{Z}}\\left\[\\log(1-f\_{\\phi}(g\_{\\theta}(z)))\\right\].

The specialty is that the two networks are *adversarials*, meaning that the generator seeks to minimize L𝖦𝖠𝖭L\_{\\mathsf{GAN}} (i.e., to fool the discriminator), while the discriminator wants to maximize L𝖦𝖠𝖭L\_{\\mathsf{GAN}} (that is, to perform correct classifications). Training thus usually alternates between gradient descent of L𝖦𝖠𝖭L\_{\\mathsf{GAN}} using the gradient with respect to θ\\theta, and gradient ascent of L𝖦𝖠𝖭L\_{\\mathsf{GAN}} using the gradient with respect to ϕ\\phi, i.e., θ(i+1)\\displaystyle\\theta^{(i+1)}

\=θ(i)−ηθ​(θ(i))​∇θL𝖦𝖠𝖭​(θ(i),ϕ(i)),\\displaystyle=\\theta^{(i)}-\\eta\_{\\theta}\\left(\\theta^{(i)}\\right)\\nabla\_{\\theta}L\_{\\mathsf{GAN}}\\left(\\theta^{(i)},\\phi^{(i)}\\right), ϕ(i+1)\\displaystyle\\phi^{(i+1)}

\=ϕ(i)+ηϕ​(ϕ(i))​∇ϕL𝖦𝖠𝖭​(θ(i),ϕ(i)),\\displaystyle=\\phi^{(i)}+\\eta\_{\\phi}\\left(\\phi^{(i)}\\right)\\nabla\_{\\phi}L\_{\\mathsf{GAN}}\\left(\\theta^{(i)},\\phi^{(i)}\\right), where ηθ\\eta\_{\\theta} and ηϕ\\eta\_{\\phi} are (potentially variable) learning rates. Training stops when this two-player game converges to an equilibrium.

### II-B Multi-objective optimization

Again, this section covers only the basic concepts of multi-objective optimization. Much more detailed overviews can be found in \[[120](#bib.bib120 ""), [43](#bib.bib43 "")\].

#### II-B1 Definition and concepts

Consider the situation where instead of a single loss function, we have a vector with KK conflicting ones, i.e., L⁡(θ)\=\[L1​(θ),…,LK​(θ)\]⊤L(\\theta)=\[L\_{1}(\\theta),\\ldots,L\_{K}(\\theta)\]^{\\top}. The task thus becomes to minimize all losses at the same time, i.e., minθ∈ℝq⁡(L1​(θ)LK​(θ)).\\min\_{\\theta\\in\\mathbb{R}^{q}}\\begin{pmatrix}L\_{1}(\\theta)\\\\ \\vdots\\\\ L\_{K}(\\theta)\\end{pmatrix}.

(MOP)

If the objectives are conflicting, then there does not exist a single optimal θ∗\\theta^{\*} that minimizes all LkL\_{k}. Instead, there exists a *Pareto set* 𝒫\\mathcal{P} with optimal trade-offs, i.e.,

𝒫\={θ∈ℝq|∄θ^:Lk​(θ^)≤Lk​(θ)for​k\=1,…,K,Lk​(θ^)<Lk​(θ)for at least one ​k}.\\mathcal{P}=\\left\\{\\theta\\in\\mathbb{R}^{q}\\penalty\\ \\middle|\\penalty\\ \\nexists\\hat{\\theta}:\\begin{array}\[\]{ll}L\_{k}(\\hat{\\theta})\\leq L\_{k}(\\theta)&\\mbox{for}\\penalty\\ k=1,\\ldots,K,\\\\ L\_{k}(\\hat{\\theta})\<L\_{k}(\\theta)&\\mbox{for at least one }\\penalty\\ k\\end{array}\\right\\}.

In other words, a point θ^\\hat{\\theta} *dominates* a point θ\\theta, if it is at least as good in all LkL\_{k}, while being strictly better with respect to at least one loss. The Pareto set 𝒫\\mathcal{P} thus consists of all non-dominated points. The corresponding set in objective space is called the *Pareto front* 𝒫ℱ\=L⁡(𝒫)\\mathcal{P}\_{\\mathcal{F}}=L(\\mathcal{P}). Under smoothness assumptions, both objects have dimension K−1K-1 \[[73](#bib.bib73 "")\], i.e., 𝒫\\mathcal{P} and 𝒫ℱ\\mathcal{P}\_{\\mathcal{F}} are lines for two objectives, 2D surfaces for three objectives, and so on. Furthermore, they are bounded by Pareto sets and fronts of the next lower number of objectives \[[59](#bib.bib59 "")\], meaning that individual minima constrain a two-objective solution, 1D fronts constrain the 2D surface of a K\=3K=3 problem, etc.

In deep learning, the most common situation is a very large number qq of trainable parameters, but a moderate number KK of objectives. It is thus much more common to visualize and study Pareto fronts instead of Pareto sets. Figure [1](#S2.F1 "Figure 1 ‣ II-B1 Definition and concepts ‣ II-B Multi-objective optimization ‣ II Preliminaries ‣ Multi-objective Deep Learning: Taxonomy and Survey of the State of the Art") shows three examples of a convex and a non-convex problem, as well as the special case of non-conflicting criteria.

Figure 1: Left: Example of a convex Pareto front, where each point has a unique tangent, i.e., weighting vector w(i)w^{(i)}. Middle: Non-convex front, where multiple points have the same tangent vector. Right: Non-conflicting objectives such that the Pareto front collapses.

#### II-B2 Gradients and optimality conditions

Closely related to single-objective optimization, there exist first order optimality conditions, referred to as the *Karush-Kuhn-Tucker (KKT)* conditions \[[120](#bib.bib120 "")\]. A point θ∗\\theta^{\*} is said to be *Pareto-critical* if there exists a convex combination of the individual gradients ∇Lk\\nabla L\_{k} that is zero. More formally, we have

∑k\=1Kαk∗∇Lk(θ∗)\=0,∑k\=1Kαk∗\=1,\\displaystyle\\sum\_{k=1}^{K}\\alpha^{\*}\_{k}\\nabla L\_{k}(\\theta^{\*})=0,\\qquad\\sum\_{k=1}^{K}\\alpha^{\*}\_{k}=1,

(KKT)

which is a natural extension of the case K\=1K=1.

Equation ([KKT](#S2.Ex13 "In II-B2 Gradients and optimality conditions ‣ II-B Multi-objective optimization ‣ II Preliminaries ‣ Multi-objective Deep Learning: Taxonomy and Survey of the State of the Art")) is the basis for most gradient-based methods (more details in Section [II-B3](#S2.SS2.SSS3.Px1 "Computing individual Pareto optima ‣ II-B3 Overview of methods ‣ II-B Multi-objective optimization ‣ II Preliminaries ‣ Multi-objective Deep Learning: Taxonomy and Survey of the State of the Art")) and thus essential in the construction of efficient multi-objective deep learning algorithms. Their goal is to compute elements from the Pareto critical set,

𝒫c\={θ∗∈ℝq|∃α∗∈ℝ≥0K,∑k\=1Kαk∗\=1:([KKT](#S2.Ex13 "In II-B2 Gradients and optimality conditions ‣ II-B Multi-objective optimization ‣ II Preliminaries ‣ Multi-objective Deep Learning: Taxonomy and Survey of the State of the Art"))holds},\\mathcal{P}\_{c}=\\left\\{\\theta^{\*}\\in\\mathbb{R}^{q}\\penalty\\ \\middle|\\penalty\\ \\exists\\alpha^{\*}\\in\\mathbb{R}\_{\\geq 0}^{K},\\,\\sum\_{k=1}^{K}\\alpha^{\*}\_{k}=1\\,:\\penalty\\ \\eqref{eq:KKT}\\,\\mbox{holds}\\right\\}, which contains excellent candidates for Pareto optima, since 𝒫c⊇𝒫\\mathcal{P}\_{c}\\supseteq\\mathcal{P}. In a similar fashion to the single-objective case, there exist extensions to constraints \[[58](#bib.bib58 ""), [52](#bib.bib52 "")\], but we will exclusively consider unconstrained problems here.

###### Remark 2 (Lipschitz continuous loss functions).

An extension is required if we consider less regularity, i.e., Lipschitz continuity. In the context of deep learning, this is the case for activation functions with kinks (e.g., ReLU⁡(x)\=max⁡{0,x}\\operatorname{ReLU}(x)=\\max\\{0,x\\}), or when considering ℓ1\\ell\_{1} regularization terms (‖θ‖1\=∑|θi|\\|\\theta\\|\_{1}=\\sum|\\theta\_{i}|), cf. Figure [2](#S2.F2 "Figure 2 ‣ Remark 2 (Lipschitz continuous loss functions). ‣ II-B2 Gradients and optimality conditions ‣ II-B Multi-objective optimization ‣ II Preliminaries ‣ Multi-objective Deep Learning: Taxonomy and Survey of the State of the Art"). Then, *subdifferentials* ∂Lk\\partial L\_{k} take the place of the gradients ∇Lk\\nabla L\_{k} (cf. \[[33](#bib.bib33 "")\] for a detailed introduction), and the condition ([KKT](#S2.Ex13 "In II-B2 Gradients and optimality conditions ‣ II-B Multi-objective optimization ‣ II Preliminaries ‣ Multi-objective Deep Learning: Taxonomy and Survey of the State of the Art")) is replaced by a non-smooth KKT condition

0∈conv⁡(⋃k\=1K∂Lk​(θ∗)),0\\in\\operatorname{conv}\\left(\\bigcup\_{k=1}^{K}\\partial L\_{k}(\\theta^{\*})\\right),

(2)

which reduces to ([KKT](#S2.Ex13 "In II-B2 Gradients and optimality conditions ‣ II-B Multi-objective optimization ‣ II Preliminaries ‣ Multi-objective Deep Learning: Taxonomy and Survey of the State of the Art")) for all θ∗\\theta^{\*} where the Lk​(θ∗)L\_{k}(\\theta^{\*}) are smooth, i.e., outside θ\=0\\theta=0 in the exemplary Figure [2](#S2.F2 "Figure 2 ‣ Remark 2 (Lipschitz continuous loss functions). ‣ II-B2 Gradients and optimality conditions ‣ II-B Multi-objective optimization ‣ II Preliminaries ‣ Multi-objective Deep Learning: Taxonomy and Survey of the State of the Art") below.

Figure 2: ℓ1\\ell\_{1} norm and ReLU activation function for θ∈ℝ\\theta\\in\\mathbb{R}.

#### II-B3 Overview of methods

There exists a large number of conceptually very different methods to find (approximate) solutions of ([MOP](#S2.Ex11 "In II-B1 Definition and concepts ‣ II-B Multi-objective optimization ‣ II Preliminaries ‣ Multi-objective Deep Learning: Taxonomy and Survey of the State of the Art")). The first key question is—and this will come up again in our taxonomy—whether we put the *decision-making*11 1 Decision-making refers to the selection of a particular element of 𝒫\\mathcal{P}, either once or interactively to react to changing circumstances. It is a topic of its own \[[182](#bib.bib182 "")\], but we will not further study the decision-making process here. before ([II-B3](#S2.SS2.SSS3.Px1 "Computing individual Pareto optima ‣ II-B3 Overview of methods ‣ II-B Multi-objective optimization ‣ II Preliminaries ‣ Multi-objective Deep Learning: Taxonomy and Survey of the State of the Art")) or after ([II-B3](#S2.SS2.SSS3.Px2 "Computing the entire set ‣ II-B3 Overview of methods ‣ II-B Multi-objective optimization ‣ II Preliminaries ‣ Multi-objective Deep Learning: Taxonomy and Survey of the State of the Art")) optimization, or even consider an interactive approach ([II-B3](#S2.SS2.SSS3.Px3 "Interactive methods ‣ II-B3 Overview of methods ‣ II-B Multi-objective optimization ‣ II Preliminaries ‣ Multi-objective Deep Learning: Taxonomy and Survey of the State of the Art")).

##### Computing individual Pareto optima

When computing a single Pareto optimal (or critical) point, this means that some decision has been made beforehand to guide which point θ∗\\theta^{\*} is sought. The two main approaches are to actively decide by means of a parameter-dependent *scalarization*, or to accept any optimal point, i.e., to leave the decision-making to chance.

In the former case, we synthesize the loss vector L⁡(θ)L(\\theta) into a single loss function L^​(θ)∈ℝ\\hat{L}(\\theta)\\in\\mathbb{R}, meaning that we scalarize the optimization problem, see \[[119](#bib.bib119 ""), [43](#bib.bib43 "")\] for detailed overviews. As a consequence, we can leverage the entire literature on single objective algorithms such as gradient descent or (Quasi-)Newton methods.

Scalarization requires the selection of a set of weights ww. In many cases, we have w∈ℝKw\\in\\mathbb{R}^{K}, i.e., as many weights as we have objectives. The simplest technique is the *weighted sum*

L^​(θ)\=∑k\=1Kwk​Lk​(θ),\\hat{L}(\\theta)=\\sum\_{k=1}^{K}w\_{k}L\_{k}(\\theta),

(WS)

which is also implicitly used in all sorts of regularization techniques (e.g., minθ⁡L⁡(θ)+λ​‖θ‖22\\min\_{\\theta}L(\\theta)+\\lambda\\|\\theta\\|\_{2}^{2}, λ∈ℝ\>0\\lambda\\in\\mathbb{R}\_{>0}. Then, w1\=1/(1+λ)w\_{1}=1/(1+\\lambda) and w2\=λ/(1+λ)w\_{2}=\\lambda/(1+\\lambda)). Geometrically, the weights ww define hyperplanes determining which point on the front we will find. This is visualized in Figure [1](#S2.F1 "Figure 1 ‣ II-B1 Definition and concepts ‣ II-B Multi-objective optimization ‣ II Preliminaries ‣ Multi-objective Deep Learning: Taxonomy and Survey of the State of the Art"), where we also see that non-convex problems immediately result in non-uniqueness. There are thus many, more sophisticated alternatives to ([WS](#S2.Ex15 "In Computing individual Pareto optima ‣ II-B3 Overview of methods ‣ II-B Multi-objective optimization ‣ II Preliminaries ‣ Multi-objective Deep Learning: Taxonomy and Survey of the State of the Art")) such as the ϵ\\epsilon-constraint method, where all objectives but one are transformed into constraints:

minθ∈ℝq⁡Lk​(θ)s.t.Li​(θ)≤ϵi​∀i∈{1,…,K}∖k.\\min\_{\\theta\\in\\mathbb{R}^{q}}L\_{k}(\\theta)\\quad\\mbox{s.t.}\\quad L\_{i}(\\theta)\\leq\\epsilon\_{i}\\penalty\\ \\forall\\penalty\\ i\\in\\{1,\\ldots,K\\}\\setminus k.

(3)

Alternatively, preference vectors \[[180](#bib.bib180 ""), [44](#bib.bib44 "")\] (also known under the name *Pascoletti-Serafini* \[[134](#bib.bib134 "")\]) are highly popular. This approach is visualized by the triangles in Figure [3](#S2.F3 "Figure 3 ‣ Computing individual Pareto optima ‣ II-B3 Overview of methods ‣ II-B Multi-objective optimization ‣ II Preliminaries ‣ Multi-objective Deep Learning: Taxonomy and Survey of the State of the Art"), where the optimization problem is transformed into stepping as far as possible along the direction defined by the preference vector. While these advanced techniques beyond ([WS](#S2.Ex15 "In Computing individual Pareto optima ‣ II-B3 Overview of methods ‣ II-B Multi-objective optimization ‣ II Preliminaries ‣ Multi-objective Deep Learning: Taxonomy and Survey of the State of the Art")) are capable of handling non-convex problems, they usually introduce more challenging single-objective problems due to additional constraints (cf. ([3](#S2.E3 "In Computing individual Pareto optima ‣ II-B3 Overview of methods ‣ II-B Multi-objective optimization ‣ II Preliminaries ‣ Multi-objective Deep Learning: Taxonomy and Survey of the State of the Art"))).

Figure 3: Sketch of different multiobjective optimization concepts. Entire front 𝒫ℱ\\mathcal{P}\_{\\mathcal{F}} via hypervolume maximization (circles) or single point via gradient descent (squares) or preference vector scalarization (triangles).

As an alternative, *multiple-gradient descent algorithms (MGDAs)* are increasingly popular, in particular when it comes to very high-dimensional problems. The key ingredient is the calculation of a *common descent direction* d⁡(θ)∈ℝqd(\\theta)\\in\\mathbb{R}^{q} that satisfies

(∇Lk​(θ))⊤​d​(θ)<0,k∈{1,…,K},\\left(\\nabla L\_{k}(\\theta)\\right)^{\\top}d(\\theta)<0,\\quad k\\in\\{1,\\ldots,K\\}, which again is a straightforward extension of single-objective descent directions. The determination of such a dd usually requires the solution of a subproblem in each step, for instance a quadratic problem of dimension KK \[[155](#bib.bib155 ""), [42](#bib.bib42 "")\], d⁡(θ)\\displaystyle d(\\theta)

\=−∑k\=1Kwk∇Lk(θ),where\\displaystyle=-\\sum\_{k=1}^{K}w\_{k}\\nabla L\_{k}(\\theta),\\qquad\\mbox{where}

(CDD)

w\\displaystyle w

\=argminw^∈\[0,1\]K∑w^k\=1‖∑k\=1Kw^k∇Lk(θ)‖22.\\displaystyle=\\arg\\min\_{\\begin{array}\[\]{c}\\scriptstyle{\\hat{w}\\in\[0,1\]^{K}}\\\\ \\scriptstyle{\\sum\\hat{w}\_{k}=1}\\end{array}}\\left\\|\\sum\_{k=1}^{K}\\hat{w}\_{k}\\nabla L\_{k}(\\theta)\\right\\|\_{2}^{2}.

However, there are various alternatives to ([CDD](#S2.Ex17 "In Computing individual Pareto optima ‣ II-B3 Overview of methods ‣ II-B Multi-objective optimization ‣ II Preliminaries ‣ Multi-objective Deep Learning: Taxonomy and Survey of the State of the Art")) such as a dual formulation \[[51](#bib.bib51 "")\] or a computationally cheaper so-called *Franke-Wolfe* approach \[[161](#bib.bib161 "")\]. Once a common descent direction d⁡(θ)d(\\theta) has been obtained, we proceed in a standard fashion by iteratively updating θ\\theta until convergence or some other stopping criterion is met, cf. Algorithm [1](#alg1 "Algorithm 1 ‣ Computing individual Pareto optima ‣ II-B3 Overview of methods ‣ II-B Multi-objective optimization ‣ II Preliminaries ‣ Multi-objective Deep Learning: Taxonomy and Survey of the State of the Art") as well as the squares in Figure [3](#S2.F3 "Figure 3 ‣ Computing individual Pareto optima ‣ II-B3 Overview of methods ‣ II-B Multi-objective optimization ‣ II Preliminaries ‣ Multi-objective Deep Learning: Taxonomy and Survey of the State of the Art") for an illustration.

Algorithm 1 Multiple-gradient descent algorithm (MGDA)

1: Initial guess θ(0)\\theta^{(0)}, learning rate η∈ℝ\>0\\eta\\in\\mathbb{R}\_{>0} (possibly adaptive), maximum number of iterations i𝗆𝖺𝗑i\_{\\mathsf{max}}, hyperparameters (depending on specific version of MGDA) 

2: θ∗∈𝒫c\\theta^{\*}\\in\\mathcal{P}\_{c} 

3: Set i\=0i=0 

4: while θ(i)∉𝒫c\\theta^{(i)}\\notin\\mathcal{P}\_{c} and i<i𝗆𝖺𝗑i\<i\_{\\mathsf{max}} do 

5:   Calculate gradients ∇Li​(θ(i))\\nabla L\_{i}\\left(\\theta^{(i)}\\right) for i\=1​…,ki=1\\ldots,k 

6:   Calculate descent direction d⁡(θ(i))d\\left(\\theta^{(i)}\\right) (e.g., via ([CDD](#S2.Ex17 "In Computing individual Pareto optima ‣ II-B3 Overview of methods ‣ II-B Multi-objective optimization ‣ II Preliminaries ‣ Multi-objective Deep Learning: Taxonomy and Survey of the State of the Art"))) 

7:   If adaptive, determine learning rate η⁡(θ(i))\\eta\\left(\\theta^{(i)}\\right) 

8:   Update θ\\theta:

θ(i+1)\=θ(i)+η⁡(θ(i))​d​(θ(i))\\theta^{(i+1)}=\\theta^{(i)}+\\eta\\left(\\theta^{(i)}\\right)d\\left(\\theta^{(i)}\\right)

9:   i\=i+1i=i+1 

10: end while 

Various extensions concern Newton \[[50](#bib.bib50 "")\] or Quasi-Newton \[[140](#bib.bib140 "")\] directions, uncertainties \[[137](#bib.bib137 "")\], momentum \[[179](#bib.bib179 ""), [168](#bib.bib168 ""), [132](#bib.bib132 "")\], or non-smoothness \[[119](#bib.bib119 ""), [57](#bib.bib57 ""), [178](#bib.bib178 ""), [179](#bib.bib179 "")\].

As laid out at the beginning of this section, this approach omits the decision-making. MGDAs yield Pareto critical points θ∗∈𝒫c\\theta^{\*}\\in\\mathcal{P}\_{c}, but one usually cannot determine which one, and how the different goals are prioritized in that point. To achieve this, one needs to resort to hybrid approaches including, e.g., preference vectors \[[208](#bib.bib208 "")\].

##### Computing the entire set

If we want to postpone the decision-making to take a more informed decision, we need to calculate the entire Pareto set 𝒫\\mathcal{P} and front 𝒫ℱ\\mathcal{P}\_{\\mathcal{F}}. The most straightforward approach is to adapt the weights ww in scalarization and solve the single-objective problem multiple times. Alternatively, one can combine MGDA with a multi-start strategy (i.e., a set of random initial guesses {θ(0,j)}j\=1M\\left\\{\\theta^{(0,j)}\\right\\}\_{j=1}^{M}) to obtain multiple points. However, in both cases, it may be very hard or even impossible to obtain a good coverage of 𝒫\\mathcal{P}, i.e., that approximates the entire set with evenly distributed points.

###### Remark 3 (Box coverings).

An alternative to approximating 𝒫\\mathcal{P} by a finite set of points is to introduce an outer box covering (see, e.g., \[[159](#bib.bib159 ""), [38](#bib.bib38 "")\]). In theory, the numerical effort for a suitably fine covering grows exponentially with the dimension of the object we want to approximate (i.e., with the number of objectives KK), but is independent of the parameter dimension qq. This is good for the common case of few objectives. However, in practice, set-based numerics often also scale with qq due to the need to represent the boxes via Monte Carlo sampling, thus rendering them too expensive for applications in machine learning.

Instead of parameter variation or using multi-start, we can directly consider a *population* of weights {θ(j)}j\=1M\\left\\{\\theta^{(j)}\\right\\}\_{j=1}^{M} that we iteratively update to improve each individual’s performance while also ensuring a suitable spread over the entire front 𝒫ℱ\\mathcal{P}\_{\\mathcal{F}}. The population’s performance is often measured by the hypervolume metric (see, e.g., \[[9](#bib.bib9 "")\]), which is also visualized in Figure [3](#S2.F3 "Figure 3 ‣ Computing individual Pareto optima ‣ II-B3 Overview of methods ‣ II-B Multi-objective optimization ‣ II Preliminaries ‣ Multi-objective Deep Learning: Taxonomy and Survey of the State of the Art"). This is defined as the union of the boxes spanned by some reference point (shown in black) and one of the population’s individuals, respectively. One can attempt to directly maximize this metric, for instance using Newton’s method \[[169](#bib.bib169 "")\], see also the survey \[[11](#bib.bib11 "")\] for for an extensive introduction.

The more popular alternative when optimizing an entire population is via *multi-objective evolutionary algorithms (MOEAs)* \[[34](#bib.bib34 "")\], see Figure [4](#S2.F4 "Figure 4 ‣ Computing the entire set ‣ II-B3 Overview of methods ‣ II-B Multi-objective optimization ‣ II Preliminaries ‣ Multi-objective Deep Learning: Taxonomy and Survey of the State of the Art") for an illustration and Algorithm [2](#alg2 "Algorithm 2 ‣ Computing the entire set ‣ II-B3 Overview of methods ‣ II-B Multi-objective optimization ‣ II Preliminaries ‣ Multi-objective Deep Learning: Taxonomy and Survey of the State of the Art") for a rough algorithmic outline. Therein, the pupulation’s fitness is increased from generation to generation by maximizing a criterion that combines optimality (or non-dominance) with a spreading criterion. The most popular and widely used algorithm in this category is likely NSGA-II \[[35](#bib.bib35 "")\], but there are many alternatives regarding the crossover step (Step 3 in Algorithm [2](#alg2 "Algorithm 2 ‣ Computing the entire set ‣ II-B3 Overview of methods ‣ II-B Multi-objective optimization ‣ II Preliminaries ‣ Multi-objective Deep Learning: Taxonomy and Survey of the State of the Art")), mutation (Step 4), the selection (Step 5), the population size, and so on. For more details, see the surveys \[[53](#bib.bib53 ""), [209](#bib.bib209 ""), [181](#bib.bib181 "")\]. Finally, many combinations exist with—for instance—preference vectors \[[180](#bib.bib180 "")\] or gradients \[[21](#bib.bib21 ""), [132](#bib.bib132 "")\], such that the creation of offspring is more directed using gradient information.

Figure 4: MOEA example, where a population of individuals is improved from one generation to the next (■→○→△→□\\blacksquare\\rightarrow\\bigcirc\\rightarrow\\triangle\\rightarrow\\square).

Algorithm 2 Multi-objective evolutionary algorithm (MOEA)

1: Initial population P⁡(0)P(0) of individuals {θ(0,j)}j\=1M\\left\\{\\theta^{(0,j)}\\right\\}\_{j=1}^{M}, number of generations i𝗆𝖺𝗑i\_{\\mathsf{max}}, hyperparameters (depending on specific version of MOEA) 

2: Set i\=0i=0 

3: while i<i𝗆𝖺𝗑i\<i\_{\\mathsf{max}} do 

4:   Create an offspring population P^​(i)\\widehat{P}{(i)} out of P⁡(i){P}{(i)}, e.g., using crossover between two individuals 

5:   Modify offspring population via mutation:

P\~​(i)\=ℳ​(P^​(i))\\widetilde{P}{(i)}=\\mathcal{M}\\left(\\widehat{P}{(i)}\\right)

6:   Selection of the next generation P⁡(i+1){P}{(i+1)} either from P\~​(i)\\widetilde{P}{(i)} or from P​(i)∪P\~​(i){P}{(i)}\\cup\\widetilde{P}{(i)} (the latter is called *elitism*) by a survival-of-the-fittest process (e.g., using a non-dominance and spread metric) 

7:   i\=i+1i=i+1 

8: end while 

A final technique falling into the category of approximating the entire Pareto set by a finite set of points is *continuation*. Rewriting the condition ([KKT](#S2.Ex13 "In II-B2 Gradients and optimality conditions ‣ II-B Multi-objective optimization ‣ II Preliminaries ‣ Multi-objective Deep Learning: Taxonomy and Survey of the State of the Art")) as a zero-finding problem, H⁡(θ∗,α∗)\=(∑k\=1Kα∗k∇Lk(θ∗)∑k\=1Kαk∗−1)\=0,H(\\theta^{\*},\\alpha^{\*})=\\begin{pmatrix}\\sum\_{k=1}^{K}\\alpha^{\*}\_{k}\\nabla L\_{k}(\\theta^{\*})\\\\ \\sum\_{k=1}^{K}\\alpha^{\*}\_{k}-1\\end{pmatrix}=0, we find that under suitable regularity assumptions (i.e., twice continuously differentiable losses), the implicit function theorem says that the zero level set of HH is a smooth manifold of dimension K−1K-1 \[[73](#bib.bib73 "")\]. The tangent space can be computed from the kernel of the Jacobian H′​(θ∗,α∗)∈ℝq+1×q+KH^{\\prime}(\\theta^{\*},\\alpha^{\*})\\in\\mathbb{R}^{q+1\\times q+K}, H′​(θ∗,α∗)\=(∑k\=1Kαk∗​∇2Lk​(θ∗)L1​(θ∗)…LK​(θ∗)0…01…1).H^{\\prime}(\\theta^{\*},\\alpha^{\*})=\\begin{pmatrix}\\sum\_{k=1}^{K}\\alpha^{\*}\_{k}\\nabla^{2}L\_{k}(\\theta^{\*})&L\_{1}(\\theta^{\*})&\\ldots&L\_{K}(\\theta^{\*})\\\\ 0\\quad\\ldots\\quad 0&1&\\ldots&1\\end{pmatrix}.

The procedure is then to start from a known Pareto optimum, compute a predictor within the tangent space and then compute the next Pareto critical point through a corrector step (e.g., using MGDA), see Figure [5](#S2.F5 "Figure 5 ‣ Computing the entire set ‣ II-B3 Overview of methods ‣ II-B Multi-objective optimization ‣ II Preliminaries ‣ Multi-objective Deep Learning: Taxonomy and Survey of the State of the Art") and Algorithm [3](#alg3 "Algorithm 3 ‣ Computing the entire set ‣ II-B3 Overview of methods ‣ II-B Multi-objective optimization ‣ II Preliminaries ‣ Multi-objective Deep Learning: Taxonomy and Survey of the State of the Art"). However, in addition to the gradients, we also require the Hessians of all loss functions. These are q×qq\\times q matrices and thus expensive to calculate as well as to store. In addition, the C2C^{2} regularity is violated by many network architectures (e.g., when using ReLU activations) or loss functions. An extension to Lipschitz continuous objectives can be found in \[[18](#bib.bib18 "")\], even though it is computationally even more expensive.

Figure 5: Continuation algorithms use predictor steps θ\~\\widetilde{\\theta} along the tangent space of 𝒫c\\mathcal{P}\_{c}, i.e., in parameter space (left). A corrector step then produces the next Pareto optimal point. The right plot shows the corresponding points in the objective space.

Algorithm 3 Continuation

1: Initial Pareto critical point θ(0)\\theta^{(0)} and KKT multiplier α(0)\\alpha^{(0)}, hyperparameters 

2: Set j\=0j=0 

3: while other end of front has not been reached do 

4:   Compute tangent space in θ(j)\\theta^{(j)} using kernel vectors of the weighted Hessian matrix H′​(θ(j),α(j))H^{\\prime}(\\theta^{(j)},\\alpha^{(j)}) 

5:   Predictor step along the tangent space →\\rightarrow θ\~(j+1)\\widetilde{\\theta}^{(j+1)} 

6:   Corrector step to obtain the next Pareto critical point θ(j+1)\\theta^{(j+1)} 

7:   j\=j+1j=j+1 

8: end while 

##### Interactive methods

Regardless of the order of decision-making and optimization, all previously mentioned approaches can be implemented in a block-wise manner, i.e., a single run of an algorithm. Instead, *interactive* methods (e.g., \[[119](#bib.bib119 ""), [47](#bib.bib47 ""), [94](#bib.bib94 ""), [158](#bib.bib158 "")\]) alternate between decision-making and optimization. The approach usually starts from a Pareto optimum. Then, based on the preference of the decision maker (e.g., “improve objective L1L\_{1}, do not get worse in L2L\_{2}, but a drop in L3L\_{3} is acceptable”), we compute another Pareto optimum that respects this prioritization. Very naturally, these algorithms have a close relation to continuation methods, where the decision maker’s preference needs to be translated into a suitable predictor direction in the tangent space.

### II-C Multi-objective machine learning

The combination of multi-objective optimization and machine learning has been studied for several decades already, see \[[80](#bib.bib80 ""), [81](#bib.bib81 "")\] for overviews. While the combination is a very natural one due to various performance criteria that are relevant in learning models from data, the focus has mostly been on other types of machine learning than deep neural networks. From the authors’ point of view, the main reason is the large computational cost that comes with both multi-objective optimization and deep learning, which renders their combination very challenging. In Table [I](#S2.T1 "Table I ‣ II-C Multi-objective machine learning ‣ II Preliminaries ‣ Multi-objective Deep Learning: Taxonomy and Survey of the State of the Art"), we list the main pros and cons of various MO techniques when it comes to deep learning, in particular concerning the large computational cost.

Table I: Pros and cons of various classes of multi-objective optimization algorithms. The main computational challenges in the context of deep learning are shown in bold.

Algorithm

Advantages

 Challenges 

class

MOEAs

+\\bm{+} often gradient-free

 −\\bm{-} large # of fcn. evals. 

+\\bm{+} global optimization

 −\\bm{-} slow convergence 

Scalarization

+\\bm{+} single-objective opt.

 −\\bm{-} parametrization via ww 

 −\\bm{-} additional constraints 

MGDA

+\\bm{+} convergence similar

 −\\bm{-} no steering 

to single-objective opt.

 −\\bm{-} sub-routine for direction 

 −\\bm{-} single optimum 

Continuation

+\\bm{+} fast convergence

 −\\bm{-} req. smoothness: C2C^{2} 

 −\\bm{-} Hessian calculation 

 −\\bm{-} only connected fronts 

Thus—before turning our focus on deep learning in the next section—we here want to give a brief list of ways to incorporate multi-objective optimization with machine learning:

*   •

```
Feature selection \[[162](#bib.bib162 ""), [60](#bib.bib60 ""), [4](#bib.bib4 "")\],
```
*   •

```
hyperparameter tuning \[[84](#bib.bib84 ""), [125](#bib.bib125 "")\],
```
*   •

```
architecture search \[[82](#bib.bib82 "")\],
```
*   •

```
data imputation \[[113](#bib.bib113 "")\]
```
*   •

```
training with respect to multiple objectives, for instance, support vector machines \[[174](#bib.bib174 ""), [10](#bib.bib10 "")\], decision trees, bayesian classifiers, radial basis function networks, or clustering (see also the references in \[[80](#bib.bib80 ""), [81](#bib.bib81 ""), [6](#bib.bib6 "")\]).
```
*   •

```
multi-objective clustering \[[7](#bib.bib7 ""), [61](#bib.bib61 ""), [122](#bib.bib122 "")\]
```
As in particular the last point sets deep learning apart from other machine learning techniques, this will be the focus of our taxonomy in the next Section.

###### Remark 4.

It should be noted that there already exist several surveys on the topic of multi-objective machine learning, specifically in the context of MOEAs \[[6](#bib.bib6 ""), [181](#bib.bib181 "")\] and hyperparameter tuning \[[125](#bib.bib125 ""), [84](#bib.bib84 "")\]. Moreover, three articles have introduced taxonomies of multi-objective reinforcement learning algorithms \[[106](#bib.bib106 ""), [70](#bib.bib70 ""), [48](#bib.bib48 "")\], which share similarities with our classification in the next section. Nevertheless, we believe that this article closes a gap in particular in the areas of deep neural network training and gradient-based approaches (i.e., MGDAs), which are still less popular in the optimization community than MOEAs.

## III A taxonomy of multi-objective deep learning

Before we introduce our taxonomy for the multi-objective training procedure of deep neural networks in Section [III-A](#S3.SS1 "III-A Taxonomy of the training procedure ‣ III A taxonomy of multi-objective deep learning ‣ Multi-objective Deep Learning: Taxonomy and Survey of the State of the Art"), let us first distinguish between the various purposes of multi-objective optimization in deep learning. These are, in no particular order,

1.  (i)

```
preprocessing steps such as feature selection or data imputation,
```
2.  (ii)

    the treatment of multiple primary tasks such

    *   •

        multi-task learning,

    *   •

        multi-class classification,

    *   •

        clustering with respect to multiple criteria,

    *   •

        multiple rewards in reinforcement learning,
3.  (iii)

    the consideration of secondary tasks, e.g.,

    *   •

        regularization,

    *   •

        sparsity,

    *   •

        fairness,

    *   •

        interpretability,

    *   •

        incorporation of prior knowledge,
4.  (iv)

```
using multi-objective optimization as a tool to improve the performance of a deep learning task.
```
While these are quite versatile tasks, they share the common structure that a multi-objective optimization problem of the form ([MOP](#S2.Ex11 "In II-B1 Definition and concepts ‣ II-B Multi-objective optimization ‣ II Preliminaries ‣ Multi-objective Deep Learning: Taxonomy and Survey of the State of the Art")) needs to be solved, for which one can pursue various strategies.

### III-A Taxonomy of the training procedure

Figure [6](#S3.F6 "Figure 6 ‣ III-A Taxonomy of the training procedure ‣ III A taxonomy of multi-objective deep learning ‣ Multi-objective Deep Learning: Taxonomy and Survey of the State of the Art") shows our taxonomy of multi-objective optimization methods for deep learning problems. In the following, let’s discuss various alternatives while taking into account the advantages and challenges highlighted in Table [I](#S2.T1 "Table I ‣ II-C Multi-objective machine learning ‣ II Preliminaries ‣ Multi-objective Deep Learning: Taxonomy and Survey of the State of the Art").

Figure 6: Taxonomy of methods. The squares denote design decisions, the ellipses denote classes of algorithms. Dashed ellipses refer to extensions building on other algorithms. In the context of deep learning, gradient-descent (i.e., MGDA) lies at the heart of most successful approaches.

Before diving in, it should be noted that scalarization takes a special role since we essentially eliminate the multi-objective nature of the problem, which allows us to leverage many techniques from single-objective optimization. In particular, the weighted sum approach ([WS](#S2.Ex15 "In Computing individual Pareto optima ‣ II-B3 Overview of methods ‣ II-B Multi-objective optimization ‣ II Preliminaries ‣ Multi-objective Deep Learning: Taxonomy and Survey of the State of the Art")) introduces no additional constraints and is implicitly being used whenever penalty terms are added, e.g., for regularization or sparsity. ϵ\\epsilon constraint ([3](#S2.E3 "In Computing individual Pareto optima ‣ II-B3 Overview of methods ‣ II-B Multi-objective optimization ‣ II Preliminaries ‣ Multi-objective Deep Learning: Taxonomy and Survey of the State of the Art")) or the Pascoletti-Serafini render the optimization significantly more challenging due to the constraints, which is why they are much less common in the context deep learning.

Turning our attention to the taxonomy in Figure [6](#S3.F6 "Figure 6 ‣ III-A Taxonomy of the training procedure ‣ III A taxonomy of multi-objective deep learning ‣ Multi-objective Deep Learning: Taxonomy and Survey of the State of the Art"), the first decision one has to take is whether the entire Pareto front is desired, or if we are instead satisfied with a single optimal compromise. The former results in substantial additional cost, which is why the deep learning community has until now mostly opted for the “decide-then-optimize” strategy. Besides the just-mentioned simple weighted-sum approach, MGDA is by far the most popular option, since—aside from the need to solve ([CDD](#S2.Ex17 "In Computing individual Pareto optima ‣ II-B3 Overview of methods ‣ II-B Multi-objective optimization ‣ II Preliminaries ‣ Multi-objective Deep Learning: Taxonomy and Survey of the State of the Art")) (or a similar sub-problem) instead of simply calculating a single gradient—it allows us to use all the existing machinery for gradient-based training, i.e., momentum, stochastic gradients, etc.

In the case of “optimize-then-decide”, there currently appear to be two similarly popular approaches. First, evolutionary algorithms are extremely popular and at the same time easy to use, which is why they are a natural choice for a first attempt at multi-objective optimization. However, due to their large cost and otherwise slow convergence, a hybridization with gradients is advisable in the context of deep learning. Second, scalarization problems can easily be used to find multiple Pareto optima simply by varying the weight vector ww. This has been used extensively, in particular in combination with the weighted sum. When it comes to continuation, there is until now little work, as the computational cost associated with second order information can quickly become prohibitively large.

The third option of interactive multi-objective deep learning has—to the best of our knowledge—not yet been explored with multiple objectives. Research in this area is also known as *interactive machine learning (IML)* \[[17](#bib.bib17 ""), [23](#bib.bib23 "")\] or *human-in-the-loop (HITL)* \[[196](#bib.bib196 ""), [126](#bib.bib126 "")\]. However, most concepts are related to active learning or to human intermediate tasks such as data annotation, or human feedback for interpretability or performance improvement.22 2 In particular in the literature on large language models, human feedback for fine tuning is an important contribution. A research direction such as interacting with a decision maker in terms of design criteria—as is popular in other multi-objective contexts \[[119](#bib.bib119 ""), [94](#bib.bib94 ""), [47](#bib.bib47 ""), [158](#bib.bib158 "")\]—remains a task for future research.

###### Remark 5 (Special role of reinforcement learning).

Due to its sequential decision-making character, reinforcement learning is fundamentally different from the other learning paradigms. The goal is to find a policy π\\pi from interaction with a system, which means that presenting a Pareto front to a human decision maker is not the scenario RL is intended for. Common themes in multi-objective RL are thus decide-then-optimize (i.e., the left branch of Figure [6](#S3.F6 "Figure 6 ‣ III-A Taxonomy of the training procedure ‣ III A taxonomy of multi-objective deep learning ‣ Multi-objective Deep Learning: Taxonomy and Survey of the State of the Art")), known under the term single-policy algorithms. Besides, multiple-policy algorithms follow the right branch in spirit, but the “population” in that context is usually a set of value or QQ functions. The decision-making is then performed online, based on an algorithmic decision rule or user-defined weighting. Due to this reason, we will cover this case separately in Section [V](#S5 "V Deep multi-objective reinforcement learning ‣ Multi-objective Deep Learning: Taxonomy and Survey of the State of the Art") as a special case.

## IV Multi-objective deep learning survey

Following the taxonomy in Figure [6](#S3.F6 "Figure 6 ‣ III-A Taxonomy of the training procedure ‣ III A taxonomy of multi-objective deep learning ‣ Multi-objective Deep Learning: Taxonomy and Survey of the State of the Art"), the goal of this section is to provide an overview of the current state of the art, and we are going to separate the contributions according to the different learning paradigms introduced in Section [II-A](#S2.SS1 "II-A Deep learning ‣ II Preliminaries ‣ Multi-objective Deep Learning: Taxonomy and Survey of the State of the Art"). Before we dive in, we would like to highlight the various types of objectives that can be pursued with multi-objective optimization. There are multiple reasons to treat deep learning as a multi-objective problem, despite the increased cost:

1.  (i)

    the consideration of multiple, equally important criteria, for example in the context of

    *   •

        multi-task learning,

    *   •

        multi-class classification,
2.  (ii)

```
trade-offs between different performance indicators, e.g., fairness versus bias,
```
3.  (iii)

```
balancing of knowledge and data, for instance in the field of physics-informed machine learning,
```
4.  (iv)

    inclusion of secondary objectives such as

    *   •

        sparsity / regularization paths,

    *   •

        interpretability,
5.  (v)

```
multi-objective optimization as a performance-enhancing paradigm.
```
###### Remark 6 (A note on overparametrization).

It is well known that many deep learning architectures possess such a large number of degrees of freedom that they can essentially fit any label structure. This phenomenon is known as *overparametrization* \[[130](#bib.bib130 "")\]. A side-effect of this phenomenon in the context of multi-objective learning is that in the case of very powerful function approximators (i.e., large networks), one can in principle resolve the conflict between different objectives. This means that for some tasks such as multi-task learning, the Pareto front collapses to a single point, cf. Figure [1](#S2.F1 "Figure 1 ‣ II-B1 Definition and concepts ‣ II-B Multi-objective optimization ‣ II Preliminaries ‣ Multi-objective Deep Learning: Taxonomy and Survey of the State of the Art") on the right. An example of this can be found in the MultiMNIST example in \[[161](#bib.bib161 "")\].

### IV-A Supervised learning

#### IV-A1 Multiobjective Gradient Descent Algorithms

The huge success of machine learning has also had a strong impact on the optimization community and the research directions pursued therein. For example, stochastic gradient descent has been widely studied. Even though these studies do not all consider deep learning problems, they have expensive problems (such as deep neural network training) in mind, for instance when developing multi-objective extensions of stochastic gradient descent \[[110](#bib.bib110 "")\], momentum-based algorithms for accelerated convergence \[[179](#bib.bib179 ""), [168](#bib.bib168 ""), [167](#bib.bib167 ""), [132](#bib.bib132 "")\], or combinations thereof as in the multi-objective Adam algorithm \[[123](#bib.bib123 "")\].

Other works directly consider deep learning applications, a very prominent example being \[[161](#bib.bib161 "")\]. Therein, a so-called *Frank Wolfe* routine was presented that replaces the sub-routine ([CDD](#S2.Ex17 "In Computing individual Pareto optima ‣ II-B3 Overview of methods ‣ II-B Multi-objective optimization ‣ II Preliminaries ‣ Multi-objective Deep Learning: Taxonomy and Survey of the State of the Art")) in Algorithm [1](#alg1 "Algorithm 1 ‣ Computing individual Pareto optima ‣ II-B3 Overview of methods ‣ II-B Multi-objective optimization ‣ II Preliminaries ‣ Multi-objective Deep Learning: Taxonomy and Survey of the State of the Art") with a cheaper version. This method is then used to train a deep neural network with a shared parameter section for representation learning, followed by task-specific layers for the individual tasks (here the identification of multiple handwritten digits). Interestingly, there appears to be no conflict between the two tasks, which we believe is due to the overparametrization effect mentioned in Remark [6](#Thmremark6 "Remark 6 (A note on overparametrization). ‣ IV Multi-objective deep learning survey ‣ Multi-objective Deep Learning: Taxonomy and Survey of the State of the Art"). Alternative gradient-based frameworks for deep multi-task are *PCGrad* \[[205](#bib.bib205 "")\]—where each task’s gradient is projected onto the normal plane of the other gradients to avoid expensive sub-problems such as ([CDD](#S2.Ex17 "In Computing individual Pareto optima ‣ II-B3 Overview of methods ‣ II-B Multi-objective optimization ‣ II Preliminaries ‣ Multi-objective Deep Learning: Taxonomy and Survey of the State of the Art"))— or *CAGrad* \[[105](#bib.bib105 "")\], where ([CDD](#S2.Ex17 "In Computing individual Pareto optima ‣ II-B3 Overview of methods ‣ II-B Multi-objective optimization ‣ II Preliminaries ‣ Multi-objective Deep Learning: Taxonomy and Survey of the State of the Art")) is replaced by a (rather similar) formulation that tries to reduce conflicts between the individual tasks. In \[[112](#bib.bib112 "")\], the *Stein Variational Gradient Descent* (SVGD) algorithm from \[[109](#bib.bib109 "")\] is extended to multiple objectives and tested on a large number of different problems such as accuracy-vs-fairness or multi-task learning. In the *NashMTL* procedure \[[129](#bib.bib129 "")\], multi-task learning is considered from a game-theoretic perspective, more precisely as a bargaining game.

Several authors have extended the basic MGDA procedure to obtain an approximation of the entire front. For instance in \[[103](#bib.bib103 "")\], a multi-start, constrained version of MGDA is developed, where disjoint feasible cones ensure a good spread, see Figure [7](#S4.F7 "Figure 7 ‣ IV-A1 Multiobjective Gradient Descent Algorithms ‣ IV-A Supervised learning ‣ IV Multi-objective deep learning survey ‣ Multi-objective Deep Learning: Taxonomy and Survey of the State of the Art") for an illustration. In \[[112](#bib.bib112 "")\], Langevin dynamics are used to ensure a good spread within a population of individuals that are iterated via the SVGD algorithm.

Figure 7: Pareto multi-task learning, where a population of points is optimized via MGDA, but each individual is restricted to an individual cone \[[103](#bib.bib103 "")\].

#### IV-A2 Scalarization

In the area of scalarization, the vast majority of deep learning algorithms simply uses the weighted sum whenever more than a single loss is considered, such as additional regularization terms, or different loss contributions such as physics and data loss in physics-informed machine learning \[[85](#bib.bib85 "")\]. Very often, this is simply done without naming it multi-objective optimization, as it is such a common theme. In the following, we will discuss scalarization techniques going beyond this weighted sum approach. For instance, in \[[104](#bib.bib104 "")\], the weighted sum is used but modified in terms of a dynamic weighting strategy over the iterations, such task losses are balanced efficiently. A so-called conic scalarizaiton techinque \[[86](#bib.bib86 "")\] which—loosely speaking—is a more sophisticated version of the weighted sum, is used in \[[75](#bib.bib75 "")\] for multi-objective encoder training to balance performance and adversarial robustness in image classification.

To obtain a good coverage of the entire front, the weighted sum weights are adaptively selected via the so-called *Non-Inferior Set Estimation* procedure in \[[142](#bib.bib142 "")\], which is a type of bisection method so that the points are better distributed. As an example, they consider a multinomial loss versus L2L\_{2} regularization. Weighted Chebyshev scalarization is used for multi-task learning in \[[74](#bib.bib74 "")\], combined with proximal gradient descent to take sparsity into account as the third objective. In \[[150](#bib.bib150 "")\] a weighted sum loss is balanced with a cosine similarity in order to improve the spread of points when varying the weight ww.

#### IV-A3 MOEAs

As laid out earlier, MOEAs are in most cases too expensive for deep neural network training. The number of generations (SS in Algorithm [2](#alg2 "Algorithm 2 ‣ Computing the entire set ‣ II-B3 Overview of methods ‣ II-B Multi-objective optimization ‣ II Preliminaries ‣ Multi-objective Deep Learning: Taxonomy and Survey of the State of the Art")) is usually very large, and so is the population size MM. As a consequence, MOEAs tend to require a very large number of function evaluations. Moreover, the crossover procedure often consists of randomly combining two individuals, which is a close-to-hopeless procedure for the parameter dimensions we find in realistic deep leraning applications. As a consequence, the usage of MOEAs is restricted to hybridized versions (cf. our taxonomy in Figure [6](#S3.F6 "Figure 6 ‣ III-A Taxonomy of the training procedure ‣ III A taxonomy of multi-objective deep learning ‣ Multi-objective Deep Learning: Taxonomy and Survey of the State of the Art")) as is for instance done in \[[112](#bib.bib112 "")\], where Langevin dynamics ensure a good spread of a population that is iterated using MGDA. Alternatively MOEAs can be used for other tasks in the context of deep learning such as neural architecture search (\[[46](#bib.bib46 "")\], more details in Section [IV-D](#S4.SS4 "IV-D Neural architecture search ‣ IV Multi-objective deep learning survey ‣ Multi-objective Deep Learning: Taxonomy and Survey of the State of the Art")) or the the selection between different networks. This was done in \[[24](#bib.bib24 "")\] for multiple association rule learning to foster interpretability.

#### IV-A4 Continuation

Similar to MOEAs—but due to a different reason—continuation methods suffer from large computational cost that tends to be prohibitive for deep learning applications. We have seen in Section [II-B3](#S2.SS2.SSS3.Px2 "Computing the entire set ‣ II-B3 Overview of methods ‣ II-B Multi-objective optimization ‣ II Preliminaries ‣ Multi-objective Deep Learning: Taxonomy and Survey of the State of the Art") that continuation requires the Hessians of all loss functions, which are expensive to calculate as well as to store. To this end, research has mostly focused on very specific settings such as the regularization path between a main loss and the sparsifying ℓ1\\ell\_{1} norm ‖θ‖1\\|\\theta\\|\_{1} \[[54](#bib.bib54 ""), [18](#bib.bib18 ""), [55](#bib.bib55 "")\]. Besides, there are two approaches trying to avoid Hessian computation. In \[[116](#bib.bib116 "")\], Hessians are approximated using the Krylov subspace method MINRES, whereas in \[[8](#bib.bib8 "")\], the minimization of one of the objectives is used as a suboptimal proxy for the predictor step.

### IV-B Unsupervised and self-supervised learning

When acquiring labeled data is expensive or impractical, unsupervised and self-supervised learning methods offer novel opportunities in deep learning to balance diverse objectives.

#### IV-B1 Scalarization

A foundational approach is the simple summation of losses, where individual losses are combined using fixed weights \[[65](#bib.bib65 ""), [166](#bib.bib166 ""), [100](#bib.bib100 ""), [115](#bib.bib115 "")\]. Beyond straightforward loss aggregation, scalarization methods provide a more sophisticated way to balance objectives by converting them into a single scalarized loss. For instance, \[[190](#bib.bib190 "")\] introduced an unsupervised active learning approach that integrates representativeness, informativeness, and diversity, while \[[31](#bib.bib31 "")\] employed a Pareto self-supervised training approach to harmonize self-supervised and supervised tasks in few-shot learning. These methods highlight the potential of scaling objectives to address problem-specific priorities.

#### IV-B2 Adaptive weighting

Unlike scalarization, adaptive weighting dynamically adjusts the importance of objectives throughout the training process. For example, \[[121](#bib.bib121 "")\] proposed self-adjusting weighted gradients to optimize the hypervolume in multi-task settings, demonstrating its application with denoising autoencoders. In \[[149](#bib.bib149 "")\], it is explored how ensemble methods can enhance self-supervised learning by adaptively weighting losses across ensembled projection heads during training.

#### IV-B3 Evolutionary algorithms

Since MOEAs tend to be too expensive for direct network optimization, \[[108](#bib.bib108 "")\] instead developed a structure-learning algorithm for deep neural networks based on multi-objective optimization, employing evolutionary strategies to identify optimal network architectures (see also the related Section [IV-D](#S4.SS4 "IV-D Neural architecture search ‣ IV Multi-objective deep learning survey ‣ Multi-objective Deep Learning: Taxonomy and Survey of the State of the Art") on neural architecture search). Expanding on this, \[[71](#bib.bib71 "")\] integrated generative adversarial networks (GANs) with evolutionary multi-objective optimization, demonstrating how GANs can enhance the exploration of diverse solutions in high-dimensional spaces. Combining self-supervised representation learning with evolutionary search, \[[197](#bib.bib197 "")\] developed a technique to discover optimal properties in an implicit chemical space.

Besides the above-mentioned references, the following studies address multi-objective optimization for unsupervised learning tasks such as feature selection \[[60](#bib.bib60 ""), [4](#bib.bib4 ""), [118](#bib.bib118 ""), [171](#bib.bib171 ""), [68](#bib.bib68 "")\] and clustering \[[61](#bib.bib61 "")\], but they do not explore their applicability to deep neural networks.

### IV-C Generative modeling

Generative modeling is currently one of the most popular areas of machine learning. Until now, the number of contributions with an explicit multi-objective treatment is rather limited, most of them in the area of generative adversarial networks. The works \[[41](#bib.bib41 ""), [5](#bib.bib5 "")\] consider multiple discriminator networks, even though they are not trained using multi-objective optimization. In \[[191](#bib.bib191 "")\], several GANs are weighted using the weighted sum method to have better control over the generated output. Adaptive weighted-sum-like approaches for multi-task training of multi-adversarial networks were developed in \[[67](#bib.bib67 ""), [56](#bib.bib56 "")\], and multi-adversarial domain adaptation via the weighted sum in \[[135](#bib.bib135 "")\].

There are even fewer articles in other domains of generative modeling. In \[[191](#bib.bib191 "")\], a reference-point scalarization—very similar to reference vector methods—was used for multi-objective training of variational autoencoders that create data according to multiple criteria \[[191](#bib.bib191 "")\]. Diffusion models were trained with respect to multiple criteria using MGDA in \[[204](#bib.bib204 "")\], and the multi-objective treatment of multiple judges for LLM feedback during the training phase was studied \[[199](#bib.bib199 "")\], where again weighted sum scalarization was used. It can thus be concluded that there is plenty of room for research in multi-objective generative modeling, be it with respect to more advanced optimization techniques beyond the weighted sum, or regarding advanced modeling approaches.

### IV-D Neural architecture search

Besides the direct use of multi-objective optimization algorithms for deep learning, which may be prohibitively expensive in many cases, one can also use multi-objective optimization on a meta level. Similar to algorithm selection techniques, *neural architecture search* (see \[[46](#bib.bib46 "")\] for a detailed overview in the single-objective case) describes the task to select the best neural network architecture for a given learning problem with respect to criteria such as predictive performance, inference time, or number of parameters. In this area, MOEAs such as NSGA-II are a popular choice, e.g., \[[82](#bib.bib82 ""), [45](#bib.bib45 ""), [201](#bib.bib201 "")\]. However, when using hypernetworks instead, this approach can be accelerated as well, for instance using MGDA \[[170](#bib.bib170 ""), [89](#bib.bib89 "")\].

### IV-E Applications

Before concluding, we here list a couple of applications, where multi-objective deep learning has proven to be helpful or even superior to the more established single-objective counterpart. This list is likely not exhaustive, but we will make an attempt to demonstrate the versatility of multi-objetive deep learning in various areas.

#### IV-E1 Language and video analysis and enhancement

Machine learning in the area of language, audio and video data is characterized by large amounts of data and long time series. Applications of multi-objective concepts to speech include the multi-target training of long-short-term memory networks for speech enhancement \[[172](#bib.bib172 ""), [200](#bib.bib200 "")\] or multi-objective speech recognition \[[153](#bib.bib153 "")\]. In the context of video data, the summarization, streaming optimization and short video generation was studied \[[39](#bib.bib39 ""), [133](#bib.bib133 "")\] and \[[198](#bib.bib198 "")\], respectively. For natural language processing, applications include text summation \[[152](#bib.bib152 "")\], text generation \[[139](#bib.bib139 "")\] prompt engineering for LLMs \[[12](#bib.bib12 "")\], meta-learning in terms of multi-objective LLM selection \[[98](#bib.bib98 "")\], and the multi-objective treatment of multiple judges for LLM feedback during the training phase \[[199](#bib.bib199 "")\]. Finally, multi-objective adversarial gesture generation by weighting various discriminators was studied in \[[49](#bib.bib49 "")\].

#### IV-E2 Engineering applications

This section covers all sorts of technical applications beyond video and language. For instance multi-task learning was used for for phoneme detection in \[[160](#bib.bib160 "")\], a generative model for air quality and weather prediction was trained in \[[67](#bib.bib67 "")\], medical image denoising by GANs trained with multiple discriminators was realized in \[[56](#bib.bib56 "")\], and multi-objective deep reinforcement learning was successfully used for workflow scheduling in \[[192](#bib.bib192 "")\].

#### IV-E3 Physics-informed machine learning

Finally, we would like to highlight the area of scientific machine learning, which has received tremendous attention in recent years. More specifically, *physics-informed neural networks (PINNs)* \[[85](#bib.bib85 "")\] are models that predict the solution of a differential equation. These are equations describing the dynamics of complex systems such as robots, fluid mechanics or nuclear fusion. The solution to such a differential equation is a function, the state of the system uu depending on time tt (and often on space ss as well), u⁡(t,s)u(t,s). The goal of PINNs is now to approximate u⁡(t,s)u(t,s) by a neural network that takes tt and ss as inputs, and produces uu as the output. Since the underlying equations are very often known, on can define a *physics loss*, meaning that the output satisfies the differential equation at a number of random points in the space-time domain. Generalization then ensures that fθ​(t,s)≈u⁡(t,s)f\_{\\theta}(t,s)\\approx u(t,s) for all tt and ss.

PINNs are a natural playground for multi-objective optimization, since we often have both physical knowledge and data, such that we have two losses that we would like to satisfy at the same time \[[85](#bib.bib85 "")\]. For clean data and exact dynamics, these objectives are not in conflict \[[3](#bib.bib3 "")\], but this is seldom the case for real applications, where data is noisy, the system equations are approximations, or the domain of interest cannot be defined exactly (e.g., the flow inside the human heart). In this area, multi-objective optimization has the potential to become quite important, even though research has now been limited to weighted sum training \[[144](#bib.bib144 "")\], as well as a comparison between MGDA and MOEA \[[3](#bib.bib3 "")\] for a few relatively small sample problems.

## V Deep multi-objective reinforcement learning

Due to its sequential decision-making nature, reinforcement learning (RL—see \[[173](#bib.bib173 ""), [16](#bib.bib16 "")\] for excellent overviews) takes a special role in the field of machine learning, with concepts that often differ significantly from the other learning paradigms. Nevertheless, it also shares features such as heavy usage of deep neural network function approximators or gradient-based learning. In the following, we thus briefly cover the basics of RL, before addressing modifications to our taxonomy in the context of RL and surveying the literature on deep multi-objective RL.

### V-A Basics

In contrast to supervised learning, RL follows a trial-and-error philosophy. That is, an agent interacts with its environment through actions a∈𝒜a\\in\\mathcal{A} and receives a reward r∈ℝr\\in\\mathbb{R}, indicating whether the action was beneficial or not. Through the action, the system state s∈𝒮s\\in\\mathcal{S} changes from time tt to t+1t+1 in a probabilistic manner according to the transition operator 𝒯:𝒮×𝒜→𝒫⁡(𝒮)\\mathcal{T}:\\mathcal{S}\\times\\mathcal{A}\\rightarrow\\mathcal{P}(\\mathcal{S}). Since the dynamics is independent of past states33 3 For instance, the transition in chess from one board position sts\_{t} to the next position st+1s\_{t+1} is independent of how sts\_{t} was reached (i.e., of st−1s\_{t-1}, st−2s\_{t-2}, …\\ldots)., this setting is referred to as a *Markov Decision Process (MDP)*.

The goal in RL is to find a *policy* π:𝒮→𝒫⁡(𝒜)\\pi:\\mathcal{S}\\rightarrow\\mathcal{P}(\\mathcal{A}) that maximizes the sum of discounted future rewards (with discount factor γ∈(0,1\]\\gamma\\in(0,1\]), also referred to as the *value*:

Vπ​(s)\=𝔼π​\[∑τ\=0∞γτ​rt+τ|st\=s\].V\_{\\pi}(s)=\\mathbb{E}\_{\\pi}\\left\[\\,\\sum\_{\\tau=0}^{\\infty}\\gamma^{\\tau}r\_{t+\\tau}\\biggm|s\_{t}=s\\,\\right\].

(4)

A closely related concept is the so-called QQ-function that determines the value of a state ss if we take action aa and then follow policy π\\pi, Qπ(s,a)\=𝔼π\[∑τ\=0∞γτrt+τ|st\=s,at\=a\].Q\_{\\pi}\\left(s,a\\right)=\\mathbb{E}\_{\\pi}\\left\[\\,\\sum\_{\\tau=0}^{\\infty}\\gamma^{\\tau}r\_{t+\\tau}\\biggm|s\_{t}=s,a\_{t}=a\\,\\right\].

(5)

We thus have the relation Vπ​(s)\=Qπ​(s,π⁡(s))V\_{\\pi}(s)=Q\_{\\pi}(s,\\pi(s)). Once we know QQ, we can determine the optimal action by evaluating QQ for all aa:

a∗\=arg⁡maxa∈𝒜​Qπ​(s,a).a^{\*}=\\arg\\max\_{a\\in\\mathcal{A}}Q\_{\\pi}\\left(s,a\\right).

(6)

If the action set is not finite, but continuous, then ([6](#S5.E6 "In V-A Basics ‣ V Deep multi-objective reinforcement learning ‣ Multi-objective Deep Learning: Taxonomy and Survey of the State of the Art")) becomes a (potentially expensive) nonlinear optimization problem itself. In such a situation, we can instead try to learn the policy π\\pi directly. Algorithms of this class usually make use of the *policy gradient theorem*, where we directly compute gradients of a performance criterion (e.g., the integral over the value function) with respect to the policy, often using a so-called *critic* to efficiently evaluate this gradient, see \[[173](#bib.bib173 "")\] for details.

To summarize, RL ultimately boils down to learning VπV\_{\\pi}, QπQ\_{\\pi} or π\\pi itself from experience, i.e., interactions with the environment. We are thus facing a dynamic optimization problem, where we iteratively update our behavior π\\pi in order to maximize our value. While this can conceptually be realized using the theory of Dynamic Programming \[[16](#bib.bib16 "")\] the complexity quickly supersedes all computing capacities. To circumvent this issue, deep reinforcement learning introduces neural network approximations of the above-mentioned functions. For discrete action spaces, *deep QQ-learning* (e.g., \[[186](#bib.bib186 "")\]) has proven very successful, whereas for continuous control tasks, policy gradient methods (e.g., the *Proximal Policy Optimization (PPO)* \[[157](#bib.bib157 "")\], the *deep deterministic policy gradient (DDPG)* \[[102](#bib.bib102 "")\] or the *Soft Actor Critic (SAC)* \[[66](#bib.bib66 "")\]) are the most prominent methods. There are numerous success stories of deep RL such as board or video games (Chess or Go \[[163](#bib.bib163 "")\], Atari \[[124](#bib.bib124 "")\]), robotics \[[93](#bib.bib93 "")\] or complex physics systems \[[37](#bib.bib37 ""), [189](#bib.bib189 ""), [138](#bib.bib138 "")\]. In particular for continuous action spaces, these algorithms heavily rely on gradient descent, similar to the supervised learning case.

### V-B MORL: adapted taxonomy and survey

As mentioned in Remark [5](#Thmremark5 "Remark 5 (Special role of reinforcement learning). ‣ III-A Taxonomy of the training procedure ‣ III A taxonomy of multi-objective deep learning ‣ Multi-objective Deep Learning: Taxonomy and Survey of the State of the Art"), the taxonomy can only be partially applied to multi-objective reinforcement learning (MORL) due to its sequential decision-making nature. In the MORL literature, the distinction between first-decide-then-optimize and first-optimize-then-decide—the top decision in our taxonomy in Figure [6](#S3.F6 "Figure 6 ‣ III-A Taxonomy of the training procedure ‣ III A taxonomy of multi-objective deep learning ‣ Multi-objective Deep Learning: Taxonomy and Survey of the State of the Art")—is also referred to as *single-policy* versus *multiple-policy* algorithms. The central deviation from our taxonomy lies in the second option, where we do not present a Pareto set to a decision maker, but calculate multiple utility functions, which are then synthesized in an automated fashion to yield a Pareto optimal policy, cf. Figure [8](#S5.F8 "Figure 8 ‣ V-B MORL: adapted taxonomy and survey ‣ V Deep multi-objective reinforcement learning ‣ Multi-objective Deep Learning: Taxonomy and Survey of the State of the Art").

Figure 8: Modification of the right branch of Figure [6](#S3.F6 "Figure 6 ‣ III-A Taxonomy of the training procedure ‣ III A taxonomy of multi-objective deep learning ‣ Multi-objective Deep Learning: Taxonomy and Survey of the State of the Art") for MORL. 

Independent of which approach we pursue, the multi-objective setting means that we have a vector-valued reward r∈ℝKr\\in\\mathbb{R}^{K}, as well as a vector-valued QQ function

Qπ​(s,a)\=(Qπ,1​(s,a)Qπ,K​(s,a)),Q\_{\\pi}(s,a)=\\begin{pmatrix}Q\_{\\pi,1}(s,a)\\\\ \\vdots\\\\ Q\_{\\pi,K}(s,a)\\end{pmatrix},

(7)

where the entries in ([7](#S5.E7 "In V-B MORL: adapted taxonomy and survey ‣ V Deep multi-objective reinforcement learning ‣ Multi-objective Deep Learning: Taxonomy and Survey of the State of the Art")) are as in ([5](#S5.E5 "In V-A Basics ‣ V Deep multi-objective reinforcement learning ‣ Multi-objective Deep Learning: Taxonomy and Survey of the State of the Art")) but with the individual rewards rkr\_{k}, respectively. The value function ([4](#S5.E4 "In V-A Basics ‣ V Deep multi-objective reinforcement learning ‣ Multi-objective Deep Learning: Taxonomy and Survey of the State of the Art")) is transformed accordingly. Quite naturally, as the goal is always to increase the value, one can now define a Pareto set 𝒫\\mathcal{P} of non-dominated value functions, see \[[145](#bib.bib145 "")\] for a detailed introduction. It should be mentioned, though, that the multi-objective treatment introduces significant additional challenges in terms of finding Pareto optimal policies, see \[[114](#bib.bib114 "")\] for details.

As mentioned earlier, there already exist several MORL overviews \[[183](#bib.bib183 ""), [145](#bib.bib145 ""), [106](#bib.bib106 ""), [70](#bib.bib70 "")\], which is why we are going to restrict our attention to the deep learning approaches. A general framework for the usage of various such deep MORL approaches has been presented in \[[131](#bib.bib131 "")\]. However, it must be said that conceptually, there is no real difference between deep MORL and other MORL algorithms—the conceptual treatment of multiple criteria is entirely in the synthesization of the policy, and thus largely independent of how the individual value or QQ functions are modeled/approximated.

The largest part of the MORL literature is concerned with QQ learning, i.e., learning approximations of QQ functions of the form ([7](#S5.E7 "In V-B MORL: adapted taxonomy and survey ‣ V Deep multi-objective reinforcement learning ‣ Multi-objective Deep Learning: Taxonomy and Survey of the State of the Art")) and then selecting the action by solving a multi-objective extension of ([6](#S5.E6 "In V-A Basics ‣ V Deep multi-objective reinforcement learning ‣ Multi-objective Deep Learning: Taxonomy and Survey of the State of the Art")). A large number of techniques relies on scalarizing the vector-valued QQ function ([7](#S5.E7 "In V-B MORL: adapted taxonomy and survey ‣ V Deep multi-objective reinforcement learning ‣ Multi-objective Deep Learning: Taxonomy and Survey of the State of the Art")) in a linear fashion using a weight vector w∈ℝKw\\in\\mathbb{R}^{K}, Q^π​(s,a)\\displaystyle\\hat{Q}\_{\\pi}(s,a)

\=w⊤​Qπ​(s,a)\=∑k\=1Kwk​Qπ,k​(s,a)\\displaystyle=w^{\\top}Q\_{\\pi}(s,a)=\\sum\_{k=1}^{K}w\_{k}Q\_{\\pi,k}(s,a)

\=∑k\=1K𝔼π\[∑τ\=0∞γτrk,t+τ|st\=s,at\=a\]\\displaystyle=\\sum\_{k=1}^{K}\\mathbb{E}\_{\\pi}\\left\[\\,\\sum\_{\\tau=0}^{\\infty}\\gamma^{\\tau}r\_{k,t+\\tau}\\penalty\\ \\middle|\\penalty\\ s\_{t}=s,a\_{t}=a\\,\\right\]

(8)

\=𝔼π\[∑τ\=0∞γτ(∑k\=1Krk,t+τ)|st\=s,at\=a\].\\displaystyle=\\mathbb{E}\_{\\pi}\\left\[\\,\\sum\_{\\tau=0}^{\\infty}\\gamma^{\\tau}\\left(\\sum\_{k=1}^{K}r\_{k,t+\\tau}\\right)\\penalty\\ \\middle|\\penalty\\ s\_{t}=s,a\_{t}=a\\,\\right\].

The distinction of single-policy versus multiple-policy then boils down to the question of when and how to select ww. Options are:

1.  (i)

```
single-policy: fix ww ahead of time (nonlinear versions of scalarization equally possible)
```
2.  (ii)

```
single-policy: determine a rule according to which ww is adjusted dynamically
```
3.  (iii)

    multiple-policy: train multiple Q^(j)\\hat{Q}^{(j)} (and consequently, V^(j)\\hat{V}^{(j)}) corresponding to weights {w(j)}j\=1M\\{w^{(j)}\\}\_{j=1}^{M}. This can also refer to individual objectives (i.e., w\=\[1,0,…,0\]w=\[1,0,\\ldots,0\]). Optionally, one can then form the so-called *convex coverage set* (CCS) \[[145](#bib.bib145 ""), [147](#bib.bib147 "")\] that interpolates linearly between the non-dominated value functions.44 4 Conceptually, there is a close connection to explicit model predictive control \[[13](#bib.bib13 "")\], where we also compute a simplex of feedback laws based on a finite set of control problems. For weighting, we have several options:

    1.  a)

        fixed weights during planning

    2.  b)

        dynamically adapted weights during planning

    3.  c)

        selection policy (e.g., greedy)
#### V-B1 Single-policy algorithms

In the single-policy setting with fixed weights (point (i) in the list above), any deep QQ learning architecture can readily be applied. In a similar fashion, the ϵ\\epsilon-constraint method ([3](#S2.E3 "In Computing individual Pareto optima ‣ II-B3 Overview of methods ‣ II-B Multi-objective optimization ‣ II Preliminaries ‣ Multi-objective Deep Learning: Taxonomy and Survey of the State of the Art")) can be transferred to the RL setting, where it is referred to as *thresholded lexicographic ordering* (TLO). Since a constraint violation is undesirable, the constraint-objectives are considered more important, thus the ordering. An alternative to avoid the decision-making is to allow non-linear scalarization and use, e.g., a neural network architecture to learn a suitable scalarizer fθ​(V⁡(s))\=V^​(s)f\_{\\theta}(V(s))=\\hat{V}(s). Under the assumption that fθf\_{\\theta} is strictly concave, one can show that the solution to this MDP is Pareto optimal \[[2](#bib.bib2 "")\].

For improved performance, a single-policy algorithm is proposed in \[[187](#bib.bib187 "")\], in which multiple single-objective QQ-learning problems are solved for various Chebyshev scalarizations. During execution, a simple greedy strategy then selects the QQ function with maximal value, thus eliminating the decision-making.

Aside from scalarization, MGDA-like approaches have been proposed, for instance in \[[88](#bib.bib88 ""), [210](#bib.bib210 "")\], where—in the context of constrained reinforcement learning—policy gradients are aggregated into a single descent direction. In \[[203](#bib.bib203 "")\], multiple constraints are transformed into objectives and then considered using an MGDA-like procedure.

#### V-B2 Multiple-policy algorithms

If we decide not to scalarize the reward or value or QQ-function before training, then we can follow one of two strategies. The first one is to simply compute the vector of VV or QQ values and then decide online which one to take. This is realized in a linear fashion using a weight vector ww in, e.g., \[[1](#bib.bib1 "")\]. In an extension of this work \[[114](#bib.bib114 "")\], the authors add a strictly concave term to the rewards, which overcomes issues with solutions that are not Pareto optimal.

A simple way to automatically choose a weight (in the context of QQ learning) is the so-called top-QQ approach \[[106](#bib.bib106 "")\] where we always select the entry of QQ that maximizes the value, i.e., maxj⁡maxa∈𝒜​Q^(j)​(s,a).\\max\_{j}\\max\_{a\\in\\mathcal{A}}\\hat{Q}^{(j)}(s,a).

Again, we can use any deep RL approach to learn the individual QQ functions. This is in line with point (iii-b) from the enumeration above. A very similar strategy was followed in \[[127](#bib.bib127 "")\], where several scalarized problems of the form ([8](#S5.E8 "In V-B MORL: adapted taxonomy and survey ‣ V Deep multi-objective reinforcement learning ‣ Multi-objective Deep Learning: Taxonomy and Survey of the State of the Art")) are solved. Their values vary linearly with the weighting of the objectives such that during planning, we select the QQ network that has the largest value for the given ww, to decide on the action.

Again in a similar fashion to \[[2](#bib.bib2 "")\], the paper \[[177](#bib.bib177 "")\] considers a set of QQ functions, but at the same time learns a so-called decision value (using temporal difference learning) according to which the individual QQ values are weighted.

The multiple-policy section of \[[187](#bib.bib187 "")\] treats the problem slightly differently, in that an entire set of non-dominated QQ functions is stored (i.e., those having non-dominated value for at least one action aa). The selection of a strategy is then performed online, either following a decision makers preference or a greedy policy.

In \[[195](#bib.bib195 "")\] several—seemingly unrelated—policies are trained using PPO, each taking into account different constraints. Depending on the decision maker’s preference in terms of constraints, a suitable policy is then selected.

## VI Deep learning for multi-objective optimization

Even though the content of this section is not at the center of our overview, we would like to briefly highlight the vice-versa combination of multi-objective optimization and deep learning, since the two are sometimes confused. Instead of treating deep learning problems using multi-objective optimization, one may also using deep learning to accelerate the solution of (classical) multi-objective optimization problems, such as the multi-criteria design of a complex technical system like an electric vehicle.

Very naturally, as solving MOPs is computationally expensive—in particular for complex and costly-to-evaluate models—there is a strong interest in accelerating the evaluation of the loss function L⁡(θ)L(\\theta) or their gradients. There has been extensive research on surrogate-assisted multi-objective optimization \[[32](#bib.bib32 ""), [176](#bib.bib176 ""), [136](#bib.bib136 ""), [36](#bib.bib36 "")\]. That is, instead of L⁡(θ)L(\\theta), we train a surrogate function fϕ​(θ)f\_{\\phi}(\\theta)—parametrized by ϕ\\phi—using a small number N𝖾𝗑𝗉N\_{\\mathsf{exp}} of expensive model evaluations at carefully chosen points {θ(i)}i\=1N𝖾𝗑𝗉\\{\\theta^{(i)}\\}\_{i=1}^{N\_{\\mathsf{exp}}}:

min⁡∑i\=1N𝖾𝗑𝗉ϕ⁡‖L⁡(θ(i))−fϕ​(θ(i))‖22.\\min\_{\\phi}\\sum\_{i=1}^{N\_{\\mathsf{exp}}}\\left\\|L\\left(\\theta^{(i)}\\right)-f\_{\\phi}\\left(\\theta^{(i)}\\right)\\right\\|\_{2}^{2}.

(9)

Depending on the type of problem and the availability of gradients, one may extend this by matching the gradients ∇L\\nabla L and ∇θfϕ\\nabla\_{\\theta}f\_{\\phi}. Modeling techniques for fϕf\_{\\phi} range from polynomials over radial basis functions and Kriging models to neural networks, and there is a distinction between global approximations of LL and ones that are valid only locally. Besides smaller models, the latter case may allow for error analysis through trust-region techniques \[[15](#bib.bib15 "")\], at the cost of requiring additional intermittent evaluations of the original loss function LL.

Not surprisingly, machine learning has found its entrance into this area of research as well, see \[[141](#bib.bib141 "")\] for a recent overview. In the spirit of ([9](#S6.E9 "In VI Deep learning for multi-objective optimization ‣ Multi-objective Deep Learning: Taxonomy and Survey of the State of the Art")), deep surrogate models were suggested in \[[22](#bib.bib22 ""), [206](#bib.bib206 "")\], and generative Kriging modeling was studied in \[[76](#bib.bib76 "")\]. Another generative modeling approach called GFlowNets was proposed in \[[77](#bib.bib77 "")\], and the usage of LLMs for solving MOPs was suggested in \[[107](#bib.bib107 "")\]. An alternative to surrogates for LL is to model the problem of hypervolume maximization by a deep neural network \[[207](#bib.bib207 "")\]. Finally, besides supervised and generative modeling, there have been approaches using reinforcement learning \[[99](#bib.bib99 ""), [211](#bib.bib211 "")\] that suggest Pareto optimal points, and also to obtain the most efficient sample sites to build surrogates from when function evaluations are very expensive \[[28](#bib.bib28 "")\].

## VII Conclusion

Multi-objective deep learning is constantly gaining attention, and we believe that the consideration of multiple conflicting criteria will become the new standard in the future, due to the ever-increasing complexity of modern-day tasks. And while researchers appear to have agreed on gradient-based approaches (MGDA) for multi-objective deep learning, there are many exciting questions for future research.

*   •

```
interactive approaches where the training procedure interacts with a decision maker have not yet been studied,
```
*   •

```
systematic usage for very high-dimensional problems with millions of parameters is still very scarce,
```
*   •

```
challenging benchmark problems would help to foster research; in particular—to the best of our knowledge—there are no deep learning test problems where the Pareto front is non-convex,
```
*   •

```
the massive trends of generative AI and large language models will likely play an important role as well—both in terms of solving multi-objective optimization problem and in using multi-objective optimization during their training.
```
## Acknowledgments

The authors acknowledge funding by the German Federal Ministry of Education and Research (BMBF) through the AI junior research group “Multicriteria Machine Learning” (Grant ID 01||S22064).

## References

*   \[1\] A. Abels, D. Roijers, T. Lenaerts, A. Nowé, and D. Steckelmacher, “Dynamic weights in multi-objective deep reinforcement learning,” in *Proceedings of the 36th International Conference on Machine Learning*, ser. Proceedings of Machine Learning Research, K. Chaudhuri and R. Salakhutdinov, Eds., vol. 97. PMLR, 09–15 Jun 2019, pp. 11–20.
*   \[2\] M. Agarwal, V. Aggarwal, and T. Lan, “Multi-objective reinforcement learning with non-linear scalarization,” in *Proceedings of the 21st International Conference on Autonomous Agents and Multiagent Systems*, 2022, pp. 9–17.
*   \[3\] J. Akhter, P. D. Fährmann, K. Sonntag, and S. Peitz, “Common pitfalls to avoid while using multiobjective optimization in machine learning,” *arXiv:2405.01480*, 2024.
*   \[4\] A. F. J. AL-Gburi, M. Z. A. Nazri, M. R. B. Yaakub, and Z. A. A. Alyasseri, “Multi-objective unsupervised feature selection and cluster based on symbiotic organism search,” *Algorithms*, vol. 17, no. 8, p. 355, 2024.
*   \[5\] I. Albuquerque, J. Monteiro, T. Doan, B. Considine, T. Falk, and I. Mitliagkas, “Multi-objective training of generative adversarial networks with multiple discriminators,” in *Proceedings of the 36th International Conference on Machine Learning*, ser. Proceedings of Machine Learning Research, K. Chaudhuri and R. Salakhutdinov, Eds., vol. 97. PMLR, 2019, pp. 202–211.
*   \[6\] S.-A. N. Alexandropoulos, C. K. Aridas, S. B. Kotsiantis, and M. N. Vrahatis, *Multi-Objective Evolutionary Optimization Algorithms for Machine Learning: A Recent Survey*. Cham: Springer International Publishing, 2019, pp. 35–55.
*   \[7\] A. K. Alok, S. Saha, and A. Ekbal, “A new semi-supervised clustering technique using multi-objective optimization,” *Applied Intelligence*, vol. 43, pp. 633–661, 2015.
*   \[8\] A. C. Amakor, S. Peitz, and K. Sonntag, “A multiobjective continuation method to compute the regularization path of deep neural networks,” *arXiv:2308.12044*, 2023.
*   \[9\] A. Auger, J. Bader, D. Brockhoff, and E. Zitzler, “Hypervolume-based multiobjective optimization: Theoretical foundations and practical implications,” *Theoretical Computer Science*, vol. 425, pp. 75–103, 2012, theoretical Foundations of Evolutionary Computation.
*   \[10\] A. Aşkan and S. Sayın, “SVM classification for imbalanced data sets using a multiobjective optimization framework,” *Annals of Operations Research*, vol. 216, no. 1, p. 191–203, Jan. 2013.
*   \[11\] J. Bader and E. Zitzler, “Hype: An algorithm for fast hypervolume-based many-objective optimization,” *Evolutionary Computation*, vol. 19, no. 1, pp. 45–76, 2011.
*   \[12\] J. Baumann and O. Kramer, “Evolutionary multi-objective optimization of large language model prompts for balancing sentiments,” in *International Conference on the Applications of Evolutionary Computation (Part of EvoStar)*. Springer, 2024, pp. 212–224.
*   \[13\] A. Bemporad, M. Morari, V. Dua, and E. N. Pistikopoulos, “The explicit linear quadratic regulator for constrained systems,” *Automatica*, vol. 38, no. 1, pp. 3–20, 2002.
*   \[14\] K. Berahmand, F. Daneshfar, E. S. Salehi, Y. Li, and Y. Xu, “Autoencoders and their applications in machine learning: a survey,” *Artificial Intelligence Review*, vol. 57, no. 2, p. 28, 2024.
*   \[15\] M. Berkemeier and S. Peitz, “Derivative-free multiobjective trust region descent method using radial basis function surrogate models,” *Mathematical and Computational Applications*, vol. 26, no. 2, 2021.
*   \[16\] D. Bertsekas, *Reinforcement learning and optimal control*. Athena Scientific, 2019, vol. 1.
*   \[17\] Y. Bian and C. North, “DeepSI: Interactive deep learning for semantic interaction,” in *Proceedings of the 26th International Conference on Intelligent User Interfaces*, ser. IUI ’21. New York, NY, USA: Association for Computing Machinery, 2021, p. 197–207.
*   \[18\] K. Bieker, B. Gebken, and S. Peitz, “On the Treatment of Optimization Problems with L1 Penalty Terms via Multiobjective Continuation,” *IEEE Transactions on Pattern Analysis and Machine Intelligence*, vol. 44, no. 11, pp. 7797–7808, 2022.
*   \[19\] C. M. Bishop, *Pattern Recognition and Machine Learning*, 1st ed., ser. Information Science and Statistics. Springer, 2006.
*   \[20\] C. M. Bishop and H. Bishop, *Deep Learning: Foundations and Concepts*. Springer International Publishing, 2024.
*   \[21\] P. A. N. Bosman, “On gradients and hybrid evolutionary algorithms for real-valued multiobjective optimization,” *IEEE Transactions on Evolutionary Computation*, vol. 16, no. 1, pp. 51–69, 2012.
*   \[22\] D. Botache, J. Decke, W. Ripken, A. Dornipati, F. Götz-Hahn, M. Ayeb, and B. Sick, “Enhancing multi-objective optimisation through machine learning-supported multiphysics simulation,” in *Machine Learning and Knowledge Discovery in Databases. Applied Data Science Track*, A. Bifet, T. Krilavičius, I. Miliou, and S. Nowaczyk, Eds. Cham: Springer Nature Switzerland, 2024, pp. 297–312.
*   \[23\] S. Budd, E. C. Robinson, and B. Kainz, “A survey on active learning and human-in-the-loop deep learning for medical image analysis,” *Medical Image Analysis*, vol. 71, p. 102062, 2021.
*   \[24\] D. Bui-Thi, P. Meysman, and K. Laukens, “MoMAC: Multi-objective optimization to combine multiple association rules into an interpretable classification,” *Applied Intelligence*, vol. 52, no. 3, pp. 3090–3102, 2022.
*   \[25\] L. Bungert, T. Roith, D. Tenbrinck, and M. Burger, “A Bregman learning framework for sparse neural networks,” *Journal of Machine Learning Research*, vol. 23, no. 192, pp. 1–43, 2022.
*   \[26\] O. A. Bustos-Brinez, J. A. Gallego-Mejia, and F. A. González, “Ad-dmkde: Anomaly detection through density matrices and fourier features,” in *International Conference on Information Technology & Systems*. Springer, 2023, pp. 327–338.
*   \[27\] M. Caron, P. Bojanowski, A. Joulin, and M. Douze, “Deep clustering for unsupervised learning of visual features,” in *Proceedings of the European conference on computer vision (ECCV)*, 2018, pp. 132–149.
*   \[28\] S. Chen, J. Wu, and X. Liu, “EMORL: Effective multi-objective reinforcement learning method for hyperparameter optimization,” *Engineering Applications of Artificial Intelligence*, vol. 104, p. 104315, 2021.
*   \[29\] Y. Chen, M. Mancini, X. Zhu, and Z. Akata, “Semi-supervised and unsupervised deep visual learning: A survey,” *IEEE transactions on pattern analysis and machine intelligence*, vol. 46, no. 3, pp. 1327–1347, 2022.
*   \[30\] Y.-C. Chen, “A tutorial on kernel density estimation and recent advances,” *Biostatistics & Epidemiology*, vol. 1, no. 1, pp. 161–187, 2017.
*   \[31\] Z. Chen, J. Ge, H. Zhan, S. Huang, and D. Wang, “Pareto self-supervised training for few-shot learning,” in *Proceedings of the IEEE/CVF conference on computer vision and pattern recognition*, 2021, pp. 13 663–13 672.
*   \[32\] T. Chugh, K. Sindhya, J. Hakanen, and K. Miettinen, “Handling computationally expensive multiobjective optimization problems with evolutionary algorithms-a survey,” *Reports of the Department of Mathematical Information Technology, Series B, Scientific Computing no. B*, vol. 4, no. 2015, p. 1957, 2015.
*   \[33\] F. H. Clarke, *Optimization and nonsmooth analysis*. SIAM, 1990.
*   \[34\] C. A. Coello Coello, G. B. Lamont, and D. A. Van Veldhuizen, *Evolutionary Algorithms for Solving Multi-Objective Problems*, 2nd ed. Springer, 2007.
*   \[35\] K. Deb, A. Pratap, S. Agarwal, and T. Meyarivan, “A fast and elitist multiobjective genetic algorithm: NSGA-II,” *IEEE Transactions on Evolutionary Computation*, vol. 6, no. 2, p. 182–197, 2002.
*   \[36\] K. Deb, P. C. Roy, and R. Hussein, “Surrogate modeling approaches for multiobjective optimization: Methods, taxonomy, and results,” *Mathematical and Computational Applications*, vol. 26, no. 1, 2021.
*   \[37\] J. Degrave, F. Felici, J. Buchli, M. Neunert, B. Tracey, F. Carpanese, T. Ewalds, R. Hafner, A. Abdolmaleki, D. de Las Casas *et al.*, “Magnetic control of tokamak plasmas through deep reinforcement learning,” *Nature*, vol. 602, no. 7897, pp. 414–419, 2022.
*   \[38\] M. Dellnitz, O. Schütze, and T. Hestermeyer, “Covering pareto sets by multilevel subdivision techniques,” *Journal of Optimization Theory and Applications*, vol. 124, no. 1, p. 113–136, 2005.
*   \[39\] M. Dhanushree, R. Priya, P. Aruna, and R. Bhavani, “Static video summarization with multi-objective constrained optimization,” *Journal of Ambient Intelligence and Humanized Computing*, vol. 15, no. 4, pp. 2621–2639, 2024.
*   \[40\] T. Dobrički, X. Zhuang, K. J. Won, and B.-W. Hong, “Survey on unsupervised learning methods for optical flow estimation,” in *2022 13th International Conference on Information and Communication Technology Convergence (ICTC)*, 2022.
*   \[41\] I. Durugkar, I. Gemp, and S. Mahadevan, “Generative multi-adversarial networks,” in *International Conference on Learning Representations*, 2017.
*   \[42\] J.-A. Désidéri, “Multiple-gradient descent algorithm (MGDA) for multiobjective optimization,” *Comptes Rendus. Mathématique*, vol. 350, no. 5–6, p. 313–318, Mar. 2012.
*   \[43\] M. Ehrgott, *Multicriteria optimization*, 2nd ed. Springer, 2005.
*   \[44\] G. Eichfelder, “An adaptive scalarization method in multiobjective optimization,” *SIAM Journal on Optimization*, vol. 19, no. 4, pp. 1694–1718, 2009.
*   \[45\] T. Elsken, J. H. Metzen, and F. Hutter, “Efficient multi-objective neural architecture search via Lamarckian evolution,” in *International Conference on Learning Representations*, 2019.
*   \[46\] ——, “Neural architecture search: A survey,” *Journal of Machine Learning Research*, vol. 20, no. 55, pp. 1–21, 2019.
*   \[47\] P. Eskelinen, K. Miettinen, K. Klamroth, and J. Hakanen, “Pareto navigator for interactive nonlinear multiobjective optimization,” *OR Spectrum*, vol. 32, no. 1, pp. 211–227, 2010.
*   \[48\] F. Felten, E.-G. Talbi, and G. Danoy, “Multi-objective reinforcement learning based on decomposition: A taxonomy and framework,” *Journal of Artificial Intelligence Research*, vol. 79, pp. 679–723, 2024.
*   \[49\] Y. Ferstl, M. Neff, and R. McDonnell, “Multi-objective adversarial gesture generation,” in *Proceedings of the 12th ACM SIGGRAPH Conference on Motion, Interaction and Games*, 2019.
*   \[50\] J. Fliege, L. M. Graña Drummond, and B. F. Svaiter, “Newton’s method for multiobjective optimization,” *SIAM Journal on Optimization*, vol. 20, no. 2, pp. 602–626, 2009.
*   \[51\] J. Fliege and B. F. Svaiter, “Steepest descent methods for multicriteria optimization,” *Mathematical Methods of Operations Research*, vol. 51, no. 3, pp. 479–494, 2000.
*   \[52\] J. Fliege and A. I. F. Vaz, “A method for constrained multiobjective optimization based on sqp techniques,” *SIAM Journal on Optimization*, vol. 26, no. 4, pp. 2091–2119, 2016.
*   \[53\] C. M. Fonseca and P. J. Fleming, “An overview of evolutionary algorithms in multiobjective optimization,” *Evolutionary Computation*, vol. 3, no. 1, pp. 1–16, 1995.
*   \[54\] Y. Fu, C. Liu, D. Li, X. Sun, J. Zeng, and Y. Yao, “DessiLBI: Exploring structural sparsity of deep networks via differential inclusion paths,” in *Proceedings of the 37th International Conference on Machine Learning*, vol. 119. PMLR, 2020, pp. 3315–3326.
*   \[55\] Y. Fu, C. Liu, D. Li, Z. Zhong, X. Sun, J. Zeng, and Y. Yao, “Exploring structural sparsity of deep networks via inverse scale spaces,” *IEEE Transactions on Pattern Analysis and Machine Intelligence*, vol. 45, no. 2, pp. 1749–1765, 2023.
*   \[56\] Y. Fu, S. Dong, Y. Huang, M. Niu, C. Ni, L. Yu, K. Shi, Z. Yao, and C. Zhuo, “Mpgan: Multi pareto generative adversarial network for the denoising and quantitative analysis of low-dose pet images of human brain,” *Medical Image Analysis*, vol. 98, p. 103306, 2024.
*   \[57\] B. Gebken and S. Peitz, “An Efficient Descent Method for Locally Lipschitz Multiobjective Optimization Problems,” *Journal of Optimization Theory and Applications*, vol. 188, pp. 696–723, 2021.
*   \[58\] B. Gebken, S. Peitz, and M. Dellnitz, “A Descent Method for Equality and Inequality Constrained Multiobjective Optimization Problems,” in *Numerical and Evolutionary Optimization – NEO 2017*, L. Trujillo, O. Schütze, Y. Maldonado, and P. Valle, Eds. Springer, Cham, 2019, pp. 29–61.
*   \[59\] ——, “On the hierarchical structure of Pareto critical sets,” *Journal of Global Optimization*, vol. 73, no. 4, pp. 891–913, 2019.
*   \[60\] M. Gong, M. Zhang, and Y. Yuan, “Unsupervised band selection based on evolutionary multiobjective optimization for hyperspectral images,” *IEEE Transactions on Geoscience and Remote Sensing*, vol. 54, no. 1, pp. 544–557, 2015.
*   \[61\] G. González-Almagro, A. Rosales-Pérez, J. Luengo, J.-R. Cano, and S. García, “Improving constrained clustering via decomposition-based multiobjective optimization with memetic elitism,” in *Proceedings of the 2020 Genetic and Evolutionary Computation Conference*, 2020, pp. 333–341.
*   \[62\] I. Goodfellow, Y. Bengio, and A. Courville, *Deep Learning*. MIT Press, 2016.
*   \[63\] I. Goodfellow, J. Pouget-Abadie, M. Mirza, B. Xu, D. Warde-Farley, S. Ozair, A. Courville, and Y. Bengio, “Generative adversarial nets,” in *Advances in Neural Information Processing Systems*, Z. Ghahramani, M. Welling, C. Cortes, N. Lawrence, and K. Weinberger, Eds., vol. 27. Curran Associates, Inc., 2014.
*   \[64\] J. Gui, T. Chen, J. Zhang, Q. Cao, Z. Sun, H. Luo, and D. Tao, “A survey on self-supervised learning: Algorithms, applications, and future trends,” *arXiv preprint arXiv:2301.05712*, 2023.
*   \[65\] T. Gui, L. Qing, Q. Zhang, J. Ye, H. Yan, Z. Fei, and X. Huang, “Constructing multiple tasks for augmentation: Improving neural image classification with k-means features,” *arXiv:1911.07518*, 2019.
*   \[66\] T. Haarnoja, A. Zhou, P. Abbeel, and S. Levine, “Soft actor-critic: Off-policy maximum entropy deep reinforcement learning with a stochastic actor,” in *International conference on machine learning*. PMLR, 2018, pp. 1861–1870.
*   \[67\] J. Han, H. Liu, H. Zhu, and H. Xiong, “Kill two birds with one stone: A multi-view multi-adversarial learning approach for joint air quality and weather prediction,” *IEEE Transactions on Knowledge and Data Engineering*, vol. 35, no. 11, pp. 11 515–11 528, 2023.
*   \[68\] J. Handl and J. Knowles, “Feature subset selection in unsupervised learning via multiobjective optimization,” *International Journal of Computational Intelligence Research*, vol. 2, no. 3, pp. 217–238, 2006.
*   \[69\] T. Hastie, R. Tibshirani, and J. Friedman, *The Elements of Statistical Learning*. Springer New York, 2009.
*   \[70\] C. F. Hayes, R. Rădulescu, E. Bargiacchi, J. Källström, M. Macfarlane, M. Reymond, T. Verstraeten, L. M. Zintgraf, R. Dazeley, F. Heintz, E. Howley, A. A. Irissappane, P. Mannion, A. Nowé, G. Ramos, M. Restelli, P. Vamplew, and D. M. Roijers, “A practical guide to multi-objective reinforcement learning and planning,” *Autonomous Agents and Multi-Agent Systems*, vol. 36, no. 1, Apr. 2022.
*   \[71\] C. He, S. Huang, R. Cheng, K. C. Tan, and Y. Jin, “Evolutionary multiobjective optimization driven by generative adversarial networks (gans),” *arXiv:1910.04966*, 2019.
*   \[72\] K. He, X. Chen, S. Xie, Y. Li, P. Dollár, and R. Girshick, “Masked autoencoders are scalable vision learners,” in *Proceedings of the IEEE/CVF conference on computer vision and pattern recognition*, 2022, pp. 16 000–16 009.
*   \[73\] C. Hillermeier, *Nonlinear Multiobjective Optimization: A Generalized Homotopy Approach*. Birkhäuser, 2001.
*   \[74\] S. S. Hotegni, M. Berkemeier, and S. Peitz, “Multi-objective optimization for sparse deep multi-task learning,” in *2024 International Joint Conference on Neural Networks (IJCNN)*, 2024, pp. 1–9.
*   \[75\] S. S. Hotegni and S. Peitz, “MOREL: Enhancing adversarial robustness through multi-objective representation learning,” *arXiv:2410.01697*, 2024.
*   \[76\] R. Hussein and K. Deb, “A generative Kriging surrogate model for constrained and unconstrained multi-objective optimization,” in *Proceedings of the Genetic and Evolutionary Computation Conference 2016*, ser. GECCO ’16. New York, NY, USA: Association for Computing Machinery, 2016, p. 573–580.
*   \[77\] M. Jain, S. C. Raparthy, A. Hernández-García, J. Rector-Brooks, Y. Bengio, S. Miret, and E. Bengio, “Multi-objective GFlowNets,” in *Proceedings of the 40th International Conference on Machine Learning*, ser. Proceedings of Machine Learning Research, A. Krause, E. Brunskill, K. Cho, B. Engelhardt, S. Sabato, and J. Scarlett, Eds., vol. 202. PMLR, 23–29 Jul 2023, pp. 14 631–14 653.
*   \[78\] H. Jang, D. Kim, J. Kim, J. Shin, P. Abbeel, and Y. Seo, “Visual representation learning with stochastic frame prediction,” *arXiv preprint arXiv:2406.07398*, 2024.
*   \[79\] W. Jia, M. Sun, J. Lian, and S. Hou, “Feature dimensionality reduction: a review,” *Complex & Intelligent Systems*, vol. 8, no. 3, pp. 2663–2693, 2022.
*   \[80\] Y. Jin, Ed., *Multi-Objective Machine Learning*. Springer Berlin Heidelberg, 2006.
*   \[81\] Y. Jin and B. Sendhoff, “Pareto-based multiobjective machine learning: An overview and case studies,” *IEEE Transactions on Systems, Man, and Cybernetics, Part C (Applications and Reviews)*, vol. 38, no. 3, pp. 397–415, 2008.
*   \[82\] Y. Jin, R. Wen, and B. Sendhoff, “Evolutionary multi-objective optimization of spiking neural networks,” in *Artificial Neural Networks – ICANN 2007*, J. M. de Sá, L. A. Alexandre, W. Duch, and D. Mandic, Eds. Berlin, Heidelberg: Springer Berlin Heidelberg, 2007, pp. 370–379.
*   \[83\] L. Jing, X. Yang, J. Liu, and Y. Tian, “Self-supervised spatiotemporal feature learning via video rotation prediction,” *arXiv preprint arXiv:1811.11387*, 2018.
*   \[84\] F. Karl, T. Pielok, J. Moosbauer, F. Pfisterer, S. Coors, M. Binder, L. Schneider, J. Thomas, J. Richter, M. Lang, E. C. Garrido-Merchán, J. Branke, and B. Bischl, “Multi-objective hyperparameter optimization in machine learning—an overview,” *ACM Transactions on Evolutionary Learning and Optimization*, vol. 3, no. 4, 2023.
*   \[85\] G. E. Karniadakis, I. G. Kevrekidis, L. Lu, P. Perdikaris, S. Wang, and L. Yang, “Physics-informed machine learning,” *Nature Reviews Physics*, vol. 3, no. 6, pp. 422–440, 2021.
*   \[86\] R. Kasimbeyli, “A conic scalarization method in multi-objective optimization,” *Journal of Global Optimization*, vol. 56, no. 2, p. 279–297, Sep. 2011.
*   \[87\] A. Khan, A. Sohail, M. Fiaz, M. Hassan, T. H. Afridi, S. U. Marwat, F. Munir, S. Ali, H. Naseem, M. Z. Zaheer *et al.*, “A survey of the self supervised learning mechanisms for vision transformers,” *arXiv preprint arXiv:2408.17059*, 2024.
*   \[88\] D. Kim, M. Hong, J. Park, and S. Oh, “Scale-invariant gradient aggregation for constrained multi-objective reinforcement learning,” *arXiv:2403.00282*, 2024.
*   \[89\] S. Kim, H. Kwon, E. Kwon, Y. Choi, T.-H. Oh, and S. Kang, “MDARTS: Multi-objective differentiable neural architecture search,” in *2021 Design, Automation & Test in Europe Conference & Exhibition (DATE)*. IEEE, 2021, pp. 1344–1349.
*   \[90\] D. P. Kingma and J. Ba, “Adam: A method for stochastic optimization,” *arXiv:1412.6980*, 2014.
*   \[91\] D. P. Kingma and M. Welling, “Auto-encoding variational Bayes,” *arXiv:1312.6114*, 2013.
*   \[92\] D. P. Kingma, M. Welling *et al.*, “An introduction to variational autoencoders,” *Foundations and Trends® in Machine Learning*, vol. 12, no. 4, pp. 307–392, 2019.
*   \[93\] J. Kober, J. A. Bagnell, and J. Peters, “Reinforcement learning in robotics: A survey,” *The International Journal of Robotics Research*, vol. 32, no. 11, pp. 1238–1274, 2013.
*   \[94\] K.-H. Küfer, A. Scherrer, M. Monz, F. Alonso, H. Trinkaus, T. Bortfeld, and C. Thieke, “Intensity-modulated radiotherapy - a large scale multi-criteria programming problem,” *OR Spectrum*, vol. 25, no. 2, p. 223–249, May 2003.
*   \[95\] P. Kumar, P. Rawat, and S. Chauhan, “Contrastive self-supervised learning: review, progress, challenges and future research directions,” *International Journal of Multimedia Information Retrieval*, vol. 11, no. 4, pp. 461–488, 2022.
*   \[96\] Y. LeCun, Y. Bengio, and G. Hinton, “Deep learning,” *Nature*, vol. 521, no. 7553, pp. 436–444, 2015.
*   \[97\] I. Lemhadri, F. Ruan, L. Abraham, and R. Tibshirani, “Lassonet: A neural network with feature sparsity,” *Journal of Machine Learning Research*, vol. 22, no. 127, pp. 1–29, 2021.
*   \[98\] B. Li, Z. Di, Y. Yang, H. Qian, P. Yang, H. Hao, K. Tang, and A. Zhou, “It’s morphing time: Unleashing the potential of multiple llms via multi-objective optimization,” *arXiv:2407.00487*, 2024.
*   \[99\] K. Li, T. Zhang, and R. Wang, “Deep reinforcement learning for multiobjective optimization,” *IEEE Transactions on Cybernetics*, vol. 51, no. 6, pp. 3103–3114, 2021.
*   \[100\] X. Li and Y. Chen, “Multi-augmentation contrastive learning as multi-objective optimization for graph neural networks,” in *Pacific-Asia Conference on Knowledge Discovery and Data Mining*. Springer, 2023, pp. 495–507.
*   \[101\] Z. Li, Y. Chen, Y. LeCun, and F. T. Sommer, “Neural manifold clustering and embedding,” *arXiv preprint arXiv:2201.10000*, 2022.
*   \[102\] T. P. Lillicrap, J. J. Hunt, A. Pritzel, N. Heess, T. Erez, Y. Tassa, D. Silver, and D. Wierstra, “Continuous control with deep reinforcement learning,” *arXiv:1509.02971*, 2015.
*   \[103\] X. Lin, H.-L. Zhen, Z. Li, Q.-F. Zhang, and S. Kwong, “Pareto multi-task learning,” in *Advances in Neural Information Processing Systems*, H. Wallach, H. Larochelle, A. Beygelzimer, F. d'Alché-Buc, E. Fox, and R. Garnett, Eds., vol. 32. Curran Associates, Inc., 2019.
*   \[104\] B. Liu, Y. Feng, P. Stone, and Q. Liu, “FAMO: Fast adaptive multitask optimization,” *Advances in Neural Information Processing Systems*, vol. 36, 2024.
*   \[105\] B. Liu, X. Liu, X. Jin, P. Stone, and Q. Liu, “Conflict-averse gradient descent for multi-task learning,” *Advances in Neural Information Processing Systems*, vol. 34, pp. 18 878–18 890, 2021.
*   \[106\] C. Liu, X. Xu, and D. Hu, “Multiobjective reinforcement learning: A comprehensive overview,” *IEEE Transactions on Systems, Man, and Cybernetics: Systems*, vol. 45, no. 3, pp. 385–398, 2015.
*   \[107\] F. Liu, X. Lin, Z. Wang, S. Yao, X. Tong, M. Yuan, and Q. Zhang, “Large language model for multi-objective evolutionary optimization,” *arXiv:2310.12541*, 2023.
*   \[108\] J. Liu, M. Gong, Q. Miao, X. Wang, and H. Li, “Structure learning for deep neural networks based on multiobjective optimization,” *IEEE transactions on neural networks and learning systems*, vol. 29, no. 6, pp. 2450–2463, 2017.
*   \[109\] Q. Liu and D. Wang, “Stein variational gradient descent: A general purpose bayesian inference algorithm,” in *Advances in Neural Information Processing Systems*, D. Lee, M. Sugiyama, U. Luxburg, I. Guyon, and R. Garnett, Eds., vol. 29. Curran Associates, Inc., 2016.
*   \[110\] S. Liu and L. N. Vicente, “The stochastic multi-gradient algorithm for multi-objective optimization and its application to supervised machine learning,” *Annals of Operations Research*, vol. 339, no. 3, p. 1119–1148, Mar. 2021.
*   \[111\] X. Liu, F. Zhang, Z. Hou, L. Mian, Z. Wang, J. Zhang, and J. Tang, “Self-supervised learning: Generative or contrastive,” *IEEE transactions on knowledge and data engineering*, vol. 35, no. 1, pp. 857–876, 2021.
*   \[112\] X. Liu, X. Tong, and Q. Liu, “Profiling Pareto front with multi-objective stein variational gradient descent,” *Advances in Neural Information Processing Systems*, vol. 34, pp. 14 721–14 733, 2021.
*   \[113\] F. Lobato, C. Sales, I. Araujo, V. Tadaiesky, L. Dias, L. Ramos, and A. Santana, “Multi-objective genetic algorithm for missing data imputation,” *Pattern Recognition Letters*, vol. 68, pp. 126–131, 2015.
*   \[114\] H. Lu, D. Herman, and Y. Yu, “Multi-objective reinforcement learning: Convexity, stationarity and pareto optimality,” in *The Eleventh International Conference on Learning Representations*, 2023.
*   \[115\] Z. Lu, B. Shuai, Y. Chen, Z. Xu, and D. Modolo, “Self-supervised multi-object tracking with path consistency,” in *CVPR 2024*, 2024.
*   \[116\] P. Ma, T. Du, and W. Matusik, “Efficient continuous pareto exploration in multi-task learning,” in *Proceedings of the 37th International Conference on Machine Learning*, ser. Proceedings of Machine Learning Research, H. D. III and A. Singh, Eds., vol. 119. PMLR, 13–18 Jul 2020, pp. 6522–6531.
*   \[117\] U. Michelucci, “An introduction to autoencoders,” *arXiv preprint arXiv:2201.03898*, 2022.
*   \[118\] I. Mierswa and M. Wurst, “Information preserving multi-objective feature selection for unsupervised learning,” in *Proceedings of the 8th annual conference on Genetic and evolutionary computation*, 2006, pp. 1545–1552.
*   \[119\] K. Miettinen and M. Mäkelä, “Interactive bundle-based method for nondifferentiable multiobjeective optimization: NIMBUS,” *Optimization*, vol. 34, no. 3, pp. 231–246, 1995.
*   \[120\] K. Miettinen, *Nonlinear Multiobjective Optimization*. Springer US, 1998.
*   \[121\] C. S. Miranda and F. J. Von Zuben, “Multi-objective optimization for self-adjusting weighted gradient in machine learning tasks,” *arXiv preprint arXiv:1506.01113*, 2015.
*   \[122\] S. Mitra, M. Hasanuzzaman, and S. Saha, “A unified multi-view clustering algorithm using multi-objective optimization coupled with generative model,” *ACM Transactions on Knowledge Discovery from Data (TKDD)*, vol. 14, no. 1, pp. 1–31, 2020.
*   \[123\] B. Mitrevski, M. Filipovic, D. Antognini, E. L. Glaude, B. Faltings, and C. Musat, “Momentum-based gradient methods in multi-objective recommendation,” *arXiv:2009.04695*, 2020.
*   \[124\] V. Mnih, K. Kavukcuoglu, D. Silver, A. A. Rusu, J. Veness, M. G. Bellemare, A. Graves, M. Riedmiller, A. K. Fidjeland, G. Ostrovski, S. Petersen, C. Beattie, A. Sadik, I. Antonoglou, H. King, D. Kumaran, D. Wierstra, S. Legg, and D. Hassabis, “Human-level control through deep reinforcement learning,” *Nature*, vol. 518, no. 7540, p. 529–533, 2015.
*   \[125\] A. Morales-Hernández, I. Van Nieuwenhuyse, and S. Rojas Gonzalez, “A survey on multi-objective hyperparameter optimization algorithms for machine learning,” *Artificial Intelligence Review*, vol. 56, no. 8, p. 8043–8093, Dec. 2022.
*   \[126\] E. Mosqueira-Rey, E. Hernandez-Pereira, D. Alonso-Rios, J. Bobes-Bascaran, and A. Fernandez-Leal, “Human-in-the-loop machine learning: a state of the art,” *Artificial Intelligence Review*, vol. 56, no. 4, p. 3005–3054, 2022.
*   \[127\] H. Mossalam, Y. M. Assael, D. M. Roijers, and S. Whiteson, “Multi-objective deep reinforcement learning,” *arXiv:1610.02707*, 2016.
*   \[128\] B. Nachman and D. Shih, “Anomaly detection with density estimation,” *Physical Review D*, vol. 101, no. 7, p. 075042, 2020.
*   \[129\] A. Navon, A. Shamsian, I. Achituve, H. Maron, K. Kawaguchi, G. Chechik, and E. Fetaya, “Multi-task learning as a bargaining game,” *arXiv:2202.01017*, 2022.
*   \[130\] B. Neyshabur, Z. Li, S. Bhojanapalli, Y. LeCun, and N. Srebro, “The role of over-parametrization in generalization of neural networks,” in *International Conference on Learning Representations*, 2019.
*   \[131\] T. T. Nguyen, N. D. Nguyen, P. Vamplew, S. Nahavandi, R. Dazeley, and C. P. Lim, “A multi-objective deep reinforcement learning framework,” *Engineering Applications of Artificial Intelligence*, vol. 96, p. 103915, 2020.
*   \[132\] F. Nikbakhtsarvestani, M. Ebrahimi, and S. Rahnamayan, “Multi-objective ADAM optimizer (MAdam),” in *2023 IEEE International Conference on Systems, Man, and Cybernetics (SMC)*, 2023, pp. 3860–3867.
*   \[133\] T. Ozçelebi, “Multi-objective optimization for video streaming,” Ph.D. dissertation, PhD Thesis, Graduate School of Sciences and Engineering, Koc University, 2006.
*   \[134\] A. Pascoletti and P. Serafini, “Scalarizing vector optimization problems,” *Journal of Optimization Theory and Applications*, vol. 42, no. 4, pp. 499–524, 1984.
*   \[135\] Z. Pei, Z. Cao, M. Long, and J. Wang, “Multi-adversarial domain adaptation,” *Proceedings of the AAAI Conference on Artificial Intelligence*, vol. 32, no. 1, 2018.
*   \[136\] S. Peitz and M. Dellnitz, “A Survey of Recent Trends in Multiobjective Optimal Control – Surrogate Models, Feedback Control and Objective Reduction,” *Mathematical and Computational Applications*, vol. 23, no. 2, 2018.
*   \[137\] ——, “Gradient-based multiobjective optimization with uncertainties,” in *NEO 2016*, Y. Maldonado, L. Trujillo, O. Schütze, A. Riccardi, and M. Vasile, Eds. Springer, 2018, vol. 731, pp. 159–182.
*   \[138\] S. Peitz, J. Stenner, V. Chidananda, O. Wallscheid, S. L. Brunton, and K. Taira, “Distributed Control of Partial Differential Equations Using Convolutional Reinforcement Learning,” *Physica D: Nonlinear Phenomena*, vol. 461, p. 134096, 2024.
*   \[139\] M. M. A. Pour, A. Pesaranghader, E. Cohen, and S. Sanner, “Gaussian process optimization for adaptable multi-objective text generation using linearly-weighted language models,” in *Findings of the Association for Computational Linguistics: NAACL 2024*, 2024, pp. 1529–1536.
*   \[140\] Z. Povalej, “Quasi-newton’s method for multiobjective optimization,” *Journal of Computational and Applied Mathematics*, vol. 255, pp. 765–777, 2014.
*   \[141\] Q. Qu, Z. Ma, A. Clausen, and B. N. Jorgensen, “A comprehensive review of machine learning in multi-objective optimization,” in *2021 IEEE 4th International Conference on Big Data and Artificial Intelligence (BDAI)*. IEEE, Jul. 2021, p. 7–14.
*   \[142\] M. M. Raimundo, T. F. Drumond, A. C. R. Marques, C. Lyra, A. Rocha, and F. J. Von Zuben, “Exploring multiobjective training in multiclass classification,” *Neurocomputing*, vol. 435, pp. 307–320, 2021.
*   \[143\] D. A. Reynolds *et al.*, “Gaussian mixture models.” *Encyclopedia of biometrics*, vol. 741, no. 659-663, 2009.
*   \[144\] F. M. Rohrhofer, S. Posch, C. Gößnitzer, and B. C. Geiger, “Data vs. physics: The apparent Pareto front of physics-informed neural networks,” *IEEE Access*, vol. 11, pp. 86 252–86 261, 2023.
*   \[145\] D. M. Roijers, P. Vamplew, S. Whiteson, and R. Dazeley, “A survey of multi-objective sequential decision-making,” *Journal of Artificial Intelligence Research*, vol. 48, p. 67–113, Oct. 2013.
*   \[146\] D. M. Roijers, S. Whiteson, P. Vamplew, and R. Dazeley, “Why multi-objective reinforcement learning?” in *European Workshop on Reinforcement Learning*, 2015, pp. 1–2.
*   \[147\] D. M. Roijers, S. Whiteson, and F. A. Oliehoek, “Computing convex coverage sets for faster multi-objective coordination,” *Journal of Artificial Intelligence Research*, vol. 52, p. 399–443, Mar. 2015.
*   \[148\] R. Rombach, A. Blattmann, D. Lorenz, P. Esser, and B. Ommer, “High-resolution image synthesis with latent diffusion models,” in *Proceedings of the IEEE/CVF conference on computer vision and pattern recognition*, 2022, pp. 10 684–10 695.
*   \[149\] Y. Ruan, S. Singh, W. Morningstar, A. A. Alemi, S. Ioffe, I. Fischer, and J. V. Dillon, “Weighted ensemble self-supervised learning,” *arXiv preprint arXiv:2211.09981*, 2022.
*   \[150\] M. Ruchte and J. Grabocka, “Scalable pareto front approximation for deep multi-objective learning,” in *2021 IEEE International Conference on Data Mining (ICDM)*, 2021, pp. 1306–1311.
*   \[151\] L. Ruthotto and E. Haber, “An introduction to deep generative modeling,” *GAMM-Mitteilungen*, vol. 44, no. 2, May 2021.
*   \[152\] S. Ryu, H. Do, Y. Kim, G. Lee, and J. Ok, “Multi-dimensional optimization for text summarization via reinforcement learning,” in *Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, L.-W. Ku, A. Martins, and V. Srikumar, Eds. Bangkok, Thailand: Association for Computational Linguistics, Aug. 2024, pp. 5858–5871.
*   \[153\] A. F. M. Saif, L. Chen, X. Cui, S. Lu, B. Kingsbury, and T. Chen, “M2ASR: Multilingual multi-task automatic speech recognition via multi-objective optimization,” in *Interspeech 2024*, 2024, pp. 1240–1244.
*   \[154\] L. K. Saul and S. T. Roweis, “Think globally, fit locally: unsupervised learning of low dimensional manifolds,” *Journal of Machine Learning Research*, vol. 4, no. null, p. 119–155, 2003.
*   \[155\] S. Schäffler, R. Schultz, and K. Weinzierl, “Stochastic method for the solution of unconstrained vector optimization problems,” *Journal of Optimization Theory and Applications*, vol. 114, no. 1, p. 209–222, 2002.
*   \[156\] L. Schmarje, M. Santarossa, S.-M. Schröder, and R. Koch, “A survey on semi-, self- and unsupervised learning for image classification,” *IEEE Access*, 2021.
*   \[157\] J. Schulman, F. Wolski, P. Dhariwal, A. Radford, and O. Klimov, “Proximal policy optimization algorithms,” *arXiv:1707.06347*, 2017.
*   \[158\] O. Schütze, O. Cuate, A. Martín, S. Peitz, and M. Dellnitz, “Pareto Explorer: a global/local exploration tool for many-objective optimization problems,” *Engineering Optimization*, vol. 52, no. 5, pp. 832–855, 2020.
*   \[159\] O. Schütze, S. Mostaghim, M. Dellnitz, and J. Teich, *Covering Pareto Sets by Multilevel Evolutionary Subdivision Techniques*. Springer Berlin Heidelberg, 2003, p. 118–132.
*   \[160\] M. L. Seltzer and J. Droppo, “Multi-task learning in deep neural networks for improved phoneme recognition,” in *2013 IEEE International Conference on Acoustics, Speech and Signal Processing*, 2013, pp. 6965–6969.
*   \[161\] O. Sener and V. Koltun, “Multi-task learning as multi-objective optimization,” *Advances in neural information processing systems*, vol. 31, 2018.
*   \[162\] W. Shu and H. Shen, “Multi-criteria feature selection on cost-sensitive data with missing values,” *Pattern Recognition*, vol. 51, pp. 268–280, 2016.
*   \[163\] D. Silver, T. Hubert, J. Schrittwieser, I. Antonoglou, M. Lai, A. Guez, M. Lanctot, L. Sifre, D. Kumaran, T. Graepel, T. Lillicrap, K. Simonyan, and D. Hassabis, “A general reinforcement learning algorithm that masters chess, shogi, and go through self-play,” *Science*, vol. 362, no. 6419, pp. 1140–1144, 2018.
*   \[164\] K. Sindhu Meena and S. Suriya, “A survey on supervised and unsupervised learning techniques,” in *Proceedings of international conference on artificial intelligence, smart grid and smart city applications: AISGSC 2019*. Springer, 2020, pp. 627–644.
*   \[165\] J. Sohl-Dickstein, E. Weiss, N. Maheswaranathan, and S. Ganguli, “Deep unsupervised learning using nonequilibrium thermodynamics,” in *Proceedings of the 32nd International Conference on Machine Learning*, ser. Proceedings of Machine Learning Research, F. Bach and D. Blei, Eds., vol. 37. Lille, France: PMLR, 07–09 Jul 2015, pp. 2256–2265.
*   \[166\] B. C. Song, D. H. Kim, and S. hyun Lee, “Metric-based regularization and temporal ensemble for multi-task learning using heterogeneous unsupervised tasks,” in *2019 IEEE/CVF International Conference on Computer Vision Workshop (ICCVW)*. IEEE, 2019, pp. 2903–2912.
*   \[167\] K. Sonntag and S. Peitz, “Fast convergence of inertial multiobjective gradient-like systems with asymptotic vanishing damping,” *SIAM Journal on Optimization*, vol. 34, pp. 2259–2286, 2024.
*   \[168\] ——, “Fast Multiobjective Gradient Methods with Nesterov Acceleration via Inertial Gradient-like Systems,” *Journal of Optimization Theory and Applications*, vol. 201, pp. 539–582, 2024.
*   \[169\] V. A. Sosa Hernández, O. Schütze, and M. Emmerich, “Hypervolume maximization via set based newton’s method,” in *EVOLVE - A Bridge between Probability, Set Oriented Numerics, and Evolutionary Computation V*, A.-A. Tantar, E. Tantar, J.-Q. Sun, W. Zhang, Q. Ding, O. Schütze, M. Emmerich, P. Legrand, P. Del Moral, and C. A. Coello Coello, Eds. Cham: Springer International Publishing, 2014, pp. 15–28.
*   \[170\] R. S. Sukthanker, A. Zela, B. Staffler, S. Dooley, J. Grabocka, and F. Hutter, “Multi-objective differentiable neural architecture search,” *arXiv:2402.18213*, 2024.
*   \[171\] C. Suman, S. Tripathy, and S. Saha, “Building an effective intrusion detection system using unsupervised feature selection in multi-objective optimization framework,” *arXiv:1905.06562*, 2019.
*   \[172\] L. Sun, J. Du, L.-R. Dai, and C.-H. Lee, “Multiple-target deep learning for lstm-rnn based speech enhancement,” in *2017 Hands-free Speech Communications and Microphone Arrays (HSCMA)*, 2017, pp. 136–140.
*   \[173\] R. S. Sutton and A. G. Barto, *Reinforcement learning: An introduction*. MIT press, 2018.
*   \[174\] T. Suttorp and C. Igel, *Multi-Objective Optimization of Support Vector Machines*. Berlin, Heidelberg: Springer Berlin Heidelberg, 2006, pp. 199–220.
*   \[175\] R. Swamy, D. M. King, and S. H. Jacobson, “Multiobjective optimization for politically fair districting: A scalable multilevel approach,” *Operations Research*, vol. 71, no. 2, pp. 536–562, 2023.
*   \[176\] M. Tabatabaei, J. Hakanen, M. Hartikainen, K. Miettinen, and K. Sindhya, “A survey on handling computationally expensive multiobjective optimization problems using surrogates: non-nature inspired methods,” *Structural and Multidisciplinary Optimization*, vol. 52, no. 1, p. 1–25, Mar. 2015.
*   \[177\] T. Tajmajer, “Modular multi-objective deep reinforcement learning with decision values,” in *2018 Federated Conference on Computer Science and Information Systems (FedCSIS)*, 2018, pp. 85–93.
*   \[178\] H. Tanabe, E. H. Fukuda, and N. Yamashita, “Proximal gradient methods for multiobjective optimization and their applications,” *Computational Optimization and Applications*, vol. 72, no. 2, p. 339–361, 2018.
*   \[179\] ——, “An accelerated proximal gradient method for multiobjective optimization,” *Computational Optimization and Applications*, vol. 86, no. 2, p. 421–455, 2023.
*   \[180\] L. Thiele, K. Miettinen, P. J. Korhonen, and J. Molina, “A preference-based evolutionary algorithm for multi-objective optimization,” *Evolutionary Computation*, vol. 17, no. 3, pp. 411–436, 2009.
*   \[181\] Y. Tian, L. Si, X. Zhang, R. Cheng, C. He, K. C. Tan, and Y. Jin, “Evolutionary large-scale multi-objective optimization: A survey,” *ACM Comput. Surv.*, vol. 54, no. 8, Oct. 2021.
*   \[182\] E. Triantaphyllou, *Multi-Criteria Decision Making Methods*. Boston, MA: Springer US, 2000, pp. 5–21.
*   \[183\] P. Vamplew, R. Dazeley, A. Berry, R. Issabekov, and E. Dekker, “Empirical evaluation methods for multiobjective reinforcement learning algorithms,” *Machine Learning*, vol. 84, no. 1-2, pp. 51–80, Jul. 2011.
*   \[184\] P. Vamplew, B. J. Smith, J. Källström, G. Ramos, R. Rădulescu, D. M. Roijers, C. F. Hayes, F. Heintz, P. Mannion, P. J. K. Libin, R. Dazeley, and C. Foale, “Scalar reward is not enough: a response to silver, singh, precup and sutton (2021),” *Autonomous Agents and Multi-Agent Systems*, vol. 36, no. 2, Oct. 2022.
*   \[185\] F. Van Breugel, J. N. Kutz, and B. W. Brunton, “Numerical differentiation of noisy data: A unifying multi-objective optimization framework,” *IEEE Access*, vol. 8, pp. 196 865–196 877, 2020.
*   \[186\] H. Van Hasselt, A. Guez, and D. Silver, “Deep reinforcement learning with double q-learning,” *Proceedings of the AAAI Conference on Artificial Intelligence*, vol. 30, no. 1, Mar. 2016.
*   \[187\] K. Van Moffaert and A. Nowé, “Multi-objective reinforcement learning using sets of Pareto dominating policies,” *The Journal of Machine Learning Research*, vol. 15, no. 1, pp. 3483–3512, 2014.
*   \[188\] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, L. u. Kaiser, and I. Polosukhin, “Attention is all you need,” in *Advances in Neural Information Processing Systems*, I. Guyon, U. V. Luxburg, S. Bengio, H. Wallach, R. Fergus, S. Vishwanathan, and R. Garnett, Eds., vol. 30. Curran Associates, Inc., 2017.
*   \[189\] C. Vignon, J. Rabault, and R. Vinuesa, “Recent advances in applying deep reinforcement learning for flow control: Perspectives and future directions,” *Physics of fluids*, vol. 35, no. 3, 2023.
*   \[190\] C. Wan, F. Jin, Z. Qiao, W. Zhang, and Y. Yuan, “Unsupervised active learning with loss prediction,” *Neural Computing and Applications*, pp. 1–9, 2023.
*   \[191\] S. Wang, X. Guo, X. Lin, B. Pan, Y. Du, Y. Wang, Y. Ye, A. Petersen, A. Leitgeb, S. Alkhalifa, K. Minbiole, W. M. Wuest, A. Shehu, and L. Zhao, “Multi-objective deep data generation with correlated property control,” in *Advances in Neural Information Processing Systems*, S. Koyejo, S. Mohamed, A. Agarwal, D. Belgrave, K. Cho, and A. Oh, Eds., vol. 35. Curran Associates, Inc., 2022, pp. 28 889–28 901.
*   \[192\] Y. Wang, H. Liu, W. Zheng, Y. Xia, Y. Li, P. Chen, K. Guo, and H. Xie, “Multi-objective workflow scheduling with deep-q-network-based multi-agent reinforcement learning,” *IEEE Access*, vol. 7, pp. 39 974–39 982, 2019.
*   \[193\] Z. Wang and D. W. Scott, “Nonparametric density estimation for high-dimensional data—algorithms and applications,” *Wiley Interdisciplinary Reviews: Computational Statistics*, vol. 11, no. 4, p. e1461, 2019.
*   \[194\] G. Wilson and D. J. Cook, “A survey of unsupervised deep domain adaptation,” *ACM Transactions on Intelligent Systems and Technology (TIST)*, vol. 11, no. 5, pp. 1–46, 2020.
*   \[195\] K. H. Wray, S. Tiomkin, M. J. Kochenderfer, and P. Abbeel, “Multi-objective policy gradients with topological constraints,” in *2022 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)*. IEEE, 2022, pp. 9034–9039.
*   \[196\] X. Wu, L. Xiao, Y. Sun, J. Zhang, T. Ma, and L. He, “A survey of human-in-the-loop for machine learning,” *Future Generation Computer Systems*, vol. 135, pp. 364–381, 2022.
*   \[197\] X. Xia, Y. Su, C. Zheng, and X. Zeng, “Molecule optimization via multi-objective evolutionary in implicit chemical space,” *arXiv preprint arXiv:2212.08826*, 2022.
*   \[198\] H. Xu, “Short video creation mode based on interactive multi-objective optimization algorithm,” in *International Conference on Frontier Computing*. Springer, 2023, pp. 9–14.
*   \[199\] T. Xu, E. Helenowski, K. A. Sankararaman, D. Jin, K. Peng, E. Han, S. Nie, C. Zhu, H. Zhang, W. Zhou *et al.*, “The perfect blend: Redefining RLHF with mixture of judges,” *arXiv:2409.20370*, 2024.
*   \[200\] Y. Xu, J. Du, Z. Huang, L.-R. Dai, and C.-H. Lee, “Multi-objective learning and mask-based post-processing for deep neural network based speech enhancement,” *arxiv:1703.07172*, 2017.
*   \[201\] Y. Xue, C. Zhu, M. Zhou, M. Wahib, and M. Gabbouj, “A pairwise comparison relation-assisted multi-objective evolutionary neural architecture search method with multi-population mechanism,” *arXiv:2407.15600*, 2024.
*   \[202\] X. Yang, C. Deng, F. Zheng, J. Yan, and W. Liu, “Deep spectral clustering using dual autoencoder network,” in *Proceedings of the IEEE/CVF conference on computer vision and pattern recognition*, 2019, pp. 4066–4075.
*   \[203\] Y. Yao, Z. Liu, Z. Cen, P. Huang, T. Zhang, W. Yu, and D. Zhao, “Gradient shaping for multi-constraint safe reinforcement learning,” in *6th Annual Learning for Dynamics & Control Conference*. PMLR, 2024, pp. 25–39.
*   \[204\] Y. Yao, Y. Pan, J. Li, I. Tsang, and X. Yao, “Proud: Pareto-guided diffusion model for multi-objective generation,” *Machine Learning*, vol. 113, no. 9, p. 6511–6538, 2024.
*   \[205\] T. Yu, S. Kumar, A. Gupta, S. Levine, K. Hausman, and C. Finn, “Gradient surgery for multi-task learning,” *Advances in Neural Information Processing Systems*, vol. 33, pp. 5824–5836, 2020.
*   \[206\] T. Zhang, F. Li, X. Zhao, W. Qi, and T. Liu, “A convolutional neural network-based surrogate model for multi-objective optimization evolutionary algorithm based on decomposition,” *Swarm and Evolutionary Computation*, vol. 72, p. 101081, Jul. 2022.
*   \[207\] X. Zhang, X. Lin, B. Xue, Y. Chen, and Q. Zhang, “Hypervolume maximization: a geometric view of Pareto set learning,” in *Proceedings of the 37th International Conference on Neural Information Processing Systems*. Red Hook, NY, USA: Curran Associates Inc., 2023.
*   \[208\] X. Zhang, X. Lin, and Q. Zhang, “PMGDA: A preference-based multiple gradient descent algorithm,” *arXiv:2402.09492*, 2024.
*   \[209\] A. Zhou, B.-Y. Qu, H. Li, S.-Z. Zhao, P. N. Suganthan, and Q. Zhang, “Multiobjective evolutionary algorithms: A survey of the state of the art,” *Swarm and Evolutionary Computation*, vol. 1, no. 1, pp. 32–49, 2011.
*   \[210\] R. Zhou, T. Liu, D. Kalathil, P. Kumar, and C. Tian, “Anchor-changing regularized natural policy gradient for multi-objective reinforcement learning,” *Advances in Neural Information Processing Systems*, vol. 35, pp. 13 584–13 596, 2022.
*   \[211\] F. Zou, G. G. Yen, L. Tang, and C. Wang, “A reinforcement learning approach for dynamic multi-objective optimization,” *Information Sciences*, vol. 546, pp. 815–834, 2021.

[](javascript:toggleReadingMode\(\); "Disable reading mode, show header and footer")