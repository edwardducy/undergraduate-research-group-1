# Proactive Gradient Conflict Mitigation in Multi-Task Learning:  
A Sparse Training Perspective

 Zhi Zhang Affiliation: ILLC, University of Amsterdam, Netherlands Email: [zhangzhizz2626@gmail.com](mailto:zhangzhizz2626@gmail.com)    Jiayi Shen Affiliation: ILLC, University of Amsterdam, Netherlands Email: [j.shen@uva.nl](mailto:j.shen@uva.nl)    Congfeng Cao Affiliation: ILLC, University of Amsterdam, Netherlands Email: [shanghang@pku.edu.cn](mailto:shanghang@pku.edu.cn)    Gaole Dai Affiliation:  State Key Laboratory of Multimedia Information ProcessingSchool of Computer Science, Peking University, China Email: [e.shutova@uva.nl](mailto:e.shutova@uva.nl)    Shiji Zhou Affiliation: Department of Automation, Tsinghua University, China    Qizhe Zhang Affiliation:  State Key Laboratory of Multimedia Information ProcessingSchool of Computer Science, Peking University, China    Shanghang Zhang11 1 Corresponding author. Affiliation:  State Key Laboratory of Multimedia Information ProcessingSchool of Computer Science, Peking University, China    Ekaterina Shutova11 1 Corresponding author. Affiliation: ILLC, University of Amsterdam, Netherlands 

###### Abstract

Advancing towards generalist agents necessitates the concurrent processing of multiple tasks using a unified model, thereby underscoring the growing significance of simultaneous model training on multiple downstream tasks. A common issue in multi-task learning is the occurrence of gradient conflict, which leads to potential competition among different tasks during joint training. This competition often results in improvements in one task at the expense of deterioration in another. Although several optimization methods have been developed to address this issue by manipulating task gradients for better task balancing, they cannot decrease the incidence of gradient conflict. In this paper, we systematically investigate the occurrence of gradient conflict across different methods and propose a strategy to reduce such conflicts through sparse training (ST), wherein only a portion of the model’s parameters are updated during training while keeping the rest unchanged. Our extensive experiments demonstrate that ST effectively mitigates conflicting gradients and leads to superior performance. Furthermore, ST can be easily integrated with gradient manipulation techniques, thus enhancing their effectiveness.

## 1 Introduction

Attaining the status of a generalist agent necessitates addressing multiple tasks within a unified architecture, thereby emphasizing the significance of multi-task learning (MTL) \[[37](#bib.bib37 "")\], which involves concurrently acquiring proficiency in multiple tasks and striving for superior overall performance compared to learning these tasks separately.

The primary concern for MTL lies in the phenomenon of task competition when the model is jointly trained by optimizing the average loss across all tasks. As a result, a subset of tasks demonstrates superior performance while others remain sub-optimized compared to their individual learning counterparts. One of the reasons behind it, from an optimization perspective, is gradient conflict (GC) \[[35](#bib.bib35 "")\], wherein the direction and magnitude of gradients between tasks differ significantly. This can result in the average gradient biasing towards optimizing one task while providing relatively smaller and sometimes even negative optimization for other tasks when updating the network \[[35](#bib.bib35 ""), [18](#bib.bib18 "")\].

Numerous works have employed the gradient manipulation method to directly or indirectly adjust the gradients of tasks to mitigate the issue of gradient conflict in tasks. The former involves direct alteration of task gradients through manually designed criteria when conflicts arise \[[35](#bib.bib35 ""), [4](#bib.bib4 ""), [17](#bib.bib17 "")\], while the latter modifies task gradients by adjusting weights of loss for each task \[[28](#bib.bib28 ""), [19](#bib.bib19 ""), [26](#bib.bib26 ""), [18](#bib.bib18 "")\]. Although these methods effectively modify the gradients conflicting with each other, they do not decrease the occurrence of conflicting gradients during training \[[29](#bib.bib29 "")\].

A simple approach to mitigate the occurrence of conflicting gradients is to convert those layers in which gradient conflict frequently arises into task-specific layers, thereby reducing the likelihood of gradient conflicts within the remaining shared layers \[[29](#bib.bib29 "")\]. However, this strategy introduces additional modules and disrupts the internal structure of the original model, resulting in increased computational costs. Furthermore, identifying frequently conflicting layers adds extra computational costs. This becomes prohibitively expensive as the model size continues to expand, and thus prompting our fundamental inquiry:

Figure 1: The average occurrence percentage of gradient conflict over epochs (all epochs/last 50% epochs) during training on the SAM model with NYUv2 datasets is evaluated using various methods, including joint training and gradient manipulation techniques. 

(*Q*)  *Is there a universally applicable approach to proactively mitigate the occurrence of gradient conflicts as well as preserve architectural integrity for MTL?*

To tackle this issue, we propose a novel perspective on mitigating gradient conflict in MTL, termed Sparse Training (ST), wherein a subset of parameters from the original model are selected to learn multiple tasks simultaneously while keeping the remaining parameters frozen. The intuition behind this lies in the reduction of a high-dimensional optimization problem to a low-dimensional one, which effectively alleviates the optimization complexity. Moreover, restricting the gradient updates of individual tasks to influence only a subset of parameters, rather than all parameters, effectively reduces potential interference between tasks.

Our key findings demonstrate that ST can effectively reduce the incidence of gradient conflict, particularly during the later stages of training, as illustrated in [Fig. 1](#S1.F1 "In 1 Introduction ‣ Proactive Gradient Conflict Mitigation in Multi-Task Learning: A Sparse Training Perspective"). A summary of our contributions is as follows: i) We provide a novel perspective, sparse training, for proactively reducing the incidence of gradient conflict during training while keeping the architecture intact; ii) Sparse training can be easily applied to improve various gradient manipulation methods by reducing the occurrence the gradient conflict over different datasets and architectures; iii) In addition to conventional research that primarily focuses on smaller models (MTAN \[[20](#bib.bib20 "")\] and SegNet \[[1](#bib.bib1 "")\]), we provide a comprehensive assessment of larger pre-trained models, including SAM \[[3](#bib.bib3 "")\], ViT \[[8](#bib.bib8 "")\], Swin Transformer \[[22](#bib.bib22 "")\], using various gradient manipulation techniques, such as PCGrad \[[35](#bib.bib35 "")\], CAGrad \[[17](#bib.bib17 "")\], GradDrop \[[4](#bib.bib4 "")\], MGDA \[[28](#bib.bib28 "")\], IMTL-G \[[19](#bib.bib19 "")\] and NashMTL \[[26](#bib.bib26 "")\], to stimulate research in the field of sparse training for MTL. Our findings demonstrate that as the model size increases, the issue of gradient conflict becomes more exacerbated, as shown in [Fig. 5(a)](#S4.F5.sf1 "In Figure 5 ‣ Effectiveness on both pre-trained and randomly initialized models ‣ 4.3 Performance on diverse benchmarks ‣ 4 Experiments ‣ Proactive Gradient Conflict Mitigation in Multi-Task Learning: A Sparse Training Perspective"), underscoring the significance of investigating the gradient conflict in large-scale models.

## 2 Related work

#### Multi-task optimization for MTL

The recent works \[[35](#bib.bib35 ""), [17](#bib.bib17 ""), [4](#bib.bib4 ""), [28](#bib.bib28 ""), [19](#bib.bib19 ""), [26](#bib.bib26 ""), [18](#bib.bib18 "")\] have achieved impressive results in addressing task imbalance issues in MTL by directly or indirectly modifying conflicting task gradients. Specifically, some works \[[35](#bib.bib35 ""), [17](#bib.bib17 ""), [4](#bib.bib4 "")\] propose to form a new update gradient at each training step by directly altering gradients based on certain criteria. Other works \[[28](#bib.bib28 ""), [19](#bib.bib19 ""), [26](#bib.bib26 ""), [18](#bib.bib18 ""), [14](#bib.bib14 "")\] learn dynamic loss scale to balance different tasks during training, and thus indirectly altering the gradient of tasks. However, these methods only address GC when it occurs and do not proactively prevent it. In this paper, we sparsely train an MTL model, effectively reducing the incidence of GC.

#### Training with subset of parameters

Several methods have already been proposed in single-task learning. Some of them select a subset of parameters based on a certain pre-defined rule, such as gradient \[[40](#bib.bib40 ""), [11](#bib.bib11 "")\] and magnitude of parameters \[[16](#bib.bib16 "")\]. In addition to selecting parameters by hand design, the works in \[[27](#bib.bib27 ""), [25](#bib.bib25 ""), [34](#bib.bib34 "")\] automatically select the subset of parameters through optimization. Although sparse training has been extensively investigated in single-task learning, its application in MTL remains relatively unexplored. [Sun et al. \[30\]](#bib.bib30 "") and [Calandriello et al. \[2\]](#bib.bib2 "") learn to share information between tasks using a sparse model instead of sparse training. Differently, we research the gradient conflict via the sparse training perspective.

![Refer to caption](2411.18615v1/images/joint_3d.drawio.png)

(a) 

![Refer to caption](2411.18615v1/images/NSF_3d.drawio.png)

(b) 

![Refer to caption](2411.18615v1/images/pcgrad_3d.drawio.png)

(c) 

![Refer to caption](2411.18615v1/images/pcgrad_NSF_3d.drawio.png)

(d) 

Figure 2: Visualization of gradients change for different methods. gig\_{i} and gjg\_{j} are two conflicting gradients, and the green arrow is the actual update vector. The process of sparse training can be interpreted as performing an orthographic/coordinate projection of conflicting gradients onto the subspace defined by the selected parameters, resulting in better alignment of the projected gradients. 

## 3 Approach

### 3.1 Background

#### Multi-task learning (MTL)

aims to learn multiple tasks simultaneously within a single model. Formally, given {𝒯t}t\=1T\\left\\{\\mathcal{T}\_{t}\\right\\}\_{t=1}^{T} tasks (≥2\\geq 2) and a model Θ\\Theta with parameters Θ\=(θsha,θsep)\\Theta=(\\theta\_{\\mathrm{sha}},\\theta\_{\\mathrm{sep}}), where θsha\\theta\_{\\mathrm{sha}} and θsep\\theta\_{\\mathrm{sep}} are shared parameter with all tasks and task-specific parameters θsep\={θsept}t\=1T\\theta\_{\\mathrm{sep}}=\\left\\{\\theta\_{\\mathrm{sep}}^{t}\\right\\}\_{t=1}^{T} respectively, the commonly used optimization method for MTL (referred to as Joint Train) is based on computing the average loss across all tasks with equal weights:

Θ∗\=arg⁡minΘ⁡ℒ⁡(Θ),\\Theta^{\*}=\\arg\\min\_{\\Theta}\\mathcal{L}(\\Theta),

(1)

ℒ⁡(Θ)\=ℒ⁡(θsha,θsep)\=1T​∑t\=1Tℒt​(θsha,θsept)\\mathcal{L}(\\Theta)=\\mathcal{L}(\\theta\_{\\mathrm{sha}},\\theta\_{\\mathrm{sep}})=\\frac{1}{T}\\sum\_{t=1}^{T}\\mathcal{L}\_{t}(\\theta\_{\\mathrm{sha}},\\theta\_{\\mathrm{sep}}^{t})

(2)

where each task tt is associated with a corresponding loss function ℒt​(θsha,θsept)\\mathcal{L}\_{t}(\\theta\_{\\mathrm{sha}},\\theta\_{\\mathrm{sep}}^{t}).

#### Gradient conflict (GC)

However, optimizing all tasks by aggregating their losses indiscriminately ([Eq. 2](#S3.E2 "In Multi-task learning (MTL) ‣ 3.1 Background ‣ 3 Approach ‣ Proactive Gradient Conflict Mitigation in Multi-Task Learning: A Sparse Training Perspective")) may lead to task competition, wherein certain tasks demonstrate improvement while others exhibit a decline compared to training them separately. From an optimization perspective, one of the reasons stems from conflicts in gradients. Formally, the update of task 𝒯i\\mathcal{T}\_{i} may potentially exert a detrimental impact on another task 𝒯j\\mathcal{T}\_{j}, namely:

Δ​ℒj\=ℒj​(θ^sha,θsepj)−ℒj​(θsha,θsepj),\\Delta\\mathcal{L}\_{j}=\\mathcal{L}\_{j}(\\hat{\\theta}\_{\\mathrm{sha}},\\theta\_{\\mathrm{sep}}^{j})-\\mathcal{L}\_{j}(\\theta\_{\\mathrm{sha}},\\theta\_{\\mathrm{sep}}^{j}),

(3)

θ^sha\=θsha−α​𝐠i\\hat{\\theta}\_{\\mathrm{sha}}=\\theta\_{\\mathrm{sha}}-\\alpha\\mathbf{g}\_{i}

(4)

where 𝐠i\=∇θshaℒi​(θsha,θsepi)\\mathbf{g}\_{i}=\\nabla\_{{\\theta}\_{\\mathrm{sha}}}\\mathcal{L}\_{i}(\\theta\_{\\mathrm{sha}},\\theta\_{\\mathrm{sep}}^{i}) is the gradient of loss on task 𝒯i\\mathcal{T}\_{i} with respect to θsha\\theta\_{\\mathrm{sha}} and α\\alpha is the learning rate. After the first-order Taylor approximation, [Eq. 3](#S3.E3 "In Gradient conflict (GC) ‣ 3.1 Background ‣ 3 Approach ‣ Proactive Gradient Conflict Mitigation in Multi-Task Learning: A Sparse Training Perspective") can be expressed as −α𝐠i⋅𝐠j+o(α)-\\alpha\\mathbf{g}\_{i}\\cdot\\mathbf{g}\_{j}+o(\\alpha). Gradient conflict arises when 𝐠i⋅𝐠j<0\\mathbf{g}\_{i}\\cdot\\mathbf{g}\_{j}<0, leading to Δ​ℒj\>0\\Delta\\mathcal{L}\_{j}>0, indicating that task 𝒯i\\mathcal{T}\_{i} has a detrimental impact on task 𝒯j\\mathcal{T}\_{j}. Following \[[35](#bib.bib35 "")\], we provide the definition of gradient conflict:

###### Definition 1 (Gradient Conflict)

If cos⁡ϕi​j<0\\cos{\\phi\_{ij}}<0, where ϕi​j\\phi\_{ij} is the angle between gradients of two tasks 𝐠i\\mathbf{g}\_{i} and 𝐠j\\mathbf{g}\_{j} (i≠j)(i\\neq j), then 𝐠i\\mathbf{g}\_{i} and 𝐠j\\mathbf{g}\_{j} are deemed to exhibit gradient conflict.

#### Gradient manipulation

To alleviate the issue of gradient conflict, gradient manipulation methods adjust conflicting gradients based on specific criteria and utilize these modified gradients for model updating. Instead of updating the model on the average gradient in [Eq. 1](#S3.E1 "In Multi-task learning (MTL) ‣ 3.1 Background ‣ 3 Approach ‣ Proactive Gradient Conflict Mitigation in Multi-Task Learning: A Sparse Training Perspective") and [Eq. 2](#S3.E2 "In Multi-task learning (MTL) ‣ 3.1 Background ‣ 3 Approach ‣ Proactive Gradient Conflict Mitigation in Multi-Task Learning: A Sparse Training Perspective"):

∇θshaℒ​(Θ)\=1T​∑t\=1T∇θshaℒt​(θsha,θsept),\\nabla\_{\\theta\_{\\mathrm{sha}}}\\mathcal{L}(\\Theta)=\\frac{1}{T}\\sum\_{t=1}^{T}\\nabla\_{\\theta\_{\\mathrm{sha}}}\\mathcal{L}\_{t}(\\theta\_{\\mathrm{sha}},\\theta\_{\\mathrm{sep}}^{t}),

(5)

the gradients of all tasks in gradient manipulation methods are modified as follows:

∇θshaℒgm​(Θ)\=1T​∑t\=1T𝒘t​∇θshaℒt​(θsha,θsept),\\nabla\_{\\theta\_{\\mathrm{sha}}}\\mathcal{L}\_{\\mathrm{gm}}(\\Theta)=\\frac{1}{T}\\sum\_{t=1}^{T}\\bm{w}\_{t}\\nabla\_{\\theta\_{\\mathrm{sha}}}\\mathcal{L}\_{t}(\\theta\_{\\mathrm{sha}},\\theta\_{\\mathrm{sep}}^{t}),

(6)

𝒘t\=f⁡(∇θshaℒ1,…,∇θshaℒT)\\bm{w}\_{t}=f\\left(\\nabla\_{\\theta\_{\\mathrm{sha}}}\\mathcal{L}\_{1},\\ldots,\\nabla\_{\\theta\_{\\mathrm{sha}}}\\mathcal{L}\_{T}\\right)

(7)

where 𝒘t\\bm{w}\_{t} can be either pre-defined or dynamically computed for tasks via ff and thus achieve the aim of adjusting the task gradient \[[18](#bib.bib18 ""), [26](#bib.bib26 ""), [28](#bib.bib28 ""), [19](#bib.bib19 ""), [35](#bib.bib35 ""), [4](#bib.bib4 ""), [17](#bib.bib17 "")\]. However, the results of our experiment suggest that these methods can only modify gradients when conflicts occur, rather than proactively reducing the occurrence of GC during training, compared with Joint Train, as shown in [Fig. 1](#S1.F1 "In 1 Introduction ‣ Proactive Gradient Conflict Mitigation in Multi-Task Learning: A Sparse Training Perspective").

### 3.2 Sparse training for multi-task learning

In this study, we investigate the gradient conflict commonly observed in multi-task learning from a novel perspective: sparse training, which selectively trains only a subset of the model parameters as opposed to full parameter training. This perspective is based on the intuition that by converting a high-dimensional space optimization problem into a lower-dimensional one, the complexity of optimization can be effectively reduced. Additionally, by limiting the impact of gradient updates to only a subset of parameters for each task instead of all parameters, potential interference between tasks can be mitigated.

#### Sparse training (ST)

entails the initial parameter selection from the original model, and then updating only these parameters while keeping other parameters fixed during model training. To clarify potential misunderstandings regarding ST—often confused with sparse networks, where parameters are abandoned for model compression—we provide the following definition to ensure consistency and ease of understanding throughout this paper.

###### Definition 2 (Sparse Training)

Given a model Θ\\Theta and a binary mask matrix MM indicating whether parameters in Θ\\Theta are selected, where M∈ℝ|Θ|×|Θ|M\\in\\mathbb{R}^{\\lvert\\Theta\\rvert\\times\\lvert\\Theta\\rvert}, Mi​i∈{0,1}M\_{ii}\\in\\{0,1\\} and Mi​j\=0M\_{ij}=0 (∀i≠j)(\\forall i\\neq j), the model is updated by Θ^\=Θ−α​M​∇Θℒ​(Θ)\\hat{\\Theta}=\\Theta-\\alpha M\\nabla\_{\\Theta}\\mathcal{L}(\\Theta). We define this training strategy as sparse training.

Typically, the model architecture in multi-task learning includes a shared encoder as a feature extractor with task-specific decoders for multiple tasks. Therefore, sparse training is used in the encoder, and full parameters training for the decoders. We detail how the mask is computed in section [Sec. 3.4](#S3.SS4 "3.4 Parameter selection per neuron (PSN) ‣ 3 Approach ‣ Proactive Gradient Conflict Mitigation in Multi-Task Learning: A Sparse Training Perspective"). We now apply sparse training for multi-task learning (Joint Train). The visualization of the gradient change can be viewed in [Fig. 2](#S2.F2 "In Training with subset of parameters ‣ 2 Related work ‣ Proactive Gradient Conflict Mitigation in Multi-Task Learning: A Sparse Training Perspective") and the update with the reformulated gradient from [Eq. 5](#S3.E5 "In Gradient manipulation ‣ 3.1 Background ‣ 3 Approach ‣ Proactive Gradient Conflict Mitigation in Multi-Task Learning: A Sparse Training Perspective") is as follows

θ^sha\=θsha−∇θshaℒ​(Θ)\=θsha−M​1T​∑t\=1T∇θshaℒt​(θsha,θsept).\\begin{split}\\hat{\\theta}\_{\\mathrm{sha}}&=\\theta\_{\\mathrm{sha}}-\\nabla\_{\\theta\_{\\mathrm{sha}}}\\mathcal{L}(\\Theta)\\\\ &=\\theta\_{\\mathrm{sha}}-M\\frac{1}{T}\\sum\_{t=1}^{T}\\nabla\_{\\theta\_{\\mathrm{sha}}}\\mathcal{L}\_{t}(\\theta\_{\\mathrm{sha}},\\theta\_{\\mathrm{sep}}^{t}).\\end{split}

(8)

#### Combination with gradient manipulation methods

The application of sparse training can be seamlessly and effectively extended to improve various gradient manipulation methods in MTL. The update with the reformulated gradient from [Eq. 6](#S3.E6 "In Gradient manipulation ‣ 3.1 Background ‣ 3 Approach ‣ Proactive Gradient Conflict Mitigation in Multi-Task Learning: A Sparse Training Perspective") is as follows

θ^sha\=θsha−∇θshaℒgm​(Θ)\=θsha−M​1T​∑t\=1T𝒘t​∇θshaℒt​(θsha,θsept).\\begin{split}\\hat{\\theta}\_{\\mathrm{sha}}&=\\theta\_{\\mathrm{sha}}-\\nabla\_{\\theta\_{\\mathrm{sha}}}\\mathcal{L}\_{\\mathrm{gm}}(\\Theta)\\\\ &=\\theta\_{\\mathrm{sha}}-M\\frac{1}{T}\\sum\_{t=1}^{T}\\bm{w}\_{t}\\nabla\_{\\theta\_{\\mathrm{sha}}}\\mathcal{L}\_{t}(\\theta\_{\\mathrm{sha}},\\theta\_{\\mathrm{sep}}^{t}).\\end{split}

(9)

### 3.3 Theoretical analysis for sparse training

After introducing sparse training into MTL, the optimization objective in [Eq. 1](#S3.E1 "In Multi-task learning (MTL) ‣ 3.1 Background ‣ 3 Approach ‣ Proactive Gradient Conflict Mitigation in Multi-Task Learning: A Sparse Training Perspective") can be formed:

Θ∗\=arg⁡minΘ⁡ℒ⁡(Θ),s.t.‖(I−M)​(θsha−θshain)‖2\=0,\\small\\Theta^{\*}=\\arg\\min\_{\\Theta}\\mathcal{L}(\\Theta),s.t.\\ \\ \\|(I-M)(\\theta\_{\\mathrm{sha}}-\\theta\_{\\mathrm{sha}}^{\\mathrm{in}})\\|^{2}=0,

(10)

where θshain\\theta\_{\\mathrm{sha}}^{\\mathrm{in}} is the initialized original model for θ\\theta and II is identity matrix. According to Lagrangian duality, [Eq. 10](#S3.E10 "In 3.3 Theoretical analysis for sparse training ‣ 3 Approach ‣ Proactive Gradient Conflict Mitigation in Multi-Task Learning: A Sparse Training Perspective") can be reformulated as:

L\=minΘ⁡maxλ⁡ℒ⁡(Θ)+λ​‖(I−M)​(θsha−θshain)‖2.L=\\min\_{\\Theta}\\max\_{\\lambda}\\mathcal{L}(\\Theta)+\\lambda\\|(I-M)(\\theta\_{\\mathrm{sha}}-\\theta\_{\\mathrm{sha}}^{\\mathrm{in}})\\|^{2}.

(11)

This can be transformed to optimize the upper bound LL of regularized problem:

Lr\=minΘ⁡ℒ⁡(Θ)+‖(I−M)​(θsha−θshain)‖2≤L.L\_{r}=\\min\_{\\Theta}\\mathcal{L}(\\Theta)+\\|(I-M)(\\theta\_{\\mathrm{sha}}-\\theta\_{\\mathrm{sha}}^{\\mathrm{in}})\\|^{2}\\leq L.

(12)

Please see the supplemental material for proof. [Fu et al. \[11\]](#bib.bib11 "") demonstrates that [Eq. 12](#S3.E12 "In 3.3 Theoretical analysis for sparse training ‣ 3 Approach ‣ Proactive Gradient Conflict Mitigation in Multi-Task Learning: A Sparse Training Perspective") has better stability and smaller generalization bound than only optimizing [Eq. 1](#S3.E1 "In Multi-task learning (MTL) ‣ 3.1 Background ‣ 3 Approach ‣ Proactive Gradient Conflict Mitigation in Multi-Task Learning: A Sparse Training Perspective"), resulting in better performance.

### 3.4 Parameter selection per neuron (PSN)

Several promising sparse training methods exist for single-task learning, but they are either time-consuming, requiring mask updates at each iteration \[[27](#bib.bib27 ""), [25](#bib.bib25 ""), [34](#bib.bib34 "")\], or memory-intensive due to gradient calculations for all parameters \[[40](#bib.bib40 ""), [11](#bib.bib11 "")\]. In MTL, where multiple tasks are trained simultaneously, time efficiency is crucial. Thus, we adopt a one-time selection method, choosing parameters before training and keeping the mask fixed throughout. We consider the following two aspects for selection, magnitude of the parameter and involvement of all neurons in the network.

#### The magnitude of parameters

Several studies have focused on model compression through the elimination of parameters with lower magnitudes \[[13](#bib.bib13 ""), [10](#bib.bib10 "")\]. This highlights the significance of parameters with larger magnitudes in neural networks, which is consistent with our experimental findings (See [Fig. 5(c)](#S4.F5.sf3 "In Figure 5 ‣ Effectiveness on both pre-trained and randomly initialized models ‣ 4.3 Performance on diverse benchmarks ‣ 4 Experiments ‣ Proactive Gradient Conflict Mitigation in Multi-Task Learning: A Sparse Training Perspective")). The intuition behind this phenomenon lies in the fact that parameters with larger magnitudes exert a greater influence on altering neuron activation states through the activation function, wherein a neuron becomes active once the input surpasses a predefined threshold. Therefore, we exclusively select parameters with the highest magnitude for training multiple tasks.

#### The involvement of all neurons

A simple idea is to select a certain proportion of parameters with the highest magnitude from the neural network (NN), but this may prevent some neurons from being engaged during training and hinder effective model training due to the dependence of the NN state on neuron activation. Motivated by studies highlighting distinct roles for different components in NN \[[32](#bib.bib32 ""), [40](#bib.bib40 ""), [9](#bib.bib9 "")\], we posit that engaging all neurons is crucial for effective model training. The rationale is that each neuron within the network possesses the inherent capability to finely adjust its activation state, thereby effectively adapting the overall NN state to the tasks, especially for learning multiple tasks simultaneously. Our experiments further substantiate this assertion, as shown in [Fig. 5(c)](#S4.F5.sf3 "In Figure 5 ‣ Effectiveness on both pre-trained and randomly initialized models ‣ 4.3 Performance on diverse benchmarks ‣ 4 Experiments ‣ Proactive Gradient Conflict Mitigation in Multi-Task Learning: A Sparse Training Perspective").

Figure 3: PSN. Top-1 highest-magnitude parameter among all input connections of each neuron is selected. 

#### PSN

By integrating the two aspects, we select the top-K connections (weight/parameters) with the highest magnitude among all input connections for each neuron in the network (Please see [Fig. 3](#S3.F3 "In The involvement of all neurons ‣ 3.4 Parameter selection per neuron (PSN) ‣ 3 Approach ‣ Proactive Gradient Conflict Mitigation in Multi-Task Learning: A Sparse Training Perspective") for top-1 example). This approach facilitates the training process for fitting tasks by ensuring that every neuron possesses activation potential, while parameters with higher magnitudes facilitate easier activation of neurons. In this paper, sparse training refers to using this method to select parameters and training the selected parameter, unless otherwise specified.

## 4 Experiments

Our experiments are conducted on comprehensive MTL benchmarks to evaluate the effectiveness of sparse training. First, we investigate if sparse training reduces gradient conflict. Subsequently, we examine its impact on performance across various MTL setups. The more details of the experiment are provided in [Appendix D](#A4 "Appendix D Detailed experiment setting ‣ Proactive Gradient Conflict Mitigation in Multi-Task Learning: A Sparse Training Perspective").

### 4.1 EXPERIMENTAL SETUP

#### Dateset

Our MTL datasets are categorized into three groups: i) Dense prediction tasks: NYUv2 \[[6](#bib.bib6 "")\]: An indoor scene understanding dataset containing 1449 RGBD images with per-pixel labels across 13 classes, including semantic segmentation, depth estimation, and surface normal prediction. CityScapes \[[5](#bib.bib5 "")\]: 5000 street-view RGBD images with per-pixel annotations for 7-class semantic segmentation and depth estimation. ii) Multiple binary-classification tasks: CelebA \[[21](#bib.bib21 "")\]: 200,000 facial images of 10,000 celebrities, each with 40 binary attributes for facial features. We use the first 10 attributes for 10 binary classification tasks due to limited computation. iii) Multiple multi-class classification tasks: VTAB \[[36](#bib.bib36 "")\]: Containing 24 image understanding tasks with 1000 training examples per task. We use four tasks from it to create two multi-task benchmarks: Clevr: Simple 3D shapes with counting and depth prediction tasks. SmallNORB: Artificial objects with object azimuth and camera elevation prediction tasks.

#### Baseline

We evaluate our approach using various baselines including i) single-task learning (STL): Each task is trained independently; ii) Joint Train: Training all tasks with average task loss; and 6 gradient manipulation methods including 3 direct and 3 indirect modification techniques. The former includes: iii) PCGrad: Projecting each task gradient onto the normal plane of other tasks \[[35](#bib.bib35 "")\]; iv) CAGrad: Enhancing the optimization of average loss by explicitly regulating the minimum decrease across tasks \[[17](#bib.bib17 "")\]; and v) GradDrop: Stochastically dropping specific dimensions of the gradients based on their level of conflict. The latter includes vi) MGDA: Identifying the same descent direction for each task \[[28](#bib.bib28 "")\]; vii) IMTL-G: Determining the update direction by ensuring equal projections on gradients \[[19](#bib.bib19 "")\]; viii) NashMTL: Treating MTL as a bargaining game to optimize all tasks \[[26](#bib.bib26 "")\].

#### Model

We experiment with several architectures including: i) CNN-based: MTAN \[[20](#bib.bib20 "")\] incorporates an attention mechanism into the SegNet \[[1](#bib.bib1 "")\]. ii) Transformer-based. SAM \[[15](#bib.bib15 "")\] is a strong visual foundation model for segmentation. ViT-B/16 \[[8](#bib.bib8 "")\] and Swin Transformer \[[22](#bib.bib22 "")\] are vision classification models pre-trained on ImageNet21K \[[7](#bib.bib7 "")\]. All experiments were conducted on pre-trained SAM, ViT and Swin (except for randomly initialized MTAN), unless otherwise specified.

#### Evaluation

i) Relative task drop (𝚫​𝒎%\\bm{\\Delta}\\bm{m}\\%). Following \[[23](#bib.bib23 "")\], we evaluate the MTL overall performance for a baseline bb by computing the average performance drop against STL ss over {𝒯t}t\=1T\\{\\mathcal{T}\_{t}\\}\_{t=1}^{T} tasks and K𝒯tK\_{\\mathcal{T}\_{t}} metrics for each 𝒯t\\mathcal{T}\_{t}: Δ​m%\=(1T​∑t\=1T1K𝒯t​∑k\=1K𝒯t(−1)δk​(Mbk−Msk)/Msk)×100\\Delta m\\%=(\\frac{1}{T}\\sum\_{t=1}^{T}\\frac{1}{K\_{\\mathcal{T}\_{t}}}\\sum\_{k=1}^{K\_{\\mathcal{T}\_{t}}}(-1)^{\\delta\_{k}}(M\_{b}^{k}-M\_{s}^{k})/M\_{s}^{k})\\times 100 where MbkM\_{b}^{k}, MskM\_{s}^{k} are the value of metrics kk evaluated with bb and ss respectively. δk\=1\\delta\_{k}=1 if the MkM^{k} is higher the better and 0 otherwise. ii) Average incidence of GC (𝒑%\\bm{p}\\%). We evaluate the extent of gradient conflict for a baseline by calculating the average incidence of GC over epochs during training. Given TT tasks, EE epochs, and II iterations per epoch, 𝒑%\=1E​I​∑e\=1E∑i\=1I(Ng​c/Na​l​l)×100\\bm{p}\\%=\\frac{1}{EI}\\sum\_{e=1}^{E}\\sum\_{i=1}^{I}(N\_{gc}/N\_{all})\\times 100, where Ng​cN\_{gc} and Na​l​lN\_{all} represent the number of occurrence of gradient conflicts between two tasks for all task combinations (T2)\\binom{T}{2} and the number of the combinations in each iteration during training, respectively.

### 4.2 Incidence of gradient conflict

We train a MTL model using the Joint Train and 6 state-of-the-art gradient manipulation techniques including PCGrad, CAGrad, GradDrop, MGDA, IMTL-G and NashMTL and then introduce our sparse training strategy to these methods. Throughout the training process, we record instances of GC between any two tasks among all tasks for each training iteration and then calculate the average incidence of GC both over all epochs and the last 50% epochs. The observations of the SAM model on the NYU-v2 dataset are provided below. Similar results on other datasets and models are shown in [Sec. F.3](#A6.SS3 "F.3 NYU-v2 on MTAN ‣ Appendix F Detailed experiment results ‣ Proactive Gradient Conflict Mitigation in Multi-Task Learning: A Sparse Training Perspective"), [Sec. F.5](#A6.SS5 "F.5 CelebA on Swin ‣ Appendix F Detailed experiment results ‣ Proactive Gradient Conflict Mitigation in Multi-Task Learning: A Sparse Training Perspective"), [Sec. F.7](#A6.SS7 "F.7 CityScapes on MTAN ‣ Appendix F Detailed experiment results ‣ Proactive Gradient Conflict Mitigation in Multi-Task Learning: A Sparse Training Perspective") and [Sec. F.6](#A6.SS6 "F.6 SmallNORB on ViT ‣ Appendix F Detailed experiment results ‣ Proactive Gradient Conflict Mitigation in Multi-Task Learning: A Sparse Training Perspective").

#### Gradient manipulation methods cannot effectively reduce the incidence of gradient conflict

The gradient manipulation methods  \[[35](#bib.bib35 ""), [4](#bib.bib4 ""), [17](#bib.bib17 ""), [26](#bib.bib26 ""), [18](#bib.bib18 ""), [19](#bib.bib19 ""), [28](#bib.bib28 "")\] aim to modify conflicting gradients that are prevalent during the joint training of MTL. As shown in [Tab. 1](#S4.T1 "In Sparse training effectively decreases the occurrence of gradient conflict ‣ 4.2 Incidence of gradient conflict ‣ 4 Experiments ‣ Proactive Gradient Conflict Mitigation in Multi-Task Learning: A Sparse Training Perspective"), the average incidence of GC using Joint train is 31.89% across all training epochs and 35.85% over the last 50% epochs. The incidence of GC cannot be effectively reduced by any gradient magnitude methods compared with the Joint train, as shown in [Fig. 1](#S1.F1 "In 1 Introduction ‣ Proactive Gradient Conflict Mitigation in Multi-Task Learning: A Sparse Training Perspective") and [Tab. 1](#S4.T1 "In Sparse training effectively decreases the occurrence of gradient conflict ‣ 4.2 Incidence of gradient conflict ‣ 4 Experiments ‣ Proactive Gradient Conflict Mitigation in Multi-Task Learning: A Sparse Training Perspective"). The reason is that these methods can only make the conflicting gradients not conflict when the GC occurs, rather than proactively prevent the occurrence of GC. The incidence of GC is even exacerbated by these methods, particularly MGDA showing a significant increase of 8.55% compared to Joint Train. Notably, these findings are consistent with \[[29](#bib.bib29 "")\], where they provide the distribution of the angles between the two task gradients.

#### Sparse training effectively decreases the occurrence of gradient conflict

As shown in [Tab. 1](#S4.T1 "In Sparse training effectively decreases the occurrence of gradient conflict ‣ 4.2 Incidence of gradient conflict ‣ 4 Experiments ‣ Proactive Gradient Conflict Mitigation in Multi-Task Learning: A Sparse Training Perspective"), after combining sparse training with all methods, including Joint Train and gradient manipulation methods, the average incidence of gradient conflict is effectively reduced over all epochs. For example, ST in Joint Train reduced the incidence over all epochs by 5.56%. The phenomenon of gradient conflict reduction is consistently observed in nearly every training epoch, as illustrated in [Fig. 4](#S4.F4 "In Sparse training effectively decreases the occurrence of gradient conflict ‣ 4.2 Incidence of gradient conflict ‣ 4 Experiments ‣ Proactive Gradient Conflict Mitigation in Multi-Task Learning: A Sparse Training Perspective"), which further demonstrates the effectiveness of ST for decreasing gradient conflict. In addition, all methods with ST exhibit a greater improvement in the average incidence of gradient conflict during the last 50% epochs compared to all epochs, which implies a greater level of prevention of gradient conflict with the progress of sparse training. For instance of NashMTL, there is a threefold improvement in the average incidence of gradient conflict during the last 50% epochs compared to all epochs.

| Methods      |                 |
| ------------ | --------------- |
| All epochs   | Last 50% epochs |
| 31.89        | 35.85           |
| 26.33 (5.56) | 29.14 (6.71)    |
| 33.69        | 38.70           |
| 30.33 (3.36) | 33.46 (5.24)    |
| 34.26        | 39.97           |
| 31.50 (2.76) | 34.68 (5.29)    |
| 33.56        | 38.45           |
| 30.95 (2.61) | 33.93 (4.52)    |
| 40.44        | 44.77           |
| 40.05 (0.39) | 42.34 (2.43)    |
| 32.15        | 37.13           |
| 28.45 (3.70) | 31.34 (5.79)    |
| 36.67        | 39.58           |
| 35.51 (1.16) | 35.48 (4.10)    |

Table 1: Average incidence of GC between tasks for different methods. We compute the average incidence of GC over all epochs and the last 50% epochs during training SAM on NYUv2. The improvement by sparse training is provided in (∙\\bullet).

(a) 

(b) 

Figure 4: The incidence of GC between tasks during training SAM on NYUv2 dataset. The top and bottom figures are Joint Train and PCGrad respectively. Please see [Fig. 7](#A6.F7 "In F.1 Ablation study ‣ Appendix F Detailed experiment results ‣ Proactive Gradient Conflict Mitigation in Multi-Task Learning: A Sparse Training Perspective") in [Sec. F.2](#A6.SS2 "F.2 NYU-v2 on SAM ‣ Appendix F Detailed experiment results ‣ Proactive Gradient Conflict Mitigation in Multi-Task Learning: A Sparse Training Perspective") for more results on other gradient manipulation methods. 

### 4.3 Performance on diverse benchmarks

It is natural to investigate whether reducing gradient conflict during training through sparsity can enhance performance on common benchmarks. In this section, we present diverse benchmarks to demonstrate the effectiveness of ST.

| Methods         |                    |                      |                      |                             |                                |            |            |            |                        |  |
| --------------- | ------------------ | -------------------- | -------------------- | --------------------------- | ------------------------------ | ---------- | ---------- | ---------- | ---------------------- |  |
| mIoU ↑\\uparrow | Pix Acc ↑\\uparrow | Abs Err ↓\\downarrow | Rel Err ↓\\downarrow | Angle Distance ↓\\downarrow | Within t∘t^{\\circ} ↑\\uparrow |            |            |            |                        |  |
| Mean            | Median             | 11.25                | 22.5                 | 30                          |                                |            |            |            |                        |  |
| 58.6258.62      | 79.2079.20         | 0.38100.3810         | 0.15530.1553         | 19.2919.29                  | 12.6412.64                     | 46.3746.37 | 72.1972.19 | 80.7380.73 | −-                     |  |
| 59.0959.09      | 79.6179.61         | 0.33480.3348         | 0.13600.1360         | 22.3422.34                  | 16.3316.33                     | 35.4635.46 | 64.0264.02 | 75.2075.20 | 6.7636.763             |  |
| 60.0360.03      | 79.9679.96         | 0.33200.3320         | 0.13530.1353         | 21.9821.98                  | 15.9215.92                     | 36.6936.69 | 64.9264.92 | 75.8275.82 | 5.3145.314             |  |
| 59.1859.18      | 80.1280.12         | 0.32580.3258         | 0.13230.1323         | 21.8121.81                  | 15.7215.72                     | 36.9236.92 | 65.4965.49 | 76.2676.26 | 4.5844.584             |  |
| 59.3759.37      | 80.33              | 0.32720.3272         | 0.13300.1330         | 21.5321.53                  | 15.3915.39                     | 38.0238.02 | 66.0966.09 | 76.7176.71 | 3.7413.741             |  |
| 59.7859.78      | 80.1680.16         | 0.3215               | 0.1305               | 19.9219.92                  | 13.4013.40                     | 43.8743.87 | 70.4770.47 | 79.5979.59 | −1.816-1.816           |  |
| 60.3360.33      | 80.2080.20         | 0.32320.3232         | 0.13060.1306         | 19.7419.74                  | 13.2013.20                     | 44.4244.42 | 71.0471.04 | 80.0280.02 | −2.423\\mathbb{-2.423} |  |
| 59.0259.02      | 79.8079.80         | 0.32830.3283         | 0.13210.1321         | 22.0322.03                  | 15.9515.95                     | 36.4236.42 | 64.9064.90 | 75.7975.79 | 5.3235.323             |  |
| 59.7459.74      | 80.3280.32         | 0.32780.3278         | 0.13220.1322         | 21.8121.81                  | 15.6315.63                     | 37.4037.40 | 65.5065.50 | 76.1576.15 | 4.3294.329             |  |
| 37.4337.43      | 67.5867.58         | 0.44270.4427         | 0.18100.1810         | 19.2319.23                  | 12.61                          | 46.4346.43 | 72.35      | 80.87      | 9.1629.162             |  |
| 41.6041.60      | 69.9669.96         | 0.44140.4414         | 0.17780.1778         | 19.22                       | 12.61                          | 46.44      | 72.2972.29 | 80.8080.80 | 7.7917.791             |  |
| 60.64           | 80.2980.29         | 0.33240.3324         | 0.13480.1348         | 19.8519.85                  | 13.3713.37                     | 43.9243.92 | 70.7870.78 | 79.979.9   | −1.537-1.537           |  |
| 60.3560.35      | 80.1880.18         | 0.33470.3347         | 0.13500.1350         | 19.6519.65                  | 13.1513.15                     | 44.6644.66 | 71.2571.25 | 80.2080.20 | −1.955-1.955           |  |
| 59.4259.42      | 80.2080.20         | 0.33030.3303         | 0.13410.1341         | 19.9019.90                  | 13.3913.39                     | 43.8643.86 | 70.6570.65 | 79.7279.72 | −1.295-1.295           |  |
| 59.3659.36      | 79.9879.98         | 0.32780.3278         | 0.13230.1323         | 19.6319.63                  | 13.0213.02                     | 45.0645.06 | 71.3171.31 | 80.1580.15 | −2.384-2.384           |  |

Table 2: The test performance on NYU-v2 dataset training on SAM model. The green cell color indicates that sparse training improves the performance of joint training or gradient manipulation methods. The best result is highlighted in bold.

CelebA

Clevr

SmallNORB

NYU-v2

CityScapes

Methods

𝚫​𝐦%↓\\mathbf{\\Delta m\\%}\\downarrow

Counting

Depth

𝚫​𝐦%↓\\mathbf{\\Delta m\\%}\\downarrow

𝚫​𝐦%↓\\mathbf{\\Delta m\\%}\\downarrow

𝚫​𝐦%↓\\mathbf{\\Delta m\\%}\\downarrow

𝚫​𝐦%↓\\mathbf{\\Delta m\\%}\\downarrow

(F1)

(Top 1 ↑\\uparrow)

(Top 1 ↑\\uparrow)

STL

−-

58.6458.64

57.6857.68

−-

−-

−-

−-

Joint Train

3.123.12

54.8654.86

54.6854.68

5.845.84

10.70

5.59

26.87

w/ ST

2.032.03

61.80

54.8154.81

−0.21-0.21

10.11

2.49

17.48

PCGrad

1.701.70

49.0149.01

53.3953.39

11.9311.93

9.99

3.97

19.96

w/ ST

1.421.42

59.0159.01

55.2955.29

1.75\\pagecolor\[HTML\]{CCFFCC}1.75

9.71

1.98

19.22

CAGrad

1.961.96

49.3349.33

53.6753.67

11.4111.41

10.50

0.20

16.26

w/ ST

1.231.23

58.5158.51

55.2755.27

2.192.19

10.22

-2.76

8.88

GradDrop

1.181.18

49.0249.02

52.8852.88

12.3612.36

11.73

3.58

20.34

w/ ST

0.830.83

58.8758.87

54.0754.07

2.942.94

10.76

1.38

17.45

MGDA

−0.41-0.41

49.5649.56

55.9755.97

9.229.22

10.15

1.38

6.91

w/ ST

-1.08

58.2358.23

56.91

1.021.02

9.79

-3.18

3.17

IMTL-G

0.970.97

54.9954.99

54.5154.51

5.875.87

10.19

-0.76

10.65

w/ ST

0.190.19

61.0561.05

56.7356.73

-1.24

10.15

-3.18

7.10

NashMTL

3.593.59

47.0447.04

53.0753.07

13.8913.89

10.84

-4.04

6.68

w/ ST

3.223.22

58.6158.61

54.9754.97

2.372.37

9.57

-5.11

3.99

Table 3: The test performance on CelebA, Clevr, SmallNORB, NYU-v2 and CityScapes dataset. CelebA is trained on Swin Transformer. Clevr and SmallNORB are trained on ViT. NYU-v2 and CityScapes are trained on MTAN. We only present Δ​m%\\Delta m\\% for limited space. Please see [Tab. 7](#A6.T7 "In F.3 NYU-v2 on MTAN ‣ Appendix F Detailed experiment results ‣ Proactive Gradient Conflict Mitigation in Multi-Task Learning: A Sparse Training Perspective"), [Tab. 12](#A6.T12 "In F.7 CityScapes on MTAN ‣ Appendix F Detailed experiment results ‣ Proactive Gradient Conflict Mitigation in Multi-Task Learning: A Sparse Training Perspective") and [Tab. 11](#A6.T11 "In F.6 SmallNORB on ViT ‣ Appendix F Detailed experiment results ‣ Proactive Gradient Conflict Mitigation in Multi-Task Learning: A Sparse Training Perspective") for detailed results in supplemental materials. The green cell color indicates that sparse training improves the performance of joint training or gradient manipulation methods. The best result is highlighted in bold. 

#### Sparse training improves the performance for all state-of-the-art methods

The performance of Joint Train and all gradient manipulation methods is consistently improved by sparse training, as demonstrated in [Tab. 2](#S4.T2 "In 4.3 Performance on diverse benchmarks ‣ 4 Experiments ‣ Proactive Gradient Conflict Mitigation in Multi-Task Learning: A Sparse Training Perspective") for NYU-v2 benchmarks. Specifically, sparse training not only enhances overall task performance but also improves individual task performance for the majority of methods. For example, in [Tab. 2](#S4.T2 "In 4.3 Performance on diverse benchmarks ‣ 4 Experiments ‣ Proactive Gradient Conflict Mitigation in Multi-Task Learning: A Sparse Training Perspective"), Joint Train demonstrates improvements across all individual tasks through sparse training. Similarly, as shown in [Tab. 3](#S4.T3 "In 4.3 Performance on diverse benchmarks ‣ 4 Experiments ‣ Proactive Gradient Conflict Mitigation in Multi-Task Learning: A Sparse Training Perspective"), all methods exhibit notable improvements by sparse training on CelebA, Clevr, SmallNORB and CityScapes benchmarks.

#### Effectiveness on both pre-trained and randomly initialized models

Our study primarily focuses on the sparse training for large pre-trained models, because leveraging prior knowledge from these models can be beneficial for MTL and our experimental results demonstrate that larger models exhibit a more severe gradient conflict, as shown in [Fig. 5(a)](#S4.F5.sf1 "In Figure 5 ‣ Effectiveness on both pre-trained and randomly initialized models ‣ 4.3 Performance on diverse benchmarks ‣ 4 Experiments ‣ Proactive Gradient Conflict Mitigation in Multi-Task Learning: A Sparse Training Perspective"). However, in order to ensure a fair comparison with related works that manipulate gradients in small and randomly initialized models, we also conduct experiments under the same setting as theirs to further demonstrate the effectiveness of sparse training. As shown in [Tab. 3](#S4.T3 "In 4.3 Performance on diverse benchmarks ‣ 4 Experiments ‣ Proactive Gradient Conflict Mitigation in Multi-Task Learning: A Sparse Training Perspective"), we observe that even for the small randomly initialized models, the performance of joint training and all gradient manipulation methods is improved by sparse training. Please see [Tab. 7](#A6.T7 "In F.3 NYU-v2 on MTAN ‣ Appendix F Detailed experiment results ‣ Proactive Gradient Conflict Mitigation in Multi-Task Learning: A Sparse Training Perspective") and [Tab. 12](#A6.T12 "In F.7 CityScapes on MTAN ‣ Appendix F Detailed experiment results ‣ Proactive Gradient Conflict Mitigation in Multi-Task Learning: A Sparse Training Perspective") for the detailed results in the Appendix.

(a) 

(b) 

(c) 

Figure 5: Ablation study for Joint Train with NYU-v2 dataset. (a) The average incidence of GC during joint training on different sizes of Swin transformers. Please see the numerical statics for all epochs in [Tab. 8](#A6.T8 "In F.4 NYU-v2 on Swin ‣ Appendix F Detailed experiment results ‣ Proactive Gradient Conflict Mitigation in Multi-Task Learning: A Sparse Training Perspective") in [Sec. F.4](#A6.SS4 "F.4 NYU-v2 on Swin ‣ Appendix F Detailed experiment results ‣ Proactive Gradient Conflict Mitigation in Multi-Task Learning: A Sparse Training Perspective"). (b) The different number of trainable parameters for MTAN and SAM models. (C) Different sparse methods training on SAM. Metrics for all tasks are min-max normalized. Please see [Tab. 5](#A5.T5 "In Training with subset of parameters ‣ Appendix E Extended related work ‣ Proactive Gradient Conflict Mitigation in Multi-Task Learning: A Sparse Training Perspective") for detailed results in [Sec. F.1](#A6.SS1 "F.1 Ablation study ‣ Appendix F Detailed experiment results ‣ Proactive Gradient Conflict Mitigation in Multi-Task Learning: A Sparse Training Perspective").

#### Generalization on different architectures and MTL tasks

To evaluate the generalization across diverse architectures and MTL tasks, we conducted experiments on both CNN-based models and transformer-based models with varying visual MTL capabilities. Specifically, our MTL tasks encompassed visual classification (CelebA, Clevr and SmallNORB) and visual dense prediction (NYU-v2 and CityScapes). For the former, we utilized Swin Transformer and ViT as backbones for multiple binary classification tasks ([Tab. 3](#S4.T3 "In 4.3 Performance on diverse benchmarks ‣ 4 Experiments ‣ Proactive Gradient Conflict Mitigation in Multi-Task Learning: A Sparse Training Perspective")) and two multi-class classification tasks ([Tab. 3](#S4.T3 "In 4.3 Performance on diverse benchmarks ‣ 4 Experiments ‣ Proactive Gradient Conflict Mitigation in Multi-Task Learning: A Sparse Training Perspective"), and [Tab. 11](#A6.T11 "In F.6 SmallNORB on ViT ‣ Appendix F Detailed experiment results ‣ Proactive Gradient Conflict Mitigation in Multi-Task Learning: A Sparse Training Perspective") in Appendix), respectively. The latter involved predicting dense masks for each task, necessitating an encoder-decoder structure to generate corresponding masks. We explored two types of structures: a symmetrical encoder-decoder structure with a CNN-based model, e.g. MTAN ([Tab. 3](#S4.T3 "In 4.3 Performance on diverse benchmarks ‣ 4 Experiments ‣ Proactive Gradient Conflict Mitigation in Multi-Task Learning: A Sparse Training Perspective"), and [Tabs. 7](#A6.T7 "In F.3 NYU-v2 on MTAN ‣ Appendix F Detailed experiment results ‣ Proactive Gradient Conflict Mitigation in Multi-Task Learning: A Sparse Training Perspective") and [12](#A6.T12 "Table 12 ‣ F.7 CityScapes on MTAN ‣ Appendix F Detailed experiment results ‣ Proactive Gradient Conflict Mitigation in Multi-Task Learning: A Sparse Training Perspective") in Appendix) and an asymmetric structure with a heavy-weight encoder and a light-weight decoder using a transformer-based model, e.g. SAM ([Tab. 2](#S4.T2 "In 4.3 Performance on diverse benchmarks ‣ 4 Experiments ‣ Proactive Gradient Conflict Mitigation in Multi-Task Learning: A Sparse Training Perspective") in Appendix). As shown in these tables, the efficacy of sparse training in improving all baselines across various architectures and MTL tasks underscores its robust generalization capability.

### 4.4 Ablation study

#### The larger the model, the more severe gradient conflicts.

In this paper, we focus more on investigating the gradient conflict in the pre-trained large models as larger models demonstrated a more severe phenomenon of gradient conflict. This can be observed in [Fig. 5(a)](#S4.F5.sf1 "In Figure 5 ‣ Effectiveness on both pre-trained and randomly initialized models ‣ 4.3 Performance on diverse benchmarks ‣ 4 Experiments ‣ Proactive Gradient Conflict Mitigation in Multi-Task Learning: A Sparse Training Perspective"), where Swin/Tiny demonstrates significantly less gradient conflict compared to Swin/Base and Swin/Large. It is worth noting that although larger models tend to experience more severe gradient conflicts, this does not necessarily lead to inferior performance compared to smaller models with milder gradient conflicts. This discrepancy can be attributed to differences in model capacity and the prior knowledge embedded through pre-training. Nevertheless, this observation underscores the importance of exploring methods to mitigate gradient conflicts in larger models. Within the same model architecture and size, reducing gradient conflicts has been shown to improve performance, as evidenced by works such as \[[35](#bib.bib35 ""), [17](#bib.bib17 "")\]. Addressing severe gradient conflicts in larger models may thus unlock their full potential, enabling better utilization of their capacity and capabilities.

#### Effortless search for the number of trainable parameters.

We explore the effect of trainable parameter numbers for ST. The results in [Fig. 5(b)](#S4.F5.sf2 "In Figure 5 ‣ Effectiveness on both pre-trained and randomly initialized models ‣ 4.3 Performance on diverse benchmarks ‣ 4 Experiments ‣ Proactive Gradient Conflict Mitigation in Multi-Task Learning: A Sparse Training Perspective") show that the pre-trained model (SAM) and the randomly initialized model (MTAN) have different optimal trainable parameter numbers. MTAN requires ∼\\sim60% of the parameters, while SAM needs only ∼\\sim30%, leveraging information from the pre-trained model. In our paper, most of the experiments use these proportions for ST and achieve better results (please see [Tab. 4](#A4.T4 "In D.1 Number of trainable parameters ‣ Appendix D Detailed experiment setting ‣ Proactive Gradient Conflict Mitigation in Multi-Task Learning: A Sparse Training Perspective") in [Sec. D.1](#A4.SS1 "D.1 Number of trainable parameters ‣ Appendix D Detailed experiment setting ‣ Proactive Gradient Conflict Mitigation in Multi-Task Learning: A Sparse Training Perspective") for the detailed number). Additionally, ST offers a wide range of trainable parameter options that outperform Joint Train, which implies that hyperparameter search for the number of trainable parameters becomes effortless. Specifically, both models have a ∼\\sim40% probability of yielding superior outcomes.

#### Effectiveness for both higher magnitude and neural-level selection.

We investigate various parameter selection approaches: Random: Randomly selecting parameters from the network; Global: Choosing parameters with the highest magnitude from the whole network instead of the input connections of each neuron in the network (Ours); Reverse: Selecting parameters with the lowest magnitude among input connections of each neuron. For a fair comparison, we maintain the same selected number. The results in [Fig. 5(c)](#S4.F5.sf3 "In Figure 5 ‣ Effectiveness on both pre-trained and randomly initialized models ‣ 4.3 Performance on diverse benchmarks ‣ 4 Experiments ‣ Proactive Gradient Conflict Mitigation in Multi-Task Learning: A Sparse Training Perspective") indicate that higher magnitude values are superior to lower ones (Ours \>\> Reverse). Furthermore, it is crucial to evenly select parameters from the entire network (Ours \>\> Random \>\> Global), as Ours ensure that the parameters of input connection for each neuron are selected, and Random guarantees an equal proportion of parameters is selected in each block of the network, whereas this is not the case for Global (see [Fig. 6](#A5.F6 "In Training with subset of parameters ‣ Appendix E Extended related work ‣ Proactive Gradient Conflict Mitigation in Multi-Task Learning: A Sparse Training Perspective") for detailed statistics in Appendix).

## 5 Conclusion

In this paper, the occurrence of gradient conflict in multi-task learning is extensively investigated from a novel perspective: sparse training. Extensive experiments demonstrate that sparse training transferring high-dimensional space into low-dimensional space effectively reduces the incidence of gradient conflict during training while preserving the integrity of the original model. Furthermore, combining sparse training with other gradient manipulation methods significantly improves performance for multi-task learning.

## References

*   \[1\] Vijay Badrinarayanan, Alex Kendall, and Roberto Cipolla. Segnet: A deep convolutional encoder-decoder architecture for image segmentation. *IEEE transactions on pattern analysis and machine intelligence*, 39(12):2481–2495, 2017.
*   \[2\] Daniele Calandriello, Alessandro Lazaric, and Marcello Restelli. Sparse multi-task reinforcement learning. *Advances in neural information processing systems*, 27, 2014.
*   \[3\] Tianrun Chen, Lanyun Zhu, Chaotao Ding, Runlong Cao, Shangzhan Zhang, Yan Wang, Zejian Li, Lingyun Sun, Papa Mao, and Ying Zang. Sam fails to segment anything? – sam-adapter: Adapting sam in underperformed scenes: Camouflage, shadow, and more, 2023.
*   \[4\] Zhao Chen, Jiquan Ngiam, Yanping Huang, Thang Luong, Henrik Kretzschmar, Yuning Chai, and Dragomir Anguelov. Just pick a sign: Optimizing deep multitask models with gradient sign dropout. *Advances in Neural Information Processing Systems*, 33:2039–2050, 2020.
*   \[5\] Marius Cordts, Mohamed Omran, Sebastian Ramos, Timo Rehfeld, Markus Enzweiler, Rodrigo Benenson, Uwe Franke, Stefan Roth, and Bernt Schiele. The cityscapes dataset for semantic urban scene understanding. In *Proceedings of the IEEE conference on computer vision and pattern recognition*, pages 3213–3223, 2016.
*   \[6\] Camille Couprie, Clément Farabet, Laurent Najman, and Yann LeCun. Indoor semantic segmentation using depth information, 2013.
*   \[7\] Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. Imagenet: A large-scale hierarchical image database. In *Proceedings of the IEEE/CVF conference on computer vision and pattern recognition*, pages 248–255. Ieee, 2009.
*   \[8\] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, et al. An image is worth 16x16 words: Transformers for image recognition at scale. *arXiv preprint arXiv:2010.11929*, 2020.
*   \[9\] Fenglei Fan, Jinjun Xiong, Mengzhou Li, and Ge Wang. On interpretability of artificial neural networks: A survey. *IEEE Transactions on Radiation and Plasma Medical Sciences*, 5:741–760, 2020.
*   \[10\] Jonathan Frankle and Michael Carbin. The lottery ticket hypothesis: Finding sparse, trainable neural networks. *arXiv preprint arXiv:1803.03635*, 2018.
*   \[11\] Zihao Fu, Haoran Yang, Anthony Man-Cho So, Wai Lam, Lidong Bing, and Nigel Collier. On the effectiveness of parameter-efficient fine-tuning. In *Proceedings of the AAAI Conference on Artificial Intelligence*, pages 12799–12807, 2023.
*   \[12\] Pengsheng Guo, Chen-Yu Lee, and Daniel Ulbricht. Learning to branch for multi-task learning. In *International conference on machine learning*, pages 3854–3863. PMLR, 2020.
*   \[13\] Song Han, Jeff Pool, John Tran, and William Dally. Learning both weights and connections for efficient neural network. *Advances in neural information processing systems*, 28, 2015.
*   \[14\] Alex Kendall, Yarin Gal, and Roberto Cipolla. Multi-task learning using uncertainty to weigh losses for scene geometry and semantics. In *Proceedings of the IEEE conference on computer vision and pattern recognition*, pages 7482–7491, 2018.
*   \[15\] Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C Berg, Wan-Yen Lo, et al. Segment anything. In *Proceedings of the IEEE/CVF International Conference on Computer Vision*, pages 4015–4026, 2023.
*   \[16\] François Lagunas, Ella Charlaix, Victor Sanh, and Alexander M Rush. Block pruning for faster transformers. *arXiv preprint arXiv:2109.04838*, 2021.
*   \[17\] Bo Liu, Xingchao Liu, Xiaojie Jin, Peter Stone, and Qiang Liu. Conflict-averse gradient descent for multi-task learning. *Advances in Neural Information Processing Systems*, 34:18878–18890, 2021a.
*   \[18\] Bo Liu, Yihao Feng, Peter Stone, and Qiang Liu. Famo: Fast adaptive multitask optimization, 2023.
*   \[19\] Liyang Liu, Yi Li, Zhanghui Kuang, J Xue, Yimin Chen, Wenming Yang, Qingmin Liao, and Wayne Zhang. Towards impartial multi-task learning. In *International Conference on Learning Representations*, 2021b.
*   \[20\] Shikun Liu, Edward Johns, and Andrew J Davison. End-to-end multi-task learning with attention. In *Proceedings of the IEEE/CVF conference on computer vision and pattern recognition*, pages 1871–1880, 2019.
*   \[21\] Ziwei Liu, Ping Luo, Xiaogang Wang, and Xiaoou Tang. Deep learning face attributes in the wild. In *Proceedings of the IEEE international conference on computer vision*, pages 3730–3738, 2015.
*   \[22\] Ze Liu, Yutong Lin, Yue Cao, Han Hu, Yixuan Wei, Zheng Zhang, Stephen Lin, and Baining Guo. Swin transformer: Hierarchical vision transformer using shifted windows. In *Proceedings of the IEEE/CVF international conference on computer vision*, pages 10012–10022, 2021c.
*   \[23\] Kevis-Kokitsi Maninis, Ilija Radosavovic, and Iasonas Kokkinos. Attentive single-tasking of multiple tasks. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pages 1851–1860, 2019.
*   \[24\] Ishan Misra, Abhinav Shrivastava, Abhinav Gupta, and Martial Hebert. Cross-stitch networks for multi-task learning. In *Proceedings of the IEEE/CVF conference on computer vision and pattern recognition*, pages 3994–4003, 2016.
*   \[25\] Hesham Mostafa and Xin Wang. Parameter efficient training of deep convolutional neural networks by dynamic sparse reparameterization. In *International Conference on Machine Learning*, pages 4646–4655. PMLR, 2019.
*   \[26\] Aviv Navon, Aviv Shamsian, Idan Achituve, Haggai Maron, Kenji Kawaguchi, Gal Chechik, and Ethan Fetaya. Multi-task learning as a bargaining game, 2022.
*   \[27\] Victor Sanh, Thomas Wolf, and Alexander Rush. Movement pruning: Adaptive sparsity by fine-tuning. *Advances in neural information processing systems*, 33:20378–20389, 2020.
*   \[28\] Ozan Sener and Vladlen Koltun. Multi-task learning as multi-objective optimization. *Advances in Neural Information Processing Systems*, 31, 2018.
*   \[29\] Guangyuan Shi, Qimai Li, Wenlong Zhang, Jiaxin Chen, and Xiao-Ming Wu. Recon: Reducing conflicting gradients from the root for multi-task learning. *arXiv preprint arXiv:2302.11289*, 2023.
*   \[30\] Tianxiang Sun, Yunfan Shao, Xiaonan Li, Pengfei Liu, Hang Yan, Xipeng Qiu, and Xuanjing Huang. Learning sparse sharing architectures for multiple tasks. In *Proceedings of the AAAI conference on artificial intelligence*, pages 8936–8943, 2020.
*   \[31\] Simon Vandenhende, Stamatios Georgoulis, Wouter Van Gansbeke, Marc Proesmans, Dengxin Dai, and Luc Van Gool. Multi-task learning for dense prediction tasks: A survey. *IEEE transactions on pattern analysis and machine intelligence*, 44(7):3614–3633, 2021.
*   \[32\] Zifeng Wang, Shao-Lun Huang, Ercan E Kuruoglu, Jimeng Sun, Xi Chen, and Yefeng Zheng. Pac-bayes information bottleneck. *arXiv preprint arXiv:2109.14509*, 2021.
*   \[33\] Dan Xu, Wanli Ouyang, Xiaogang Wang, and Nicu Sebe. Pad-net: Multi-tasks guided prediction-and-distillation network for simultaneous depth estimation and scene parsing. In *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition*, pages 675–684, 2018.
*   \[34\] Runxin Xu, Fuli Luo, Zhiyuan Zhang, Chuanqi Tan, Baobao Chang, Songfang Huang, and Fei Huang. Raise a child in large language model: Towards effective and generalizable fine-tuning. *arXiv preprint arXiv:2109.05687*, 2021.
*   \[35\] Tianhe Yu, Saurabh Kumar, Abhishek Gupta, Sergey Levine, Karol Hausman, and Chelsea Finn. Gradient surgery for multi-task learning. *Advances in Neural Information Processing Systems*, 33:5824–5836, 2020.
*   \[36\] Xiaohua Zhai, Joan Puigcerver, Alexander Kolesnikov, Pierre Ruyssen, Carlos Riquelme, Mario Lucic, Josip Djolonga, Andre Susano Pinto, Maxim Neumann, Alexey Dosovitskiy, et al. A large-scale study of representation learning with the visual task adaptation benchmark. *arXiv preprint arXiv:1910.04867*, 2019.
*   \[37\] Yu Zhang and Qiang Yang. A survey on multi-task learning. *IEEE Transactions on Knowledge and Data Engineering*, 34(12):5586–5609, 2021.
*   \[38\] Zhenyu Zhang, Zhen Cui, Chunyan Xu, Zequn Jie, Xiang Li, and Jian Yang. Joint task-recursive learning for semantic segmentation and depth estimation. In *Proceedings of the European Conference on Computer Vision (ECCV)*, pages 235–251, 2018.
*   \[39\] Zhenyu Zhang, Zhen Cui, Chunyan Xu, Yan Yan, Nicu Sebe, and Jian Yang. Pattern-affinitive propagation across depth, surface normal and semantic segmentation. In *Proceedings of the IEEE/CVF conference on computer vision and pattern recognition*, pages 4106–4115, 2019.
*   \[40\] Zhi Zhang, Qizhe Zhang, Zijun Gao, Renrui Zhang, Ekaterina Shutova, Shiji Zhou, and Shanghang Zhang. Gradient-based parameter selection for efficient fine-tuning. *arXiv preprint arXiv:2312.10136*, 2023.

  
 

Supplementary Material  

In this supplemental material, we provide extra details about the content in the main body of the paper. First, we provide detailed proof for [Equation 12](#S3.E12 "In 3.3 Theoretical analysis for sparse training ‣ 3 Approach ‣ Proactive Gradient Conflict Mitigation in Multi-Task Learning: A Sparse Training Perspective") in [Appendix A](#A1 "Appendix A Proof for ‣ Proactive Gradient Conflict Mitigation in Multi-Task Learning: A Sparse Training Perspective"). Then, we discuss the limitations of our work in [Appendix B](#A2 "Appendix B Limitations ‣ Proactive Gradient Conflict Mitigation in Multi-Task Learning: A Sparse Training Perspective"). Moreover, the broader impacts of our research are discussed in [Appendix C](#A3 "Appendix C Broader Impacts ‣ Proactive Gradient Conflict Mitigation in Multi-Task Learning: A Sparse Training Perspective"). In addition, we present all hyperparameters and experiment settings in [Appendix F](#A6 "Appendix F Detailed experiment results ‣ Proactive Gradient Conflict Mitigation in Multi-Task Learning: A Sparse Training Perspective") for a better understanding of the experiments and reproduction to the readers. We also provide the extended related works in [Appendix E](#A5 "Appendix E Extended related work ‣ Proactive Gradient Conflict Mitigation in Multi-Task Learning: A Sparse Training Perspective"). Finally, the additional experiment results are demonstrated in [Appendix F](#A6 "Appendix F Detailed experiment results ‣ Proactive Gradient Conflict Mitigation in Multi-Task Learning: A Sparse Training Perspective"), which further indicate the effectiveness of our proposed method and the consistency with the claim in the main body of the paper.

## Appendix A Proof for [Equation 12](#S3.E12 "In 3.3 Theoretical analysis for sparse training ‣ 3 Approach ‣ Proactive Gradient Conflict Mitigation in Multi-Task Learning: A Sparse Training Perspective")

According to Lagrangian duality, [Eq. 10](#S3.E10 "In 3.3 Theoretical analysis for sparse training ‣ 3 Approach ‣ Proactive Gradient Conflict Mitigation in Multi-Task Learning: A Sparse Training Perspective") can be reformulated as:

L\=\\displaystyle L=

minΘ⁡maxλ⁡ℒ⁡(Θ)+λ​‖(I−M)​(θsha−θshain)‖2\\displaystyle\\min\_{\\Theta}\\max\_{\\lambda}\\mathcal{L}(\\Theta)+\\lambda\\|(I-M)(\\theta\_{\\mathrm{sha}}-\\theta\_{\\mathrm{sha}}^{\\mathrm{in}})\\|^{2}

≥\\displaystyle\\geq

maxλ⁡minΘ⁡ℒ⁡(Θ)+λ​‖(I−M)​(θsha−θshain)‖2\\displaystyle\\max\_{\\lambda}\\min\_{\\Theta}\\mathcal{L}(\\Theta)+\\lambda\\|(I-M)(\\theta\_{\\mathrm{sha}}-\\theta\_{\\mathrm{sha}}^{\\mathrm{in}})\\|^{2}

≥\\displaystyle\\geq

minΘ⁡ℒ⁡(Θ)+‖(I−M)​(θsha−θshain)‖2\\displaystyle\\min\_{\\Theta}\\mathcal{L}(\\Theta)+\\|(I-M)(\\theta\_{\\mathrm{sha}}-\\theta\_{\\mathrm{sha}}^{\\mathrm{in}})\\|^{2}

where λ\\lambda is the Lagrangian multiplier.

## Appendix B Limitations

Due to the limited computational resources, we employ grid searches in the Joint train method to determine the optimal hyperparameter for the number of trainable parameters, which is then utilized across all gradient manipulation methods. However, it is possible that these methods may benefit from a more optimized hyperparameter selection for the number of trainable parameters. Furthermore, sparse training can effectively mitigate gradient conflicts between tasks in MTL by reducing the dimensionality of parameter space and limiting their impact on updates between tasks. The regularization constitutes one of the theory’s reasons. Nevertheless, we anticipate that our future research will contribute to a deeper comprehension of multi-task learning and subsequently enhance the performance of MTL.

## Appendix C Broader Impacts

The nature of our research does not directly contribute to societal impact; however, like any machine learning paper, it has the potential to adversely affect society through automation and job displacement. While it is challenging to predict specific risks, similar to any technology, inadequate regulation may lead to an exacerbation of social and economic inequality. The positive aspect lies in the potential environmental impact of our work, as multi-task learning enables information sharing among tasks, thereby reducing data requirements and further minimizing energy consumption during training.

## Appendix D Detailed experiment setting

### D.1 Number of trainable parameters

We provide the number of trainable parameters for all experiments conducted in our paper. As shown in [Table 4](#A4.T4 "In D.1 Number of trainable parameters ‣ Appendix D Detailed experiment setting ‣ Proactive Gradient Conflict Mitigation in Multi-Task Learning: A Sparse Training Perspective"), most of them have the same percentage of trainable parameters within a model across different methods. In addition, in general, we can observe that sparse training for the pre-trained model needs ∼\\sim30% while that for random initialized model needs ∼\\sim60%.

|        |        |       |           |        |            |
| ------ | ------ | ----- | --------- | ------ | ---------- |
| SAM    | Swin   | ViT   | MTAN      |        |            |
| NYU-v2 | CelebA | Clevr | SmallNORB | NYU-v2 | CityScapes |
| 30.97  | 37.60  | 29.38 | 29.38     | 62.19  | 76.02      |
| 30.97  | 37.60  | 29.38 | 19.63     | 62.19  | 76.02      |
| 30.97  | 72.85  | 29.38 | 29.38     | 62.19  | 76.02      |
| 30.97  | 49.58  | 29.38 | 29.38     | 62.19  | 76.02      |
| 30.97  | 37.60  | 29.38 | 29.38     | 62.19  | 62.19      |
| 30.97  | 37.60  | 29.38 | 29.38     | 62.19  | 62.19      |
| 30.97  | 37.60  | 29.38 | 29.38     | 62.19  | 83.48      |
| 30.97  | 37.60  | 29.38 | 29.38     | 62.19  | 62.19      |

Table 4: Number of trainable parameters. The values in the table are expressed as percentages (%). As we select Top-K input parameters among all input connections for each neuron, therefore the same K might lead to different percentages of trainable parameters for different models. For example, K=300 results in 30.97% in SAM, 37.60% in Swin, and 29.38% in ViT for the pre-trained model. 

### D.2 Implementation details

Following the work of Nash \[[26](#bib.bib26 "")\], we apply all gradient manipulation techniques to the gradients of the shared weights. We set the hyperparameter c of CAGrad to 0.4, as it has been reported to yield optimal performance for NYUv2 and Cityscapes datasets \[[17](#bib.bib17 "")\]. The experiments were conducted on the A100 80G GPU. Typically, training with SAM using NYU-v2 and Swin with CelebA requires approximately 1 day for a gradient manipulation method. Training ViT with SmallNORB takes around 18 minutes for a gradient manipulation method, while training ViT with Clevr takes about 30 minutes. On the other hand, training MTAN with NYU-v2 demands roughly 18 hours for a gradient manipulation method, whereas training MTAN with CityScapes necessitates approximately 12 hours.

#### SAM, ViT, Swin

For all methods, including single-task learning, the gradient manipulation method, and our sparse training, we employed a batch size of 3 and searched for the optimal learning rate from the set {2e-4, 5e-5}, and then the best results are reported. The reason is that we find the optimal learning rate for sparse training is bigger than that for full parameters training. Therefore, for most methods, the optimal learning rate for sparse training is 2e-4 and that for the full parameters training is 5e-5. we also use data augmentations for all methods, following \[[17](#bib.bib17 "")\]. The batch size used is set to be 3 for NYUv2 dataset, and 256 for CelebA, and 128 for SmallNORB and Clevr.

#### MTAN

Following the works in \[[26](#bib.bib26 ""), [17](#bib.bib17 "")\], we incorporate data augmentations during training for both Joint Train method and all gradient manipulation methods. Each method is trained for 200 epochs with an initial learning rate of 0.0001, which is then reduced to 0.00005 after 100 epochs. For Multi-Task Learning (MTL) methods, we utilize a Multi-Task Attention Network (MTAN) \[[20](#bib.bib20 "")\] based on SegNet architecture proposed by \[[1](#bib.bib1 "")\]. Similar to previous studies \[[17](#bib.bib17 "")\], the STL baseline refers to training task-specific SegNet models. The batch size used is set to be 2 for NYUv2 dataset and 8 for CityScapes dataset respectively. To align with prior research on MTL including \[[17](#bib.bib17 ""), [35](#bib.bib35 ""), [1](#bib.bib1 "")\], we report the test performance averaged over the last 10 epochs.

## Appendix E Extended related work

#### Multi-task learning

Multi-task learning \[[37](#bib.bib37 "")\] aims to improve the overall performance of all tasks. In this work, we focus on a conventional setup of multi-task learning  \[[31](#bib.bib31 "")\]: given a single input, multi-task models perform different and related predictions, such as segmentation, depth and surface normal. In other words, the input is shared by different tasks. In this paper, we roughly divide existing MTL into two categories:

i) Multi-task optimization. Recent works \[[35](#bib.bib35 ""), [17](#bib.bib17 ""), [4](#bib.bib4 ""), [28](#bib.bib28 ""), [19](#bib.bib19 ""), [26](#bib.bib26 ""), [18](#bib.bib18 "")\] provide impressive results in solving the task imbalance during optimization. The rationale behind these works is that re-weighting all task gradients or losses helps multi-task models reduce conflicting gradients among tasks \[[17](#bib.bib17 ""), [28](#bib.bib28 "")\]. Specifically, some works \[[4](#bib.bib4 ""), [28](#bib.bib28 ""), [19](#bib.bib19 "")\] propose to form a new update gradient at each optimization by linearly combining task gradients. Other works \[[18](#bib.bib18 ""), [14](#bib.bib14 "")\] learn dynamic loss scale to balance different tasks during training. However, it is challenging to scale up most existing optimization works to giant foundation models due to non-trivial computational and memory costs. In this paper, we propose a neuron-based parameter selection to sparsely fine-tune the pre-trained model, which boosts the performance of most optimization methods.

ii) Multi-task architecture In this branch, multi-task methods design different architectures to improve the exchanging or sharing of information among tasks \[[31](#bib.bib31 "")\]. Regarding where tasks interact, multi-task architectures are separated into encoder-focused and decoder-focused. The former shares the information in the encoder by the transformation of activations among tasks \[[24](#bib.bib24 "")\], learnable task-specific attention modules \[[20](#bib.bib20 "")\], branching networks for similar tasks \[[12](#bib.bib12 "")\] and so on. The latter recursively uses task predictions to improve overall performance \[[33](#bib.bib33 ""), [39](#bib.bib39 ""), [38](#bib.bib38 "")\]. However, these architectures still suffer from the task imbalance issue during multi-task optimization. In this paper, our work focuses on boosting multi-task optimization. As one of the multi-task optimization methods, our method can seamlessly generalize to different backbone models.

#### Training with subset of parameters

several methods already proposed in single-task learning. several methods select a subset of parameters based on a certain pre-defined rule, such as gradient \[[40](#bib.bib40 ""), [11](#bib.bib11 "")\] and magnitude of parameters \[[16](#bib.bib16 "")\]. In addition to selecting parameters by hand design, \[[27](#bib.bib27 ""), [25](#bib.bib25 ""), [34](#bib.bib34 "")\] automatically select the subset of parameters through optimization. Although sparse training has been extensively investigated in single-task learning, its application in multi-task learning remains relatively unexplored. \[[30](#bib.bib30 ""), [2](#bib.bib2 "")\] learning to share information between tasks using a sparse model. Differing from them, in this paper, we systematically research the gradient conflict via the sparse training perspective.

| Methods | Segmentation | Depth      | Surface Normal | 𝚫​𝐦%↓\\mathbf{\\Delta m\\%}\\downarrow |            |            |            |            |            |            |  |
| ------- | ------------ | ---------- | -------------- | -------------------------------------- | ---------- | ---------- | ---------- | ---------- | ---------- | ---------- |  |
|         |              |            |                |                                        |            |            |            |            |            |            |  |
|         |              |            |                |                                        |            |            |            |            |            |            |  |
| Random  | 59.8559.85   | 80.0980.09 | 0.33570.3357   | 0.13590.1359                           | 22.1722.17 | 16.1216.12 | 36.0836.08 | 64.5064.50 | 75.5675.56 | 6.0146.014 |  |
| Global  | 59.5359.53   | 79.3879.38 | 0.33800.3380   | 0.13730.1373                           | 22.3222.32 | 16.3216.32 | 35.6235.62 | 63.9363.93 | 75.1675.16 | 6.8556.855 |  |
| Reverse | 59.3559.35   | 79.5779.57 | 0.34170.3417   | 0.13960.1396                           | 22.3122.31 | 16.2516.25 | 35.9835.98 | 64.0764.07 | 75.1675.16 | 6.9606.960 |  |
| Ours    | 60.0360.03   | 79.9679.96 | 0.33200.3320   | 0.13530.1353                           | 21.9821.98 | 15.9215.92 | 36.6936.69 | 64.9264.92 | 75.8275.82 | 5.3145.314 |  |

Table 5: Different sparse training methods on SAM model with NYU-v2 datasets.

(a) 

(b) 

(c) 

Figure 6: The distribution of selected trainable parameters for different sparse training methods over different blocks. The experiments are conducted on SAM model with NYU-v2 dataset.

## Appendix F Detailed experiment results

In this section, we provide the detailed experiment results conducted in the main body of our paper, including the average incident of gradient conflict, the incident of gradient conflict for all epochs, and visualization of the gradient conflict for Joint Train and all gradient manipulation methods.

### F.1 Ablation study

The detailed results for various sparse methods are provided in [Tab. 5](#A5.T5 "In Training with subset of parameters ‣ Appendix E Extended related work ‣ Proactive Gradient Conflict Mitigation in Multi-Task Learning: A Sparse Training Perspective"), which is the full version of [Fig. 5(c)](#S4.F5.sf3 "In Figure 5 ‣ Effectiveness on both pre-trained and randomly initialized models ‣ 4.3 Performance on diverse benchmarks ‣ 4 Experiments ‣ Proactive Gradient Conflict Mitigation in Multi-Task Learning: A Sparse Training Perspective"). It can be observed that, with the exception of Pix Acc in segmentation, our sparse method outperforms other methods. In addition, we provide the distribution of the selected parameters using different sparse training over different blocks of the model. As shown in [Fig. 6](#A5.F6 "In Training with subset of parameters ‣ Appendix E Extended related work ‣ Proactive Gradient Conflict Mitigation in Multi-Task Learning: A Sparse Training Perspective"), the parameters selected by our sparse training method and Random are evenly distributed over the whole network. As for Global selecting the parameters with the highest magnitude, the distribution of selected parameters is largely different over different blocks

(a) 

(b) 

(c) 

(d) 

(e) 

(f) 

(g) 

Figure 7: The number of occurrence gradient conflictions between tasks during training SAM on NYUv2 dataset.

Figure 8: The average occurrence percentage of gradient conflict over epochs (all epochs/last 50% epochs) during training on MTAN model with NYU-v2 datasets was evaluated using various methods, including joint training and gradient manipulation techniques. 

### F.2 NYU-v2 on SAM

The incidence of gradient conflict for Joint Train and gradient manipulation method over all epochs are shown in [Fig. 7](#A6.F7 "In F.1 Ablation study ‣ Appendix F Detailed experiment results ‣ Proactive Gradient Conflict Mitigation in Multi-Task Learning: A Sparse Training Perspective"), which is the full version of [Fig. 4](#S4.F4 "In Sparse training effectively decreases the occurrence of gradient conflict ‣ 4.2 Incidence of gradient conflict ‣ 4 Experiments ‣ Proactive Gradient Conflict Mitigation in Multi-Task Learning: A Sparse Training Perspective") in the main body of the paper.

### F.3 NYU-v2 on MTAN

| Methods       |                 |
| ------------- | --------------- |
| All epochs    | Last 50% epochs |
| 36.01         | 39.87           |
| 33.86 (2.15)  | 36.45 (3.42)    |
| 35.71         | 39.51           |
| 34.05 (1.66)  | 37.25 (2.26)    |
| 37.21         | 40.93           |
| 34.14 (3.07)  | 37.04 (3.89)    |
| 36.37         | 39.71           |
| 34.42 (1.95)  | 37.10(2.61)     |
| 37.76         | 42.1            |
| 37.15 ( 0.61) | 41.25 (0.85)    |
| 37.14         | 41.22           |
| 35.81 (1.33)  | 39.17 (2.05)    |
| 37.19         | 40.79           |
| 35.83 (1.36)  | 39.0 (1.79)     |

Table 6: Average incidence of gradient conflict between tasks over epochs for different methods. The improvement by sparse training is provided in (∙\\bullet). We calculate the average incidence of gradient conflict over all epochs and the last 50% epochs during training MTAN on NYUv2.

We also conduct experiments on MTAN with NYU-v2 dataset. MTAN is a random initialized model. As we can see in [Tab. 6](#A6.T6 "In F.3 NYU-v2 on MTAN ‣ Appendix F Detailed experiment results ‣ Proactive Gradient Conflict Mitigation in Multi-Task Learning: A Sparse Training Perspective"), even for the random initialized model, sparse training can also reduce the incidence of gradient conflict. The visualization of the occurrence of gradient conflict for each epoch is shown in [Fig. 9](#A6.F9 "In F.3 NYU-v2 on MTAN ‣ Appendix F Detailed experiment results ‣ Proactive Gradient Conflict Mitigation in Multi-Task Learning: A Sparse Training Perspective") and the average incidence of gradient conflict across all epochs for different methods is shown in [Fig. 8](#A6.F8 "In F.1 Ablation study ‣ Appendix F Detailed experiment results ‣ Proactive Gradient Conflict Mitigation in Multi-Task Learning: A Sparse Training Perspective"). As for the performance of the overall tasks on NYU-v2, the sparse training improves not only the overall performance (Δ​m%\\Delta m\\%) but also the performance of each task for all methods including Joint Train and all gradient manipulation methods, as shown in [Tab. 7](#A6.T7 "In F.3 NYU-v2 on MTAN ‣ Appendix F Detailed experiment results ‣ Proactive Gradient Conflict Mitigation in Multi-Task Learning: A Sparse Training Perspective"). In addition, following \[[26](#bib.bib26 "")\], we conduct the experiments three times with three different seeds. The m​e​a​nmean ± s​t​dstd is presented in [Tab. 7](#A6.T7 "In F.3 NYU-v2 on MTAN ‣ Appendix F Detailed experiment results ‣ Proactive Gradient Conflict Mitigation in Multi-Task Learning: A Sparse Training Perspective"), we can observe that the sparse training is robust to the random seed.

(a) 

(b) 

(c) 

(d) 

(e) 

(f) 

(g) 

Figure 9: The number of occurrence gradient conflictions between tasks during tuning MTAN on NYUv2 dataset.

Methods

Segmentation

Depth

Surface Normal

𝚫​𝐦%↓\\mathbf{\\Delta m\\%}\\downarrow

mIoU ↑\\uparrow

Pix Acc ↑\\uparrow

Abs Err ↓\\downarrow

Rel Err ↓\\downarrow

Angle Distance ↓\\downarrow

Within t∘t^{\\circ} ↑\\uparrow

Mean

Median

11.25

22.5

30

STL

38.30

63.76

0.6754

0.2780

25.01

19.21

30.14

57.20

69.15

−-

Joint Train

39.29

65.33

0.5493

0.2263

28.15

23.96

22.09

47.50

61.08

5.59

w/ ST

41.04 (±0.28)

66.05 (±0.12)

0.5417 (±0.0008)

0.2232 (±0.0011)

27.40(±0.05)

22.90(±0.12)

23.58(±0.13)

49.59(±0.14)

63.01(±0.09)

2.49(±0.11)

PCGrad

38.06

64.64

0.5550

0.2325

27.41

22.80

23.86

49.83

63.14

3.97

w/ ST

40.49 (±0.32)

66.17(±0.23)

0.5441 (±0.0023)

0.2264 (±0.0030)

27.09 (±0.08)

22.55(±0.03)

24.22(±0.12)

50.34(±0.17)

63.63(±0.12)

1.98(±0.12)

CAGrad

39.79

65.49

0.5486

0.2250

26.31

21.58

25.61

52.36

65.58

0.20

w/ ST

39.93(±0.33)

66.19(±0.16)

0.5299(±0.0025)

0.2097(±0.0038)

25.71(±0.02)

20.70(±0.03)

26.86(±0.13)

54.22(±0.15)

67.30(±0.13)

-2.76(±0.10)

GradDrop

39.39

65.12

0.5455

0.2279

27.48

22.96

23.38

49.44

62.87

3.58

w/ ST

40.84(±0.35)

66.84(±0.24)

0.5288(±0.0021)

0.2209(±0.0021)

27.18(±0.03)

22.56(±0.07)

24.10(±0.11)

50.33(±0.14)

63.67(±0.13)

1.38(±0.12)

MGDA

30.47

59.90

0.6070

0.2555

24.88

19.45

29.18

56.88

69.36

1.38

w/ ST

32.42(±0.41)

61.61(±0.21)

0.5851(±0.0015)

0.2239 (±0.0032)

24.35(±0.02)

18.61(±0.03)

31.14(±0.12)

58.63(±0.15)

70.62(±0.13)

-3.09(±0.14)

IMTL-G

39.35

65.60

0.5426

0.2256

26.02

21.19

26.20

53.13

66.24

-0.76

w/ ST

40.73(±0.33)

66.00(±0.17)

0.5219(±0.0015)

0.2100(±0.0021)

25.6(±0.05)

20.64(±0.04)

26.81(±0.16)

54.38(±0.15)

67.49(±0.12)

-3.18(±0.11)

NashMTL

40.13

65.93

0.5261

0.2171

25.26

20.08

28.40

55.47

68.15

-4.04

w/ ST

39.75(±0.21)

66.45(±0.05)

0.5156(±0.0006)

0.2121(±0.0009)

24.96(±0.01)

19.80(±0.05)

28.80(±0.11)

56.20(±0.10)

68.93(±0.09)

-5.11(±0.07)

Table 7: The test performance on NYU-v2 dataset training on MTAN model, involving three tasks: semantic segmentation, depth estimation and surface normal. The result is the mean over three random seeds (std is presented in (± ∙\\bullet). The green cell color indicates that sparse training improves the performance of joint training or gradient manipulation methods. The best result is highlighted in bold.

### F.4 NYU-v2 on Swin

In order to investigate how the incidence of gradient conflict changes with varying model sizes, we conduct experiments on Swin/Tiny, Swin/Base and Swin/Large through the Joint Train. As depicted in [Tab. 8](#A6.T8 "In F.4 NYU-v2 on Swin ‣ Appendix F Detailed experiment results ‣ Proactive Gradient Conflict Mitigation in Multi-Task Learning: A Sparse Training Perspective"), there is an observed increase in the incidence of gradient conflict as the model size increases. Additionally, the performance of tasks improves as the model size increases [Tab. 9](#A6.T9 "In F.4 NYU-v2 on Swin ‣ Appendix F Detailed experiment results ‣ Proactive Gradient Conflict Mitigation in Multi-Task Learning: A Sparse Training Perspective").

| Model / Size | Average incidence of GC (%) |
| ------------ | --------------------------- |
| Swin / Tiny  | 37.42                       |
| Swin / Base  | 40.34                       |
| Swin / Large | 41.84                       |

Table 8: The average incidence of gradient conflict across all epochs during joint training with NYU-v2 on different sizes of Swin transformer.

| Model      |       | Segmentation | Depth  | Surface Normal |       |       |       |       |       |
| ---------- | ----- | ------------ | ------ | -------------- | ----- | ----- | ----- | ----- | ----- |
| Swin/Tiny  | 55.22 | 76.54        | 0.3746 | 0.1542         | 27.47 | 21.70 | 27.81 | 52.40 | 64.05 |
| Swin/Base  | 59.60 | 79.16        | 0.3419 | 0.1388         | 25.88 | 19.74 | 31.23 | 56.24 | 67.32 |
| Swin/Large | 61.34 | 80.28        | 0.3321 | 0.1345         | 25.09 | 18.73 | 33.05 | 58.12 | 68.86 |

Table 9: The test performance on NYU-v2 dataset jointly training on Swin models. 

### F.5 CelebA on Swin

Following \[[26](#bib.bib26 "")\], we train CelebA on Swin for only 30 epochs, because there are many more tasks in this dataset compared with other datasets, which leads to a significant increase in computation. As we can observe in [Tab. 10](#A6.T10 "In F.5 CelebA on Swin ‣ Appendix F Detailed experiment results ‣ Proactive Gradient Conflict Mitigation in Multi-Task Learning: A Sparse Training Perspective"), most of the methods including Joint Train and gradient manipulation methods can be improved by sparse training in terms of average incidence of gradient conflict between tasks over epochs. It is noted that the improvement by sparse training here is not significant, which is because of the limited training epoch. Specifically, as shown in [Tab. 1](#S4.T1 "In Sparse training effectively decreases the occurrence of gradient conflict ‣ 4.2 Incidence of gradient conflict ‣ 4 Experiments ‣ Proactive Gradient Conflict Mitigation in Multi-Task Learning: A Sparse Training Perspective") and [Tab. 6](#A6.T6 "In F.3 NYU-v2 on MTAN ‣ Appendix F Detailed experiment results ‣ Proactive Gradient Conflict Mitigation in Multi-Task Learning: A Sparse Training Perspective"), our sparse training improves more for later epochs. As for the performance of CelebA on Swin, please refer to [Tab. 3](#S4.T3 "In 4.3 Performance on diverse benchmarks ‣ 4 Experiments ‣ Proactive Gradient Conflict Mitigation in Multi-Task Learning: A Sparse Training Perspective"). The visualization for the occurrence of gradient conflict for each epoch and average incidence of gradient conflict over all epochs for different methods, including Joint Train and all gradient manipulation methods, are shown in [Fig. 10](#A6.F10 "In F.5 CelebA on Swin ‣ Appendix F Detailed experiment results ‣ Proactive Gradient Conflict Mitigation in Multi-Task Learning: A Sparse Training Perspective") and [Fig. 11](#A6.F11 "In F.5 CelebA on Swin ‣ Appendix F Detailed experiment results ‣ Proactive Gradient Conflict Mitigation in Multi-Task Learning: A Sparse Training Perspective")

| Methods       |                 |
| ------------- | --------------- |
| All epochs    | Last 50% epochs |
| 47.61         | 48.78           |
| 46.96 (0.65)  | 48.48 (0.30)    |
| 48.48         | 50.83           |
| 47.24 (1.24)  | 48.88 (1.95)    |
| 48.21         | 50.23           |
| 48.33 (-0.12) | 50.40(-0.17)    |
| 47.36         | 48.72           |
| 47.13 (0.23)  | 48.57 (0.15)    |
| 44.56         | 45.65           |
| 44.30 (0.26)  | 44.26(1.39)     |
| 46.89         | 47.77           |
| 45.03 (1.86)  | 46.32(1.45)     |
| 46.83         | 47.67           |
| 46.78(0.05)   | 47.34(0.33)     |

Table 10: Average incidence of gradient conflict between tasks over epochs for different methods. The improvement by sparse training is provided in (∙\\bullet). We calculate the average incidence of gradient conflict over all epochs and the last 50% epochs during training Swin on CelebA.

(a) 

(b) 

(c) 

(d) 

(e) 

(f) 

(g) 

Figure 10: The number of occurrence gradient conflictions between tasks during tuning Swin on CelebA dataset.

Figure 11: The average occurrence percentage of gradient conflict over epochs (all epochs/last 50% epochs) during training on Swin model with CelebA datasets was evaluated using various methods, including joint training and gradient manipulation techniques. 

### F.6 SmallNORB on ViT

SmallNORB is a much more difficult benchmark compared to other benchmarks in this paper. It comprises artificial objects observed under varying conditions and includes two tasks: object azimuth and camera-elevation prediction. As shown in [Tab. 11](#A6.T11 "In F.6 SmallNORB on ViT ‣ Appendix F Detailed experiment results ‣ Proactive Gradient Conflict Mitigation in Multi-Task Learning: A Sparse Training Perspective"), even for the STL, the Top 1 accuracy only achieves ∼\\sim30%, therefore, we use Top 5 as an extra metric here. We observed that even for this difficult task, sparse training can still achieve better performance compared with Joint Train and all gradient manipulation methods.

| Methods          |                  |                  |                  |                                     |  |
| ---------------- | ---------------- | ---------------- | ---------------- | ----------------------------------- |  |
| Top 1 ↑\\uparrow | Top 5 ↑\\uparrow | Top 1 ↑\\uparrow | Top 5 ↑\\uparrow |                                     |  |
| 32.9232.92       | 70.0670.06       | 36.5636.56       | 94.6794.67       | −-                                  |  |
| 28.0128.01       | 67.0567.05       | 29.8429.84       | 89.7589.75       | 10.7010.70                          |  |
| 27.3327.33       | 68.3568.35       | 30.7330.73       | 89.8789.87       | 10.1110.11                          |  |
| 28.7928.79       | 67.8567.85       | 30.1030.10       | 88.4488.44       | 9.999.99                            |  |
| 27.53927.539     | 67.9267.92       | 31.1831.18       | 90.1890.18       | 9.71\\pagecolor\[HTML\]{CCFFCC}9.71 |  |
| 28.7228.72       | 68.4268.42       | 29.3329.33       | 87.9387.93       | 10.5010.50                          |  |
| 28.5928.59       | 68.2168.21       | 29.8229.82       | 88.3788.37       | 10.2210.22                          |  |
| 27.5027.50       | 66.1366.13       | 29.8629.86       | 88.5088.50       | 11.7311.73                          |  |
| 28.3428.34       | 67.7967.79       | 29.5229.52       | 88.3888.38       | 10.7610.76                          |  |
| 30.8230.82       | 70.1370.13       | 27.2927.29       | 86.1686.16       | 10.1510.15                          |  |
| 28.2828.28       | 68.8868.88       | 30.0130.01       | 89.4789.47       | 9.799.79                            |  |
| 29.5729.57       | 69.9269.92       | 28.5128.51       | 86.7486.74       | 10.1910.19                          |  |
| 27.6527.65       | 69.0969.09       | 30.0130.01       | 89.6689.66       | 10.1510.15                          |  |
| 27.0227.02       | 66.8866.88       | 30.8330.83       | 89.7489.74       | 10.8410.84                          |  |
| 28.1728.17       | 67.9367.93       | 31.0131.01       | 89.3589.35       | 9.579.57                            |  |

Table 11: The test performance on SmallNORB dataset trained on ViT. The green cell color indicates that sparse training improves the performance of joint training or gradient manipulation methods. The best result is highlighted in bold. 

### F.7 CityScapes on MTAN

We also conduct experiments on MTAN with CityScapes dataset. MTAN is a random initialized model. As we can see in [Tab. 13](#A6.T13 "In F.7 CityScapes on MTAN ‣ Appendix F Detailed experiment results ‣ Proactive Gradient Conflict Mitigation in Multi-Task Learning: A Sparse Training Perspective"), even for the random initialized model, sparse training can also reduce the incidence of gradient conflict. The reduction in the incidence of gradient conflict for CityScapes is observed to be comparatively smaller than that for NYU-v2. This discrepancy can be attributed to the fact that CityScapes, which involves only two tasks, has a lower likelihood of encountering gradient conflicts between tasks compared to NYU-v2, which encompasses three tasks. The visualization of the occurrence of gradient conflict for each epoch is shown in [Fig. 12](#A6.F12 "In F.7 CityScapes on MTAN ‣ Appendix F Detailed experiment results ‣ Proactive Gradient Conflict Mitigation in Multi-Task Learning: A Sparse Training Perspective") and the average incidence of gradient conflict across all epochs for different methods is shown in [Fig. 13](#A6.F13 "In F.7 CityScapes on MTAN ‣ Appendix F Detailed experiment results ‣ Proactive Gradient Conflict Mitigation in Multi-Task Learning: A Sparse Training Perspective") . As for the performance of the overall tasks on CityScapes, the sparse training improves all methods including Joint Train and all gradient manipulation methods, as shown in [Tab. 12](#A6.T12 "In F.7 CityScapes on MTAN ‣ Appendix F Detailed experiment results ‣ Proactive Gradient Conflict Mitigation in Multi-Task Learning: A Sparse Training Perspective").

Methods

Segmentation

Depth

𝚫​𝐦%↓\\mathbf{\\Delta m\\%}\\downarrow

mIoU ↑\\uparrow

Pix Acc ↑\\uparrow

Abs Err ↓\\downarrow

Rel Err ↓\\downarrow

STL

77.61

94.15

0.0122

35.68

−-

Joint Train

78.14

94.29

0.0174

59.21

26.87

w/ ST

78.34

94.34

0.0143

55.00

17.48

PCGrad

77.79

94.21

0.0155

51.99

19.96

w/ ST

77.79

94.26

0.0160

51.99

19.22

CAGrad

76.82

93.70

0.0138

53.74

16.26

w/ ST

77.20

94.01

0.0150

39.85

8.88

GradDrop

77.91

94.28

0.0154

55.58

20.34

w/ ST

78.34

94.38

0.0163

48.95

17.45

MGDA

69.91

92.17

0.0124

40.68

6.91

w/ ST

68.38

91.91

0.0128

33.19

3.17

IMTL-G

77.55

94.10

0.0135

47.17

10.65

w/ ST

75.75

93.98

0.0138

40.16

7.10

NashMTL

77.51

94.22

0.0152

36.36

6.68

w/ ST

76.87

94.09

0.0148

33.30

3.99

Table 12: The test performance on CityScapes dataset training on MTAN model. The green cell color indicates that sparse training improves the performance of joint training or gradient manipulation methods. The best result is highlighted in bold.

| Methods      |                 |
| ------------ | --------------- |
| All epochs   | Last 50% epochs |
| 39.72        | 40.99           |
| 38.79 (0.93) | 40.02 (0.97)    |
| 39.98        | 41.06           |
| 38.66(1.32)  | 39.97(1.09)     |
| 39.39        | 40.94           |
| 37.77(1.62)  | 39.42(1.52)     |
| 39.32        | 40.72           |
| 39.03(0.29)  | 40.12(0.60)     |
| 36.37        | 39.69           |
| 36.14(0.23)  | 39.38(0.31)     |
| 37.72        | 39.51           |
| 36.83(0.89)  | 38.72(0.79)     |
| 38.40        | 40.69           |
| 38.04(0.36)  | 40.26(0.43)     |

Table 13: Average incidence of gradient conflict between tasks over epochs for different methods. The improvement by sparse training is provided in (∙\\bullet). We calculate the average incidence of gradient conflict over all epochs and the last 50% epochs during training MTAN on CityScapes.

(a) 

(b) 

(c) 

(d) 

(e) 

(f) 

(g) 

Figure 12: The number of occurrence gradient conflictions between tasks during tuning MTAN on CityScapes dataset.

Figure 13: The average occurrence percentage of gradient conflict over epochs (all epochs/last 50% epochs) during training on MTAN model with CityScapes datasets was evaluated using various methods, including joint training and gradient manipulation techniques. 

### F.8 FAMO

FAMO \[[18](#bib.bib18 "")\] is an approximation method for gradient manipulation by using the history of loss to compute the current task weight. We also try our sparse training with FAMO on NYU-v2, CelebA, Clevr, SmallORB datasets with ViT, SAM, MTAN and Swin models. As shown in [Tab. 14](#A6.T14 "In F.8 FAMO ‣ Appendix F Detailed experiment results ‣ Proactive Gradient Conflict Mitigation in Multi-Task Learning: A Sparse Training Perspective"), [Tab. 16](#A6.T16 "In F.8 FAMO ‣ Appendix F Detailed experiment results ‣ Proactive Gradient Conflict Mitigation in Multi-Task Learning: A Sparse Training Perspective"), [Tab. 17](#A6.T17 "In F.8 FAMO ‣ Appendix F Detailed experiment results ‣ Proactive Gradient Conflict Mitigation in Multi-Task Learning: A Sparse Training Perspective") and [Tab. 15](#A6.T15 "In F.8 FAMO ‣ Appendix F Detailed experiment results ‣ Proactive Gradient Conflict Mitigation in Multi-Task Learning: A Sparse Training Perspective"), even for the approximation method, sparse training method achieves the best results and further show the effectiveness of our sparse training methods.

Methods

Segmentation

Depth

Surface Normal

𝚫​𝐦%↓\\mathbf{\\Delta m\\%}\\downarrow

mIoU ↑\\uparrow

Pix Acc ↑\\uparrow

Abs Err ↓\\downarrow

Rel Err ↓\\downarrow

Angle Distance ↓\\downarrow

Within t∘t^{\\circ} ↑\\uparrow

Mean

Median

11.25

22.5

30

FAMO

57.6457.64

78.5978.59

0.35740.3574

0.14630.1463

19.39619.396

12.84612.846

45.6145.61

71.8771.87

80.5980.59

−0.5669-0.5669

w/ ST

57.6857.68

78.7978.79

0.35200.3520

0.14300.1430

19.27919.279

12.71112.711

46.1246.12

72.0672.06

80.7280.72

−1.353-1.353

Table 14: The test performance on NYU-v2 dataset training on SAM model. The green cell color indicates that sparse training improves the performance of joint training or gradient manipulation methods. The best result is highlighted in bold.

CelebA

Clevr

NYU-v2

Methods

𝚫​𝐦%↓\\mathbf{\\Delta m\\%}\\downarrow

Counting

Depth

𝚫​𝐦%↓\\mathbf{\\Delta m\\%}\\downarrow

𝚫​𝐦%↓\\mathbf{\\Delta m\\%}\\downarrow

(F1)

(Top 1 ↑\\uparrow)

(Top 1 ↑\\uparrow)

FAMO

2.352.35

55.8355.83

56.8056.80

3.163.16

-4.10

w/ ST

2.322.32

62.57

56.0456.04

-1.93

-4.46

Table 15: The test performance on CelebA, Clevr and NYU-v2 dataset. CelebA is trained on Swin Transformer and Clevr is trained on ViT. NYU-v2 is trained on MTAN. The green cell color indicates that sparse training improves the performance of joint training or gradient manipulation methods. The best result is highlighted in bold. 

Methods

Segmentation

Depth

Surface Normal

𝚫​𝐦%↓\\mathbf{\\Delta m\\%}\\downarrow

mIoU ↑\\uparrow

Pix Acc ↑\\uparrow

Abs Err ↓\\downarrow

Rel Err ↓\\downarrow

Angle Distance ↓\\downarrow

Within t∘t^{\\circ} ↑\\uparrow

Mean

Median

11.25

22.5

30

FAMO

38.88

64.90

0.5474

0.2194

25.06

19.57

29.21

56.61

68.98

-4.10

w/ ST

37.85

65.27

0.5543

0.2215

25.09

19.15

30.03

57.49

69.52

-4.46

Table 16: The test performance on NYU-v2 dataset training on MTAN model. The green cell color indicates that sparse training improves the performance of joint training or gradient manipulation methods. The best result is highlighted in bold.

| Methods |            | Object Azimuth |  | Camera Elevation | 𝚫​𝐦%↓\\mathbf{\\Delta m\\%}\\downarrow |            |
| ------- | ---------- | -------------- |  | ---------------- | -------------------------------------- | ---------- |
| FAMO    | 24.6824.68 | 63.6963.69     |  | 34.3534.35       | 92.1392.13                             | 10.7110.71 |
| w/ ST   | 26.3826.38 | 66.5466.54     |  | 32.0532.05       | 91.0291.02                             | 10.2710.27 |

Table 17: The test performance on SmallNORB dataset trained on ViT. The green cell color indicates that sparse training improves the performance of joint training or gradient manipulation methods. The best result is highlighted in bold. 

[](javascript:toggleReadingMode\(\); "Disable reading mode, show header and footer")