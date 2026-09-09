# Recon: Reducing Conflicting Gradients from the Root for Multi-Task Learning

 Guangyuan Shi Email: [guang-yuan.shi@connect.polyu.hk](mailto:guang-yuan.shi@connect.polyu.hk)    Qimai Li Email: [qee-mai.li@connect.polyu.hk](mailto:qee-mai.li@connect.polyu.hk)    Wenlong Zhang Email: [wenlong.zhang@connect.polyu.hk](mailto:wenlong.zhang@connect.polyu.hk)    Jiaxin Chen Email: [jiax.chen@connect.polyu.hk](mailto:jiax.chen@connect.polyu.hk)    Xiao-Ming WuDepartment of Computing, The Hong Kong Polytechnic University, Hong Kong S.A.R., China Email: [xiao-ming.wu@polyu.edu.hk](mailto:xiao-ming.wu@polyu.edu.hk) 

###### Abstract

A fundamental challenge for multi-task learning is that different tasks may conflict with each other when they are solved jointly, and a cause of this phenomenon is *conflicting gradients* during optimization. Recent works attempt to mitigate the influence of conflicting gradients by directly altering the gradients based on some criteria. However, our empirical study shows that “gradient surgery” cannot effectively reduce the occurrence of conflicting gradients. In this paper, we take a different approach to reduce conflicting gradients *from the root*. In essence, we investigate the task gradients w.r.t. each *shared* network layer, select the layers with high conflict scores, and turn them to *task-specific* layers. Our experiments show that such a simple approach can greatly reduce the occurrence of conflicting gradients in the remaining shared layers and achieve better performance, with only a slight increase in model parameters in many cases. Our approach can be easily applied to improve various state-of-the-art methods including gradient manipulation methods and branched architecture search methods. Given a network architecture (e.g., ResNet18), it only needs to search for the conflict layers once, and the network can be modified to be used with different methods on the same or even different datasets to gain performance improvement. The source code is available at [https://github.com/moukamisama/Recon](https://github.com/moukamisama/Recon "").

## 1 Introduction

Multi-task learning (MTL) is a learning paradigm in which multiple different but correlated tasks are jointly trained with a shared model ([Caruana, 1997](#bib.bib4 "")), in the hope of achieving better performance with an overall smaller model size than learning each task independently. By discovering shared structures across tasks and leveraging domain-specific training signals of related tasks, MTL can achieve efficiency and effectiveness. Indeed, MTL has been successfully applied in many domains including natural language processing ([Hashimoto et al., 2017](#bib.bib16 "")), reinforcement learning ([Parisotto et al., 2016](#bib.bib32 ""); [D’Eramo et al., 2020](#bib.bib10 "")) and computer vision ([Vandenhende et al., 2021](#bib.bib41 "")).

A major challenge for multi-task learning is *negative transfer* ([Ruder, 2017](#bib.bib34 "")), which refers to the performance drop on a task caused by the learning of other tasks, resulting in worse overall performance than learning them separately. This is caused by *task conflicts*, i.e., tasks compete with each other and unrelated information of individual tasks may impede the learning of common structures. From the optimization point of view, a cause of negative transfer is *conflicting gradients* ([Yu et al., 2020](#bib.bib43 "")), which refers to two task gradients pointing away from each other and the update of one task will have a negative effect on the other. Conflicting gradients make it difficult to optimize the multi-task objective, since task gradients with larger magnitude may dominate the update vector, making the optimizer prioritize some tasks over others and struggle to converge to a desirable solution.

Prior works address task/gradient conflicts mainly by balancing the tasks via task reweighting or gradient manipulation. Task reweighting methods adaptively re-weight the loss functions by homoscedastic uncertainty ([Kendall et al., 2018](#bib.bib19 "")), balancing the pace at which tasks are learned [Chen et al. (2018)](#bib.bib5 ""); [Liu et al. (2019)](#bib.bib24 ""), or learning a loss weight parameter ([Liu et al., 2021b](#bib.bib23 "")). Gradient manipulation methods reduce the influence of conflicting gradients by directly altering the gradients based on different criteria ([Sener & Koltun, 2018](#bib.bib37 ""); [Yu et al., 2020](#bib.bib43 ""); [Chen et al., 2020](#bib.bib6 ""); [Liu et al., 2021a](#bib.bib22 "")) or rotating the shared features ([Javaloy & Valera, 2022](#bib.bib18 "")). While these methods have demonstrated effectiveness in different scenarios, in our empirical study, we find that they cannot reduce the occurrence of conflicting gradients (see Sec. [3.3](#S3.SS3 "3.3 Gradient Surgery Cannot Effectively Reduce Conflicting Gradients ‣ 3 Pilot Study: Task Conflicts in Multi-Task Learning ‣ Recon: Reducing Conflicting Gradients from the Root for Multi-Task Learning") for more discussion).

We propose a different approach to reduce conflicting gradients for MTL. Specifically, we investigate layer-wise conflicting gradients, i.e., the task gradients w.r.t. each shared network layer. We first train the network with a regular MTL algorithm (e.g., joint-training) for a number of iterations, compute the conflict scores for all shared layers, and select those with highest conflict scores (indicating severe conflicts). We then set the selected shared layers task-specific and train the modified network from scratch. As demonstrated by comprehensive experiments and analysis, our simple approach Recon has the following key advantages: (1) Recon can greatly reduce conflicting gradients with only a slight increase in model parameters (less than 1% in some cases) and lead to significantly better performance. (2) Recon can be easily applied to improve various gradient manipulation methods and branched architecture search methods. Given a network architecture, it only needs to search for the conflict layers once, and the network can be modified to be used with different methods and even on different datasets to gain performance improvement. (3) Recon can achieve better performance than branched architecture search methods with a much smaller model.

## 2 Related Works

In this section, we briefly review related works in multi-task learning in four categories: tasks clustering, architecture design, architecture search, and task balancing. *Tasks clustering methods* mainly focus on identifying which tasks should be learned together ([Thrun & O’Sullivan, 1996](#bib.bib40 ""); [Zamir et al., 2018](#bib.bib44 ""); [Standley et al., 2020](#bib.bib39 ""); [Shen et al., 2021](#bib.bib38 ""); [Fifty et al., 2021](#bib.bib11 "")).

*Architecture design methods* include hard parameter sharing methods ([Kokkinos, 2017](#bib.bib20 ""); [Long et al., 2017](#bib.bib26 ""); [Bragman et al., 2019](#bib.bib2 "")), which learn a shared feature extractor and task-specific decoders, and soft parameters sharing methods ([Misra et al., 2016](#bib.bib30 ""); [Ruder et al., 2019](#bib.bib35 ""); [Gao et al., 2019](#bib.bib12 ""); [Gao et al., 2020](#bib.bib13 ""); [Liu et al., 2019](#bib.bib24 "")), where some parameters of each task are assigned to do cross-task talk via a sharing mechanism. Compared with soft parameters sharing methods, our approach Recon has much better scalability when dealing with a large number of tasks.

Instead of designing a fixed network structure, some methods ([Rosenbaum et al., 2018](#bib.bib33 ""); [Meyerson & Miikkulainen, 2018](#bib.bib29 ""); [Yang et al., 2020](#bib.bib42 "")) propose to dynamically self-organize the network for different tasks. Among them, *branched architecture search* ([Guo et al., 2020](#bib.bib15 ""); [Bruggemann et al., 2020](#bib.bib3 "")) methods are more related to our work. They propose an automated architecture search algorithm to build a tree-structured network by learning where to branch. In contrast, our method Recon decides which layers to be shared across tasks by considering the severity of layer-wise conflicting gradients, resulting in a more compact architecture with lower time cost and better performance.

Another line of research is *task balancing* methods. To address task/gradient conflicts, some methods attempt to re-weight the multi-task loss function using homoscedastic uncertainty ([Kendall et al., 2018](#bib.bib19 "")), task prioritization ([Guo et al., 2018](#bib.bib14 "")), or similar learning pace ([Liu et al., 2019](#bib.bib24 ""); [Liu et al., 2021b](#bib.bib23 "")). GradNorm ([Chen et al., 2018](#bib.bib5 "")) learns task weights by dynamically tuning gradient magnitudes. MGDA ([Sener & Koltun, 2018](#bib.bib37 "")) find the weights by minimizing the norm of the weighted sum of task gradients. To reduce the influence of conflicting gradients, PCGrad ([Yu et al., 2020](#bib.bib43 "")) projects each gradient onto the normal plane of another gradient and uses the average of projected gradients for update. Graddrop ([Chen et al., 2020](#bib.bib6 "")) randomly drops some elements of gradients based on element-wise conflict. CAGrad ([Liu et al., 2021a](#bib.bib22 "")) ensures convergence to a minimum of the average loss across tasks by gradient manipulation. RotoGrad ([Javaloy & Valera, 2022](#bib.bib18 "")) re-weights task gradients and rotates the shared feature space. Instead of manipulating gradients, our method Recon leverages gradient information to modify network structure to mitigate task conflicts from the root.

## 3 Pilot Study: Task Conflicts in Multi-Task Learning

### 3.1 Multi-task Learning: Problem Definition

Multi-task learning (MTL) aims to learn a set of correlated tasks {𝒯i}i\=1T\\{\\mathcal{T}\_{i}\\}\_{i=1}^{T} simultaneously. For each task 𝒯i\\mathcal{T}\_{i}, the empirical loss function is ℒi​(θsh,θi)\\mathcal{L}\_{i}(\\theta\_{\\mathrm{sh}},\\theta\_{i}), where θsh\\theta\_{\\mathrm{sh}} are parameters shared among all tasks and θi\\theta\_{i} are task-specific parameters. The goal is to find optimal parameters θ\={θsh,θ1,θ2,⋯,θT}\\theta=\\{\\theta\_{\\mathrm{sh}},\\theta\_{1},\\theta\_{2},\\cdots,\\theta\_{T}\\} to achieve high performance across all tasks. Formally, it aims to minimize a multi-task objective:

θ∗\=arg⁡min⁡∑iTθ⁡wi​ℒi​(θsh,θi),\\theta^{\*}=\\arg\\min\_{\\theta}\\sum\_{i}^{T}w\_{i}\\mathcal{L}\_{i}(\\theta\_{\\mathrm{sh}},\\theta\_{i}),

(1)

where wiw\_{i} are pre-defined or dynamically computed weights for different tasks. A popular choice is to use the average loss (i.e., equal weights). However, optimizing the multi-task objective is difficult, and a known cause is conflicting gradients.

Figure 1: The distributions of gradient conflicts (in terms of cos⁡ϕi​j\\cos\\phi\_{ij}) of the joint-training baseline and state-of-the-art gradient manipulation methods on Multi-Fashion+MNIST benchmark. 

### 3.2 Conflicting Gradients

Let 𝐠i\=∇θshℒi​(θsh,θi)\\mathbf{g}\_{i}=\\nabla\_{\\theta\_{\\mathrm{sh}}}\\mathcal{L}\_{i}(\\theta\_{\\mathrm{sh}},\\theta\_{i}) denote the gradient of task 𝒯i\\mathcal{T}\_{i} w.r.t. the shared parameters θsh\\theta\_{\\mathrm{sh}} (i.e., a vector of the partial derivatives of ℒi\\mathcal{L}\_{i} w.r.t. θsh\\theta\_{\\mathrm{sh}}) and gits\=∇θiℒi​(θsh,θi)g\_{i}^{\\mathrm{ts}}=\\nabla\_{\\theta\_{i}}\\mathcal{L}\_{i}(\\theta\_{\\mathrm{sh}},\\theta\_{i}) denote the gradient w.r.t. the task-specific parameters θi\\theta\_{i}. A small change of θsh\\theta\_{\\mathrm{sh}} in the direction of negative 𝐠i\\mathbf{g}\_{i} is θsh←θsh−α​𝐠i\\theta\_{\\mathrm{sh}}\\leftarrow\\theta\_{\\mathrm{sh}}-\\alpha\\mathbf{g}\_{i}, with a sufficiently small step size α\\alpha. The effect of this change on the performance of another task 𝒯j\\mathcal{T}\_{j} is measured by:

Δℒj\=ℒj(θsh−α𝐠i,θj)−ℒj(θsh,θj)\=−α𝐠i⋅𝐠j+o(α),\\Delta\\mathcal{L}\_{j}=\\mathcal{L}\_{j}(\\theta\_{\\mathrm{sh}}-\\alpha\\mathbf{g}\_{i},\\theta\_{j})-\\mathcal{L}\_{j}(\\theta\_{\\mathrm{sh}},\\theta\_{j})=-\\alpha\\mathbf{g}\_{i}\\cdot\\mathbf{g}\_{j}+o(\\alpha),

(2)

where the second equality is obtained by first order Taylor approximation. Likewise, the effect of a small update of θsh\\theta\_{\\mathrm{sh}} in the direction of the negative gradient of task 𝒯j\\mathcal{T}\_{j} (i.e., −𝐠j-\\mathbf{g}\_{j}) on the performance of task 𝒯i\\mathcal{T}\_{i} is Δℒi\=−α𝐠i⋅𝐠j+o(α)\\Delta\\mathcal{L}\_{i}=-\\alpha\\mathbf{g}\_{i}\\cdot\\mathbf{g}\_{j}+o(\\alpha). Notably, the model update for task 𝒯i\\mathcal{T}\_{i} is considered to have a negative effect on task 𝒯j\\mathcal{T}\_{j} when 𝐠i⋅𝐠j<0\\mathbf{g}\_{i}\\cdot\\mathbf{g}\_{j}<0, since it increases the loss of task 𝒯j\\mathcal{T}\_{j}, and vice versa. A formal definition of conflicting gradients is given as follows ([Yu et al., 2020](#bib.bib43 "")).

###### Definition 1 (Conflicting Gradients).

The gradients 𝐠i\\mathbf{g}\_{i} and 𝐠j​(i≠j)\\mathbf{g}\_{j}(i\\neq j) are said to be conflicting with each other if cos⁡ϕi​j<0\\cos{\\phi\_{ij}}<0, where ϕi​j\\phi\_{ij} is the angle between 𝐠i\\mathbf{g}\_{i} and 𝐠j\\mathbf{g}\_{j}.

As shown in [Yu et al. (2020)](#bib.bib43 ""), conflicts in gradient pose serious challenges for optimizing the multi-task objective (Eq. [1](#S3.E1 "In 3.1 Multi-task Learning: Problem Definition ‣ 3 Pilot Study: Task Conflicts in Multi-Task Learning ‣ Recon: Reducing Conflicting Gradients from the Root for Multi-Task Learning")). Using the average gradient (i.e., 1T​∑i\=1T𝐠i\\frac{1}{T}\\sum\_{i=1}^{T}\\mathbf{g}\_{i}) for gradient decent may hurt the performance of individual tasks, especially when there is a large difference in gradient magnitudes, which will make the optimizer struggle to converge to a desirable solution.

### 3.3 Gradient Surgery Cannot Effectively Reduce Conflicting Gradients

To mitigate the influence of conflicting gradients, several methods ([Yu et al., 2020](#bib.bib43 ""); [Chen et al., 2020](#bib.bib6 ""); [Liu et al., 2021a](#bib.bib22 "")) have been proposed to perform “gradient surgery”. Instead of following the average gradient direction, they alter conflicting gradients based on some criteria and use the modified gradients for model update. We conduct a pilot study to investigate whether gradient manipulation can effectively reduce the occurrence of conflicting gradients. For each training iteration, we first calculate the task gradients of all tasks w.r.t. the shared parameters (i.e., 𝐠i\\mathbf{g}\_{i} for any task ii) and compute the conflict angle between any two task gradients 𝐠i\\mathbf{g}\_{i} and 𝐠j\\mathbf{g}\_{j} in terms of c​o​s​ϕi​jcos{\\phi\_{ij}}. We then count and draw the distribution of c​o​s​ϕi​jcos{\\phi\_{ij}} in all training iterations. We provide the statistics of the joint-training baseline (i.e., training all tasks jointly with equal loss weights and all parameters shared) and several state-of-the-art gradient manipulation methods including GradDrop ([Chen et al., 2020](#bib.bib6 "")), PCGrad ([Yu et al., 2020](#bib.bib43 "")), CAGrad ([Liu et al., 2021a](#bib.bib22 "")), and MGDA ([Sener & Koltun, 2018](#bib.bib37 "")) on Multi-Fashion+MNIST ([Lin et al., 2019](#bib.bib21 "")), CityScapes, NYUv2, and PASCAL-Context datasets. The results are provided in Fig. [1](#S3.F1 "Figure 1 ‣ 3.1 Multi-task Learning: Problem Definition ‣ 3 Pilot Study: Task Conflicts in Multi-Task Learning ‣ Recon: Reducing Conflicting Gradients from the Root for Multi-Task Learning"), Fig. [5](#A3.F5 "Figure 5 ‣ Appendix C Additional Ablation Study ‣ Recon: Reducing Conflicting Gradients from the Root for Multi-Task Learning"), Fig. [6](#A3.F6 "Figure 6 ‣ Appendix C Additional Ablation Study ‣ Recon: Reducing Conflicting Gradients from the Root for Multi-Task Learning"), Fig. [7](#A3.F7 "Figure 7 ‣ Appendix C Additional Ablation Study ‣ Recon: Reducing Conflicting Gradients from the Root for Multi-Task Learning"), Table [6](#S5.T6 "Table 6 ‣ 5.2 Comparison with the State-of-the-Art ‣ 5 Experiments ‣ Recon: Reducing Conflicting Gradients from the Root for Multi-Task Learning"), and Tables [8](#A3.T8 "Table 8 ‣ Appendix C Additional Ablation Study ‣ Recon: Reducing Conflicting Gradients from the Root for Multi-Task Learning")-[10](#A3.T10 "Table 10 ‣ Appendix C Additional Ablation Study ‣ Recon: Reducing Conflicting Gradients from the Root for Multi-Task Learning"). It can be seen that gradient manipulation methods can only slightly reduce the occurrence of conflicting gradients (compared to joint-training) in some cases, and in some other cases they even increase it.

## 4 Our Approach: Reducing Conflicting Gradients from the Root

![Refer to caption](2302.11289v1/GD.png)

(a) Joint-train

![Refer to caption](2302.11289v1/PCGrad.png)

(b) PCGrad

![Refer to caption](2302.11289v1/Recon.png)

(c) Recon

(d) Recon

Figure 2: Illustration of the differences between joint-training, gradient manipulation, and our approach. (a) In joint-training, the update vector (in green) is the average gradient 12​(𝐠i+𝐠j)\\frac{1}{2}(\\mathbf{g}\_{i}+\\mathbf{g}\_{j}). Due to the conflict between 𝐠i\\mathbf{g}\_{i} and 𝐠j\\mathbf{g}\_{j}, the update vector is dominated by 𝐠i\\mathbf{g}\_{i} (in red). (b) PCGrad ([Yu et al., 2020](#bib.bib43 "")) projects each gradient onto the normal plane of the other one and uses the average of the projected gradients (indicated by dashed grey arrows) as the update vector (in green). As such, the update vector is less dominated by 𝐠i\\mathbf{g}\_{i}. (c) Our approach Recon finds the parameters contributing most (e.g., θ3\\theta\_{3}) to gradient conflicts and turns them into task specific ones. In effect, it performs an orthographic/coordinate projection of conflicting gradients to the space of the rest parameters (e.g., θ1\\theta\_{1} and θ2\\theta\_{2}) such that the projected gradients 𝐠ifix\\mathbf{g}\_{i}^{\\mathrm{fix}} and 𝐠jfix\\mathbf{g}\_{j}^{\\mathrm{fix}} are better aligned. (d) Illustration of Recon turning a shared layer with high conflict score to task-specific layers.

Our pilot study shows that adjusting gradients for model update cannot effectively prevent the occurrence of conflicting gradients in MTL, which suggests that the root causes of this phenomenon may be closely related to the nature of different tasks and the way how model parameters are shared among them. Therefore, to mitigate task conflicts for MTL, in this paper, we take a different approach to reduce the occurrence of conflicting gradients from the root.

### 4.1 Recon: Removing Layer-wise Conflicting Gradients

Our approach is extremely simple and intuitive. We first identify the shared network layers where conflicts occur most frequently and then turn them into task-specific parameters. Suppose the shared model parameters θsh\\theta\_{\\mathrm{sh}} are composed of nn layers, i.e., θsh\={θsh(k)}k\=1n\\theta\_{\\mathrm{sh}}=\\{\\theta^{(k)}\_{\\mathrm{sh}}\\}\_{k=1}^{n}, where θsh(k)\\theta^{(k)}\_{\\mathrm{sh}} is the kthk^{\\mathrm{th}} shared layer. Let 𝐠i(k)\\mathbf{g}\_{i}^{(k)} denote the gradient of task 𝒯i\\mathcal{T}\_{i} w.r.t. the kthk^{\\mathrm{th}} shared layer θsh(k)\\theta^{(k)}\_{\\mathrm{sh}}, i.e., 𝐠i(k)\\mathbf{g}\_{i}^{(k)} is a vector of the partial derivatives of ℒi\\mathcal{L}\_{i} w.r.t. the parameters of θsh(k)\\theta^{(k)}\_{\\mathrm{sh}}. Let ϕi​j(k)\\phi\_{ij}^{(k)} denote the angle between 𝐠i(k)\\mathbf{g}\_{i}^{(k)} and 𝐠j(k)\\mathbf{g}\_{j}^{(k)}. We define layer-wise conflicting gradients and SS-conflict score as follows.

###### Definition 2 (Layer-wise Conflicting Gradients).

The gradients 𝐠i(k)\\mathbf{g}\_{i}^{(k)} and 𝐠j(k)\\mathbf{g}\_{j}^{(k)} (i≠ji\\neq j) are said to be conflicting with each other if cos⁡ϕi​j(k)<0\\cos{\\phi\_{ij}^{(k)}}<0.

###### Definition 3 (SS-Conflict Score).

For any −1<S≤0-1\<S\\leq 0, the SS-conflict score for the kthk^{\\mathrm{th}} shared layer is the number of different pairs (i,j)​(i≠j𝐶𝐿𝑂𝑆𝐸(i,j)(i\\neq j) s.t. cos⁡ϕi​j(k)<S\\cos{\\phi\_{ij}^{(k)}}\<S, denoted as s(k)s^{(k)}.

SS indicates the severity of conflicts, and setting SS smaller means we care about cases of more severe conflicts. The SS-conflict score s(k)s^{(k)} indicates the occurrence of conflicting gradients at severity level SS for the kthk^{\\mathrm{th}} shared layer. If s(k)\=(T2)s^{(k)}=\\binom{T}{2}, it means that for any two different tasks, there is a conflict in their gradients w.r.t. the kthk^{\\mathrm{th}} shared layer. By computing SS-conflict scores, we can identify the shared layers where conflicts occur most frequently.

We describe our method Recon in Algorithm [1](#algorithm1 "In 4.1 Recon: Removing Layer-wise Conflicting Gradients ‣ 4 Our Approach: Reducing Conflicting Gradients from the Root ‣ Recon: Reducing Conflicting Gradients from the Root for Multi-Task Learning"). First, we train the network for II iterations and compute SS-conflict scores for each shared layer θ(k)\\theta^{(k)} in every iteration, denoted by {si(k)}i\=1I\\{s^{(k)}\_{i}\\}\_{i=1}^{I}. Then, we sum up the scores in all iterations, i.e., s(k)\=∑i\=1Isi(k)s^{(k)}=\\sum\_{i=1}^{I}s^{(k)}\_{i}, and find the layers with highest s(k)s^{(k)} scores. Next, we set these layers to be task-specific and train the modified network from scratch. We demonstrate the effectiveness of Recon by a theoretical analysis in Sec. [4.2](#S4.SS2 "4.2 Theoretical Analysis ‣ 4 Our Approach: Reducing Conflicting Gradients from the Root ‣ Recon: Reducing Conflicting Gradients from the Root for Multi-Task Learning") and comprehensive experiments in Sec. [5](#S5 "5 Experiments ‣ Recon: Reducing Conflicting Gradients from the Root for Multi-Task Learning"). The results show that Recon can effectively reduce the occurrence of conflicting gradients in the remaining shared layers and lead to substantial improvements over state-of-the-art.

Algorithm 1 Recon: Removing Layer-wise Conflicting Gradients

Input: Model parameters θ\\theta, learning rate α\\alpha, a set of tasks {𝒯i}i\=1T\\{\\mathcal{T}\_{i}\\}\_{i=1}^{T}, number of iterations II for computing conflict scores, conflict severity level SS, number of selected layers KK. 

// Train the network and compute conflict scores for all layers 

for *iteration i = 1, 2, …, II* do 

   for *i = 1 , 2, …, TT* do 

      Compute the gradients of task 𝒯i\\mathcal{T}\_{i} w.r.t. all shared layers, i.e., {𝐠i(k)}k\=1n\\{\\mathbf{g}\_{i}^{(k)}\\}\_{k=1}^{n} ; 

   end for 

   Calculate the SS-conflict scores for all shared layers in the current iteration, i.e., {si(k)}k\=1n\\{s^{(k)}\_{i}\\}\_{k=1}^{n}; 

   Update θ\\theta with joint-training or any gradient manipulation method ; 

end for 

// Set layers with top conflict scores task-specific 

For each layer kk, calculate the sum of SS-conflict scores in all iterations, i.e., s(k)\=∑i\=1Isi(k)s^{(k)}=\\sum\_{i=1}^{I}s^{(k)}\_{i}; 

Select the top KK layers with highest s(k)s^{(k)} and set them task-specific; 

// Train the modified network from scratch 

for *iteration i = 1, 2, …* do 

   Update θ\\theta with joint-training or any gradient manipulation method; 

end for 

Output: Model parameters θ\\theta. 

### 4.2 Theoretical Analysis

Here, we provide a theoretical analysis of Recon. Let θsh\={θshfix,θshcf}\\theta\_{\\mathrm{sh}}=\\{\\theta\_{\\mathrm{sh}}^{\\mathrm{fix}},\\theta\_{\\mathrm{sh}}^{\\mathrm{cf}}\\}, where θshfix\\theta\_{\\mathrm{sh}}^{\\mathrm{fix}} are the remaining shared parameters, and θshcf\\theta\_{\\mathrm{sh}}^{\\mathrm{cf}} are those that will be turned to task-specific parameters θ1cf,θ2cf,⋯,θTcf\\theta\_{1}^{\\mathrm{cf}},\\theta\_{2}^{\\mathrm{cf}},\\cdots,\\theta\_{T}^{\\mathrm{cf}}. Notice that θ1cf,θ2cf,⋯,θTcf\\theta\_{1}^{\\text{cf}},\\theta\_{2}^{\\text{cf}},\\cdots,\\theta\_{T}^{\\text{cf}} will all be initialized with θshcf\\theta\_{\\mathrm{sh}}^{\\mathrm{cf}}. Therefore, after applying Recon, the model parameters are θr\={θshfix,θ1cf,…,θTcf,θ1ts,…,θTts}.\\theta\_{r}=\\{\\theta\_{\\mathrm{sh}}^{\\mathrm{fix}},\\theta\_{1}^{\\text{cf}},\\dots,\\theta\_{T}^{\\text{cf}},\\theta\_{1}^{\\mathrm{ts}},\\dots,\\theta\_{T}^{\\mathrm{ts}}\\}. An one-step gradient update of θr\\theta\_{r} is:

θ^shfix\=θshfix−α∑i\=1Twi𝐠ifix,θ^icf\=θicf−α𝐠icf,θ^its\=θits−α𝐠its,i\=1,…,T,\\hat{\\theta}\_{\\mathrm{sh}}^{\\mathrm{fix}}=\\theta\_{\\mathrm{sh}}^{\\mathrm{fix}}-\\alpha\\sum\_{i=1}^{T}w\_{i}\\mathbf{g}\_{i}^{\\mathrm{fix}},\\quad\\hat{\\theta}\_{i}^{\\mathrm{cf}}=\\theta\_{i}^{\\mathrm{cf}}-\\alpha\\mathbf{g}\_{i}^{\\mathrm{cf}},\\quad\\hat{\\theta}\_{i}^{\\mathrm{ts}}=\\theta\_{i}^{\\mathrm{ts}}-\\alpha\\mathbf{g}\_{i}^{\\mathrm{ts}},\\quad i=1,\\dots,T,

(3)

where wiw\_{i} are weight parameters, 𝐠its\=∇θitsℒi\\mathbf{g}\_{i}^{\\mathrm{ts}}=\\nabla\_{\\theta\_{i}^{\\mathrm{ts}}}\\mathcal{L}\_{i}, 𝐠icf\=∇θshcfℒi\\mathbf{g}\_{i}^{\\mathrm{cf}}=\\nabla\_{\\theta\_{\\mathrm{sh}}^{\\mathrm{cf}}}\\mathcal{L}\_{i} and 𝐠ifix\=∇θshfixℒi\\mathbf{g}\_{i}^{\\mathrm{fix}}=\\nabla\_{\\theta\_{\\mathrm{sh}}^{\\mathrm{fix}}}\\mathcal{L}\_{i}. Notice that different methods such as joint-training, MGDA [Sener & Koltun (2018)](#bib.bib37 ""), PCGrad [Yu et al. (2020)](#bib.bib43 ""), and CAGrad [Liu et al. (2021a)](#bib.bib22 "") choose different wiw\_{i} dynamically.

Without applying Recon, the model parameters are θ\={θshfix,θshcf,θ1ts,…,θTts}.\\theta=\\{\\theta\_{\\mathrm{sh}}^{\\mathrm{fix}},\\theta\_{\\mathrm{sh}}^{\\mathrm{cf}},\\theta\_{1}^{\\mathrm{ts}},\\dots,\\theta\_{T}^{\\mathrm{ts}}\\}. An one-step gradient update of θ\\theta is given by

θ^shfix\=θshfix−α∑i\=1Twi𝐠ifix,θ^shcf\=θshcf−α∑i\=1Twi𝐠icf,θ^its\=θits−α𝐠its,i\=1,…,T.\\hat{\\theta}\_{\\mathrm{sh}}^{\\mathrm{fix}}=\\theta\_{\\mathrm{sh}}^{\\mathrm{fix}}-\\alpha\\sum\_{i=1}^{T}w\_{i}\\mathbf{g}\_{i}^{\\mathrm{fix}},\\quad\\hat{\\theta}\_{\\mathrm{sh}}^{\\mathrm{cf}}=\\theta\_{\\mathrm{sh}}^{\\mathrm{cf}}-\\alpha\\sum\_{i=1}^{T}w\_{i}\\mathbf{g}\_{i}^{\\mathrm{cf}},\\quad\\hat{\\theta}\_{i}^{\\mathrm{ts}}=\\theta\_{i}^{\\mathrm{ts}}-\\alpha\\mathbf{g}\_{i}^{\\mathrm{ts}},\\quad i=1,\\dots,T.

(4)

After the one-step updates, the loss functions with the updated parameters θ^r\\hat{\\theta}\_{r} and θ^\\hat{\\theta} respectively are:

ℒ⁡(θ^r)\=∑i\=1Tℒi​(θ^shfix,θ^icf,θ^its),and,ℒ⁡(θ^)\=∑i\=1Tℒi​(θ^shfix,θ^shcf,θ^its),\\mathcal{L}(\\hat{\\theta}\_{r})=\\sum\_{i=1}^{T}\\mathcal{L}\_{i}\\left(\\hat{\\theta}\_{\\mathrm{sh}}^{\\mathrm{fix}},\\hat{\\theta}\_{i}^{\\mathrm{cf}},\\hat{\\theta}\_{i}^{\\mathrm{ts}}\\right),\\>\\mathrm{and},\\>\\mathcal{L}(\\hat{\\theta})=\\sum\_{i=1}^{T}\\mathcal{L}\_{i}\\left(\\hat{\\theta}\_{\\mathrm{sh}}^{\\mathrm{fix}},\\hat{\\theta}\_{\\mathrm{sh}}^{\\mathrm{cf}},\\hat{\\theta}\_{i}^{\\mathrm{ts}}\\right),

(5)

where ℒi\\mathcal{L}\_{i} is the loss function of task 𝒯i\\mathcal{T}\_{i}. Denote the set of indices of the layers turned task-specific by ℙ\\mathbb{P}, then θshcf\={θsh(k)},k∈ℙ\\theta\_{\\mathrm{sh}}^{\\mathrm{cf}}=\\{\\theta\_{\\mathrm{sh}}^{(k)}\\},k\\in\\mathbb{P}. Assume that ∑i\=1Twi\=1\\sum\_{i=1}^{T}w\_{i}=1, then we have the following theorem.

###### Theorem 4.1.

Assume that ℒ\\mathcal{L} is differentiable and for any two different tasks 𝒯i\\mathcal{T}\_{i} and 𝒯j\\mathcal{T}\_{j}, it satisfies

cos⁡ϕi​j(k)​‖𝐠i(k)‖<‖𝐠j(k)‖,∀k∈ℙ\\cos{\\phi\_{ij}^{(k)}}\\|\\mathbf{g}\_{i}^{(k)}\\|<\\|\\mathbf{g}\_{j}^{(k)}\\|,\\quad\\forall k\\in\\mathbb{P}

(6)

then for any sufficiently small learning rate α\>0\\alpha>0, ℒ⁡(θ^r)<ℒ⁡(θ^).\\mathcal{L}(\\hat{\\theta}\_{r})<\\mathcal{L}(\\hat{\\theta}).

(7)

The theorem indicates that a single gradient update on the model parameters of Recon achieves lower loss than that on the original model parameters. The proof is provided in Appendix [A](#A1 "Appendix A Proof of Theorem ‣ Recon: Reducing Conflicting Gradients from the Root for Multi-Task Learning")

## 5 Experiments

In this section, we conduct extensive experiments to evaluate our approach Recon for multi-task learning and demonstrate its effectiveness, efficiency and generality.

Table 1: Multi-task learning results on Multi-Fashion+MNIST dataset. All experiments are repeated over 𝟑\\mathbf{3} random seeds and the mean values are reported. Δ​m%\\Delta m\\% denotes the average relative improvement of all tasks. #P denotes model size (MB). The grey cell color indicates that Recon improves the result of the base model. The best average result is marked in bold.

| Method | Single-task | RotoGrad | BMTAS | Joint-train | w/ Recon | MGDA  | w/ Recon | PCGrad | w/ Recon | GradDrop | w/ Recon | CAGrad | w/ Recon | MMoE   | w/ Recon |
| ------ | ----------- | -------- | ----- | ----------- | -------- | ----- | -------- | ------ | -------- | -------- | -------- | ------ | -------- | ------ | -------- |
| 98.37  | 98.10       | 98.20    | 97.42 | 98.13       | 95.19    | 98.33 | 97.37    | 98.30  | 97.38    | 98.25    | 97.47    | 98.28  | 98.27    | 98.25  |          |
| 89.63  | 88.25       | 89.71    | 88.82 | 89.26       | 89.46    | 89.28 | 88.68    | 89.77  | 88.57    | 89.51    | 88.85    | 89.65  | 89.51    | 89.67  |          |
| -      | -0.91       | -0.04    | -0.94 | -0.33       | -1.71    | -0.22 | -1.04    | 0.04   | -1.10    | -0.13    | -0.90    | -0.04  | -0.12    | -0.04  |          |
| 85.62  | 42.81       | 85.61    | 42.81 | 43.43       | 42.81    | 43.43 | 42.81    | 43.43  | 42.81    | 43.43    | 42.81    | 43.43  | 85.62    | 105.70 |          |

Table 2: Multi-task learning results on CelebA dataset. All experiments are repeated over 𝟑\\mathbf{3} random seeds and the mean values are reported. Δ​m%\\Delta m\\% denotes the average relative improvement of all tasks. #P denotes model size (MB). The grey cell color indicates that Recon improves the result of the base model. The best average result is marked in bold.

| Method | Single-task | Joint-train | w/ Recon | CAGrad | w/ Recon | Graddrop | w/ Recon | PCGrad | w/ Recon |
| ------ | ----------- | ----------- | -------- | ------ | -------- | -------- | -------- | ------ | -------- |
| 8.33   | 8.22        | 8.31        | 8.23     | 8.33   | 8.20     | 8.64     | 8.36     |        |          |
| 0.55   | 1.92        | 0.79        | 1.74     | 0.23   | 2.13     | -3.14    | 0.24     |        |          |
| 43.26  | 68.03       | 43.26       | 68.03    | 43.26  | 68.03    | 43.26    | 68.03    |        |          |

### 5.1 Experimental Setup

Datasets. We evaluate Recon on 4 multi-task datasets, namely Multi-Fashion+MNIST ([Lin et al., 2019](#bib.bib21 "")), CityScapes ([Cordts et al., 2016](#bib.bib7 "")), NYUv2 ([Couprie et al., 2013](#bib.bib8 "")), PASCAL-Context ([Mottaghi et al., 2014](#bib.bib31 "")), and CelebA ([Liu et al., 2015](#bib.bib25 "")). The tasks of each dataset are described as follows. 1) Multi-Fashion+MNIST contains two image classification tasks. Each image consists of an item from FashionMNIST and an item from MNIST. 2) CityScapes contains 2 vision tasks: 7-class semantic segmentation and depth estimation. 3) NYUv2 contains 3 tasks: 13-class semantic segmentation, depth estimation and normal prediction. 4) PASCAL-Context consists of 5 tasks: semantic segmentation, human parts segmentation and saliency estimation, surface normal estimation, and edge detection. 5) CelebA contains 40 binary classification tasks.

Baselines. The baselines include 1) single-task learning (single-task): training all tasks independently; 2) joint-training (joint-train): training all tasks together with equal loss weights and all parameters shared; 3) gradient manipulation methods: MGDA ([Sener & Koltun, 2018](#bib.bib37 "")), PCGrad ([Yu et al., 2020](#bib.bib43 "")), GradDrop ([Chen et al., 2020](#bib.bib6 "")), CAGrad ([Liu et al., 2021a](#bib.bib22 "")), RotoGrad ([Javaloy & Valera, 2022](#bib.bib18 "")); 4) branched architecture search methods: BMTAS ([Bruggemann et al., 2020](#bib.bib3 "")); 5) Architecture design methods: Cross-Stitch ([Misra et al., 2016](#bib.bib30 "")), MMoE ([Ma et al., 2018](#bib.bib27 "")). Following [Liu et al. (2021a)](#bib.bib22 ""), we implement Cross-Stitch based on SegNet ([Badrinarayanan et al., 2017](#bib.bib1 "")). For a fair comparison, all methods use same configurations and random seeds. We run all experiments 3 times with different random seeds. More experimental details are provided in Appendix [B](#A2 "Appendix B Experimental setup ‣ Recon: Reducing Conflicting Gradients from the Root for Multi-Task Learning").

Relative task improvement. Following [Maninis et al. (2019)](#bib.bib28 ""), we compute the relative task improvement with respect to the single-task baseline for each task. Given a task 𝒯j\\mathcal{T}\_{j}, the relative task improvement is Δ​m𝒯j\=1K​∑i\=1K(−1)li​(Mi−Si)/Si\\Delta m\_{\\mathcal{T}\_{j}}=\\frac{1}{K}\\sum\_{i=1}^{K}(-1)^{l\_{i}}(M\_{i}-S\_{i})/S\_{i}, where MiM\_{i}, SiS\_{i} refer to metrics for the ithi^{\\text{th}} criterion obtained by objective model and single-task model respectively, li\=1l\_{i}=1 if a lower value for the criterion is better and 00 otherwise. The average relative task improvement is Δ​m\=1T​∑j\=1TΔ​m𝒯j\\Delta m=\\frac{1}{T}\\sum\_{j=1}^{T}\\Delta m\_{\\mathcal{T}\_{j}}.

Table 3: Multi-task learning results on CityScapes dataset. All experiments are repeated over 𝟑\\mathbf{3} random seeds and the mean values are reported. Δ​m%\\Delta m\\% denotes the average relative improvement of all tasks. #P denotes the model size (MB). The grey cell color indicates that Recon improves the result of the base model. The best average result is marked in bold. 

|                 | Segmentation   | Depth   |         |                            |        |
| --------------- | -------------- | ------- | ------- | -------------------------- | ------ |
| (Higher Better) | (Lower Better) |         |         |                            |        |
| mIoU            | Pix Acc        | Abs Err | Rel Err | Δ​m%↑\\Delta m\\%\\uparrow | #P.    |
| 74.05           | 93.17          | 0.0162  | 116.66  | -79.04                     | 190.59 |
| 73.38           | 92.97          | 0.0147  | 82.31   | -47.81                     | 103.43 |
| 74.13           | 93.13          | 0.0166  | 116.00  | -79.32                     | 95.43  |
| 74.17           | 93.21          | 0.0136  | 43.18   | -12.63                     | 108.44 |
| 70.74           | 92.19          | 0.0130  | 47.09   | -16.22                     | 95.43  |
| 71.01           | 92.17          | 0.0129  | 33.41   | -4.46                      | 108.44 |
| 74.08           | 93.08          | 0.0173  | 115.79  | -80.48                     | 95.43  |
| 74.17           | 93.11          | 0.0134  | 41.37   | -10.69                     | 108.44 |
| 73.98           | 93.08          | 0.02    | 114.50  | -78.39                     | 95.43  |
| 74.18           | 93.14          | 0.0136  | 46.02   | -14.92                     | 108.44 |
| 73.81           | 93.02          | 0.0153  | 88.29   | -53.81                     | 95.43  |
| 74.22           | 93.10          | 0.0130  | 38.27   | -7.38                      | 108.44 |

Figure 3: The performance of CAGrad combined with Recon on the Multi-Fashion+MNIST benchmark with (a) different number of selected layers KK (b) different severity value SS for computing conflict scores.

### 5.2 Comparison with the State-of-the-Art

Recon improves the performance of all base models. The main results on Multi-Fashion+MNIST, and CelebA, CityScapes, PASCAL-Context, and NYUv2, are presented in Table [1](#S5.T1 "Table 1 ‣ 5 Experiments ‣ Recon: Reducing Conflicting Gradients from the Root for Multi-Task Learning"), Table [2](#S5.T2 "Table 2 ‣ 5 Experiments ‣ Recon: Reducing Conflicting Gradients from the Root for Multi-Task Learning"), Table [3](#S5.F3 "Figure 3 ‣ 5.1 Experimental Setup ‣ 5 Experiments ‣ Recon: Reducing Conflicting Gradients from the Root for Multi-Task Learning"), Table [4](#S5.T4 "Table 4 ‣ 5.2 Comparison with the State-of-the-Art ‣ 5 Experiments ‣ Recon: Reducing Conflicting Gradients from the Root for Multi-Task Learning"), and Table [5](#S5.T5 "Table 5 ‣ 5.2 Comparison with the State-of-the-Art ‣ 5 Experiments ‣ Recon: Reducing Conflicting Gradients from the Root for Multi-Task Learning") respectively. (1) Compared to gradient manipulation methods, Recon consistently improves their performance in most evaluation metrics, and achieve comparable performance on the rest of evaluation metrics. (2) Compared with branched architecture search methods and architecture design methods, Recon can further improve the performance of BMTAS and MMoE. Besides, Recon combined with other gradient manipulation methods with small model size can achieve better results than branched architecture search methods with much bigger models.

Small increases in model parameters can lead to good performance gains. Note that Recon only changes a small portion of shared parameters to task-specific. As shown in Table [1](#S5.T1 "Table 1 ‣ 5 Experiments ‣ Recon: Reducing Conflicting Gradients from the Root for Multi-Task Learning")-[5](#S5.T5 "Table 5 ‣ 5.2 Comparison with the State-of-the-Art ‣ 5 Experiments ‣ Recon: Reducing Conflicting Gradients from the Root for Multi-Task Learning"), Recon increases the model size by 0.52% to 57.25%. Recon turns 1.42%, 1.46%, 12.77%, 0.26%, 9.80% shared parameters to task-specific on Multi-Fashion+MNIST, CelebA, CityScapes, NYUv2 and PASCAL-Context respectively. The results suggest that the gradient conflicts in a small portion (less than 13%) of shared parameters impede the training of the model for multi-task learning.

Recon is compatible with various neural network architectures. We use ResNet18 on Multi-Fashion+MNIST, SegNet ([Badrinarayanan et al., 2017](#bib.bib1 "")) on CityScapes, MTAN ([Liu et al., 2019](#bib.bib24 "")) on NYUv2, and MobileNetV2 ([Sandler et al., 2018](#bib.bib36 "")) on PASCAL-Context. Recon improves the performance of baselines with different neural network architectures, including the architecture search method BMTAS ([Bruggemann et al., 2020](#bib.bib3 "")) which finds a tree-like structure for multi-task learning.

Only one search of conflict layers is needed for the same network architecture. An interesting observation from our experiments is that network architecture seems to be the deciding factor for the conflict layers found by Recon. With the same network architecture (e.g., ResNet18), the found conflict layers are quite consistent w.r.t. (1) different training stages (e.g., the first 25% iterations, or the middle or last ones) (see Table [12](#A3.T12 "Table 12 ‣ Appendix C Additional Ablation Study ‣ Recon: Reducing Conflicting Gradients from the Root for Multi-Task Learning") and Table [13](#A3.T13 "Table 13 ‣ Appendix C Additional Ablation Study ‣ Recon: Reducing Conflicting Gradients from the Root for Multi-Task Learning") and discussion in Appendix [C](#A3 "Appendix C Additional Ablation Study ‣ Recon: Reducing Conflicting Gradients from the Root for Multi-Task Learning")), (2) different MTL methods (e.g., joint-training or gradient manipulation methods) (see Table [14](#A3.T14 "Table 14 ‣ Appendix C Additional Ablation Study ‣ Recon: Reducing Conflicting Gradients from the Root for Multi-Task Learning") and discussion in Appendix [C](#A3 "Appendix C Additional Ablation Study ‣ Recon: Reducing Conflicting Gradients from the Root for Multi-Task Learning")), and (3) different datasets (see Table [15](#A3.T15 "Table 15 ‣ Appendix C Additional Ablation Study ‣ Recon: Reducing Conflicting Gradients from the Root for Multi-Task Learning") and Table [16](#A3.T16 "Table 16 ‣ Appendix C Additional Ablation Study ‣ Recon: Reducing Conflicting Gradients from the Root for Multi-Task Learning") and discussion in Appendix [C](#A3 "Appendix C Additional Ablation Study ‣ Recon: Reducing Conflicting Gradients from the Root for Multi-Task Learning")). Hence, in our experiments, we only search for the conflict layers *once* with the joint-training baseline in the first 25% training iterations and modify the network to improve various methods on the same dataset. We also find that the conflict layers found on one dataset can be used to modify the network to be directly applied on another dataset to gain performance improvement.

Table 4: Multi-task learning results on PASCAL-Context dataset with 4-task setting. All experiments are repeated over 𝟑\\mathbf{3} random seeds and the mean values are reported. Δ​m%\\Delta m\\% denotes the average relative improvement of all tasks. #P denotes the model size (MB). The grey cell color indicates Recon improves the result of the base model. The best average result is marked in bold.

|                 |                |                 |                                |                                      |       |        |       |       |                            |       |
| --------------- | -------------- | --------------- | ------------------------------ | ------------------------------------ | ----- | ------ | ----- | ----- | -------------------------- | ----- |
| (Higher Better) | (Lower Better) | (Higher Better) | Angle Distance  (Lower Better) | Within t∘t^{\\circ}  (Higher Better) |       |        |       |       |                            |       |
| mIoU            | Pix Acc        | mIoU            | Pix Acc                        | mIoU                                 | Mean  | Median | 11.25 | 22.5  | Δ​m%↑\\Delta m\\%\\uparrow | #P.   |
| 65.00           | 90.53          | 59.59           | 92.61                          | 65.61                                | 14.55 | 12.36  | 46.51 | 81.29 |                            | 30.09 |
| 64.06           | 90.45          | 57.91           | 92.17                          | 62.71                                | 16.40 | 14.23  | 39.38 | 75.93 | -4.82                      | 8.04  |
| 64.73           | 90.50          | 59.00           | 92.44                          | 66.17                                | 14.99 | 12.68  | 44.82 | 80.11 | -0.66                      | 10.20 |
| 46.05           | 86.62          | 54.82           | 91.39                          | 64.76                                | 15.77 | 13.54  | 41.98 | 77.82 | -7.67                      | 8.04  |
| 55.82           | 87.73          | 56.31           | 91.67                          | 64.91                                | 15.12 | 12.88  | 44.36 | 79.81 | -4.14                      | 10.20 |
| 63.91           | 90.45          | 58.01           | 92.19                          | 63.09                                | 16.34 | 14.19  | 39.62 | 76.06 | -4.59                      | 8.04  |
| 65.02           | 90.45          | 59.22           | 92.46                          | 66.14                                | 14.95 | 12.73  | 44.96 | 80.22 | -0.55                      | 10.20 |
| 64.14           | 90.34          | 57.62           | 92.12                          | 62.64                                | 16.46 | 14.28  | 39.29 | 75.71 | -5.00                      | 8.04  |
| 64.48           | 90.45          | 59.08           | 92.46                          | 66.23                                | 14.94 | 12.72  | 45.03 | 80.25 | -0.63                      | 10.20 |
| 63.37           | 90.17          | 57.49           | 92.07                          | 64.16                                | 16.30 | 14.12  | 39.80 | 76.23 | -4.37                      | 8.04  |
| 64.60           | 90.40          | 59.27           | 92.47                          | 65.67                                | 14.92 | 12.71  | 45.10 | 80.33 | -0.76                      | 10.20 |
| 64.89           | 90.44          | 58.87           | 92.36                          | 63.42                                | 15.66 | 13.44  | 42.29 | 78.14 | -2.89                      | 15.18 |
| 64.78           | 90.46          | 59.96           | 92.58                          | 65.96                                | 14.74 | 12.57  | 45.62 | 80.84 | -0.19                      | 16.83 |

Table 5: Multi-task learning results on NYUv2 dataset with MTAN as backbone. All experiments are repeated over 𝟑\\mathbf{3} random seeds and the mean values are reported. Δ​m%\\Delta m\\% denotes the average relative improvement of all tasks. #P denotes the model size (MB). The grey cell color indicates that Recon improves the result of the base model. The best average result is marked in bold.

|                 |                |                                |                                      |       |        |       |       |       |                            |        |
| --------------- | -------------- | ------------------------------ | ------------------------------------ | ----- | ------ | ----- | ----- | ----- | -------------------------- | ------ |
| (Higher Better) | (Lower Better) | Angle Distance  (Lower Better) | Within t∘t^{\\circ}  (Higher Better) |       |        |       |       |       |                            |        |
| mIoU            | Pix Acc        | Abs Err                        | Rel Err                              | Mean  | Median | 11.25 | 22.5  | 30    | Δ​m%↑\\Delta m\\%\\uparrow | #P.    |
| 38.67           | 64.27          | 0.6881                         | 0.2788                               | 24.87 | 18.99  | 30.43 | 57.81 | 69.70 |                            | 285.88 |
| 40.45           | 66.15          | 0.5051                         | 0.2134                               | 27.58 | 23.00  | 24.69 | 49.47 | 62.36 | 4.16                       | 285.88 |
| 39.48           | 65.23          | 0.5491                         | 0.2235                               | 27.87 | 23.76  | 22.68 | 47.91 | 61.58 | 0.75                       | 168.72 |
| 39.54           | 65.20          | 0.5312                         | 0.2234                               | 26.55 | 21.40  | 26.53 | 52.60 | 65.31 | 4.14                       | 169.59 |
| 29.28           | 60.30          | 0.6027                         | 0.2515                               | 24.89 | 19.32  | 29.85 | 57.18 | 69.38 | -2.26                      | 168.72 |
| 32.82           | 61.26          | 0.5884                         | 0.2295                               | 25.17 | 19.72  | 28.18 | 56.49 | 68.96 | 0.53                       | 169.59 |
| 38.70           | 64.97          | 0.5565                         | 0.2333                               | 27.41 | 23.00  | 23.79 | 49.45 | 62.87 | 0.49                       | 168.72 |
| 40.14           | 66.08          | 0.5265                         | 0.2241                               | 26.51 | 21.45  | 26.51 | 52.48 | 65.26 | 4.67                       | 169.59 |
| 38.55           | 65.07          | 0.54                           | 0.23                                 | 26.90 | 22.05  | 24.98 | 51.36 | 64.41 | 2.02                       | 168.72 |
| 38.61           | 65.48          | 0.5350                         | 0.2271                               | 26.31 | 21.11  | 26.90 | 53.21 | 65.95 | 3.87                       | 169.59 |
| 39.89           | 66.47          | 0.5496                         | 0.2281                               | 26.36 | 21.47  | 25.50 | 52.68 | 65.90 | 3.74                       | 168.72 |
| 39.92           | 66.07          | 0.5320                         | 0.2200                               | 25.80 | 20.59  | 27.60 | 54.31 | 67.05 | 5.80                       | 169.59 |

Figure 4: The distribution of gradient conflicts (in terms of cos⁡ϕi​j\\cos{\\phi}\_{ij}) of baselines and baselines with Recon on Multi-Fashion+MNIST dataset.

Table 6: The distribution of gradient conflicts (in terms of cos⁡ϕi​j\\cos{\\phi}\_{ij}) w.r.t. the shared parameters on Multi-Fashion+MNIST dataset. “Reduction” means the percentage of conflicting gradients in the interval of (−0.01,−1.0\](-0.01,-1.0\] reduced by the model compared with joint-training. The grey cell color indicates Recon greatly reduces the conflicting gradients (more than 50%). In contrast, gradient manipulation methods only slightly decrease their occurrence, and some method even increases it. 

| cos⁡ϕi​j\\cos{\\phi\_{ij}} | Joint-train | w/ RSL | w/ RSP | w/ Recon | MGDA  | w/ Recon | Graddrop | w/ Recon | PCGrad | w/ Recon | CAGrad | w/ Recon |
| -------------------------- | ----------- | ------ | ------ | -------- | ----- | -------- | -------- | -------- | ------ | -------- | ------ | -------- |
| 56.56                      | 53.44       | 58.15  | 58.53  | 56.06    | 56.50 | 57.26    | 57.61    | 56.72    | 57.75  | 56.18    | 59.06  |          |
| 31.25                      | 27.35       | 34.33  | 37.67  | 32.36    | 40.93 | 31.06    | 38.28    | 31.19    | 38.76  | 31.25    | 37.84  |          |
| 9.26                       | 13.45       | 6.38   | 3.04   | 8.87     | 2.12  | 8.93     | 3.32     | 9.09     | 2.87   | 9.37     | 2.44   |          |
| 2.05                       | 4.18        | 0.8    | 0.5    | 1.71     | 0.26  | 1.72     | 0.54     | 1.90     | 0.42   | 2.00     | 0.41   |          |
| 1.25                       | 1.58        | 0.34   | 0.25   | 1.0      | 0.18  | 1.03     | 0.26     | 1.10     | 0.2    | 1.20     | 0.25   |          |
| -                          | -52.94      | 40.13  | 69.82  | 7.80     | 79.62 | 7.01     | 67.20    | 3.74     | 72.21  | -0.08    | 75.32  |          |

Table 7: Comparison of Recon with RSL and RSP. PD: performance drop compared to Recon.

| Seed | w/ RSL | w/ RSP | w/ Recon | CAGrad | PCGrad |       |       |       |       |      |       |       |       |
| ---- | ------ | ------ | -------- | ------ | ------ | ----- | ----- | ----- | ----- | ---- | ----- | ----- | ----- |
| 0    | ✓      |        |          | 97.60  | 0.68   | 64.39 | 25.26 | 73.02 | 97.43 | 0.87 | 65.57 | 24.21 | 73.02 |
| 1    | ✓      |        |          | 97.11  | 1.18   | 87.61 | 2.04  | 83.63 | 94.92 | 3.39 | 87.31 | 2.46  | 83.63 |
| 2    | ✓      |        |          | 94.62  | 3.66   | 87.68 | 1.96  | 76.33 | 92.90 | 5.40 | 87.41 | 2.36  | 76.33 |
| 0    |        | ✓      |          | 97.11  | 1.18   | 85.57 | 4.07  | 52.25 | 96.93 | 1.38 | 88.16 | 1.62  | 52.25 |
| 1    |        | ✓      |          | 97.81  | 0.47   | 88.28 | 1.36  | 51.96 | 97.63 | 0.68 | 88.55 | 1.22  | 51.96 |
| 2    |        | ✓      |          | 81.18  | 17.10  | 76.56 | 13.09 | 47.50 | 88.71 | 9.59 | 84.51 | 5.27  | 47.50 |
| -    | -      | -      | ✓        | 98.28  | 0      | 89.65 | 0     | 43.42 | 98.30 | 0    | 89.77 | 0     | 43.42 |

### 5.3 Ablation Study and Analysis

Recon greatly reduces the occurrence of conflicting gradients. In Fig. [4](#S5.F4 "Figure 4 ‣ 5.2 Comparison with the State-of-the-Art ‣ 5 Experiments ‣ Recon: Reducing Conflicting Gradients from the Root for Multi-Task Learning") and Table [6](#S5.T6 "Table 6 ‣ 5.2 Comparison with the State-of-the-Art ‣ 5 Experiments ‣ Recon: Reducing Conflicting Gradients from the Root for Multi-Task Learning"), we compare the distribution of cos⁡ϕi​j\\cos{\\phi\_{ij}} before and after applying Recon on Multi-Fashion+MNIST (the results on other datasets are provided in Appendix [C](#A3 "Appendix C Additional Ablation Study ‣ Recon: Reducing Conflicting Gradients from the Root for Multi-Task Learning")). It can be seen that Recon greatly reduces the numbers of gradient pairs with severe conflicts (cosϕi​j∈(−0.01,−1\]\\cos{\\phi\_{ij}}\\in(-0.01,-1\]) by at least 67% and up to 79% when compared with joint-training, while gradient manipulation methods only slightly reduce the percentage and some even increases it. Similar observations can be made from Tables [8](#A3.T8 "Table 8 ‣ Appendix C Additional Ablation Study ‣ Recon: Reducing Conflicting Gradients from the Root for Multi-Task Learning")-[10](#A3.T10 "Table 10 ‣ Appendix C Additional Ablation Study ‣ Recon: Reducing Conflicting Gradients from the Root for Multi-Task Learning").

Randomly selecting conflict layers does not work. To show that the performance gain of Recon comes from selecting the layers with most severe conflicts instead of merely increasing model parameters, we further compare Recon with the following two baselines. RSL: randomly selecting same number of layers as Recon and set them task-specific. RSP: randomly selecting similar amount of parameters as Recon and set them task-specific. The results in Table [7](#S5.T7 "Table 7 ‣ 5.2 Comparison with the State-of-the-Art ‣ 5 Experiments ‣ Recon: Reducing Conflicting Gradients from the Root for Multi-Task Learning") show that both RSL and RSP lead to significant performance drops, which verifies the effectiveness of the selection strategy of Recon. We compare Recon with the baselines that selects the first or last KK layers in Appendix [C](#A3 "Appendix C Additional Ablation Study ‣ Recon: Reducing Conflicting Gradients from the Root for Multi-Task Learning").

Ablation study on hyperparameters. We study the influence of the conflict severity SS and the number of selected layers KK on the performance of CAGrad w/ Recon on Multi-Fashion+MNIST. As shown in Fig. [3](#S5.F3 "Figure 3 ‣ 5.1 Experimental Setup ‣ 5 Experiments ‣ Recon: Reducing Conflicting Gradients from the Root for Multi-Task Learning"), a small KK leads to a significant performance drop, which indicates that there are still some shared network layers suffering from severe gradient conflicts, while a large KK will not lead to further performance improvement since severe conflicts have been resolved. For the conflict severity SS, we find that a high value of SS (e.g., 0.00.0) leads to performance drops since it includes too many gradient pairs with small conflicts, while some of them are helpful for learning common structures and should not be removed. In the meantime, a too small SS (e.g., −0.15-0.15) also leads to performance degradation because it ignores too many gradient pairs with large conflicts, which may be detrimental to learning. While KK and SS are sensitive, we may only need to tune them once for a given network architecture, as discussed in Sec. [5.2](#S5.SS2 "5.2 Comparison with the State-of-the-Art ‣ 5 Experiments ‣ Recon: Reducing Conflicting Gradients from the Root for Multi-Task Learning").

## 6 Conclusion

We have proposed a very simple yet effective approach, namely Recon, to reduce the occurrence of conflicting gradients for multi-task learning. By considering layer-wise gradient conflicts and identifying the shared layers with severe conflicts and setting them task-specific, Recon can significantly reduce the occurrence of severe conflicting gradients and boost the performance of existing methods with only a reasonable increase in model parameters. We have demonstrated the effectiveness, efficiency, and generality of Recon via extensive experiments and analysis.

#### Acknowledgments

The authors would like to thank Lingzi Jin for checking the proof of Theorem [A.1](#A1.Thmtheorem1 "Theorem A.1. ‣ Appendix A Proof of Theorem ‣ Recon: Reducing Conflicting Gradients from the Root for Multi-Task Learning") and the anonymous reviewers for their insightful and helpful comments.

## References

*   Badrinarayanan et al. (2017) Vijay Badrinarayanan, Alex Kendall, and Roberto Cipolla. Segnet: A deep convolutional encoder-decoder architecture for image segmentation. *IEEE transactions on pattern analysis and machine intelligence*, 39(12):2481–2495, 2017.
*   Bragman et al. (2019) Felix JS Bragman, Ryutaro Tanno, Sebastien Ourselin, Daniel C Alexander, and Jorge Cardoso. Stochastic filter groups for multi-task cnns: Learning specialist and generalist convolution kernels. In *Proceedings of the IEEE/CVF International Conference on Computer Vision*, pp. 1385–1394, 2019.
*   Bruggemann et al. (2020) David Bruggemann, Menelaos Kanakis, Stamatios Georgoulis, and Luc Van Gool. Automated search for resource-efficient branched multi-task networks. *British Machine Vision Conference (BMVC)*, 2020.
*   Caruana (1997) Rich Caruana. Multitask learning. *Machine learning*, 28(1):41–75, 1997.
*   Chen et al. (2018) Zhao Chen, Vijay Badrinarayanan, Chen-Yu Lee, and Andrew Rabinovich. Gradnorm: Gradient normalization for adaptive loss balancing in deep multitask networks. In *International Conference on Machine Learning*, pp. 794–803. PMLR, 2018.
*   Chen et al. (2020) Zhao Chen, Jiquan Ngiam, Yanping Huang, Thang Luong, Henrik Kretzschmar, Yuning Chai, and Dragomir Anguelov. Just pick a sign: Optimizing deep multitask models with gradient sign dropout. *Advances in Neural Information Processing Systems*, 33:2039–2050, 2020.
*   Cordts et al. (2016) Marius Cordts, Mohamed Omran, Sebastian Ramos, Timo Rehfeld, Markus Enzweiler, Rodrigo Benenson, Uwe Franke, Stefan Roth, and Bernt Schiele. The cityscapes dataset for semantic urban scene understanding. In *Proceedings of the IEEE/CVF conference on computer vision and pattern recognition*, 2016.
*   Couprie et al. (2013) Camille Couprie, Clément Farabet, Laurent Najman, and Yann LeCun. Indoor semantic segmentation using depth information. *CoRR*, abs/1301.3572, 2013.
*   Deng et al. (2009) Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. Imagenet: A large-scale hierarchical image database. In *Proceedings of the IEEE/CVF conference on computer vision and pattern recognition*, pp. 248–255. Ieee, 2009.
*   D’Eramo et al. (2020) Carlo D’Eramo, Davide Tateo, Andrea Bonarini, Marcello Restelli, Jan Peters, et al. Sharing knowledge in multi-task deep reinforcement learning. In *International Conference on Learning Representations*, pp. 1–11. OpenReview. net, 2020.
*   Fifty et al. (2021) Chris Fifty, Ehsan Amid, Zhe Zhao, Tianhe Yu, Rohan Anil, and Chelsea Finn. Efficiently identifying task groupings for multi-task learning. *Advances in Neural Information Processing Systems*, 34:27503–27516, 2021.
*   Gao et al. (2019) Yuan Gao, Jiayi Ma, Mingbo Zhao, Wei Liu, and Alan L Yuille. Nddr-cnn: Layerwise feature fusing in multi-task cnns by neural discriminative dimensionality reduction. In *Proceedings of the IEEE/CVF conference on computer vision and pattern recognition*, pp. 3205–3214, 2019.
*   Gao et al. (2020) Yuan Gao, Haoping Bai, Zequn Jie, Jiayi Ma, Kui Jia, and Wei Liu. Mtl-nas: Task-agnostic neural architecture search towards general-purpose multi-task learning. In *Proceedings of the IEEE/CVF Conference on computer vision and pattern recognition*, pp. 11543–11552, 2020.
*   Guo et al. (2018) Michelle Guo, Albert Haque, De-An Huang, Serena Yeung, and Li Fei-Fei. Dynamic task prioritization for multitask learning. In *Proceedings of the European conference on computer vision (ECCV)*, pp. 270–287, 2018.
*   Guo et al. (2020) Pengsheng Guo, Chen-Yu Lee, and Daniel Ulbricht. Learning to branch for multi-task learning. In *International Conference on Machine Learning*, pp. 3854–3863. PMLR, 2020.
*   Hashimoto et al. (2017) Kazuma Hashimoto, Caiming Xiong, Yoshimasa Tsuruoka, and Richard Socher. A joint many-task model: Growing a neural network for multiple nlp tasks. *Empirical Methods in Natural Language Processing (EMNLP)*, 2017.
*   He et al. (2016) Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In *Proceedings of the IEEE/CVF conference on computer vision and pattern recognition*, pp. 770–778, 2016.
*   Javaloy & Valera (2022) Adrián Javaloy and Isabel Valera. Rotograd: Gradient homogenization in multitask learning. In *International Conference on Learning Representations*, 2022.
*   Kendall et al. (2018) Alex Kendall, Yarin Gal, and Roberto Cipolla. Multi-task learning using uncertainty to weigh losses for scene geometry and semantics. In *Proceedings of the IEEE/CVF conference on computer vision and pattern recognition*, pp. 7482–7491, 2018.
*   Kokkinos (2017) Iasonas Kokkinos. Ubernet: Training a universal convolutional neural network for low-, mid-, and high-level vision using diverse datasets and limited memory. In *Proceedings of the IEEE/CVF conference on computer vision and pattern recognition*, pp. 6129–6138, 2017.
*   Lin et al. (2019) Xi Lin, Hui-Ling Zhen, Zhenhua Li, Qing-Fu Zhang, and Sam Kwong. Pareto multi-task learning. *Advances in Neural Information Processing Systems*, 32, 2019.
*   Liu et al. (2021a) Bo Liu, Xingchao Liu, Xiaojie Jin, Peter Stone, and Qiang Liu. Conflict-averse gradient descent for multi-task learning. *Advances in Neural Information Processing Systems*, 34:18878–18890, 2021a.
*   Liu et al. (2021b) Liyang Liu, Yi Li, Zhanghui Kuang, J Xue, Yimin Chen, Wenming Yang, Qingmin Liao, and Wayne Zhang. Towards impartial multi-task learning. In *International Conference on Learning Representations*, 2021b.
*   Liu et al. (2019) Shikun Liu, Edward Johns, and Andrew J Davison. End-to-end multi-task learning with attention. In *Proceedings of the IEEE/CVF conference on computer vision and pattern recognition*, pp. 1871–1880, 2019.
*   Liu et al. (2015) Ziwei Liu, Ping Luo, Xiaogang Wang, and Xiaoou Tang. Deep learning face attributes in the wild. In *Proceedings of the IEEE international conference on computer vision*, pp. 3730–3738, 2015.
*   Long et al. (2017) Mingsheng Long, Zhangjie Cao, Jianmin Wang, and Philip S Yu. Learning multiple tasks with multilinear relationship networks. *Advances in Neural Information Processing Systems*, 30, 2017.
*   Ma et al. (2018) Jiaqi Ma, Zhe Zhao, Xinyang Yi, Jilin Chen, Lichan Hong, and Ed H Chi. Modeling task relationships in multi-task learning with multi-gate mixture-of-experts. In *Proceedings of the 24th ACM SIGKDD international conference on knowledge discovery & data mining*, pp. 1930–1939, 2018.
*   Maninis et al. (2019) Kevis-Kokitsi Maninis, Ilija Radosavovic, and Iasonas Kokkinos. Attentive single-tasking of multiple tasks. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pp. 1851–1860, 2019.
*   Meyerson & Miikkulainen (2018) Elliot Meyerson and Risto Miikkulainen. Beyond shared hierarchies: Deep multitask learning through soft layer ordering. In *International Conference on Learning Representations*, 2018.
*   Misra et al. (2016) Ishan Misra, Abhinav Shrivastava, Abhinav Gupta, and Martial Hebert. Cross-stitch networks for multi-task learning. In *Proceedings of the IEEE/CVF conference on computer vision and pattern recognition*, pp. 3994–4003, 2016.
*   Mottaghi et al. (2014) Roozbeh Mottaghi, Xianjie Chen, Xiaobai Liu, Nam-Gyu Cho, Seong-Whan Lee, Sanja Fidler, Raquel Urtasun, and Alan Yuille. The role of context for object detection and semantic segmentation in the wild. In *Proceedings of the IEEE/CVF conference on computer vision and pattern recognition*, pp. 891–898, 2014.
*   Parisotto et al. (2016) Emilio Parisotto, Lei Jimmy Ba, and Ruslan Salakhutdinov. Actor-mimic: Deep multitask and transfer reinforcement learning. In *International Conference on Learning Representations*, 2016.
*   Rosenbaum et al. (2018) Clemens Rosenbaum, Tim Klinger, and Matthew Riemer. Routing networks: Adaptive selection of non-linear functions for multi-task learning. In *International Conference on Learning Representations*, 2018.
*   Ruder (2017) Sebastian Ruder. An overview of multi-task learning in deep neural networks. *arXiv preprint arXiv:1706.05098*, 2017.
*   Ruder et al. (2019) Sebastian Ruder, Joachim Bingel, Isabelle Augenstein, and Anders Søgaard. Latent multi-task architecture learning. In *Proceedings of the AAAI Conference on Artificial Intelligence*, volume 33, pp. 4822–4829, 2019.
*   Sandler et al. (2018) Mark Sandler, Andrew Howard, Menglong Zhu, Andrey Zhmoginov, and Liang-Chieh Chen. Mobilenetv2: Inverted residuals and linear bottlenecks. In *Proceedings of the IEEE/CVF conference on computer vision and pattern recognition*, pp. 4510–4520, 2018.
*   Sener & Koltun (2018) Ozan Sener and Vladlen Koltun. Multi-task learning as multi-objective optimization. *Advances in Neural Information Processing Systems*, 31, 2018.
*   Shen et al. (2021) Jiayi Shen, Xiantong Zhen, Marcel Worring, and Ling Shao. Variational multi-task learning with gumbel-softmax priors. *Advances in Neural Information Processing Systems*, 34:21031–21042, 2021.
*   Standley et al. (2020) Trevor Standley, Amir Zamir, Dawn Chen, Leonidas Guibas, Jitendra Malik, and Silvio Savarese. Which tasks should be learned together in multi-task learning? In *International Conference on Machine Learning*, pp. 9120–9132. PMLR, 2020.
*   Thrun & O’Sullivan (1996) Sebastian Thrun and Joseph O’Sullivan. Discovering structure in multiple learning tasks: The tc algorithm. In *International Conference on Machine Learning*, volume 96, pp. 489–497, 1996.
*   Vandenhende et al. (2021) Simon Vandenhende, Stamatios Georgoulis, Wouter Van Gansbeke, Marc Proesmans, Dengxin Dai, and Luc Van Gool. Multi-task learning for dense prediction tasks: A survey. *IEEE transactions on pattern analysis and machine intelligence*, 2021.
*   Yang et al. (2020) Ruihan Yang, Huazhe Xu, Yi Wu, and Xiaolong Wang. Multi-task reinforcement learning with soft modularization. *Advances in Neural Information Processing Systems*, 33:4767–4777, 2020.
*   Yu et al. (2020) Tianhe Yu, Saurabh Kumar, Abhishek Gupta, Sergey Levine, Karol Hausman, and Chelsea Finn. Gradient surgery for multi-task learning. *Advances in Neural Information Processing Systems*, 33:5824–5836, 2020.
*   Zamir et al. (2018) Amir R Zamir, Alexander Sax, William Shen, Leonidas J Guibas, Jitendra Malik, and Silvio Savarese. Taskonomy: Disentangling task transfer learning. In *Proceedings of the IEEE/CVF conference on computer vision and pattern recognition*, pp. 3712–3722, 2018.

## Appendix A Proof of Theorem [A.1](#A1.Thmtheorem1 "Theorem A.1. ‣ Appendix A Proof of Theorem ‣ Recon: Reducing Conflicting Gradients from the Root for Multi-Task Learning")

###### Theorem A.1.

Assume that ℒ\\mathcal{L} is differentiable and for any two different tasks 𝒯i\\mathcal{T}\_{i} and 𝒯j\\mathcal{T}\_{j}, it satisfies

cos⁡ϕi​j(k)​‖𝐠i(k)‖<‖𝐠j(k)‖,∀k∈ℙ\\cos{\\phi\_{ij}^{(k)}}\\|\\mathbf{g}\_{i}^{(k)}\\|<\\|\\mathbf{g}\_{j}^{(k)}\\|,\\quad\\forall k\\in\\mathbb{P}

(8)

then for any sufficiently small learning rate α\>0\\alpha>0, ℒ⁡(θ^r)<ℒ⁡(θ^).\\mathcal{L}(\\hat{\\theta}\_{r})<\\mathcal{L}(\\hat{\\theta}).

(9)

###### Proof.

We consider the first order Taylor approximation of ℒi\\mathcal{L}\_{i}. For normal update, we have

ℒi​(θ^shfix,θ^shcf,θ^its)\=\\displaystyle\\mathcal{L}\_{i}\\left(\\hat{\\theta}\_{\\mathrm{sh}}^{\\mathrm{fix}},\\hat{\\theta}\_{\\mathrm{sh}}^{\\mathrm{cf}},\\hat{\\theta}\_{i}^{\\mathrm{ts}}\\right)=

ℒi​(θshfix,θshcf,θits)+(θ^shfix−θshfix)⊤​𝐠ifix\\displaystyle\\mathcal{L}\_{i}\\left(\\theta\_{\\mathrm{sh}}^{\\mathrm{fix}},\\theta\_{\\mathrm{sh}}^{\\mathrm{cf}},\\theta\_{i}^{\\mathrm{ts}}\\right)+(\\hat{\\theta}\_{\\mathrm{sh}}^{\\mathrm{fix}}-\\theta\_{\\mathrm{sh}}^{\\mathrm{fix}})^{\\top}\\mathbf{g}\_{i}^{\\mathrm{fix}}

(10)

+(θ^shcf−θshcf)⊤​𝐠icf+(θ^its−θits)⊤​𝐠its+o⁡(α).\\displaystyle+(\\hat{\\theta}\_{\\mathrm{sh}}^{\\mathrm{cf}}-\\theta\_{\\mathrm{sh}}^{\\mathrm{cf}})^{\\top}\\mathbf{g}\_{i}^{\\mathrm{cf}}+(\\hat{\\theta}\_{i}^{\\mathrm{ts}}-\\theta\_{i}^{\\mathrm{ts}})^{\\top}\\mathbf{g}\_{i}^{\\mathrm{ts}}+o(\\alpha).

(11)

For Recon update, we have

ℒi​(θ^shfix,θ^icf,θ^its)\=\\displaystyle\\mathcal{L}\_{i}\\left(\\hat{\\theta}\_{\\mathrm{sh}}^{\\mathrm{fix}},\\hat{\\theta}\_{i}^{\\mathrm{cf}},\\hat{\\theta}\_{i}^{\\mathrm{ts}}\\right)=

ℒi​(θshfix,θshcf,θits)+(θ^shfix−θshfix)⊤​𝐠its\\displaystyle\\mathcal{L}\_{i}\\left(\\theta\_{\\mathrm{sh}}^{\\mathrm{fix}},\\theta\_{\\mathrm{sh}}^{\\mathrm{cf}},\\theta\_{i}^{\\mathrm{ts}}\\right)+(\\hat{\\theta}\_{\\mathrm{sh}}^{\\mathrm{fix}}-\\theta\_{\\mathrm{sh}}^{\\mathrm{fix}})^{\\top}\\mathbf{g}\_{i}^{\\mathrm{ts}}

(12)

+(θ^icf−θshcf)⊤​𝐠icf+(θ^its−θits)⊤​𝐠its+o⁡(α).\\displaystyle+(\\hat{\\theta}\_{i}^{\\mathrm{cf}}-\\theta\_{\\mathrm{sh}}^{\\mathrm{cf}})^{\\top}\\mathbf{g}\_{i}^{\\mathrm{cf}}+(\\hat{\\theta}\_{i}^{\\mathrm{ts}}-\\theta\_{i}^{\\mathrm{ts}})^{\\top}\\mathbf{g}\_{i}^{\\mathrm{ts}}+o(\\alpha).

(13)

The difference between the two loss functions after the update is

ℒi​(θ^shfix,θ^icf,θ^its)−ℒi​(θ^shfix,θ^shcf,θ^its)\=\\displaystyle\\mathcal{L}\_{i}\\left(\\hat{\\theta}\_{\\mathrm{sh}}^{\\mathrm{fix}},\\hat{\\theta}\_{i}^{\\mathrm{cf}},\\hat{\\theta}\_{i}^{\\mathrm{ts}}\\right)-\\mathcal{L}\_{i}\\left(\\hat{\\theta}\_{\\mathrm{sh}}^{\\mathrm{fix}},\\hat{\\theta}\_{\\mathrm{sh}}^{\\mathrm{cf}},\\hat{\\theta}\_{i}^{\\mathrm{ts}}\\right)=

(θ^icf−θ^shcf)⊤​𝐠icf+o⁡(α)\\displaystyle(\\hat{\\theta}\_{i}^{\\mathrm{cf}}-\\hat{\\theta}\_{\\mathrm{sh}}^{\\mathrm{cf}})^{\\top}\\mathbf{g}\_{i}^{\\mathrm{cf}}+o(\\alpha)

(14)

\=\\displaystyle=

−α​(𝐠icf−∑j\=1Twj​𝐠jcf)⊤​𝐠icf+o⁡(α)\\displaystyle-\\alpha\\left(\\mathbf{g}\_{i}^{\\mathrm{cf}}-\\sum\_{j=1}^{T}w\_{j}\\mathbf{g}\_{j}^{\\mathrm{cf}}\\right)^{\\top}\\mathbf{g}\_{i}^{\\mathrm{cf}}+o(\\alpha)

(15)

\=\\displaystyle=

−α∑j\=1Twj(𝐠icf−𝐠jcf)⊤𝐠icf+o(α)\\displaystyle-\\alpha\\sum\_{j=1}^{T}w\_{j}\\left(\\mathbf{g}\_{i}^{\\mathrm{cf}}-\\mathbf{g}\_{j}^{\\mathrm{cf}}\\right)^{\\top}\\mathbf{g}\_{i}^{\\mathrm{cf}}+o(\\alpha)

(16)

\=\\displaystyle=

−α∑j\=1Twj(∥𝐠icf∥2−𝐠jcf𝐠icf⊤)+o(α).\\displaystyle-\\alpha\\sum\_{j=1}^{T}w\_{j}\\left(\\|\\mathbf{g}\_{i}^{\\mathrm{cf}}\\|^{2}-\\mathbf{g}\_{j}^{\\mathrm{cf}}{}^{\\top}\\mathbf{g}\_{i}^{\\mathrm{cf}}\\right)+o(\\alpha).

(17)

Assume, without loss of generality, that ‖𝐠ic​f‖≠0\\|\\mathbf{g}\_{i}^{cf}\\|\\neq 0, then

‖𝐠icf‖2−𝐠jcf𝐠icf⊤\\displaystyle\\left\\|\\mathbf{g}\_{i}^{\\mathrm{cf}}\\right\\|^{2}-\\mathbf{g}\_{j}^{\\mathrm{cf}}{}^{\\top}\\mathbf{g}\_{i}^{\\mathrm{cf}}

\=∑k∈ℙ(‖𝐠i(k)‖2−𝐠i(k)𝐠j(k)⊤)\\displaystyle=\\sum\_{k\\in\\mathbb{P}}\\left(\\left\\|\\mathbf{g}\_{i}^{(k)}\\right\\|^{2}-\\mathbf{g}\_{i}^{(k)}{}^{\\top}\\mathbf{g}\_{j}^{(k)}\\right)

(18)

\=∑k∈ℙ‖𝐠i(k)‖​(‖𝐠i(k)‖−cos⁡ϕi​j(k)​‖𝐠j(k)‖)\\displaystyle=\\sum\_{k\\in\\mathbb{P}}\\left\\|\\mathbf{g}\_{i}^{(k)}\\right\\|\\left(\\left\\|\\mathbf{g}\_{i}^{(k)}\\right\\|-\\cos\\phi\_{ij}^{(k)}\\left\\|\\mathbf{g}\_{j}^{(k)}\\right\\|\\right)

(19)

\>0.\\displaystyle>0.

(20)

Hence, the above difference is negative, if α\\alpha is sufficiently small. As such, the difference between the multi-task loss functions is also negative, if α\\alpha is sufficiently small.

ℒ⁡(θ^r)−ℒ⁡(θ^)\=∑i\=1Tℒi​(θ^shfix,θ^icf,θ^its)−∑i\=1Tℒi​(θ^shfix,θ^shcf,θ^its)<0\\mathcal{L}(\\hat{\\theta}\_{r})-\\mathcal{L}(\\hat{\\theta})=\\sum\_{i=1}^{T}\\mathcal{L}\_{i}\\left(\\hat{\\theta}\_{\\mathrm{sh}}^{\\mathrm{fix}},\\hat{\\theta}\_{i}^{\\mathrm{cf}},\\hat{\\theta}\_{i}^{\\mathrm{ts}}\\right)-\\sum\_{i=1}^{T}\\mathcal{L}\_{i}\\left(\\hat{\\theta}\_{\\mathrm{sh}}^{\\mathrm{fix}},\\hat{\\theta}\_{\\mathrm{sh}}^{\\mathrm{cf}},\\hat{\\theta}\_{i}^{\\mathrm{ts}}\\right)<0

(21)

∎

## Appendix B Experimental setup

### B.1 Multi-Fashion+MNIST

Model. We adopt ResNet18 ([He et al., 2016](#bib.bib17 "")) without pre-training as the backbone and modify the dimension of the output features to 100 for the last linear layer. For the task-specific heads, we define two linear layers followed by a ReLU function.

Tasks, losses, and metrics. Each task is a classification problem with 1010 classes and we use the cross-entropy loss as the classification loss. For evaluation, we use the classification accuracy as the metric for each task.

Model hyperparameters. We train the model for 120120 epochs with the batch size of 256256. We adopt SGD with an initial learning rate of 0.10.1 and decay the learning rate by 0.10.1 at the 60th60^{\\text{th}} and 90th90^{\\text{th}} epoch.

Baseline hyperparameters. For CAGrad, we set α\=0.2\\alpha=0.2. For BMTAS, we set the resource loss weight to 1.01.0, and we search the architecture for 100100 epochs. For RotoGrad, we set Rk\=100R\_{k}=100 which is equal to the dimension of shared features and set the learning rate of rotation parameters as learning rate of the neural networks. For MMoE, the initial learning rate of expert networks and gates are 0.10.1 and 1​e−31e-3 respectively.

Recon hyperparameters. We use CAGrad to train the model for 3030 epochs and compute the conflict score of each shared layer. We set S\=−0.1S=-0.1 for computing the scores. We select 2525 layers with the highest conflict scores and turn them into task-specific layers.

### B.2 CityScapes

Model. We adopt SegNet ([Badrinarayanan et al., 2017](#bib.bib1 "")) as the backbone where the decoder is split into two convolutional heads.

Model hyperparameters. We train the model for 200200 epochs with the batch size of 88. We adopt Adam with an initial learning rate of 5​e−55e-5 and decay the learning rate by 0.50.5 at the 100th100^{\\text{th}} epoch.

Baselines hyperparameters. For CAGrad, we set α\=0.2\\alpha=0.2. For RotoGrad, we set Rk\=1024R\_{k}=1024 and set the learning rate of rotation parameters as 10 times less than the learning rate of the neural networks.

Recon hyperparameters. We use joint-train to train the model for 4040 epochs and compute the conflict score of each shared layer. We set S\=0.0S=0.0 for computing the scores. We select 3939 layers with the highest conflict scores and turn them into task-specific layers.

### B.3 NYUv2

Model. We adopt MTAN ([Liu et al., 2019](#bib.bib24 "")) – the SegNet combined with task-specific attention modules on the encoder.

Model hyperparameters. We train the model for 200200 epochs with the batch size of 22. We adopt Adam with an initial learning rate of 1​e−41e-4 and decay the learning rate by 0.50.5 at the 100th100^{\\text{th}} epoch.

Baseline hyperparameters. For CAGrad, we set α\=0.4\\alpha=0.4 similar with [Liu et al. (2021a)](#bib.bib22 "").

Recon hyperparameters. We use joint-train to train the model for 4040 epochs and compute the conflict score of each shared layer. We set S\=−0.02S=-0.02 for computing the scores. We select 2222 layers with the highest conflict scores and turn them into task-specific layers.

### B.4 PASCAL-Context

Model. Following [Bruggemann et al. (2020)](#bib.bib3 ""), we employ MobileNetv2 [Sandler et al. (2018)](#bib.bib36 "") as the backbone with a reduced design of the ASPP module (R-ASPP) ([Sandler et al., 2018](#bib.bib36 "")). We pre-train the model on ImageNet ([Deng et al., 2009](#bib.bib9 "")).

Model hyperparameters. We train the model for 130130 epochs with the batch size of 66. We adopt Adam with an initial learning rate of 1​e−41e-4 and decay the learning rate by 0.10.1 at the 70th70^{\\text{th}} and 100th100^{\\text{th}} epoch.

Baselines hyperparameters. For CAGrad, we set α\=0.1\\alpha=0.1. For BMTAS, we set the resoure loss weight to 0.1, and we search the architecture for 130130 epochs.

Recon hyperparameters. We use joint-train to train the model for 4040 epochs and compute the conflict score of each shared layer. We set S\=−0.02S=-0.02 for computing the scores. We select 8585 layers with the highest conflict scores and turn them into task-specific layers.

### B.5 CelebA

Model. Following [Sener & Koltun (2018)](#bib.bib37 ""), we use ResNet18 ([He et al., 2016](#bib.bib17 "")) as the backbone network. We pre-train the model on ImageNet ([Deng et al., 2009](#bib.bib9 "")).

Model hyperparameters. We train the model for 55 epochs. We adopt Adam with an initial learning rate of 5​e−55e-5 and decay the learning rate by 0.50.5 at the 3th3^{\\text{th}} epoch.

Baselines hyperparameters. For CAGrad, we set α\=0.1\\alpha=0.1.

Recon hyperparameters. We use joint-train to train the model for 22 epochs and compute the conflict score of each shared layer. We set S\=−0.05S=-0.05. We select 2525 layers with the highest conflict scores and turn them into task-specific layers.

## Appendix C Additional Ablation Study

The distribution of gradient conflicts. In addition to the statistics on Multi-Fashion+MNIST, we further show the distributions of gradient conflicts of various baselines on CityScapes, NYUv2, and PASCAL-Context in Fig [5](#A3.F5 "Figure 5 ‣ Appendix C Additional Ablation Study ‣ Recon: Reducing Conflicting Gradients from the Root for Multi-Task Learning"), Fig [6](#A3.F6 "Figure 6 ‣ Appendix C Additional Ablation Study ‣ Recon: Reducing Conflicting Gradients from the Root for Multi-Task Learning"), and Fig [7](#A3.F7 "Figure 7 ‣ Appendix C Additional Ablation Study ‣ Recon: Reducing Conflicting Gradients from the Root for Multi-Task Learning") respectively. We compare the distributions with those of baselines w/ Recon on the three datasets in Fig. [8](#A3.F8 "Figure 8 ‣ Appendix C Additional Ablation Study ‣ Recon: Reducing Conflicting Gradients from the Root for Multi-Task Learning"), Fig. [9](#A3.F9 "Figure 9 ‣ Appendix C Additional Ablation Study ‣ Recon: Reducing Conflicting Gradients from the Root for Multi-Task Learning"), and Fig. [10](#A3.F10 "Figure 10 ‣ Appendix C Additional Ablation Study ‣ Recon: Reducing Conflicting Gradients from the Root for Multi-Task Learning") respectively. The detailed statistics are provided in Tables [8](#A3.T8 "Table 8 ‣ Appendix C Additional Ablation Study ‣ Recon: Reducing Conflicting Gradients from the Root for Multi-Task Learning")-[10](#A3.T10 "Table 10 ‣ Appendix C Additional Ablation Study ‣ Recon: Reducing Conflicting Gradients from the Root for Multi-Task Learning").

Figure 5: The distributions of gradient conflicts (in terms of cos⁡ϕi​j\\cos\\phi\_{ij}) of the joint-training baseline and state-of-the-art gradient manipulation methods on CityScapes dataset. 

Figure 6: The distributions of gradient conflicts (in terms of cos⁡ϕi​j\\cos\\phi\_{ij}) of the joint-training baseline and state-of-the-art gradient manipulation methods on NYUv2 dataset. 

Figure 7: The distributions of gradient conflicts (in terms of cos⁡ϕi​j\\cos\\phi\_{ij}) of the joint-training baseline and state-of-the-art gradient manipulation methods on PASCAL-Context dataset. 

Figure 8: The distribution of gradient conflicts (in terms of cos⁡ϕi​j\\cos{\\phi}\_{ij}) w.r.t. the shared parameters on CityScapes. RSL: randomly selecting same number of layers as Recon and set them task-specific. RSP: randomly selecting similar amount of parameters as Recon and set them task-specific.

Figure 9: The distribution of gradient conflicts (in terms of cos⁡ϕi​j\\cos{\\phi}\_{ij}) of baselines and baselines with Recon on NYUv2. RSL: randomly selecting same number of layers as Recon and set them task-specific. RSP: randomly selecting similar amount of parameters as Recon and set them task-specific. 

Figure 10: The distribution of gradient conflicts (in terms of cos⁡ϕi​j\\cos{\\phi}\_{ij}) of baselines and baselines with Recon on PASCAL-Context. RSL: randomly selecting same number of layers as Recon and set them task-specific. RSP: randomly selecting similar amount of parameters as Recon and set them task-specific. 

Table 8: The distribution of gradient conflicts (in terms of cos⁡ϕi​j\\cos{\\phi\_{ij}}) w.r.t. the shared parameters on CityScapes dataset. “Reduction” means the percentage of conflicting gradients in the interval of (−0.02,−1.0\](-0.02,-1.0\] reduced by the model compared with joint-training. The grey cell color indicates Recon greatly reduces the conflicting gradients (more than 50%). In contrast, gradient manipulation methods only moderately decrease their occurrence (MGDA deceases it by 22%), and some methods even increase it. 

| cos⁡ϕi​j\\cos{\\phi\_{ij}} | Joint-train | w/ RSL | w/ RSP | w/ Recon | MGDA  | w/ Recon | Graddrop | w/ Recon | PCGrad | w/ Recon | CAGrad | w/ Recon |
| -------------------------- | ----------- | ------ | ------ | -------- | ----- | -------- | -------- | -------- | ------ | -------- | ------ | -------- |
| 59.55                      | 53.16       | 58.29  | 73.62  | 63.9     | 78.27 | 59.56    | 73.82    | 59.85    | 74.52  | 60.79    | 74.54  |          |
| 10.14                      | 9.01        | 10.77  | 20.13  | 12.51    | 12.54 | 9.61     | 19.75    | 9.58     | 19.43  | 11.13    | 19.77  |          |
| 8.52                       | 7.34        | 8.72   | 5.13   | 8.59     | 5.54  | 8.19     | 5.17     | 7.94     | 4.89   | 8.83     | 4.62   |          |
| 6.45                       | 5.69        | 6.48   | 0.94   | 5.39     | 2.23  | 6.49     | 1.05     | 6.24     | 0.96   | 6.05     | 0.89   |          |
| 4.79                       | 4.53        | 4.61   | 0.14   | 3.29     | 0.85  | 4.76     | 0.16     | 4.41     | 0.15   | 4.06     | 0.13   |          |
| 10.54                      | 20.26       | 11.13  | 0.03   | 6.33     | 0.56  | 11.38    | 0.05     | 11.98    | 0.06   | 9.13     | 0.04   |          |
| -                          | -24.82      | -2.11  | 79.41  | 22.11    | 69.70 | -1.72    | 78.78    | -0.89    | 80.03  | 7.36     | 81.22  |          |

Table 9: The distribution of gradient conflicts (in terms of cos⁡ϕi​j\\cos{\\phi}\_{ij}) w.r.t. the shared parameters on NYUv2 dataset. “Reduction” means the percentage of conflicting gradients in the interval of (−0.04,−1.0\](-0.04,-1.0\] reduced by the model compared with joint-training. The grey cell color indicates Recon greatly reduces the conflicting gradients (more than 50%). In contrast, gradient manipulation methods only slightly decrease their occurrence, and some methods even increase it.

| cos⁡ϕi​j\\cos{\\phi\_{ij}} | Joint-train | w/ RSL | w/ RSP | w/ Recon | MGDA  | w/ Recon | Graddrop | w/ Recon | PCGrad | w/ Recon | CAGrad | w/ Recon |
| -------------------------- | ----------- | ------ | ------ | -------- | ----- | -------- | -------- | -------- | ------ | -------- | ------ | -------- |
| \[1.0, 0)                  | 61.96       | 52.61  | 59.70  | 73.99    | 61.28 | 74.08    | 62.93    | 75.35    | 63.25  | 75.54    | 61.95  | 74.49    |
| (0, -0.02\]                | 3.85        | 3.75   | 3.47   | 14.17    | 2.97  | 13.38    | 3.83     | 13.50    | 3.61   | 12.66    | 3.53   | 14.20    |
| (-0.02, -0.04\]            | 3.63        | 3.60   | 3.41   | 7.07     | 2.77  | 7.21     | 3.70     | 6.71     | 3.62   | 6.66     | 3.39   | 6.96     |
| (-0.04, -0.06\]            | 3.39        | 3.43   | 3.11   | 2.89     | 2.81  | 3.19     | 3.45     | 2.71     | 3.26   | 2.98     | 3.21   | 2.71     |
| (-0.06, -0.08\]            | 3.11        | 3.30   | 2.94   | 1.13     | 2.64  | 1.28     | 3.16     | 1.03     | 3.06   | 1.25     | 3.05   | 1.01     |
| (-0.08, -1.0\]             | 24.05       | 33.31  | 27.37  | 0.76     | 27.53 | 0.87     | 22.92    | 0.70     | 23.20  | 0.90     | 24.88  | 0.63     |
| Reduction (%)              | -           | -31.06 | -9.39  | 84.35    | -7.95 | 82.52    | 3.34     | 85.47    | 3.37   | 83.21    | -1.93  | 85.76    |

Table 10: The distribution of gradient conflicts (in terms of cos⁡ϕi​j\\cos{\\phi}\_{ij}) w.r.t. the shared parameters on PASCAL-Context dataset. “Reduction” means the percentage of conflicting gradients in the interval of (−0.02,−1.0\](-0.02,-1.0\] reduced by the model compared with joint-training. The grey cell color indicates Recon greatly reduces the conflicting gradients (more than 50%). In contrast, gradient manipulation methods only slightly decrease their occurrence, and some methods even increase it. 

| cos⁡ϕi​j\\cos{\\phi\_{ij}} | Joint-train | w/ RSL | w/ RSP | w/ Recon | MGDA  | w/ Recon | Graddrop | w/ Recon | PCGrad | w/ Recon | CAGrad | w/ Recon |
| -------------------------- | ----------- | ------ | ------ | -------- | ----- | -------- | -------- | -------- | ------ | -------- | ------ | -------- |
| \[1.0, 0)                  | 61.26       | 59.20  | 60.47  | 63.99    | 60.40 | 63.61    | 61.18    | 63.76    | 61.35  | 63.83    | 60.99  | 63.78    |
| (0, -0.02\]                | 9.66        | 21.01  | 18.25  | 23.57    | 8.51  | 33.53    | 9.66     | 23.41    | 9.83   | 23.61    | 9.95   | 24.04    |
| (-0.02, -0.04\]            | 7.90        | 9.91   | 9.10   | 7.65     | 7.27  | 2.04     | 7.89     | 7.83     | 7.90   | 7.65     | 8.03   | 7.53     |
| (-0.04, -0.06\]            | 5.85        | 3.05   | 3.88   | 2.59     | 5.68  | 0.45     | 5.80     | 2.71     | 5.82   | 2.66     | 5.91   | 2.51     |
| (-0.06, -0.08\]            | 4.16        | 1.32   | 1.79   | 1.07     | 4.35  | 0.17     | 4.21     | 1.12     | 4.13   | 1.10     | 4.23   | 1.04     |
| (-0.08, -1.0\]             | 11.16       | 1.30   | 2.29   | 1.13     | 13.80 | 0.20     | 11.24    | 1.16     | 10.97  | 1.16     | 10.88  | 1.08     |
| Reduction (%)              | -           | 46.41  | 41.31  | 57.21    | -6.98 | 90.16    | -0.24    | 55.90    | 0.86   | 56.76    | 0.07   | 58.20    |

Table 11: Multi-task learning results on Multi-Fashion+MNIST dataset. LSK refers to turning the fist KK layers into task-specific layers. FSK refers to turning the last KK layers into task-specific layers. PD denotes the performance drop compared with Recon.

| LSK | FSK | w/ Recon | CAGrad | PCGrad |       |      |       |       |      |       |      |       |
| --- | --- | -------- | ------ | ------ | ----- | ---- | ----- | ----- | ---- | ----- | ---- | ----- |
| ✓   |     |          | 97.63  | 0.66   | 89.14 | 0.50 | 84.17 | 97.63 | 0.65 | 88.98 | 0.66 | 84.17 |
|     | ✓   |          | 98.21  | 0.07   | 89.15 | 0.50 | 48.90 | 98.19 | 0.09 | 89.51 | 0.13 | 48.90 |
|     | -   | ✓        | 98.28  | 0      | 89.65 | 0    | 43.42 | 98.30 | 0    | 89.77 | 0    | 43.42 |

Selecting the first KK layers and the last KK Layers as conflict layers does not work. To further support the conclusion that the selection of parameters with higher probability of conflicting gradients contributes most to the performance gain rather than the increase in model capacity. We compare Recon with two baselines: (1) Select the first KK neural network layers and turn them into task-specific layers. (2) Select the last KK neural network layers and turn them into task-specific layers. The multi-task learning results on the Multi-Fashion+MNIST benchmark are presented in Table [11](#A3.T11 "Table 11 ‣ Appendix C Additional Ablation Study ‣ Recon: Reducing Conflicting Gradients from the Root for Multi-Task Learning"). The results show that if we directly turn the top or the bottom of the neural network into task-specific parameters, it still will lead to performance degradation compared to Recon.

Table 12: The distance between the layer permutations (rankings) obtained in different training stages on Multi-Fashion+MNIST dataset. “Iter.” denotes iterations. 

| Training Stage | 1st 25% Iter. | 2nd 25% Iter. | 3rd 25% Iter. | 4th 25% Iter. | All Iter. |
| -------------- | ------------- | ------------- | ------------- | ------------- | --------- |
| 1st 25% Iter.  | 0             | -             | -             | -             | -         |
| 2nd 25% Iter.  | 2.39          | 0             | -             | -             | -         |
| 3rd 25% Iter.  | 1.85          | 2.14          | 0             | -             | -         |
| 4th 25% Iter.  | 1.95          | 2.24          | 0.68          | 0             | -         |
| All Iter.      | 1.36          | 1.95          | 0.82          | 0.97          | 0         |

Table 13: Performance of the networks modified by Recon with conflict layers found in different training stages of joint-training on CityScapes dataset. Δ​m%\\Delta m\\% denotes the average relative improvement of all tasks. #P denotes the model size (MB). The best result is marked in bold. 

 Model 

Segmentation

Depth

Δ​m%\\Delta m\\%

#P.

(Higher Better)

(Lower Better)

mIoU

Pix Acc

Abs Err

Rel Err

Single-task

74.36

93.22

0.0128

29.98

190.59

1st 25% Iterations

74.17

93.21

0.0136

43.18

-12.63

108.439

2nd 25% Iterations

74.20

93.19

0.0135

42.45

-11.83

108.440

3rd 25% Iterations

74.80

93.19

0.0136

41.34

-10.90

109.567

4th 25% Iterations

74.80

93.19

0.0136

41.34

-10.90

109.567

All Iterations

74.80

93.19

0.0136

41.34

-10.90

109.567

Recon finds similar layers in different training stages. Recon ranks the network layers according to the computed SS-conflict scores. The ranking result can be represented as a layer permutation, denoted as π\\pi, and π⁡(l)\\pi(l) is the position of layer ll. The similarity between two rankings πi\\pi\_{i} and πj\\pi\_{j} can be measured as:

d⁡(πi,πj)\=1|𝕃|​∑l∈𝕃|πi​(l)−πj​(l)|,d(\\pi\_{i},\\pi\_{j})=\\frac{1}{|\\mathbb{L}|}\\sum\_{l\\in\\mathbb{L}}{|\\pi\_{i}(l)-\\pi\_{j}(l)|},

(22)

where 𝕃\\mathbb{L} denotes the set of neural network layers. In Table [12](#A3.T12 "Table 12 ‣ Appendix C Additional Ablation Study ‣ Recon: Reducing Conflicting Gradients from the Root for Multi-Task Learning"), we measure the differences in rankings obtained in different training stages (e.g., in the first 25% iterations or the second 25% iterations) on Multi-Fashion+MNIST by Eq. [22](#A3.E22 "In Appendix C Additional Ablation Study ‣ Recon: Reducing Conflicting Gradients from the Root for Multi-Task Learning"). The small distances (less than 2.4) indicate that the layers found in different training stages are quite similar. In Table [13](#A3.T13 "Table 13 ‣ Appendix C Additional Ablation Study ‣ Recon: Reducing Conflicting Gradients from the Root for Multi-Task Learning"), we compare the performance of the networks modified by Recon with conflict layers found in different training stages on CityScapes. It can be seen that the results of the last three rows are the same, which is because the layers found in the 3rd 25% iterations, 4th 25% iterations, and all iterations are *exactly the same* (the rankings may be slightly different though). The layers found in the later stages lead to slightly better performance than those found in the early stages (i.e., 1st 25% iterations and 2nd 25% iterations), indicating the conflict scores in early iterations might be a little noisy. However, since the performance gaps are acceptably small, to save time, we use the initial 25% training iterations to find conflict layers.

Table 14: The distance between the layer permutations (rankings) obtained by Recon with different methods on Multi-Fashion+MNIST dataset.

| Method | Joint-train | CAGrad | PCGrad | Gradrop | MGDA |
| ------ | ----------- | ------ | ------ | ------- | ---- |
| 0      | -           | -      | -      | -       |      |
| 1.07   | 0           | -      | -      | -       |      |
| 0.78   | 1.17        | 0      | -      | -       |      |
| 0.59   | 0.83        | 0.68   | 0      | -       |      |
| 1.71   | 1.32        | 1.90   | 1.56   | 0       |      |

Table 15: Multi-task learning results on NYUv2 dataset with SegNet as backbone. Recon∗\\ast denotes setting the layers found on CityScapes to task-specific. Δ​m%\\Delta m\\% denotes the average relative improvement of all tasks. #P denotes the model size (MB). The grey cell color indicates that Recon or Recon∗\\ast improves the result of the base model. 

Segmentation

Depth

Surface Normal

(Higher Better)

(Lower Better)

Angle Distance

(Lower Better)

Within t∘t^{\\circ}

(Higher Better)

Method

mIoU

Pix Acc

Abs Err

Rel Err

Mean

Median

11.25

22.5

30

Δ​m%↑\\Delta m\\%\\uparrow

#P.

Single-task

38.67

64.27

0.6881

0.2788

24.8683

18.9919

30.43

57.81

69.7

285.88

Joint-train

38.62

65.36

0.5378

0.2273

29.92

25.82

20.79

44.29

57.36

-1.62

95.58

w/ Recon

40.68

66.12

0.5786

0.2558

26.72

21.41

26.58

52.58

65.20

2.15

139.59

w/ Recon∗\\ast

38.81

63.69

0.5637

0.2413

26.75

21.73

26.16

51.80

64.64

1.59

121.59

MGDA

25.71

57.72

0.6033

0.2358

24.53

18.65

31.22

58.46

70.21

-2.15

95.58

w/ Recon

36.64

62.36

0.5613

0.2255

24.66

18.66

31.30

58.47

70.16

5.37

139.59

w/ Recon∗\\ast

36.85

63.51

0.5760

0.2362

24.89

18.96

30.53

57.94

69.82

4.34

121.59

Graddrop

39.01

66.13

0.5462

0.2296

29.72

25.51

19.87

44.68

58.12

-1.52

95.58

w/ Recon

39.78

65.63

0.5460

0.2280

26.42

21.16

26.89

53.16

65.84

4.45

139.59

w/ Recon∗\\ast

39.97

65.71

0.5544

0.2261

26.52

21.37

26.65

52.65

65.46

4.21

121.59

PCGrad

40.01

65.77

0.5349

0.2227

28.53

24.08

22.33

47.42

60.69

1.43

95.58

w/ Recon

40.03

65.92

0.5523

0.2384

26.24

20.89

27.30

53.66

66.25

4.19

139.59

w/ Recon∗\\ast

39.93

65.46

0.5494

0.2315

26.82

21.70

26.34

52.04

64.74

3.53

121.59

CAGrad

38.87

66.54

0.5331

0.2289

25.85

20.60

27.50

54.41

67.10

5.60

95.58

w/ Recon

40.68

66.12

0.5372

0.2266

25.44

19.87

28.96

56.00

68.28

6.99

139.59

w/ Recon∗\\ast

39.97

65.92

0.5298

0.2273

25.56

20.11

28.69

55.37

67.75

6.47

121.59

Table 16: Multi-task learning results on CityScapes dataset with MTAN as backbone. Recon∗\\ast denotes setting the layers found on NYUv2 to task-specific. Δ​m%\\Delta m\\% denotes the average relative improvement of all tasks. #P denotes the model size (MB). The grey cell color indicates that Recon or Recon∗\\ast improves the result of the base model.

Segmentation

Depth

(Higher Better)

(Lower Better)

Method

mIoU

Pix Acc

Abs Err

Rel Err

Δ​m%↑\\Delta m\\%\\uparrow

#P.

Single-task

73.74

93.05

0.0129

27.71

190.58

Joint-train

75.35

93.55

0.0169

45.64

-23.26

157.19

w/ Recon

75.72

93.74

0.0130

40.90

-11.36

196.32

w/ Recon∗\\ast

76.32

93.76

0.0132

46.40

-16.44

159.19

MGDA

70.46

91.75

0.0224

34.33

-26.02

157.19

w/ Recon

72.23

92.60

0.0122

26.93

1.37

196.32

w/ Recon∗\\ast

70.83

92.14

0.0125

25.69

1.31

159.19

Graddrop

75.19

93.53

0.0168

46.35

-23.90

157.19

w/ Recon

75.60

93.72

0.0127

38.55

-8.71

196.32

w/ Recon∗\\ast

76.49

93.82

0.0129

47.54

-16.81

159.19

PCGrad

75.64

93.54

0.02

43.53

-23.60

157.19

w/ Recon

75.89

93.71

0.0129

40.05

-10.35

196.32

w/ Recon∗\\ast

76.24

93.69

0.0128

45.24

-14.66

159.19

CAGrad

75.26

93.50

0.0176

44.23

-23.40

157.19

w/ Recon

75.65

93.71

0.0125

36.23

-6.15

196.32

w/ Recon∗\\ast

76.25

93.74

0.0123

40.05

-8.99

159.19

Recon finds similar layers with different MTL methods. In Table [14](#A3.T14 "Table 14 ‣ Appendix C Additional Ablation Study ‣ Recon: Reducing Conflicting Gradients from the Root for Multi-Task Learning"), we measure the differences in layer permutations (rankings) obtained by Recon with different methods (e.g., CAGrad and PCGrad) on Multi-Fashion+MNIST by Eq. [22](#A3.E22 "In Appendix C Additional Ablation Study ‣ Recon: Reducing Conflicting Gradients from the Root for Multi-Task Learning"). The small distances (less than 1.9) indicate that the layers found by Recon with different methods are quite similar. Therefore, in our experiments, we only use joint-training to search for the conflict layers once, and directly apply the modified network to improve different gradient manipulation methods as shown in Tables [1](#S5.T1 "Table 1 ‣ 5 Experiments ‣ Recon: Reducing Conflicting Gradients from the Root for Multi-Task Learning")-[5](#S5.T5 "Table 5 ‣ 5.2 Comparison with the State-of-the-Art ‣ 5 Experiments ‣ Recon: Reducing Conflicting Gradients from the Root for Multi-Task Learning").

The conflict layers found by Recon with the same architecture are transferable between different datasets. We conduct experiments with three different architectures: ResNet18, SegNet, and MTAN. (1) For Resnet18, we find that the layers found by Recon on CelebA and those found on Multi-Fashion+MNIST are *exactly the same*. (2) For SegNet, we find that 95% layers (38 out of 40) found on NYUv2 are identical to those found on CityScapes. On NYUv2, we compare the performance of using conflict layers found on NYUv2 (baselines w/ Recon) to that of using conflict layers found on CityScapes (i.e., baselines w/ Recon∗\\ast), as shown in Table [15](#A3.T15 "Table 15 ‣ Appendix C Additional Ablation Study ‣ Recon: Reducing Conflicting Gradients from the Root for Multi-Task Learning"). (3) For MTAN (SegNet with attention), we find that 68% layers (17 out of 25) found on CityScapes are identical to those found on NYUv2. On CityScapes, we compare the performance of using conflict layers found on CityScapes (baselines w/ Recon) to that of using conflict layers found on NYUv2 (i.e., baselines w/ Recon∗\\ast), as shown in Table [16](#A3.T16 "Table 16 ‣ Appendix C Additional Ablation Study ‣ Recon: Reducing Conflicting Gradients from the Root for Multi-Task Learning"). The results show that the conflict layers found on one dataset can be used to modify the network to be directly used on another dataset to consistently improve the performance of various baselines, while searching for the conflict layers again on the new dataset may lead to better performance.

Figure 11: Comparison of running time (one iteration, excludes data fetching) on CelebA dataset.

Analysis of running time. We evaluate how Recon scales with the number of tasks on CelebA dataset, by comparing the running time of one iteration used by Recon in computing gradient conflict scores (the most time-consuming part of Recon) to that of the baselines. The results in Fig. [11](#A3.F11 "Figure 11 ‣ Appendix C Additional Ablation Study ‣ Recon: Reducing Conflicting Gradients from the Root for Multi-Task Learning") show that Recon is as fast as other gradient manipulation methods such as CAGrad ([Liu et al., 2021a](#bib.bib22 "")) and Graddrop ([Chen et al., 2020](#bib.bib6 "")), but much slower than joint-training especially when the number of tasks is large, which is natural since Recon needs to compute pariwise cosine similarity of task gradients. However, since Recon only needs to search for the conflict layers once for a given network architecture, as discussed above, the running time is not a problem.

[](javascript:toggleReadingMode\(\); "Disable reading mode, show header and footer")