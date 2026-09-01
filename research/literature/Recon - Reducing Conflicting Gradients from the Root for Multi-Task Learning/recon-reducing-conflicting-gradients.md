# RECON: REDUCING CONFLICTING GRADIENTS FROM THE ROOT FOR MULTI-TASK LEARNING

Guangyuan Shi, Qimai Li, Wenlong Zhang, Jiaxin Chen, Xiao-Ming Wu

Department of Computing, The Hong Kong Polytechnic University, Hong Kong S.A.R., China {guang-yuan.shi, qee-mai.li, wenlong.zhang}@connect.polyu.hk, jiax.chen@connect.polyu.hk, xiao-ming.wu@polyu.edu.hk

### ABSTRACT

A fundamental challenge for multi-task learning is that different tasks may conflict with each other when they are solved jointly, and a cause of this phenomenon is *conflicting gradients* during optimization. Recent works attempt to mitigate the influence of conflicting gradients by directly altering the gradients based on some criteria. However, our empirical study shows that "gradient surgery" cannot effectively reduce the occurrence of conflicting gradients. In this paper, we take a different approach to reduce conflicting gradients *from the root*. In essence, we investigate the task gradients w.r.t. each *shared* network layer, select the layers with high conflict scores, and turn them to *task-specific* layers. Our experiments show that such a simple approach can greatly reduce the occurrence of conflicting gradients in the remaining shared layers and achieve better performance, with only a slight increase in model parameters in many cases. Our approach can be easily applied to improve various state-of-the-art methods including gradient manipulation methods and branched architecture search methods. Given a network architecture (e.g., ResNet18), it only needs to search for the conflict layers once, and the network can be modified to be used with different methods on the same or even different datasets to gain performance improvement. The source code is available at <https://github.com/moukamisama/Recon>.

# 1 INTRODUCTION

Multi-task learning (MTL) is a learning paradigm in which multiple different but correlated tasks are jointly trained with a shared model [\(Caruana, 1997\)](#page-9-0), in the hope of achieving better performance with an overall smaller model size than learning each task independently. By discovering shared structures across tasks and leveraging domain-specific training signals of related tasks, MTL can achieve efficiency and effectiveness. Indeed, MTL has been successfully applied in many domains including natural language processing [\(Hashimoto et al., 2017\)](#page-10-0), reinforcement learning [\(Parisotto](#page-10-1) [et al., 2016;](#page-10-1) [D'Eramo et al., 2020\)](#page-9-1) and computer vision [\(Vandenhende et al., 2021\)](#page-11-0).

A major challenge for multi-task learning is *negative transfer* [\(Ruder, 2017\)](#page-11-1), which refers to the performance drop on a task caused by the learning of other tasks, resulting in worse overall performance than learning them separately. This is caused by *task conflicts*, i.e., tasks compete with each other and unrelated information of individual tasks may impede the learning of common structures. From the optimization point of view, a cause of negative transfer is *conflicting gradients* [\(Yu et al.,](#page-11-2) [2020\)](#page-11-2), which refers to two task gradients pointing away from each other and the update of one task will have a negative effect on the other. Conflicting gradients make it difficult to optimize the multitask objective, since task gradients with larger magnitude may dominate the update vector, making the optimizer prioritize some tasks over others and struggle to converge to a desirable solution.

Prior works address task/gradient conflicts mainly by balancing the tasks via task reweighting or gradient manipulation. Task reweighting methods adaptively re-weight the loss functions by homoscedastic uncertainty [\(Kendall et al., 2018\)](#page-10-2), balancing the pace at which tasks are learned [Chen](#page-9-2) [et al.](#page-9-2) [\(2018\)](#page-9-2); [Liu et al.](#page-10-3) [\(2019\)](#page-10-3), or learning a loss weight parameter [\(Liu et al., 2021b\)](#page-10-4). Gradient manipulation methods reduce the influence of conflicting gradients by directly altering the gradients based on different criteria [\(Sener & Koltun, 2018;](#page-11-3) [Yu et al., 2020;](#page-11-2) [Chen et al., 2020;](#page-9-3) [Liu et al.,](#page-10-5) [2021a\)](#page-10-5) or rotating the shared features [\(Javaloy & Valera, 2022\)](#page-10-6). While these methods have demonstrated effectiveness in different scenarios, in our empirical study, we find that they cannot reduce the occurrence of conflicting gradients (see Sec. [3.3](#page-2-0) for more discussion).

We propose a different approach to reduce conflicting gradients for MTL. Specifically, we investigate layer-wise conflicting gradients, i.e., the task gradients w.r.t. each shared network layer. We first train the network with a regular MTL algorithm (e.g., joint-training) for a number of iterations, compute the conflict scores for all shared layers, and select those with highest conflict scores (indicating severe conflicts). We then set the selected shared layers task-specific and train the modified network from scratch. As demonstrated by comprehensive experiments and analysis, our simple approach Recon has the following key advantages: (1) Recon can greatly reduce conflicting gradients with only a slight increase in model parameters (less than 1% in some cases) and lead to significantly better performance. (2) Recon can be easily applied to improve various gradient manipulation methods and branched architecture search methods. Given a network architecture, it only needs to search for the conflict layers once, and the network can be modified to be used with different methods and even on different datasets to gain performance improvement. (3) Recon can achieve better performance than branched architecture search methods with a much smaller model.

# 2 RELATED WORKS

In this section, we briefly review related works in multi-task learning in four categories: tasks clustering, architecture design, architecture search, and task balancing. *Tasks clustering methods* mainly focus on identifying which tasks should be learned together [\(Thrun & O'Sullivan, 1996;](#page-11-4) [Zamir et al.,](#page-11-5) [2018;](#page-11-5) [Standley et al., 2020;](#page-11-6) [Shen et al., 2021;](#page-11-7) [Fifty et al., 2021\)](#page-9-4).

*Architecture design methods* include hard parameter sharing methods [\(Kokkinos, 2017;](#page-10-7) [Long et al.,](#page-10-8) [2017;](#page-10-8) [Bragman et al., 2019\)](#page-9-5), which learn a shared feature extractor and task-specific decoders, and soft parameters sharing methods [\(Misra et al., 2016;](#page-10-9) [Ruder et al., 2019;](#page-11-8) [Gao et al., 2019;](#page-9-6) [2020;](#page-9-7) [Liu](#page-10-3) [et al., 2019\)](#page-10-3), where some parameters of each task are assigned to do cross-task talk via a sharing mechanism. Compared with soft parameters sharing methods, our approach Recon has much better scalability when dealing with a large number of tasks.

Instead of designing a fixed network structure, some methods [\(Rosenbaum et al., 2018;](#page-11-9) [Meyerson &](#page-10-10) [Miikkulainen, 2018;](#page-10-10) [Yang et al., 2020\)](#page-11-10) propose to dynamically self-organize the network for different tasks. Among them, *branched architecture search* [\(Guo et al., 2020;](#page-9-8) [Bruggemann et al., 2020\)](#page-9-9) methods are more related to our work. They propose an automated architecture search algorithm to build a tree-structured network by learning where to branch. In contrast, our method Recon decides which layers to be shared across tasks by considering the severity of layer-wise conflicting gradients, resulting in a more compact architecture with lower time cost and better performance.

Another line of research is *task balancing* methods. To address task/gradient conflicts, some methods attempt to re-weight the multi-task loss function using homoscedastic uncertainty [\(Kendall et al.,](#page-10-2) [2018\)](#page-10-2), task prioritization [\(Guo et al., 2018\)](#page-9-10), or similar learning pace [\(Liu et al., 2019;](#page-10-3) [2021b\)](#page-10-4). GradNorm [\(Chen et al., 2018\)](#page-9-2) learns task weights by dynamically tuning gradient magnitudes. MGDA [\(Sener & Koltun, 2018\)](#page-11-3) find the weights by minimizing the norm of the weighted sum of task gradients. To reduce the influence of conflicting gradients, PCGrad [\(Yu et al., 2020\)](#page-11-2) projects each gradient onto the normal plane of another gradient and uses the average of projected gradients for update. Graddrop [\(Chen et al., 2020\)](#page-9-3) randomly drops some elements of gradients based on element-wise conflict. CAGrad [\(Liu et al., 2021a\)](#page-10-5) ensures convergence to a minimum of the average loss across tasks by gradient manipulation. RotoGrad [\(Javaloy & Valera, 2022\)](#page-10-6) re-weights task gradients and rotates the shared feature space. Instead of manipulating gradients, our method Recon leverages gradient information to modify network structure to mitigate task conflicts from the root.

# 3 PILOT STUDY: TASK CONFLICTS IN MULTI-TASK LEARNING

# 3.1 MULTI-TASK LEARNING: PROBLEM DEFINITION

Multi-task learning (MTL) aims to learn a set of correlated tasks {Ti} T <sup>i</sup>=1 simultaneously. For each task T<sup>i</sup> , the empirical loss function is Li(θsh, θi), where θsh are parameters shared among all tasks

![](_page_2_Figure_1.jpeg)

Figure 1: The distributions of gradient conflicts (in terms of cos φij ) of the joint-training baseline and state-of-the-art gradient manipulation methods on Multi-Fashion+MNIST benchmark.

and θ<sup>i</sup> are task-specific parameters. The goal is to find optimal parameters θ = {θsh, θ1, θ2, · · · , θ<sup>T</sup> } to achieve high performance across all tasks. Formally, it aims to minimize a multi-task objective:

<span id="page-2-2"></span><span id="page-2-1"></span>θ <sup>∗</sup> = arg min θ X T i wiLi(θsh, θi), (1)

where w<sup>i</sup> are pre-defined or dynamically computed weights for different tasks. A popular choice is to use the average loss (i.e., equal weights). However, optimizing the multi-task objective is difficult, and a known cause is conflicting gradients.

#### 3.2 CONFLICTING GRADIENTS

Let g<sup>i</sup> = ∇<sup>θ</sup>sh Li(θsh, θi) denote the gradient of task T<sup>i</sup> w.r.t. the shared parameters θsh (i.e., a vector of the partial derivatives of L<sup>i</sup> w.r.t. θsh) and g ts <sup>i</sup> = ∇<sup>θ</sup>iLi(θsh, θi) denote the gradient w.r.t. the task-specific parameters θ<sup>i</sup> . A small change of θsh in the direction of negative g<sup>i</sup> is θsh ← θsh−αg<sup>i</sup> , with a sufficiently small step size α. The effect of this change on the performance of another task T<sup>j</sup> is measured by:

∆L<sup>j</sup> = L<sup>j</sup> (θsh − αg<sup>i</sup> , θ<sup>j</sup> ) − L<sup>j</sup> (θsh, θ<sup>j</sup> ) = −αg<sup>i</sup> · g<sup>j</sup> + o(α), (2)

where the second equality is obtained by first order Taylor approximation. Likewise, the effect of a small update of θsh in the direction of the negative gradient of task T<sup>j</sup> (i.e., −g<sup>j</sup> ) on the performance of task T<sup>i</sup> is ∆L<sup>i</sup> = −αg<sup>i</sup> · g<sup>j</sup> + o(α). Notably, the model update for task T<sup>i</sup> is considered to have a negative effect on task T<sup>j</sup> when g<sup>i</sup> · g<sup>j</sup> < 0, since it increases the loss of task T<sup>j</sup> , and vice versa. A formal definition of conflicting gradients is given as follows [\(Yu et al., 2020\)](#page-11-2).

Definition 1 (Conflicting Gradients). *The gradients* g<sup>i</sup> *and* g<sup>j</sup> (i 6= j) *are said to be conflicting with each other if* cos φij < 0*, where* φij *is the angle between* g<sup>i</sup> *and* g<sup>j</sup> *.*

As shown in [Yu et al.](#page-11-2) [\(2020\)](#page-11-2), conflicts in gradient pose serious challenges for optimizing the multitask objective (Eq. [1\)](#page-2-1). Using the average gradient (i.e., <sup>1</sup> T P<sup>T</sup> <sup>i</sup>=1 gi) for gradient decent may hurt the performance of individual tasks, especially when there is a large difference in gradient magnitudes, which will make the optimizer struggle to converge to a desirable solution.

#### <span id="page-2-0"></span>3.3 GRADIENT SURGERY CANNOT EFFECTIVELY REDUCE CONFLICTING GRADIENTS

To mitigate the influence of conflicting gradients, several methods [\(Yu et al., 2020;](#page-11-2) [Chen et al.,](#page-9-3) [2020;](#page-9-3) [Liu et al., 2021a\)](#page-10-5) have been proposed to perform "gradient surgery". Instead of following the average gradient direction, they alter conflicting gradients based on some criteria and use the modified gradients for model update. We conduct a pilot study to investigate whether gradient manipulation can effectively reduce the occurrence of conflicting gradients. For each training iteration, we first calculate the task gradients of all tasks w.r.t. the shared parameters (i.e., g<sup>i</sup> for any task i) and compute the conflict angle between any two task gradients g<sup>i</sup> and g<sup>j</sup> in terms of cosφij . We then count and draw the distribution of cosφij in all training iterations. We provide the statistics of the joint-training baseline (i.e., training all tasks jointly with equal loss weights and all parameters shared) and several state-of-the-art gradient manipulation methods including GradDrop [\(Chen et al.,](#page-9-3) [2020\)](#page-9-3), PCGrad [\(Yu et al., 2020\)](#page-11-2), CAGrad [\(Liu et al., 2021a\)](#page-10-5), and MGDA [\(Sener & Koltun, 2018\)](#page-11-3) on Multi-Fashion+MNIST [\(Lin et al., 2019\)](#page-10-11), CityScapes, NYUv2, and PASCAL-Context datasets.

![](_page_3_Diagram_1.jpeg)

Figure 2: Illustration of the differences between joint-training, gradient manipulation, and our approach. (a) In joint-training, the update vector (in green) is the average gradient <sup>1</sup> 2 (g<sup>i</sup> + g<sup>j</sup> ). Due to the conflict between g<sup>i</sup> and g<sup>j</sup> , the update vector is dominated by g<sup>i</sup> (in red). (b) PCGrad [\(Yu](#page-11-2) [et al., 2020\)](#page-11-2) projects each gradient onto the normal plane of the other one and uses the average of the projected gradients (indicated by dashed grey arrows) as the update vector (in green). As such, the update vector is less dominated by g<sup>i</sup> . (c) Our approach Recon finds the parameters contributing most (e.g., θ3) to gradient conflicts and turns them into task specific ones. In effect, it performs an orthographic/coordinate projection of conflicting gradients to the space of the rest parameters (e.g., θ<sup>1</sup> and θ2) such that the projected gradients g fix i and g fix j are better aligned. (d) Illustration of Recon turning a shared layer with high conflict score to task-specific layers.

The results are provided in Fig. [1,](#page-2-2) Fig. [5,](#page-14-0) Fig. [6,](#page-14-1) Fig. [7,](#page-14-2) Table [6,](#page-8-0) and Tables [8-](#page-15-0)[10.](#page-16-0) It can be seen that gradient manipulation methods can only slightly reduce the occurrence of conflicting gradients (compared to joint-training) in some cases, and in some other cases they even increase it.

## 4 OUR APPROACH: REDUCING CONFLICTING GRADIENTS FROM THE ROOT

Our pilot study shows that adjusting gradients for model update cannot effectively prevent the occurrence of conflicting gradients in MTL, which suggests that the root causes of this phenomenon may be closely related to the nature of different tasks and the way how model parameters are shared among them. Therefore, to mitigate task conflicts for MTL, in this paper, we take a different approach to reduce the occurrence of conflicting gradients from the root.

#### 4.1 RECON: REMOVING LAYER-WISE CONFLICTING GRADIENTS

Our approach is extremely simple and intuitive. We first identify the shared network layers where conflicts occur most frequently and then turn them into task-specific parameters. Suppose the shared model parameters θsh are composed of n layers, i.e., θsh = {θ (k) sh } n <sup>k</sup>=1, where θ (k) sh is the k th shared layer. Let g (k) i denote the gradient of task T<sup>i</sup> w.r.t. the k th shared layer θ (k) sh , i.e., g (k) i is a vector of the partial derivatives of L<sup>i</sup> w.r.t. the parameters of θ (k) sh . Let φ (k) ij denote the angle between g (k) i and g (k) j . We define layer-wise conflicting gradients and S-conflict score as follows.

Definition 2 (Layer-wise Conflicting Gradients). *The gradients* g (k) i *and* g (k) j *(*i 6= j*) are said to be conflicting with each other if* cos φ (k) ij < 0*.*

Definition 3 (S-Conflict Score). *For any* −1 < S ≤ 0*, the* S*-conflict score for the* k th *shared layer is the number of different pairs* (i, j)(i 6= j*) s.t.* cos φ (k) ij < S*, denoted as* s (k) *.*

S indicates the severity of conflicts, and setting S smaller means we care about cases of more severe conflicts. The S-conflict score s (k) indicates the occurrence of conflicting gradients at severity level S for the k th shared layer. If s (k) = T 2 , it means that for any two different tasks, there is a conflict in their gradients w.r.t. the k th shared layer. By computing S-conflict scores, we can identify the shared layers where conflicts occur most frequently.

We describe our method Recon in Algorithm [1.](#page-4-0) First, we train the network for I iterations and compute S-conflict scores for each shared layer θ (k) in every iteration, denoted by {s (k) i } I <sup>i</sup>=1. Then,

### Algorithm 1: Recon: Removing Layer-wise Conflicting Gradients

<span id="page-4-0"></span>Input: Model parameters θ, learning rate α, a set of tasks {Ti} T <sup>i</sup>=1, number of iterations I for computing conflict scores, conflict severity level S, number of selected layers K.

// Train the network and compute conflict scores for all layers for iteration *i =* 1, 2, . . . , I do

for *i =* 1 , 2, . . . , T do

Compute the gradients of task T<sup>i</sup> w.r.t. all shared layers, i.e., {g (k) i } n <sup>k</sup>=1 ;

end

Calculate the S-conflict scores for all shared layers in the current iteration, i.e., {s (k) i } n <sup>k</sup>=1; Update θ with joint-training or any gradient manipulation method ;

end

// Set layers with top conflict scores task-specific For each layer k, calculate the sum of S-conflict scores in all iterations, i.e., s (k) = P<sup>I</sup> <sup>i</sup>=1 s (k) i ; Select the top K layers with highest s (k) and set them task-specific;

// Train the modified network from scratch

for iteration *i =* 1, 2, . . . do Update θ with joint-training or any gradient manipulation method;

end

Output: Model parameters θ.

we sum up the scores in all iterations, i.e., s (k) = P<sup>I</sup> <sup>i</sup>=1 s (k) i , and find the layers with highest s (k) scores. Next, we set these layers to be task-specific and train the modified network from scratch. We demonstrate the effectiveness of Recon by a theoretical analysis in Sec. [4.2](#page-4-1) and comprehensive experiments in Sec. [5.](#page-5-0) The results show that Recon can effectively reduce the occurrence of conflicting gradients in the remaining shared layers and lead to substantial improvements over state-of-the-art.

### <span id="page-4-1"></span>4.2 THEORETICAL ANALYSIS

Here, we provide a theoretical analysis of Recon. Let θsh = {θ fix sh , θcf sh}, where θ fix sh are the remaining shared parameters, and θ cf sh are those that will be turned to task-specific parameters θ cf 1 , θcf 2 , · · · , θcf T . Notice that θ cf 1 , θcf 2 , · · · , θcf <sup>T</sup> will all be initialized with θ cf sh. Therefore, after applying Recon, the model parameters are θ<sup>r</sup> = {θ fix sh , θcf , . . . , θcf T , θts 1 , . . . , θts T }. An one-step gradient update of θ<sup>r</sup> is:

ˆθ fix sh = θ fix sh − α X T i=1 wig fix i , ˆθ cf <sup>i</sup> = θ cf <sup>i</sup> − αg cf i , ˆθ ts <sup>i</sup> = θ ts <sup>i</sup> − αg ts i , i = 1, . . . , T, (3)

where w<sup>i</sup> are weight parameters, g ts <sup>i</sup> = ∇<sup>θ</sup> ts Li , g cf <sup>i</sup> = ∇<sup>θ</sup> cf sh L<sup>i</sup> and g fix <sup>i</sup> = ∇<sup>θ</sup> fix sh Li . Notice that different methods such as joint-training, MGDA [Sener & Koltun](#page-11-3) [\(2018\)](#page-11-3), PCGrad [Yu et al.](#page-11-2) [\(2020\)](#page-11-2), and CAGrad [Liu et al.](#page-10-5) [\(2021a\)](#page-10-5) choose different w<sup>i</sup> dynamically.

Without applying Recon, the model parameters are θ = {θ fix sh , θcf sh, θts 1 , . . . , θts T }. An one-step gradient update of θ is given by

ˆθ fix sh = θ fix sh − α X T i=1 wig fix i , ˆθ cf sh = θ cf sh − α X T i=1 wig cf i , ˆθ ts <sup>i</sup> = θ ts <sup>i</sup> − αg ts i , i = 1, . . . , T. (4)

After the one-step updates, the loss functions with the updated parameters ˆθ<sup>r</sup> and ˆθ respectively are:

L( <sup>ˆ</sup>θr) = X T i=1 Li ˆθ fix sh , ˆθ cf i , ˆθ ts i , and, L( <sup>ˆ</sup>θ) = X T i=1 Li ˆθ fix sh , ˆθ cf sh, ˆθ ts i , (5)

where L<sup>i</sup> is the loss function of task T<sup>i</sup> . Denote the set of indices of the layers turned task-specific by P, then θ cf sh = {θ (k) sh }, k <sup>∈</sup> <sup>P</sup>. Assume that P<sup>T</sup> <sup>i</sup>=1 w<sup>i</sup> = 1, then we have the following theorem.

<span id="page-5-1"></span>Table 1: Multi-task learning results on Multi-Fashion+MNIST dataset. All experiments are repeated over 3 random seeds and the mean values are reported. ∆m% denotes the average relative improvement of all tasks. #P denotes model size (MB). The grey cell color indicates that Recon improves the result of the base model. The best average result is marked in bold.

<span id="page-5-2"></span>Table 2: Multi-task learning results on CelebA dataset. All experiments are repeated over 3 random seeds and the mean values are reported. ∆m% denotes the average relative improvement of all tasks. #P denotes model size (MB). The grey cell color indicates that Recon improves the result of the base model. The best average result is marked in bold.

Theorem 4.1. *Assume that* L *is differentiable and for any two different tasks* T<sup>i</sup> *and* T<sup>j</sup> *, it satisfies*

cos φ (k) ij kg (k) i k < kg (k) j k, ∀k ∈ <sup>P</sup> (6)

*then for any sufficiently small learning rate* α > 0*,*

L( ˆθr) < L( ˆθ). (7)

The theorem indicates that a single gradient update on the model parameters of Recon achieves lower loss than that on the original model parameters. The proof is provided in Appendix [A](#page-12-0)

# <span id="page-5-0"></span>5 EXPERIMENTS

In this section, we conduct extensive experiments to evaluate our approach Recon for multi-task learning and demonstrate its effectiveness, efficiency and generality.

### 5.1 EXPERIMENTAL SETUP

Datasets. We evaluate Recon on 4 multi-task datasets, namely Multi-Fashion+MNIST [\(Lin et al.,](#page-10-11) [2019\)](#page-10-11), CityScapes [\(Cordts et al., 2016\)](#page-9-11), NYUv2 [\(Couprie et al., 2013\)](#page-9-12), PASCAL-Context [\(Mot](#page-10-12)[taghi et al., 2014\)](#page-10-12), and CelebA [\(Liu et al., 2015\)](#page-10-13). The tasks of each dataset are described as follows. 1) Multi-Fashion+MNIST contains two image classification tasks. Each image consists of an item from FashionMNIST and an item from MNIST. 2) CityScapes contains 2 vision tasks: 7-class semantic segmentation and depth estimation. 3) NYUv2 contains 3 tasks: 13-class semantic segmentation, depth estimation and normal prediction. 4) PASCAL-Context consists of 5 tasks: semantic segmentation, human parts segmentation and saliency estimation, surface normal estimation, and edge detection. 5) CelebA contains 40 binary classification tasks.

Baselines. The baselines include 1) single-task learning (single-task): training all tasks independently; 2) joint-training (joint-train): training all tasks together with equal loss weights and all parameters shared; 3) gradient manipulation methods: MGDA [\(Sener & Koltun, 2018\)](#page-11-3), PCGrad [\(Yu](#page-11-2) [et al., 2020\)](#page-11-2), GradDrop [\(Chen et al., 2020\)](#page-9-3), CAGrad [\(Liu et al., 2021a\)](#page-10-5), RotoGrad [\(Javaloy & Valera,](#page-10-6) [2022\)](#page-10-6); 4) branched architecture search methods: BMTAS [\(Bruggemann et al., 2020\)](#page-9-9); 5) Architecture design methods: Cross-Stitch [\(Misra et al., 2016\)](#page-10-9), MMoE [\(Ma et al., 2018\)](#page-10-14). Following [Liu](#page-10-5) [et al.](#page-10-5) [\(2021a\)](#page-10-5), we implement Cross-Stitch based on SegNet [\(Badrinarayanan et al., 2017\)](#page-9-13). For a fair comparison, all methods use same configurations and random seeds. We run all experiments 3 times with different random seeds. More experimental details are provided in Appendix [B.](#page-12-1)

Relative task improvement. Following [Maninis et al.](#page-10-15) [\(2019\)](#page-10-15), we compute the relative task improvement with respect to the single-task baseline for each task. Given a task T<sup>j</sup> , the relative task

<span id="page-6-0"></span>Table 3: Multi-task learning results on CityScapes dataset. All experiments are repeated over 3 random seeds and the mean values are reported. ∆m% denotes the average relative improvement of all tasks. #P denotes the model size (MB). The grey cell color indicates that Recon improves the result of the base model. The best average result is marked in bold.

![](_page_6_Figure_2.jpeg)

<span id="page-6-1"></span>Figure 3: The performance of CAGrad combined with Recon on the Multi-Fashion+MNIST benchmark with (a) different number of selected layers K (b) different severity value S for computing conflict scores.

improvement is ∆mT<sup>j</sup> = K P<sup>K</sup> <sup>i</sup>=1(−1)<sup>l</sup><sup>i</sup> (M<sup>i</sup> −Si)/S<sup>i</sup> , where M<sup>i</sup> , S<sup>i</sup> refer to metrics for the i th criterion obtained by objective model and single-task model respectively, l<sup>i</sup> = 1 if a lower value for the criterion is better and 0 otherwise. The average relative task improvement is ∆m = T P<sup>T</sup> <sup>j</sup>=1 ∆mT<sup>j</sup> .

#### <span id="page-6-2"></span>5.2 COMPARISON WITH THE STATE-OF-THE-ART

Recon improves the performance of all base models. The main results on Multi-Fashion+MNIST, and CelebA, CityScapes, PASCAL-Context, and NYUv2, are presented in Table [1,](#page-5-1) Table [2,](#page-5-2) Table [3,](#page-6-0) Table [4,](#page-7-0) and Table [5](#page-7-1) respectively. (1) Compared to gradient manipulation methods, Recon consistently improves their performance in most evaluation metrics, and achieve comparable performance on the rest of evaluation metrics. (2) Compared with branched architecture search methods and architecture design methods, Recon can further improve the performance of BMTAS and MMoE. Besides, Recon combined with other gradient manipulation methods with small model size can achieve better results than branched architecture search methods with much bigger models.

Small increases in model parameters can lead to good performance gains. Note that Recon only changes a small portion of shared parameters to task-specific. As shown in Table [1](#page-5-1)[-5,](#page-7-1) Recon increases the model size by 0.52% to 57.25%. Recon turns 1.42%, 1.46%, 12.77%, 0.26%, 9.80% shared parameters to task-specific on Multi-Fashion+MNIST, CelebA, CityScapes, NYUv2 and PASCAL-Context respectively. The results suggest that the gradient conflicts in a small portion (less than 13%) of shared parameters impede the training of the model for multi-task learning.

Recon is compatible with various neural network architectures. We use ResNet18 on Multi-Fashion+MNIST, SegNet [\(Badrinarayanan et al., 2017\)](#page-9-13) on CityScapes, MTAN [\(Liu et al., 2019\)](#page-10-3) on NYUv2, and MobileNetV2 [\(Sandler et al., 2018\)](#page-11-11) on PASCAL-Context. Recon improves the performance of baselines with different neural network architectures, including the architecture search method BMTAS [\(Bruggemann et al., 2020\)](#page-9-9) which finds a tree-like structure for multi-task learning.

Only one search of conflict layers is needed for the same network architecture. An interesting observation from our experiments is that network architecture seems to be the deciding factor for the conflict layers found by Recon. With the same network architecture (e.g., ResNet18), the found conflict layers are quite consistent w.r.t. (1) different training stages (e.g., the first 25% iterations, or the middle or last ones) (see Table [12](#page-17-0) and Table [13](#page-17-1) and discussion in Appendix [C\)](#page-14-3), (2) different MTL methods (e.g., joint-training or gradient manipulation methods) (see Table [14](#page-17-2) and discussion in Appendix [C\)](#page-14-3), and (3) different datasets (see Table [15](#page-18-0) and Table [16](#page-18-1) and discussion in Appendix [C\)](#page-14-3).

<span id="page-7-0"></span>Table 4: Multi-task learning results on PASCAL-Context dataset with 4-task setting. All experiments are repeated over 3 random seeds and the mean values are reported. ∆m% denotes the average relative improvement of all tasks. #P denotes the model size (MB). The grey cell color indicates Recon improves the result of the base model. The best average result is marked in bold.

<span id="page-7-1"></span>Table 5: Multi-task learning results on NYUv2 dataset with MTAN as backbone. All experiments are repeated over 3 random seeds and the mean values are reported. ∆m% denotes the average relative improvement of all tasks. #P denotes the model size (MB). The grey cell color indicates that Recon improves the result of the base model. The best average result is marked in bold.

Hence, in our experiments, we only search for the conflict layers *once* with the joint-training baseline in the first 25% training iterations and modify the network to improve various methods on the same dataset. We also find that the conflict layers found on one dataset can be used to modify the network to be directly applied on another dataset to gain performance improvement.

#### 5.3 ABLATION STUDY AND ANALYSIS

Recon greatly reduces the occurrence of conflicting gradients. In Fig. [4](#page-8-1) and Table [6,](#page-8-0) we compare the distribution of cos φij before and after applying Recon on Multi-Fashion+MNIST (the results on other datasets are provided in Appendix [C\)](#page-14-3). It can be seen that Recon greatly reduces the numbers of gradient pairs with severe conflicts (cos φij ∈ (−0.01, −1]) by at least 67% and up to 79% when compared with joint-training, while gradient manipulation methods only slightly reduce the percentage and some even increases it. Similar observations can be made from Tables [8](#page-15-0)[-10.](#page-16-0)

Randomly selecting conflict layers does not work. To show that the performance gain of Recon comes from selecting the layers with most severe conflicts instead of merely increasing model parameters, we further compare Recon with the following two baselines. RSL: randomly selecting

<span id="page-8-1"></span>![](_page_8_Figure_1.jpeg)

Figure 4: The distribution of gradient conflicts (in terms of cos φij ) of baselines and baselines with Recon on Multi-Fashion+MNIST dataset.

<span id="page-8-0"></span>Table 6: The distribution of gradient conflicts (in terms of cos φij ) w.r.t. the shared parameters on Multi-Fashion+MNIST dataset. "Reduction" means the percentage of conflicting gradients in the interval of (−0.01, −1.0] reduced by the model compared with joint-training. The grey cell color indicates Recon greatly reduces the conflicting gradients (more than 50%). In contrast, gradient manipulation methods only slightly decrease their occurrence, and some method even increases it.

<span id="page-8-2"></span>Table 7: Comparison of Recon with RSL and RSP. PD: performance drop compared to Recon.

same number of layers as Recon and set them task-specific. RSP: randomly selecting similar amount of parameters as Recon and set them task-specific. The results in Table [7](#page-8-2) show that both RSL and RSP lead to significant performance drops, which verifies the effectiveness of the selection strategy of Recon. We compare Recon with the baselines that selects the first or last K layers in Appendix [C.](#page-14-3)

Ablation study on hyperparameters. We study the influence of the conflict severity S and the number of selected layers K on the performance of CAGrad w/ Recon on Multi-Fashion+MNIST. As shown in Fig. [3,](#page-6-1) a small K leads to a significant performance drop, which indicates that there are still some shared network layers suffering from severe gradient conflicts, while a large K will not lead to further performance improvement since severe conflicts have been resolved. For the conflict severity S, we find that a high value of S (e.g., 0.0) leads to performance drops since it includes too many gradient pairs with small conflicts, while some of them are helpful for learning common structures and should not be removed. In the meantime, a too small S (e.g., −0.15) also leads to performance degradation because it ignores too many gradient pairs with large conflicts, which may be detrimental to learning. While K and S are sensitive, we may only need to tune them once for a given network architecture, as discussed in Sec. [5.2.](#page-6-2)

# 6 CONCLUSION

We have proposed a very simple yet effective approach, namely Recon, to reduce the occurrence of conflicting gradients for multi-task learning. By considering layer-wise gradient conflicts and identifying the shared layers with severe conflicts and setting them task-specific, Recon can significantly reduce the occurrence of severe conflicting gradients and boost the performance of existing methods with only a reasonable increase in model parameters. We have demonstrated the effectiveness, efficiency, and generality of Recon via extensive experiments and analysis.

### ACKNOWLEDGMENTS

The authors would like to thank Lingzi Jin for checking the proof of Theorem [A.1](#page-12-2) and the anonymous reviewers for their insightful and helpful comments.

# REFERENCES

<span id="page-9-14"></span><span id="page-9-13"></span><span id="page-9-12"></span><span id="page-9-11"></span><span id="page-9-10"></span><span id="page-9-9"></span><span id="page-9-8"></span><span id="page-9-7"></span><span id="page-9-6"></span><span id="page-9-5"></span><span id="page-9-4"></span><span id="page-9-3"></span><span id="page-9-2"></span><span id="page-9-1"></span><span id="page-9-0"></span>Vijay Badrinarayanan, Alex Kendall, and Roberto Cipolla. Segnet: A deep convolutional encoderdecoder architecture for image segmentation. *IEEE transactions on pattern analysis and machine intelligence*, 39(12):2481–2495, 2017. Felix JS Bragman, Ryutaro Tanno, Sebastien Ourselin, Daniel C Alexander, and Jorge Cardoso. Stochastic filter groups for multi-task cnns: Learning specialist and generalist convolution kernels. In *Proceedings of the IEEE/CVF International Conference on Computer Vision*, pp. 1385–1394, 2019. David Bruggemann, Menelaos Kanakis, Stamatios Georgoulis, and Luc Van Gool. Automated search for resource-efficient branched multi-task networks. *British Machine Vision Conference (BMVC)*, 2020. Rich Caruana. Multitask learning. *Machine learning*, 28(1):41–75, 1997. Zhao Chen, Vijay Badrinarayanan, Chen-Yu Lee, and Andrew Rabinovich. Gradnorm: Gradient normalization for adaptive loss balancing in deep multitask networks. In *International Conference on Machine Learning*, pp. 794–803. PMLR, 2018. Zhao Chen, Jiquan Ngiam, Yanping Huang, Thang Luong, Henrik Kretzschmar, Yuning Chai, and Dragomir Anguelov. Just pick a sign: Optimizing deep multitask models with gradient sign dropout. *Advances in Neural Information Processing Systems*, 33:2039–2050, 2020. Marius Cordts, Mohamed Omran, Sebastian Ramos, Timo Rehfeld, Markus Enzweiler, Rodrigo Benenson, Uwe Franke, Stefan Roth, and Bernt Schiele. The cityscapes dataset for semantic urban scene understanding. In *Proceedings of the IEEE/CVF conference on computer vision and pattern recognition*, 2016. Camille Couprie, Clement Farabet, Laurent Najman, and Yann LeCun. Indoor semantic segmenta- ´ tion using depth information. *CoRR*, abs/1301.3572, 2013. Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. Imagenet: A large-scale hierarchical image database. In *Proceedings of the IEEE/CVF conference on computer vision and pattern recognition*, pp. 248–255. Ieee, 2009. Carlo D'Eramo, Davide Tateo, Andrea Bonarini, Marcello Restelli, Jan Peters, et al. Sharing knowledge in multi-task deep reinforcement learning. In *International Conference on Learning Representations*, pp. 1–11. OpenReview. net, 2020. Chris Fifty, Ehsan Amid, Zhe Zhao, Tianhe Yu, Rohan Anil, and Chelsea Finn. Efficiently identifying task groupings for multi-task learning. *Advances in Neural Information Processing Systems*, 34:27503–27516, 2021. Yuan Gao, Jiayi Ma, Mingbo Zhao, Wei Liu, and Alan L Yuille. Nddr-cnn: Layerwise feature fusing in multi-task cnns by neural discriminative dimensionality reduction. In *Proceedings of the IEEE/CVF conference on computer vision and pattern recognition*, pp. 3205–3214, 2019. Yuan Gao, Haoping Bai, Zequn Jie, Jiayi Ma, Kui Jia, and Wei Liu. Mtl-nas: Task-agnostic neural architecture search towards general-purpose multi-task learning. In *Proceedings of the IEEE/CVF Conference on computer vision and pattern recognition*, pp. 11543–11552, 2020. Michelle Guo, Albert Haque, De-An Huang, Serena Yeung, and Li Fei-Fei. Dynamic task prioritization for multitask learning. In *Proceedings of the European conference on computer vision (ECCV)*, pp. 270–287, 2018. Pengsheng Guo, Chen-Yu Lee, and Daniel Ulbricht. Learning to branch for multi-task learning. In *International Conference on Machine Learning*, pp. 3854–3863. PMLR, 2020.

<span id="page-10-16"></span><span id="page-10-15"></span><span id="page-10-14"></span><span id="page-10-13"></span><span id="page-10-12"></span><span id="page-10-11"></span><span id="page-10-10"></span><span id="page-10-9"></span><span id="page-10-8"></span><span id="page-10-7"></span><span id="page-10-6"></span><span id="page-10-5"></span><span id="page-10-4"></span><span id="page-10-3"></span><span id="page-10-2"></span><span id="page-10-1"></span><span id="page-10-0"></span>Kazuma Hashimoto, Caiming Xiong, Yoshimasa Tsuruoka, and Richard Socher. A joint many-task model: Growing a neural network for multiple nlp tasks. *Empirical Methods in Natural Language Processing (EMNLP)*, 2017. Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In *Proceedings of the IEEE/CVF conference on computer vision and pattern recognition*, pp. 770–778, 2016. Adrian Javaloy and Isabel Valera. Rotograd: Gradient homogenization in multitask learning. In ´ *International Conference on Learning Representations*, 2022. Alex Kendall, Yarin Gal, and Roberto Cipolla. Multi-task learning using uncertainty to weigh losses for scene geometry and semantics. In *Proceedings of the IEEE/CVF conference on computer vision and pattern recognition*, pp. 7482–7491, 2018. Iasonas Kokkinos. Ubernet: Training a universal convolutional neural network for low-, mid-, and high-level vision using diverse datasets and limited memory. In *Proceedings of the IEEE/CVF conference on computer vision and pattern recognition*, pp. 6129–6138, 2017. Xi Lin, Hui-Ling Zhen, Zhenhua Li, Qing-Fu Zhang, and Sam Kwong. Pareto multi-task learning. *Advances in Neural Information Processing Systems*, 32, 2019. Bo Liu, Xingchao Liu, Xiaojie Jin, Peter Stone, and Qiang Liu. Conflict-averse gradient descent for multi-task learning. *Advances in Neural Information Processing Systems*, 34:18878–18890, 2021a. Liyang Liu, Yi Li, Zhanghui Kuang, J Xue, Yimin Chen, Wenming Yang, Qingmin Liao, and Wayne Zhang. Towards impartial multi-task learning. In *International Conference on Learning Representations*, 2021b. Shikun Liu, Edward Johns, and Andrew J Davison. End-to-end multi-task learning with attention. In *Proceedings of the IEEE/CVF conference on computer vision and pattern recognition*, pp. 1871–1880, 2019. Ziwei Liu, Ping Luo, Xiaogang Wang, and Xiaoou Tang. Deep learning face attributes in the wild. In *Proceedings of the IEEE international conference on computer vision*, pp. 3730–3738, 2015. Mingsheng Long, Zhangjie Cao, Jianmin Wang, and Philip S Yu. Learning multiple tasks with multilinear relationship networks. *Advances in Neural Information Processing Systems*, 30, 2017. Jiaqi Ma, Zhe Zhao, Xinyang Yi, Jilin Chen, Lichan Hong, and Ed H Chi. Modeling task relationships in multi-task learning with multi-gate mixture-of-experts. In *Proceedings of the 24th ACM SIGKDD international conference on knowledge discovery & data mining*, pp. 1930–1939, 2018. Kevis-Kokitsi Maninis, Ilija Radosavovic, and Iasonas Kokkinos. Attentive single-tasking of multiple tasks. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pp. 1851–1860, 2019. Elliot Meyerson and Risto Miikkulainen. Beyond shared hierarchies: Deep multitask learning through soft layer ordering. In *International Conference on Learning Representations*, 2018. Ishan Misra, Abhinav Shrivastava, Abhinav Gupta, and Martial Hebert. Cross-stitch networks for multi-task learning. In *Proceedings of the IEEE/CVF conference on computer vision and pattern recognition*, pp. 3994–4003, 2016. Roozbeh Mottaghi, Xianjie Chen, Xiaobai Liu, Nam-Gyu Cho, Seong-Whan Lee, Sanja Fidler, Raquel Urtasun, and Alan Yuille. The role of context for object detection and semantic segmentation in the wild. In *Proceedings of the IEEE/CVF conference on computer vision and pattern recognition*, pp. 891–898, 2014. Emilio Parisotto, Lei Jimmy Ba, and Ruslan Salakhutdinov. Actor-mimic: Deep multitask and transfer reinforcement learning. In *International Conference on Learning Representations*, 2016.

<span id="page-11-11"></span><span id="page-11-10"></span><span id="page-11-9"></span><span id="page-11-8"></span><span id="page-11-7"></span><span id="page-11-6"></span><span id="page-11-5"></span><span id="page-11-4"></span><span id="page-11-3"></span><span id="page-11-2"></span><span id="page-11-1"></span><span id="page-11-0"></span>Clemens Rosenbaum, Tim Klinger, and Matthew Riemer. Routing networks: Adaptive selection of non-linear functions for multi-task learning. In *International Conference on Learning Representations*, 2018. Sebastian Ruder. An overview of multi-task learning in deep neural networks. *arXiv preprint [arXiv:1706.05098](http://arxiv.org/abs/1706.05098)*, 2017. Sebastian Ruder, Joachim Bingel, Isabelle Augenstein, and Anders Søgaard. Latent multi-task architecture learning. In *Proceedings of the AAAI Conference on Artificial Intelligence*, volume 33, pp. 4822–4829, 2019. Mark Sandler, Andrew Howard, Menglong Zhu, Andrey Zhmoginov, and Liang-Chieh Chen. Mobilenetv2: Inverted residuals and linear bottlenecks. In *Proceedings of the IEEE/CVF conference on computer vision and pattern recognition*, pp. 4510–4520, 2018. Ozan Sener and Vladlen Koltun. Multi-task learning as multi-objective optimization. *Advances in Neural Information Processing Systems*, 31, 2018. Jiayi Shen, Xiantong Zhen, Marcel Worring, and Ling Shao. Variational multi-task learning with gumbel-softmax priors. *Advances in Neural Information Processing Systems*, 34:21031–21042, 2021. Trevor Standley, Amir Zamir, Dawn Chen, Leonidas Guibas, Jitendra Malik, and Silvio Savarese. Which tasks should be learned together in multi-task learning? In *International Conference on Machine Learning*, pp. 9120–9132. PMLR, 2020. Sebastian Thrun and Joseph O'Sullivan. Discovering structure in multiple learning tasks: The tc algorithm. In *International Conference on Machine Learning*, volume 96, pp. 489–497, 1996. Simon Vandenhende, Stamatios Georgoulis, Wouter Van Gansbeke, Marc Proesmans, Dengxin Dai, and Luc Van Gool. Multi-task learning for dense prediction tasks: A survey. *IEEE transactions on pattern analysis and machine intelligence*, 2021. Ruihan Yang, Huazhe Xu, Yi Wu, and Xiaolong Wang. Multi-task reinforcement learning with soft modularization. *Advances in Neural Information Processing Systems*, 33:4767–4777, 2020. Tianhe Yu, Saurabh Kumar, Abhishek Gupta, Sergey Levine, Karol Hausman, and Chelsea Finn. Gradient surgery for multi-task learning. *Advances in Neural Information Processing Systems*, 33:5824–5836, 2020. Amir R Zamir, Alexander Sax, William Shen, Leonidas J Guibas, Jitendra Malik, and Silvio Savarese. Taskonomy: Disentangling task transfer learning. In *Proceedings of the IEEE/CVF conference on computer vision and pattern recognition*, pp. 3712–3722, 2018.

# <span id="page-12-0"></span>A PROOF OF THEOREM [A.1](#page-12-2)

<span id="page-12-2"></span>Theorem A.1. *Assume that* L *is differentiable and for any two different tasks* T<sup>i</sup> *and* T<sup>j</sup> *, it satisfies*

cos φ (k) ij kg (k) i k < kg (k) j k, ∀k ∈ <sup>P</sup> (8)

*then for any sufficiently small learning rate* α > 0*,*

L( ˆθr) < L( ˆθ). (9)

*Proof.* We consider the first order Taylor approximation of L<sup>i</sup> . For normal update, we have

Li ˆθ fix sh , ˆθ cf sh, ˆθ ts i =L<sup>i</sup> θ fix sh , θcf sh, θts i + (ˆθ fix sh − θ fix sh ) <sup>&</sup>gt;g fix i (10)

+ (ˆθ cf sh − θ cf sh) <sup>&</sup>gt;g cf <sup>i</sup> + (ˆ<sup>θ</sup> ts <sup>i</sup> − θ ts i ) <sup>&</sup>gt;g ts <sup>i</sup> + o(α). (11)

For Recon update, we have

Li ˆθ fix sh , ˆθ cf i , ˆθ ts i =L<sup>i</sup> θ fix sh , θcf sh, θts i + (ˆθ fix sh − θ fix sh ) <sup>&</sup>gt;g ts i (12)

+ (ˆθ cf <sup>i</sup> − θ cf sh) <sup>&</sup>gt;g cf <sup>i</sup> + (ˆ<sup>θ</sup> ts <sup>i</sup> − θ ts i ) <sup>&</sup>gt;g ts <sup>i</sup> + o(α). (13)

The difference between the two loss functions after the update is

Li ˆθ fix sh , ˆθ cf i , ˆθ ts i − L<sup>i</sup> ˆθ fix sh , ˆθ cf sh, ˆθ ts i =(ˆθ cf <sup>i</sup> − <sup>ˆ</sup><sup>θ</sup> cf sh) <sup>&</sup>gt;g cf <sup>i</sup> + o(α) (14)

= − α g cf <sup>i</sup> − X T j=1 wjg cf j > g cf <sup>i</sup> + o(α) (15)

= − α X T j=1 w<sup>j</sup> g cf <sup>i</sup> − g cf j > g cf <sup>i</sup> + o(α) (16)

= − α X T j=1 w<sup>j</sup> kg cf <sup>i</sup> k <sup>2</sup> − g cf j <sup>&</sup>gt;g cf i + o(α). (17)

Assume, without loss of generality, that kg cf i k 6= 0, then

 g cf i 2 − g cf j <sup>&</sup>gt;g cf <sup>i</sup> = X g (k) i − g (k) i <sup>&</sup>gt;g (k) j (18)

k∈P = X k∈P g (k) i g (k) i <sup>−</sup> cos <sup>φ</sup> (k) ij g (k) j (19)

> 0. (20)

Hence, the above difference is negative, if α is sufficiently small. As such, the difference between the multi-task loss functions is also negative, if α is sufficiently small.

L( ˆθr) − L( <sup>ˆ</sup>θ) = X T i=1 Li ˆθ fix sh , ˆθ cf i , ˆθ ts i − X T i=1 Li ˆθ fix sh , ˆθ cf sh, ˆθ ts i < 0 (21)

# <span id="page-12-1"></span>B EXPERIMENTAL SETUP

# B.1 MULTI-FASHION+MNIST

Model. We adopt ResNet18 [\(He et al., 2016\)](#page-10-16) without pre-training as the backbone and modify the dimension of the output features to 100 for the last linear layer. For the task-specific heads, we define two linear layers followed by a ReLU function.

Tasks, losses, and metrics. Each task is a classification problem with 10 classes and we use the cross-entropy loss as the classification loss. For evaluation, we use the classification accuracy as the metric for each task.

Model hyperparameters. We train the model for 120 epochs with the batch size of 256. We adopt SGD with an initial learning rate of 0.1 and decay the learning rate by 0.1 at the 60th and 90th epoch.

Baseline hyperparameters. For CAGrad, we set α = 0.2. For BMTAS, we set the resource loss weight to 1.0, and we search the architecture for 100 epochs. For RotoGrad, we set R<sup>k</sup> = 100 which is equal to the dimension of shared features and set the learning rate of rotation parameters as learning rate of the neural networks. For MMoE, the initial learning rate of expert networks and gates are 0.1 and 1e − 3 respectively.

Recon hyperparameters. We use CAGrad to train the model for 30 epochs and compute the conflict score of each shared layer. We set S = −0.1 for computing the scores. We select 25 layers with the highest conflict scores and turn them into task-specific layers.

# B.2 CITYSCAPES

Model. We adopt SegNet [\(Badrinarayanan et al., 2017\)](#page-9-13) as the backbone where the decoder is split into two convolutional heads.

Model hyperparameters. We train the model for 200 epochs with the batch size of 8. We adopt Adam with an initial learning rate of 5e − 5 and decay the learning rate by 0.5 at the 100th epoch.

Baselines hyperparameters. For CAGrad, we set α = 0.2. For RotoGrad, we set R<sup>k</sup> = 1024 and set the learning rate of rotation parameters as 10 times less than the learning rate of the neural networks.

Recon hyperparameters. We use joint-train to train the model for 40 epochs and compute the conflict score of each shared layer. We set S = 0.0 for computing the scores. We select 39 layers with the highest conflict scores and turn them into task-specific layers.

## B.3 NYUV2

Model. We adopt MTAN [\(Liu et al., 2019\)](#page-10-3) – the SegNet combined with task-specific attention modules on the encoder.

Model hyperparameters. We train the model for 200 epochs with the batch size of 2. We adopt Adam with an initial learning rate of 1e − 4 and decay the learning rate by 0.5 at the 100th epoch.

Baseline hyperparameters. For CAGrad, we set α = 0.4 similar with [Liu et al.](#page-10-5) [\(2021a\)](#page-10-5).

Recon hyperparameters. We use joint-train to train the model for 40 epochs and compute the conflict score of each shared layer. We set S = −0.02 for computing the scores. We select 22 layers with the highest conflict scores and turn them into task-specific layers.

### B.4 PASCAL-CONTEXT

Model. Following [Bruggemann et al.](#page-9-9) [\(2020\)](#page-9-9), we employ MobileNetv2 [Sandler et al.](#page-11-11) [\(2018\)](#page-11-11) as the backbone with a reduced design of the ASPP module (R-ASPP) [\(Sandler et al., 2018\)](#page-11-11). We pre-train the model on ImageNet [\(Deng et al., 2009\)](#page-9-14).

Model hyperparameters. We train the model for 130 epochs with the batch size of 6. We adopt Adam with an initial learning rate of 1e − 4 and decay the learning rate by 0.1 at the 70th and 100th epoch.

Baselines hyperparameters. For CAGrad, we set α = 0.1. For BMTAS, we set the resoure loss weight to 0.1, and we search the architecture for 130 epochs.

Recon hyperparameters. We use joint-train to train the model for 40 epochs and compute the conflict score of each shared layer. We set S = −0.02 for computing the scores. We select 85 layers with the highest conflict scores and turn them into task-specific layers.

### B.5 CELEBA

Model. Following [Sener & Koltun](#page-11-3) [\(2018\)](#page-11-3), we use ResNet18 [\(He et al., 2016\)](#page-10-16) as the backbone network. We pre-train the model on ImageNet [\(Deng et al., 2009\)](#page-9-14).

Model hyperparameters. We train the model for 5 epochs. We adopt Adam with an initial learning rate of 5e − 5 and decay the learning rate by 0.5 at the 3 th epoch.

Baselines hyperparameters. For CAGrad, we set α = 0.1.

Recon hyperparameters. We use joint-train to train the model for 2 epochs and compute the conflict score of each shared layer. We set S = −0.05. We select 25 layers with the highest conflict scores and turn them into task-specific layers.

# <span id="page-14-3"></span>C ADDITIONAL ABLATION STUDY

The distribution of gradient conflicts. In addition to the statistics on Multi-Fashion+MNIST, we further show the distributions of gradient conflicts of various baselines on CityScapes, NYUv2, and PASCAL-Context in Fig [5,](#page-14-0) Fig [6,](#page-14-1) and Fig [7](#page-14-2) respectively. We compare the distributions with those of baselines w/ Recon on the three datasets in Fig. [8,](#page-15-1) Fig. [9,](#page-15-2) and Fig. [10](#page-15-3) respectively. The detailed statistics are provided in Tables [8-](#page-15-0)[10.](#page-16-0)

![](_page_14_Figure_8.jpeg)

<span id="page-14-0"></span>Figure 5: The distributions of gradient conflicts (in terms of cos φij ) of the joint-training baseline and state-of-the-art gradient manipulation methods on CityScapes dataset.

![](_page_14_Figure_10.jpeg)

<span id="page-14-1"></span>Figure 6: The distributions of gradient conflicts (in terms of cos φij ) of the joint-training baseline and state-of-the-art gradient manipulation methods on NYUv2 dataset.

![](_page_14_Figure_12.jpeg)

<span id="page-14-2"></span>Figure 7: The distributions of gradient conflicts (in terms of cos φij ) of the joint-training baseline and state-of-the-art gradient manipulation methods on PASCAL-Context dataset.

![](_page_15_Figure_1.jpeg)

<span id="page-15-1"></span>Figure 8: The distribution of gradient conflicts (in terms of cos φij ) w.r.t. the shared parameters on CityScapes. RSL: randomly selecting same number of layers as Recon and set them task-specific. RSP: randomly selecting similar amount of parameters as Recon and set them task-specific.

![](_page_15_Figure_3.jpeg)

<span id="page-15-2"></span>Figure 9: The distribution of gradient conflicts (in terms of cos φij ) of baselines and baselines with Recon on NYUv2. RSL: randomly selecting same number of layers as Recon and set them taskspecific. RSP: randomly selecting similar amount of parameters as Recon and set them task-specific.

![](_page_15_Figure_5.jpeg)

<span id="page-15-3"></span>Figure 10: The distribution of gradient conflicts (in terms of cos φij ) of baselines and baselines with Recon on PASCAL-Context. RSL: randomly selecting same number of layers as Recon and set them task-specific. RSP: randomly selecting similar amount of parameters as Recon and set them task-specific.

<span id="page-15-0"></span>Table 8: The distribution of gradient conflicts (in terms of cos φij ) w.r.t. the shared parameters on CityScapes dataset. "Reduction" means the percentage of conflicting gradients in the interval of (−0.02, −1.0] reduced by the model compared with joint-training. The grey cell color indicates Recon greatly reduces the conflicting gradients (more than 50%). In contrast, gradient manipulation methods only moderately decrease their occurrence (MGDA deceases it by 22%), and some methods even increase it.

Table 9: The distribution of gradient conflicts (in terms of cos φij ) w.r.t. the shared parameters on NYUv2 dataset. "Reduction" means the percentage of conflicting gradients in the interval of (−0.04, −1.0] reduced by the model compared with joint-training. The grey cell color indicates Recon greatly reduces the conflicting gradients (more than 50%). In contrast, gradient manipulation methods only slightly decrease their occurrence, and some methods even increase it.

<span id="page-16-0"></span>Table 10: The distribution of gradient conflicts (in terms of cos φij ) w.r.t. the shared parameters on PASCAL-Context dataset. "Reduction" means the percentage of conflicting gradients in the interval of (−0.02, −1.0] reduced by the model compared with joint-training. The grey cell color indicates Recon greatly reduces the conflicting gradients (more than 50%). In contrast, gradient manipulation methods only slightly decrease their occurrence, and some methods even increase it.

<span id="page-16-1"></span>Table 11: Multi-task learning results on Multi-Fashion+MNIST dataset. LSK refers to turning the fist K layers into task-specific layers. FSK refers to turning the last K layers into task-specific layers. PD denotes the performance drop compared with Recon.

Selecting the first K layers and the last K Layers as conflict layers does not work. To further support the conclusion that the selection of parameters with higher probability of conflicting gradients contributes most to the performance gain rather than the increase in model capacity. We compare Recon with two baselines: (1) Select the first K neural network layers and turn them into task-specific layers. (2) Select the last K neural network layers and turn them into task-specific layers. The multi-task learning results on the Multi-Fashion+MNIST benchmark are presented in Table [11.](#page-16-1) The results show that if we directly turn the top or the bottom of the neural network into task-specific parameters, it still will lead to performance degradation compared to Recon.

Recon finds similar layers in different training stages. Recon ranks the network layers according to the computed S-conflict scores. The ranking result can be represented as a layer permutation, denoted as π, and π(l) is the position of layer l. The similarity between two rankings π<sup>i</sup> and π<sup>j</sup> can be measured as:

<span id="page-16-2"></span>d(π<sup>i</sup> , π<sup>j</sup> ) = <sup>1</sup> |L| X l∈L |πi(l) − π<sup>j</sup> (l)|, (22)

where L denotes the set of neural network layers. In Table [12,](#page-17-0) we measure the differences in rankings obtained in different training stages (e.g., in the first 25% iterations or the second 25% iterations)

<span id="page-17-0"></span>Table 12: The distance between the layer permutations (rankings) obtained in different training stages on Multi-Fashion+MNIST dataset. "Iter." denotes iterations.

<span id="page-17-1"></span>Table 13: Performance of the networks modified by Recon with conflict layers found in different training stages of joint-training on CityScapes dataset. ∆m% denotes the average relative improvement of all tasks. #P denotes the model size (MB). The best result is marked in bold.

on Multi-Fashion+MNIST by Eq. [22.](#page-16-2) The small distances (less than 2.4) indicate that the layers found in different training stages are quite similar. In Table [13,](#page-17-1) we compare the performance of the networks modified by Recon with conflict layers found in different training stages on CityScapes. It can be seen that the results of the last three rows are the same, which is because the layers found in the 3rd 25% iterations, 4th 25% iterations, and all iterations are *exactly the same* (the rankings may be slightly different though). The layers found in the later stages lead to slightly better performance than those found in the early stages (i.e., 1st 25% iterations and 2nd 25% iterations), indicating the conflict scores in early iterations might be a little noisy. However, since the performance gaps are acceptably small, to save time, we use the initial 25% training iterations to find conflict layers.

<span id="page-17-2"></span>Table 14: The distance between the layer permutations (rankings) obtained by Recon with different methods on Multi-Fashion+MNIST dataset.

Recon finds similar layers with different MTL methods. In Table [14,](#page-17-2) we measure the differences in layer permutations (rankings) obtained by Recon with different methods (e.g., CAGrad and PC-Grad) on Multi-Fashion+MNIST by Eq. [22.](#page-16-2) The small distances (less than 1.9) indicate that the layers found by Recon with different methods are quite similar. Therefore, in our experiments, we only use joint-training to search for the conflict layers once, and directly apply the modified network to improve different gradient manipulation methods as shown in Tables [1](#page-5-1)[-5.](#page-7-1)

The conflict layers found by Recon with the same architecture are transferable between different datasets. We conduct experiments with three different architectures: ResNet18, SegNet, and MTAN. (1) For Resnet18, we find that the layers found by Recon on CelebA and those found on Multi-Fashion+MNIST are *exactly the same*. (2) For SegNet, we find that 95% layers (38 out of 40)

<span id="page-18-0"></span>Table 15: Multi-task learning results on NYUv2 dataset with SegNet as backbone. Recon<sup>∗</sup> denotes setting the layers found on CityScapes to task-specific. ∆m% denotes the average relative improvement of all tasks. #P denotes the model size (MB). The grey cell color indicates that Recon or Recon<sup>∗</sup> improves the result of the base model.

<span id="page-18-1"></span>Table 16: Multi-task learning results on CityScapes dataset with MTAN as backbone. Recon<sup>∗</sup> denotes setting the layers found on NYUv2 to task-specific. ∆m% denotes the average relative improvement of all tasks. #P denotes the model size (MB). The grey cell color indicates that Recon or Recon<sup>∗</sup> improves the result of the base model.

found on NYUv2 are identical to those found on CityScapes. On NYUv2, we compare the performance of using conflict layers found on NYUv2 (baselines w/ Recon) to that of using conflict layers found on CityScapes (i.e., baselines w/ Recon<sup>∗</sup> ), as shown in Table [15.](#page-18-0) (3) For MTAN (SegNet with attention), we find that 68% layers (17 out of 25) found on CityScapes are identical to those found on NYUv2. On CityScapes, we compare the performance of using conflict layers found on CityScapes (baselines w/ Recon) to that of using conflict layers found on NYUv2 (i.e., baselines w/ Recon<sup>∗</sup> ), as shown in Table [16.](#page-18-1) The results show that the conflict layers found on one dataset can be used to modify the network to be directly used on another dataset to consistently improve the performance of various baselines, while searching for the conflict layers again on the new dataset may lead to better performance.

<span id="page-19-0"></span>![](_page_19_Figure_2.jpeg)

Figure 11: Comparison of running time (one iteration, excludes data fetching) on CelebA dataset.

Analysis of running time. We evaluate how Recon scales with the number of tasks on CelebA dataset, by comparing the running time of one iteration used by Recon in computing gradient conflict scores (the most time-consuming part of Recon) to that of the baselines. The results in Fig. [11](#page-19-0) show that Recon is as fast as other gradient manipulation methods such as CAGrad [\(Liu et al., 2021a\)](#page-10-5) and Graddrop [\(Chen et al., 2020\)](#page-9-3), but much slower than joint-training especially when the number of tasks is large, which is natural since Recon needs to compute pariwise cosine similarity of task gradients. However, since Recon only needs to search for the conflict layers once for a given network architecture, as discussed above, the running time is not a problem.