This CVPR paper is the Open Access version, provided by the Computer Vision Foundation. Except for this watermark, it is identical to the accepted version; the final published version of the proceedings is available on IEEE Xplore. 

# **Quantifying Task Priority for Multi-Task Optimization** 

## Wooseong Jeong KAIST 

stk14570@kaist.ac.kr 

Kuk-Jin Yoon KAIST 

kjyoon@kaist.ac.kr 

## **Abstract** 

_The goal of multi-task learning is to learn diverse tasks within a single unified network. As each task has its own unique objective function, conflicts emerge during training, resulting in negative transfer among them. Earlier research identified these conflicting gradients in shared parameters between tasks and attempted to realign them in the same direction. However, we prove that such optimization strategies lead to sub-optimal Pareto solutions due to their inability to accurately determine the individual contributions of each parameter across various tasks. In this paper, we propose the concept of task priority to evaluate parameter contributions across different tasks. To learn task priority, we identify the type of connections related to links between parameters influenced by task-specific losses during backpropagation. The strength of connections is gauged by the magnitude of parameters to determine task priority. Based on these, we present a new method named connection strength-based optimization for multi-task learning which consists of two phases. The first phase learns the task priority within the network, while the second phase modifies the gradients while upholding this priority. This ultimately leads to finding new Pareto optimal solutions for multiple tasks. Through extensive experiments, we show that our approach greatly enhances multi-task performance in comparison to earlier gradient manipulation methods._ 

## **1. Introduction** 

Multi-task learning (MTL) is a learning paradigm that handles multiple different tasks in a single model [2]. Compared to learning tasks individually, MTL can effectively reduce the number of parameters, leading to less memory usage and computation with a higher convergence rate. Furthermore, it leverages multiple tasks as an inductive bias, enabling the learning of generalized features while reducing overfitting. Complex systems such as robot vision and autonomous driving require the ability to perform multiple tasks within a single system. Thus, MTL can be a first step in finding general architecture for computer vision. 

A primary goal of MTL is minimizing _negative transfer_ [6] and finding _Pareto-optimal solutions_ [36] for multiple tasks. Negative transfer is a phenomenon where the learning of one task adversely affects the performance of other tasks. Since each task has its own objective, this can potentially result in a trade-off among tasks. A condition in which enhancing one task is not possible without detriment to another is called _Pareto optimality_ . A commonly understood cause of this trade-off is _conflicting gradients_ [45] that arise during the optimization process. When the gradients of two tasks move in opposing directions, the task with larger magnitudes dominates the other, disrupting the search for Pareto-optimal solutions. The situation becomes more complex due to imbalances in loss scales across tasks. The way we weigh task losses is crucial for multi-task performance. When there is a significant disparity in the magnitudes of losses, the task with a larger loss would dominate the entire network. Hence, the optimal strategy for MTL should efficiently handle conflicting gradients across different loss scales. 

Previous studies address negative transfer by manipulating gradients or balancing tasks’ losses. Solutions for handling conflicting gradients are explored in [26, 36, 37, 45]. These approaches aim to align conflicting gradients towards a cohesive direction within a shared network space. However, these techniques are not effective at preventing negative transfer, as they don’t pinpoint which shared parameters are crucial for the tasks. This results in sub-optimal Pareto solutions for MTL, leading to pool multi-task performance. Balancing task losses is a strategy that can be applied independently from gradient manipulation methods. It includes scaling the loss according to homoscedastic uncertainty [22], or dynamically finding loss weights by considering the rate at which the loss decreases [29]. 

In this paper, we propose the concept of _task priority_ to address negative transfer in MTL and suggest _connection strength_ as a quantifiable measure for this purpose. The task priority is defined over shared parameters by comparing the influence of each task’s gradient on the overall multi-task loss. This reveals the relative importance of shared parameters to various tasks. To learn and conserve the task pri- 

363 

ority throughout the optimization process, we propose the concept of _task-specific connections_ and their _strength_ in the context of MTL. A _task-specific connection_ denotes the link between shared and task-specific parameters during the backpropagation of each task-specific loss. The strength of this connection can be quantified by measuring the scale of the parameters involved. Based on the types of connections and their respective strengths, we apply two distinct optimization phases. The goal of the first phase is to find new Pareto-optimal solutions for multiple tasks by learning task priorities through the use of specific connection types. The second phase aims to maintain the task priorities learned from varying loss scales by quantifying the strength of these connections. Our method outperforms previous optimization techniques that relied on gradient manipulation, consistently discovering new Pareto optimal solutions for various tasks, thereby improving multi-task performance. Our contributions are summarized as follows: 

- We propose the concept of task priority within a shared network to assess the relative importance of parameters across different tasks and to uncover the limitation inherent in traditional multi-task optimization. 

- We reinterpret connection strength within the context of MTL to quantify task priority. Based on this reinterpretation, we propose a new multi-task optimization approach called connection strength-based optimization to learn and preserve task priorities. 

- To demonstrate the robustness of our method, we perform extensive experiments. Our results consistently reveal substantial enhancements in multi-task performance when compared to prior research. 

## **2. Related Work** 

**Optimization for MTL** aims to mitigate negative transfer between tasks. Some of them [8, 26, 28, 34, 36, 37, 45] directly modify gradients to address task conflicts. MGDA [8, 36] views MTL as a multi-objective problem and minimizes the norm point in the convex hull to find a Pareto optimal set. PCGrad [45] introduces the concept of conflicting gradients and employs gradient projection to handle them. CAGrad [26] minimizes the multiple loss functions and regularizes the trajectory by leveraging the worst local improvement of individual tasks. Aligned-MTL [37] stabilize optimization by aligning the principal components of the gradient matrix. Recon [13] uses an approach similar to Neural Architecture Search (NAS) to address conflicting gradients. Some approaches use normalized gradients [3] to prevent spillover of tasks or assign stochasticity on the network’s parameter based on the level of consistency in the sign of gradients [4]. RotoGrad [21] rotates the feature space of the network to narrow the gap between tasks. Unlike earlier methods that guided gradients towards an intermediate direction (as illustrated in Fig. 1(a)), our approach 

identifies task priority in shared parameters to update gradients, leading to finding new Pareto-optimal solutions. 

**Scaling task-specific loss** largely influences multi-task performance since the task with a significant loss would dominate the whole training process and cause severe task interference. To address the task unbalancing problem in the training, some approaches re-weight the multi-task loss by measuring homoscedastic uncertainty [22], prioritizing tasks based on task difficulty [14], or balancing multi-task loss dynamically by considering the descending rate of loss [29]. We perform extensive experiments involving different loss-scaling methods to demonstrate the robustness of our approach across various loss-weighting scenarios. 

**MTL architectures** can be classified depending on the extent of network sharing across tasks. The shared trunk consists of a shared encoder followed by an individual decoder for each task [7, 30, 39, 46]. Multi-modal distillation methods [9, 40, 43, 47] have been proposed, which can be used at the end of the shared trunk for distillation to propagate task information effectively. On the other hand, crosstalk architecture uses separate networks for each task and allows parallel information flow between layers [12]. Our optimization approach can be applied to any model to mitigate task conflicts and enhance multi-task performance. 

## **3. Preliminaries** 

### **3.1. Problem Definition for Multi-task Learning** 

In multi-task learning (MTL), the network learns a set of tasks _T_ = _{τ_ 1 _, τ_ 2 _, ..., τK}_ jointly, where _K_ is the number of tasks. Each task _τi_ has its own loss function _Li_ (Θ) where Θ is the parameter of the network. The network parameter Θ can be classified into Θ = _{_ Θ _s,_ Θ1 _,_ Θ2 _, ...,_ Θ _K}_ where Θ _s_ is shared parameter across all tasks and Θ _i_ is task-specific parameters devoted to task _τi_ . Then, the objective function of multi-task learning is to minimize the weighted sum of all tasks’ losses: 



The performance in multi-task scenarios is affected by the weighting _wi_ of the task-specific loss _Li_ . 

### **3.2. Prior Approach for Multi-Task Optimization** 

From an optimization perspective, MTL seeks Pareto optimal solutions for multiple tasks. 

**Definition 1** (Pareto optimality) **.** _For a given network parameter_ Θ _, if we get_ Θ _new such that Li_ (Θ) _> Li_ (Θ _new_ ) _holds for any task τi, while ensuring that Lj_ (Θ) _≥ Lj_ (Θ _new_ ) _is satisfied for all other tasks τj (j̸_ = _i), then the situation is termed a Pareto improvement. In this context,_ 

364 



Figure 1. Overview of our connection strength-based optimization. (a) Previous methods [26, 36, 37, 45] modify gradients in shared parameters to converge toward an intermediate direction without considering the task priority, which leads to sub-optimal Pareto solutions. (b) Our method divides the optimization process into two distinct phases. In Phase 1, task priority is learned through task-specific connections, leading to the identification of a new Pareto optimal solution. In Phase 2, task priority is gauged using the connection strength between shared and task-specific nodes. Subsequently, gradients in shared parameters are aligned with the direction of the highest-priority task’s gradients. This phase ensures that priorities established in Phase 1 are maintained, thus reducing potential negative transfer. 

Θ _new is said to dominate_ Θ _. A parameter_ Θ<sup>_∗_</sup> _is Paretooptimal if no further Pareto improvements are possible. A set of Pareto optimal solutions is called a Pareto frontier._ 

Earlier research [26, 36, 37] interprets MTL in the context of multi-objective optimization, aiming for Pareto optimality. They present a theoretical analysis that demonstrates the convergence of optimization towards Pareto stationary points. Nevertheless, their analysis is constrained when applied to real-world scenarios due to its assumption of convex loss functions, which conflicts with the non-convex nature of neural networks. Also, their demonstration of optimization converging to Pareto stationary points doesn’t necessarily guarantee reaching Paretooptimal points, as the former are necessary but not sufficient conditions for Pareto optimality. We delineate their limitations theoretically by introducing the concept of task priority and empirically validate them by analyzing training loss and multi-task performance. On the other hand, Yu _et al_ . [45] emphasize the conflicting gradients. 

_gradient of task τi with respect to the shared parameters_ Θ _s as gi_ = _∇_ Θ _s Li_ (Θ _s,_ Θ _i_ ) _. And gi and gj are gradients of a pair of tasks τi and τj where i̸_ = _j. If gi · gj ≤_ 0 _, then the two gradients are called conflicting gradients._ 

Previous approaches [26, 36, 37, 45] address the issue of conflicting gradients in shared parameters Θ _s_ by aligning the gradients in a consistent direction as shown in Fig. 1(a). Nonetheless, they face challenges in minimizing negative transfer, as they cannot discern which parameters in Θ _s_ are most important to tasks. We refer to the relative importance of a task in the shared parameter as task priority. Previous studies aligned gradients without taking into account task priority, inadvertently resulting in negative transfer and reduced multi-task performance. In contrast, we introduce the notion of connection strength to determine task priority in the shared space and propose new gradient update rules based on this priority. 

## **4. Method** 

**Definition 2** (Conflicting gradients) **.** _Conflicting gradients are defined in the shared space of the network. Denote the_ 

In this section, we introduce the concept of task priority to minimize negative transfer between tasks. To measure 

365 

task priority, we establish connections in the network and assess their strength. Following that, we propose a novel optimization method for MTL termed connection strengthbased optimization. Our approach breaks down the optimization process into two phases as shown in Fig. 1(b). In Phase 1, we focus on instructing the network to catch task-specific details by learning task priority. In Phase 2, task priority within the shared parameters is determined and project gradients to preserve the priority. 

### **4.1. Motivation: Task priority** 

We propose a straightforward theoretical analysis of our approach, using the notation given in Sec. 3. Before diving deeper, we first introduce the definition of task priority. 

**Definition 3** (Task priority) **.** _Assume that the task losses Li for i_ = 1 _,_ 2 _, ..., K are differentiable. Consider X_<sup>_t_</sup> _as the input data at time t. We initiate with shared parameters_ Θ<sup>_t_</sup> _s_<sup>_andtask-specificparameters_Θ</sup><sup>_t_</sup> _i_<sup>_withsufficientlysmall_</sup> _learning rate η >_ 0 _. A subset of shared parameters at time t is denoted as θ_<sup>_t_</sup> _, such that θ_<sup>_t_</sup> _⊂_ Θ<sup>_t_</sup> _s_<sup>_.For any task τi∈T ,_</sup> _the task’s gradient for θ_<sup>_t_</sup> _is as follows:_ 



_where_ Θ<sup>˜</sup><sup>_t_</sup> _s_<sup>_represents the parameters that are part of_Θ</sup><sup>_t_</sup> _s_<sup>_but_</sup> _not in θ_<sup>_t_</sup> _. For two distinct tasks τm, τn ∈T , if τm holds priority over τn in θ_<sup>_t_</sup> _, then the following inequality holds:_ 



Our motivation is to divide shared parameters Θ _s_ into subsets _{θs,_ 1 _, θs,_ 2 _, ..., θs,K}_ based on task priority. Specifically, _θs,i_ represents a set of parameters that have a greater influence on task _τi_ compared to other tasks. From the task priority, we can derive the following theorem. 

**Theorem 1.** _Updating gradients based on task priority for shared parameters_ Θ _s (update gi for each θs,i) results in a smaller multi-task loss_<sup>�</sup><sup>_K_</sup> _i_ =1<sup>_wiLicomparedtoup-_</sup> _dating the weighted summation of task-specific gradients_ � _Ki_ =1<sup>_∇wiLiwithout considering task priority._</sup> 

The theorem suggests that by identifying the task priority within the shared parameter Θ _s_ , we can further expand the known Pareto frontier compared to neglecting that priority. A detailed proof and theoretical analysis are provided in Appendix A. However, identifying task priority in realworld scenarios is highly computationally demanding. Because it requires evaluating priorities for each subset of the parameter Θ _s_ through pairwise comparisons among multiple tasks. Instead, we prioritize tasks based on connection strength for practical purposes. 

### **4.2. Type and Strength of Connection** 

If we think of each input and output of the network’s component as a node, we can depict the computation flow by establishing connections between them, and then evaluate the strength of these connections to measure their interconnectedness. The idea of connection strength initially emerged in the field of network compression by pruning connections in expansive CNNs [35]. This notion stems from the intuition that larger parameters have a greater influence on the model’s output. Numerous studies [15, 16, 18, 19, 24, 25, 44] have reinforced this hypothesis. In our study, we re-interpret this intuition for MTL to determine task priority in shared parameters of the network. 

Before we dive in, we divide network connections based on the type of task. Conventionally, connection in a network refers to the connectivity between nodes, quantified by the magnitude of parameters. However, we regrouped the network connection based on which task’s loss influences on the connection in backpropagation. 

**Definition 4** (Task-specific connection) **.** _The connection of task τi includes a set of parameters and their interconnections, specifically those involved in the backpropagation process related to the loss function Li for task τi._ 

In the context of MTL, where each task has its own distinct objective function, diverse connections are formed during the backpropagation. Such connections are determined by the specific loss associated with each task, leading us to term them _task-specific connections_ . A set of shared and task-specific parameters, Θ _s_ and Θ _i_ , establishes a unique connection. The connection strength can be measured by the scale of parameters, mirroring the conventional notion. In this instance, we employ task-specific batch normalization to determine the task priority of the output channel of the shared convolutional layer. To establish connection strength, we initiate with a convolutional layer where the input is represented as _x ∈_ **R**<sup>_NI×H×W_</sup> and the weight is denoted by _W ∈_ **R**<sup>_NO×NI×K×K_</sup> . Here, _NI_ stands for the number of input channels, _NO_ for the number of output channels, and _K_ indicates the kernel size. Suppose we have output channel set _C_<sup>_out_</sup> = _{c_<sup>_out_</sup> _p }_<sup>_N_</sup> _p_ =1<sup>_O_and input channel set</sup> _C_<sup>_in_</sup> = _{c_<sup>_in_</sup> _q_<sup>_}N_</sup> _q_ =1<sup>_I_.Foranygivenpairofoutputandinput</sup> channels _c_<sup>_out_</sup> _p ∈C_<sup>_out_</sup> , _c_<sup>_in_</sup> _q ∈C_<sup>_in_</sup> , the connection strength _sp,q_ is defined as: 



The variables _m_ and _n_ correspond to the indices of the convolutional kernel. We explore the convolutional layer followed by task-specific batch normalization, which plays a key role in determining task priority for each output channel. We revisit the equation for batch normalization with 

366 

input _y_ and output _z_ of batch normalization [20]: 



The coefficient of _y_ has a direct correlation with the kernel’s relevance to the task since it directly modulates the output _y_ . Therefore, for task _τi_ , we re-conceptualize the connection strength at the intersection of the convolutional layer and task-specific batch normalization in the following way: 



where _γτi,p_ is a scale factor of the task-specific batch normalization. _Sp_<sup>_τi_measuresthecontributionofeachoutput</sup> channel _c_<sup>_out_</sup> _p_ to the output of task _τi_ . However, it is not possible to directly compare _Sp_<sup>_τi_across tasks because the tasks</sup> exhibit different output scales. Hence, we employ a normalized version of connection strength that takes into account the relative scale differences among tasks: 



Comparing Eq. (7) for each task allows us to determine task priority. Since normalized connection strength represents the relative contribution of each channel across the entire layer, using it to determine task priority also has the advantage of preventing a specific task from having priority over the entire layer. Connection strength depends on network parameters, necessitating design considerations based on the network structure. While this paper provides an example for convolutional layers, a similar application can be extended to transformer blocks or linear layers. In the following optimization, we employ task-specific connections and their strength to learn task priority and conserve it. 

**Algorithm 1:** Connection Strength-based Optimization for Multi-task Learning **Data:** output channel set _{c_<sup>_out_</sup> _p }_<sup>_N_</sup> _p_ =1<sup>_O_,</sup> task set _{τi}_<sup>_K_</sup> _i_ =1<sup>, loss function set</sup><sup>_{Li}K_</sup> _i_ =1<sup>,</sup> channel group _{CGi}_<sup>_K_</sup> _i_ =1<sup>,</sup> number of epochs _E_ , current epoch _e_ 



1 Randomly choose  P ∼ U (0 ,  1)<br>// Phase 1: Learning the task priority<br>2 if  P ≥ e/E then<br>3 for  i ← 1  to K  do<br>4 update: gi ←∇θLi // Update task’s<br>gradients one-by-one<br>// Phase 2: Conserving the task priority<br>5 else<br>6 Initialize  all  CGi as empty set  { }  in the shared<br>convolutional layer<br>7 for  p ← 1 to NO do<br>8 ν = arg max i S ˆ p τi<br>// Determine the top priority task ν<br>9 CGν =  CGν +  {c out p }<br>// Classify channel with task ν<br>10 for  i ← 1  to K  do<br>11 Let  {Gi, 1 , ..., Gi,K}  are gradients of  CGi<br>12 for  j ← 1  to K and i̸  =  j do<br>13 if  Gi,i · Gi,j <  0  then<br>14 Gi,j =  Gi,j - G || i G ,i i,i ·G || i,j 2 ·Gi,i<br>// Project gradients with<br>priorities<br>15 update: gfinal = � K i =1 gi<br>// Update modified gradients<br>
### **4.3. Phase 1: Learning the task priority** 

Our first approach is very simple and intuitive. Here, the notation follows Sec. 3.1 and Sec. 4.1. For simplicity, we assume all tasks’ losses are equally weighted _w_ 1 = _w_ 2 = _..._ = _wK_ = 1 _/K_ . According to conventional gradient descent (GD), we have 



for _i_ = 1 _, ..., K_ . In standard GD, the network struggles to prioritize tasks since all tasks’ gradients are updated simultaneously at each step. Instead, we sequentially update each task’s gradients, as outlined below: 



for _i_ = 1 _, ..., K_ . The intuition behind this optimization is to let the network divide shared parameters Θ _s_ into _{θs,_ 1 _, θs,_ 2 _, ..., θs,K}_ based on task priority by updating each task-specific connection sequentially. After the initial gradient descent step modifies both Θ _s_ and Θ1, _θs,_ 1 start to better align with _τ_ 1. In the second step, the network can determine whether _θs,_ 1 would be beneficial for _τ_ 2. Throughout this process, task priorities are learned by updating the task’s loss in turn. Recognizing task priority effectively enables the tasks to parse out task-specific information. 

### **4.4. Phase 2: Conserving the task priority** 

Due to negative transfer between tasks, task losses fluctuate during training, resulting in variations in multi-task performance. Therefore, we introduce a secondary optimization phase to update gradients preserving task priority. For this phase, we employ the connection strength defined 

367 

Table 1. The experimental results of different multi-task learning optimization methods on NYUD-v2 with HRNet-18. The weights of tasks are manually tuned. Experiments are repeated over 3 random seeds and average values are presented. 

|Tasks|D|epth||SemSeg|||Sur|face Nor|mal|||
|---|---|---|---|---|---|---|---|---|---|---|---|
|Method|Dis<br>(Lowe<br>rmse|tance<br>r Better)<br>abs<br>~~r~~el|(H<br>mIoU|(%)<br>igher Be<br>PAcc|tter)<br>mAcc|Angle<br>(Low<br>mean|Distance<br>er Better)<br>median|With<br>(H<br>11.25|in t degre<br>igher Bett<br>22.5|e (%)<br>er)<br>30|MTP<br>_△m ↑_(%)|
|Independent|0.667|0.186|33.18|65.04|45.07|20.75|14.04|41.32|68.26|78.04|+ 0.00|
|GD|0.594|0.150|38.67|69.16|51.12|20.52|13.46|42.63|69.00|78.42|+ 9.53|
|MGDA [36]|0.603|0.159|38.89|69.39|51.53|20.58|13.56|42.28|68.79|78.33|+ 9.21|
|PCGrad [45]|0.596|0.149|38.61|69.30|51.51|20.50|13.54|42.56|69.14|78.55|+ 9.40|
|CAGrad [26]|0.595|0.153|38.80|68.95|50.78|20.38|13.53|42.89|69.33|78.71|+ 9.84|
|Aligned-MTL [37]|0.592|0.150|39.02|68.98|51.83|20.40|13.57|42.83|69.26|78.69|+ 10.17|
|Ours|**0.565**|**0.148**|**41.10**|**70.37**|**53.74**|**19.54**|**12.45**|**46.11**|**71.54**|**80.12**|**+ 15.00**|



in Eq. (7). Because of its normalization, individual tasks cannot be highly prioritized across the entire network. The top priority task _τν_ for the channel _c_<sup>_out_</sup> _p_ is determined by evaluating the connection strength as follows: 



After determining the priority of tasks in each output channel, the gradient vector of each task is aligned with the gradient of the top priority task. In detail, we categorize output channel _{c_<sup>_out_</sup> _p }_<sup>_N_</sup> _p_ =1<sup>_O_into channel groups</sup><sup>_{CGi}_</sup> _i_<sup>_K_</sup> =1<sup>based on</sup> their top priority task. The parameter of each channel group _CGi_ corresponds to _θs,i_ in Θ _s_ = _{θs,_ 1 _, θs,_ 2 _, ..., θs,K}_ . Let _{Gi,_ 1 _, Gi,_ 2 _, ..., Gi,K}_ are task-specific gradients of _CGi_ . Then _Gi,i_ acts as the reference vector for identifying conflicting gradients. When another gradient vector _Gi,j_ , where _i̸_ = _j_ , clashes with _Gi,i_ , we adjust _Gi,j_ to lie on the perpendicular plane of the reference vector _Gi,i_ to minimize negative transfer. After projecting gradients based on task priority, the sum of them is finally updated. 

In the final step, we blend two optimization stages by picking a number _P_ from a uniform distribution spanning from 0 to 1. We define _E_ as the total number of epochs and _e_ as the current epoch. The choice of optimization for that epoch hinges on whether _P_ exceeds _e/E_ . As we approach the end of the training, the probability of selecting Phase 2 increases. This is to preserve the task priority learned in Phase 1 while updating the gradient in Phase 2. A detailed view of the optimization process is provided in Algorithm 1. The reason for mixing two phases instead of completely separating them is that the speed of learning task priority varies depending on the position within the network. 

Previous studies [26, 36, 37, 45] deal with conflicting gradients by adjusting them to align in the same direction. These studies attempt to find an intermediate point among gradient vectors, which often leads to negative transfer due to the influence of the dominant task. In comparison, our approach facilitates the network’s understanding of which shared parameter holds greater significance for a given task, thereby minimizing negative transfer more efficiently. The key distinction between earlier methods and ours is the inclusion of task priority. 

## **5. Experiments** 

### **5.1. Experimental Setup** 

**Datasets.** Our method is evaluated on three multi-task datasets: NYUD-v2 [38], PASCAL-Context [33], and Cityscapes [5]. These datasets contain different kinds of vision tasks. NYUD-v2 contains 4 vision tasks: Our evaluation is based on depth estimation, semantic segmentation, and surface normal prediction, with edge detection as an auxiliary task. PASCAL-Context contains 5 tasks: We evaluate semantic segmentation, human parts estimation, saliency estimation, and surface normal prediction, with edge detection as an auxiliary task. Cityscapes contains 2 tasks: We use semantic segmentation and depth estimation. **Baselines.** We conduct extensive experiments with the following baselines: 1) single-task learning: training each task separately; 2) GD: simply updating all tasks’ gradients jointly without any manipulation; 3) multi-task optimization methods with gradient manipulation: MGDA [36], PCGrad [45], CAGrad [26], Aligned-MTL [37]; 3) loss scaling methods: We consider 4 types of loss weighting where two of them are fixed during training and the other two use dynamically varying weights. Static setting includes equal loss: all tasks are weighted equally; manually tuned loss: all tasks are weighted manually following works in [40, 43]. Dynamic setting includes uncertainty-based approach [22]: tasks’ weights are determined dynamically based on homoscedastic uncertainty; DWA [29]: tasks’ losses are determined considering the descending rate of loss to determine tasks’ weight dynamically. 4) Architecture design methods including NAS-like approaches: Cross-Stitch [32] architecture based on SegNet [1]; Recon [13]: turn shared layers into task-specific layers when conflicting gradients are detected. All experiments are conducted 3 times with different random seeds for a fair comparison. 

**Evaluation Metrics.** To evaluate the multi-task performance (MTP), we utilized the metric proposed in [31]. It measures the per-task performance by averaging it with respect to the single-task baseline b, as shown in _△m_ = (1 _/T_ )<sup>�</sup><sup>_T_</sup> _i_ =1<sup>(</sup><sup>_−_1)</sup><sup>_li_(</sup><sup>_Mm,i−Mb,i_)</sup><sup>_/Mb,i_where</sup><sup>_li_=1if</sup> a lower value of measure _Mi_ means better performance for 

368 

Table 2. The experimental results of different multi-task learning optimization methods on PASCAL-Context with HRNet-18. The weights of tasks are manually tuned. Experiments are repeated over 3 random seeds and average values are presented. 

|Tasks|Se|mSeg|PartSeg|Sal|iency||Sur|face Nor|mal|||
|---|---|---|---|---|---|---|---|---|---|---|---|
|Method|(Highe|r Better)|(Higher Better)|(Highe|r Better)|Angle<br>(Low|Distance<br>er Better)|With<br>(H|in t degre<br>igher Bett|e (%)<br>er)|MTP|
||mIoU|PAcc|mIoU|mIoU|maxF|mean|median|<br>11.25|<br>22.5|<br>30|_△m ↑_(%)|
|Independent|60.30|89.88|60.56|67.05|78.98|14.76|11.92|47.61|81.02|90.65|+ 0.00|
|GD|62.17|90.27|61.15|67.99|79.60|14.70|11.81|47.55|80.97|90.56|+ 1.47|
|MGDA [36]|61.75|89.98|61.69|67.32|78.98|14.77|12.22|47.02|80.91|90.14|+ 1.15|
|PCGrad [45]|62.47|90.57|61.46|67.86|79.38|14.59|11.77|47.72|81.28|90.81|+ 1.86|
|CAGrad [26]|62.22|90.01|61.89|67.46|79.12|14.97|12.10|47.23|80.54|90.30|+ 1.14|
|Aligned-MTL [37]|62.43|90.51|62.05|67.94|79.57|14.76|11.86|47.44|80.78|90.46|+ 1.83|
|Ours|**63.86**|**90.65**|**63.05**|**68.30**|**79.26**|**14.33**|**11.45**|**49.08**|**81.86**|**91.05**|**+ 3.70**|



Table 3. The comparison of multi-task performance on Cityscapes. Ours demonstrate competitive results without any significant addition to the network’s parameters. 

|Method|Segm<br>(Highe|entation<br>r Better)|De<br>(Lower|pth<br>Better)|_△m ↑_(%)|#P.|
|---|---|---|---|---|---|---|
||mIoU|Pix Acc|Abs Err|Rel Err|||
|Single-task|74.36|93.22|0.0128|29.98||190.59|
|Cross-Stitch [32]|74.05|93.17|0.0162|116.66|- 79.04|190.59|
|RotoGrad [21]|73.38|92.97|0.0147|82.31|- 47.81|103.43|
|GD|74.13|93.13|0.0166|116.00|- 79.32|95.43|
|w/ Recon [13]|71.17|93.21|0.0136|43.18|- 12.63|108.44|
|MGDA [36]|70.74|92.19|0.0130|47.09|- 16.22|95.43|
|w/ Recon [13]|71.01|92.17|0.0129|**33.41**|**- 4.46**|108.44|
|Graddrop [4]|74.08|93.08|0.0173|115.79|- 80.48|95.43|
|w/ Recon [13]|74.17|93.11|0.0134|41.37|- 10.69|108.44|
|PCGrad [45]|73.98|93.08|0.02|114.50|- 78.39|95.43|
|w/ Recon [13]|74.18|93.14|0.0136|46.02|- 14.92|108.44|
|CAGrad [26]|73.81|93.02|0.0153|88.29|- 53.81|95.43|
|w/ Recon [13]|74.22|93.10|0.0130|38.27|- 7.38|108.44|
|Ours|**74.75**|**93.39**|**0.0125**|41.60|- 10.08|95.48|







(a) NYUD-v2<br>(b) PASCAL-Context<br>
Figure 2. The comparison of training losses on the NYUDv2 and PASCAL-Context. Ours find a new Pareto optimal solution for multiple tasks. 

task _i_ , and 0 otherwise. We measured the single-task performance of each task _i_ with the same backbone as baseline _b_ . To evaluate the performance of tasks, we employed widely used metrics. More details are provided in Appendix C. 

### **5.2. Experimental Results** 

**Our method achieves the largest improvements in multitask performance.** The main results on NYUD-v2, PASCAL-Context are presented in Tab. 1 and Tab. 2 respectively. For a fair comparison, we compare various optimization methods on exactly the same architecture with identical task-specific layers. Tasks’ losses are tuned manually following the setting in [40, 43]. Compared to previous methods, our approach shows better performance on most tasks and datasets. It proves our method tends to induce less task interference. 

**Proposed optimization works robustly on various loss scaling methods.** To prove the generality of our method, we conduct extensive experiments on NYUD-v2 as shown in Tabs. 1 and 5 to 7 (Appendix D.1) and PASCAL-Context as shown in Tabs. 2 and 12 to 14 (Appendix D.3). In almost all types of loss scaling, our method shows the best multitask performance. Unlike conventional approaches where the effectiveness of optimization varies depending on the 

loss scaling method, ours can be applied to various types of loss weighting and shows robust results. 

**Our method can be applied to various types of network architecture.** We use MTI-Net [40] with HRNet-18 [41] and ResNet-18 [17] on NYUD-v2 and PASCAL-Context. HRNet-18 and ResNet-18 are pre-trained on ImageNet [23]. On the other hand, we use SegNet [1] for Cityscapes from scratch following the experiments setting in [13, 26]. Our optimization shows robustly better performance with different neural network architectures. The results with ResNet18 are also experimented with various loss scaling as shown in Tabs. 8 to 11 (Appendix D.2). 

**Results are compatible with various architectures with fewer parameters.** In Tab. 3, we evaluate our methods in different aspects by considering the various types of architecture. In the table, we include the results of Recon [13] to show our method can mitigate negative transfer between tasks more parameter efficiently. Compared to CrossStitch [32] and RotoGrad [21], ours show better multi-task performance with fewer parameters. Compared to Recon, our method is more parameter efficient as it increases the number of parameters by about 0.05% with the use of taskspecific batch normalization. Our method shows comparable performance on Cityscapes with fewer parameters. 

369 

Table 4. Comparison of multi-task performance using each phase individually, sequentially, and by the proposed mixing method on NYUD-v2. 

|Ph|ase|Depth|Seg|Norm|MTP|Averaged|
|---|---|---|---|---|---|---|
|1|2|rmse|mIoU|mean|_△m ↑_|Loss|
|✓||0.581|40.36|19.55|+ 13.44|**0.5396**|
||✓|0.597|39.23|20.39|+ 10.32|0.6519|
|✓_seq_|✓_seq_|0.574|40.38|19.56|+ 13.79|0.5788|
|✓_mix_|✓_mix_|**0.565**|**41.10**|**19.54**|**+ 15.50**|0.5942|





Figure 3. Correlation of loss trends across tasks during the epochs. a) Phase 1, b) Phase 2. 

**Our method finds new Pareto optimal solutions for multiple tasks.** The final task-specific loss and their average are shown in Fig. 2 for NYUD-v2 and PASCAL-Context. We compare our method with previous gradient manipulation techniques and repeat the experiments over 3 random seeds. For both NYUD-v2 and PASCAL-Context, ours show the lowest average training loss. When comparing each task individually, ours still shows the lowest final loss on every task. This provides proof that our method leads to the expansion of the Pareto frontier of previous approaches. 

### **5.3. Ablation Study** 

**Phase 1 learns task priority to find Pareto-optimal solutions.** We perform ablation studies on each stage of optimization as shown in Tab. 4. When solely utilizing phase 2, its performance has no big difference from the previous optimization techniques. However, when the first phase was used, the lowest averaged multi-task loss was achieved. Additionally, we show the correlation of loss trends in Fig. 3. The closer the value is to 1, the more it means that the loss of the task pair decreases together. In the initial stages of optimization, phase 1 appears to align the loss more effectively than solely relying on phase 2. This shows that phase 1 aids the network in differentiating task-specific details, leading to the identification of optimal Pareto solutions. 

**During Phase 2, the task’s priority is likely to be maintained.** We evaluate the top priority task within the shared space of network using Eq. (10). Subsequently, we visualized the percentage of top priority tasks in Fig. 4. It illustrates how much of the output channels in the shared convolutional layer each task has priority over. We compared when we used only Phase 1 and when we used both Phase 1 and Phase 2. We found Phase 2 at the latter half of the optimization has an effect on conserving learned task priority. This method of priority allocation prevents a specific task from exerting a dominant influence over the entire network as discussed with Eq. (7). 

**Mixing two phases shows higher performance than us-** 



Figure 4. Visualization of the percentage of top-priority tasks over training epoch. a) Phase 1, b) Mixing Phase 1 and Phase 2 

**ing each phase separately.** In Tab. 4, using only Phase 1 results in a lower multi-task loss than when mixing the two phases. Nonetheless, combining both phases enhances multi-task performance. This improvement can be attributed to the normalized connection strength (refer to Eq. (7)), which ensures that no single task dominates the entire network during Phase 2. When the two phases are applied sequentially, performance declines compared to our mixing strategy. The reason for this performance degradation seems to be the application of Phase 1 at the later stages of Optimization. This continuously alters the established task priority, which in turn disrupts the gradient’s proper updating based on the learned priority. 

## **6. Conclusion** 

In this paper, we present a novel optimization technique for multi-task learning named connection strengthbased optimization. By recognizing task priority within shared network parameters and measuring it using connection strength, we pinpoint which parameters are crucial for distinct tasks. By learning and preserving this task priority during optimization, we are able to identify new Pareto optimal solutions, boosting multi-task performance. We validate the efficacy of our approaches through comprehensive experiments and analysis. 

**Acknowledgements** This research was supported by National Research Foundation of Korea (NRF) grant funded by the Korea government (MSIT) (NRF2022R1A2B5B03002636) and the Challengeable Future Defense Technology Research, Development Program through the Agency For Defense Development (ADD) funded by the Defense Acquisition Program Administration (DAPA) in 2024 (No.912768601), and the Technology Innovation Program (1415187329, 20024355, Development of autonomous driving connectivity technology based on sensor-infrastructure cooperation) funded By the Ministry of Trade, Industry Energy(MOTIE, Korea). 

370 

## **References** 

- [1] Vijay Badrinarayanan, Alex Kendall, and Roberto Cipolla. Segnet: A deep convolutional encoder-decoder architecture for image segmentation. _IEEE transactions on pattern analysis and machine intelligence_ , 39(12):2481–2495, 2017. 6, 7, 8 

- [2] Rich Caruana. Multitask learning. _Machine learning_ , 28: 41–75, 1997. 1 

- [3] Zhao Chen, Vijay Badrinarayanan, Chen-Yu Lee, and Andrew Rabinovich. Gradnorm: Gradient normalization for adaptive loss balancing in deep multitask networks. In _International conference on machine learning_ , pages 794–803. PMLR, 2018. 2 

- [4] Zhao Chen, Jiquan Ngiam, Yanping Huang, Thang Luong, Henrik Kretzschmar, Yuning Chai, and Dragomir Anguelov. Just pick a sign: Optimizing deep multitask models with gradient sign dropout. _Advances in Neural Information Processing Systems_ , 33:2039–2050, 2020. 2, 7 

- [5] Marius Cordts, Mohamed Omran, Sebastian Ramos, Timo Rehfeld, Markus Enzweiler, Rodrigo Benenson, Uwe Franke, Stefan Roth, and Bernt Schiele. The cityscapes dataset for semantic urban scene understanding. In _Proceedings of the IEEE conference on computer vision and pattern recognition_ , pages 3213–3223, 2016. 6 

- [6] Michael Crawshaw. Multi-task learning with deep neural networks: A survey. _arXiv preprint arXiv:2009.09796_ , 2020. 1 

- [7] Jifeng Dai, Kaiming He, and Jian Sun. Instance-aware semantic segmentation via multi-task network cascades. In _Proceedings of the IEEE conference on computer vision and pattern recognition_ , pages 3150–3158, 2016. 2 

- [8] Jean-Antoine D´esid´eri. Multiple-gradient descent algorithm (mgda) for multiobjective optimization. _Comptes Rendus Mathematique_ , 350(5-6):313–318, 2012. 2 

- [9] David Eigen and Rob Fergus. Predicting depth, surface normals and semantic labels with a common multi-scale convolutional architecture. In _Proceedings of the IEEE international conference on computer vision_ , pages 2650–2658, 2015. 2 

- [10] David Eigen and Rob Fergus. Predicting depth, surface normals and semantic labels with a common multi-scale convolutional architecture. In _Proceedings of the IEEE international conference on computer vision_ , pages 2650–2658, 2015. 8 

- [11] David Eigen, Christian Puhrsch, and Rob Fergus. Depth map prediction from a single image using a multi-scale deep network. _Advances in neural information processing systems_ , 27, 2014. 8 

- [12] Yuan Gao, Jiayi Ma, Mingbo Zhao, Wei Liu, and Alan L Yuille. Nddr-cnn: Layerwise feature fusing in multi-task cnns by neural discriminative dimensionality reduction. In _Proceedings of the IEEE/CVF conference on computer vision and pattern recognition_ , pages 3205–3214, 2019. 2 

- [13] SHI Guangyuan, Qimai Li, Wenlong Zhang, Jiaxin Chen, and Xiao-Ming Wu. Recon: Reducing conflicting gradients from the root for multi-task learning. In _The Eleventh Inter-_ 

_national Conference on Learning Representations_ , 2022. 2, 6, 7, 8 

- [14] Michelle Guo, Albert Haque, De-An Huang, Serena Yeung, and Li Fei-Fei. Dynamic task prioritization for multitask learning. In _Proceedings of the European conference on computer vision (ECCV)_ , pages 270–287, 2018. 2 

- [15] Yiwen Guo, Anbang Yao, and Yurong Chen. Dynamic network surgery for efficient dnns. _Advances in neural information processing systems_ , 29, 2016. 4 

- [16] Song Han, Jeff Pool, John Tran, and William Dally. Learning both weights and connections for efficient neural network. _Advances in neural information processing systems_ , 28, 2015. 4 

- [17] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In _Proceedings of the IEEE conference on computer vision and pattern recognition_ , pages 770–778, 2016. 7 

- [18] Yang He, Guoliang Kang, Xuanyi Dong, Yanwei Fu, and Yi Yang. Soft filter pruning for accelerating deep convolutional neural networks. _arXiv preprint arXiv:1808.06866_ , 2018. 4 

- [19] Yang He, Ping Liu, Ziwei Wang, Zhilan Hu, and Yi Yang. Filter pruning via geometric median for deep convolutional neural networks acceleration. In _Proceedings of the IEEE/CVF conference on computer vision and pattern recognition_ , pages 4340–4349, 2019. 4 

- [20] Sergey Ioffe and Christian Szegedy. Batch normalization: Accelerating deep network training by reducing internal covariate shift. In _International conference on machine learning_ , pages 448–456. PMLR, 2015. 5 

- [21] Adri´an Javaloy and Isabel Valera. Rotograd: Gradient homogenization in multitask learning. _arXiv preprint arXiv:2103.02631_ , 2021. 2, 7 

- [22] Alex Kendall, Yarin Gal, and Roberto Cipolla. Multi-task learning using uncertainty to weigh losses for scene geometry and semantics. In _Proceedings of the IEEE conference on computer vision and pattern recognition_ , pages 7482–7491, 2018. 1, 2, 6, 8 

- [23] Alex Krizhevsky, Ilya Sutskever, and Geoffrey E Hinton. Imagenet classification with deep convolutional neural networks. _Communications of the ACM_ , 60(6):84–90, 2017. 7 

- [24] Hao Li, Asim Kadav, Igor Durdanovic, Hanan Samet, and Hans Peter Graf. Pruning filters for efficient convnets. _arXiv preprint arXiv:1608.08710_ , 2016. 4 

- [25] Mingbao Lin, Liujuan Cao, Shaojie Li, Qixiang Ye, Yonghong Tian, Jianzhuang Liu, Qi Tian, and Rongrong Ji. Filter sketch for network pruning. _IEEE Transactions on Neural Networks and Learning Systems_ , 2021. 4 

- [26] Bo Liu, Xingchao Liu, Xiaojie Jin, Peter Stone, and Qiang Liu. Conflict-averse gradient descent for multi-task learning. _Advances in Neural Information Processing Systems_ , 34:18878–18890, 2021. 1, 2, 3, 6, 7, 8, 9, 10, 11, 12, 13 

- [27] Fayao Liu, Chunhua Shen, and Guosheng Lin. Deep convolutional neural fields for depth estimation from a single image. In _Proceedings of the IEEE conference on computer vision and pattern recognition_ , pages 5162–5170, 2015. 8 

- [28] Liyang Liu, Yi Li, Zhanghui Kuang, J Xue, Yimin Chen, Wenming Yang, Qingmin Liao, and Wayne Zhang. Towards impartial multi-task learning. iclr, 2021. 2 

371 

- [29] Shikun Liu, Edward Johns, and Andrew J Davison. Endto-end multi-task learning with attention. In _Proceedings of the IEEE/CVF conference on computer vision and pattern recognition_ , pages 1871–1880, 2019. 1, 2, 6, 8 

- [30] Jiaqi Ma, Zhe Zhao, Xinyang Yi, Jilin Chen, Lichan Hong, and Ed H Chi. Modeling task relationships in multi-task learning with multi-gate mixture-of-experts. In _Proceedings of the 24th ACM SIGKDD international conference on knowledge discovery & data mining_ , pages 1930–1939, 2018. 2 

- [31] Kevis-Kokitsi Maninis, Ilija Radosavovic, and Iasonas Kokkinos. Attentive single-tasking of multiple tasks. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pages 1851–1860, 2019. 6 

- [32] Ishan Misra, Abhinav Shrivastava, Abhinav Gupta, and Martial Hebert. Cross-stitch networks for multi-task learning. In _Proceedings of the IEEE conference on computer vision and pattern recognition_ , pages 3994–4003, 2016. 6, 7 

- [33] Roozbeh Mottaghi, Xianjie Chen, Xiaobai Liu, Nam-Gyu Cho, Seong-Whan Lee, Sanja Fidler, Raquel Urtasun, and Alan Yuille. The role of context for object detection and semantic segmentation in the wild. In _Proceedings of the IEEE conference on computer vision and pattern recognition_ , pages 891–898, 2014. 6 

- [34] Aviv Navon, Aviv Shamsian, Idan Achituve, Haggai Maron, Kenji Kawaguchi, Gal Chechik, and Ethan Fetaya. Multitask learning as a bargaining game. _arXiv preprint arXiv:2202.01017_ , 2022. 2 

- [35] Shreyas Saxena and Jakob Verbeek. Convolutional neural fabrics. _Advances in neural information processing systems_ , 29, 2016. 4 

- [36] Ozan Sener and Vladlen Koltun. Multi-task learning as multi-objective optimization. _Advances in neural information processing systems_ , 31, 2018. 1, 2, 3, 6, 7, 9, 10, 11, 12, 13 

      - Tan, Xinggang Wang, et al. Deep high-resolution representation learning for visual recognition. _IEEE transactions on pattern analysis and machine intelligence_ , 43(10):3349– 3364, 2020. 7 

   - [42] Dan Xu, Elisa Ricci, Wanli Ouyang, Xiaogang Wang, and Nicu Sebe. Multi-scale continuous crfs as sequential deep networks for monocular depth estimation. In _Proceedings of the IEEE conference on computer vision and pattern recognition_ , pages 5354–5362, 2017. 8 

   - [43] Dan Xu, Wanli Ouyang, Xiaogang Wang, and Nicu Sebe. Pad-net: Multi-tasks guided prediction-and-distillation network for simultaneous depth estimation and scene parsing. In _Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition_ , pages 675–684, 2018. 2, 6, 7, 8 

   - [44] Ruichi Yu, Ang Li, Chun-Fu Chen, Jui-Hsin Lai, Vlad I Morariu, Xintong Han, Mingfei Gao, Ching-Yung Lin, and Larry S Davis. Nisp: Pruning networks using neuron importance score propagation. In _Proceedings of the IEEE conference on computer vision and pattern recognition_ , pages 9194–9203, 2018. 4 

   - [45] Tianhe Yu, Saurabh Kumar, Abhishek Gupta, Sergey Levine, Karol Hausman, and Chelsea Finn. Gradient surgery for multi-task learning. _Advances in Neural Information Processing Systems_ , 33:5824–5836, 2020. 1, 2, 3, 6, 7, 9, 10, 11, 12, 13 

   - [46] Zhanpeng Zhang, Ping Luo, Chen Change Loy, and Xiaoou Tang. Facial landmark detection by deep multi-task learning. In _Computer Vision–ECCV 2014: 13th European Conference, Zurich, Switzerland, September 6-12, 2014, Proceedings, Part VI 13_ , pages 94–108. Springer, 2014. 2 

   - [47] Zhenyu Zhang, Zhen Cui, Chunyan Xu, Yan Yan, Nicu Sebe, and Jian Yang. Pattern-affinitive propagation across depth, surface normal and semantic segmentation. In _Proceedings of the IEEE/CVF conference on computer vision and pattern recognition_ , pages 4106–4115, 2019. 2 

- [37] Dmitry Senushkin, Nikolay Patakin, Arseny Kuznetsov, and Anton Konushin. Independent component alignment for multi-task learning. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pages 20083–20093, 2023. 1, 2, 3, 6, 7, 9, 10, 11, 12, 13 

- [38] Nathan Silberman, Derek Hoiem, Pushmeet Kohli, and Rob Fergus. Indoor segmentation and support inference from rgbd images. In _Computer Vision–ECCV 2012: 12th European Conference on Computer Vision, Florence, Italy, October 7-13, 2012, Proceedings, Part V 12_ , pages 746–760. Springer, 2012. 6 

- [39] Karen Simonyan and Andrew Zisserman. Very deep convolutional networks for large-scale image recognition. _arXiv preprint arXiv:1409.1556_ , 2014. 2 

- [40] Simon Vandenhende, Stamatios Georgoulis, and Luc Van Gool. Mti-net: Multi-scale task interaction networks for multi-task learning. In _Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part IV 16_ , pages 527–543. Springer, 2020. 2, 6, 7, 8 

- [41] Jingdong Wang, Ke Sun, Tianheng Cheng, Borui Jiang, Chaorui Deng, Yang Zhao, Dong Liu, Yadong Mu, Mingkui 

372 

