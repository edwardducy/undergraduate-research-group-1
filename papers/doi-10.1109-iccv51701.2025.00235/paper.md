# **Beyond Losses Reweighting: Empowering Multi-Task Learning via the Generalization Perspective** 

Hoang Phan<sup>1</sup><sup>_,∗_</sup> , Lam Tran<sup>2</sup><sup>_,∗_</sup> , Quyen Tran<sup>2</sup> , Ngoc Tran<sup>3</sup> , Tuan Truong<sup>2</sup><sup>_,†_</sup> , Qi Lei<sup>1</sup> , Nhat Ho<sup>4</sup> , Dinh Phung<sup>2</sup><sup>_,_5</sup> , Trung Le<sup>5</sup> 

1New York University, 2Qualcomm AI Research+, 3Vanderbilt University, 4University of Texas at Austin, 5Monash University 

## **Abstract** 

_Multi-task learning (MTL) trains deep neural networks to optimize several objectives simultaneously using a shared backbone, which leads to reduced computational costs, improved data efficiency, and enhanced performance through cross-task knowledge sharing. Although recent gradient manipulation techniques aim to find a common descent direction that benefits all tasks, conventional empirical loss minimization still leaves models vulnerable to overfitting and gradient conflicts. To address this, we introduce a novel MTL framework that leverages weight perturbation to regulate gradient norms, thus improving generalization. By carefully modulating weight perturbations, our approach harmonizes task-specific gradients, reducing conflicts and encouraging more robust learning across tasks. Theoretical insights reveal that controlling the gradient norm through weight perturbation directly contributes to better generalization. Extensive experiments across diverse applications demonstrate that our method significantly outperforms existing gradientbased MTL techniques in terms of task performance and overall model robustness._ 

## **1. Introduction** 

Over the past few years, deep learning has emerged as a powerful tool for functional approximation, demonstrating superior performance and even surpassing human abilities in various applications. Despite these impressive achievements, training massive independent neural networks for individual tasks demands significant computational and storage resources, as well as extended runtime. Consequently, multitask learning has become a preferable approach in many situations [47, 74, 80] as it aims to learn a shared network among tasks, reducing redundant feature calculations while 

> _∗_ Equal contributions. 

> _†_ Work done while at Qualcomm AI Research. 

> + Qualcomm AI Research is an initiative of Qualcomm Technologies, Inc. 

promoting positive task transfer. 

However, learning such a shared backbone faces performance degradation due to gradient conflict [78], where task-specific gradients may differ in direction and magnitude, resulting in tasks canceling each other and a subset of tasks being under-optimized. To tackle this, a common approach is to manipulate task gradients to find a better update direction so that all task losses decrease in a more balanced manner. This has been found to consistently exhibit improved performance [5, 45]. However, existing state-ofthe-art methods in this vein [28, 44, 46, 62, 71, 78] often overlook the geometrical properties of the loss landscape, focusing solely on minimizing the empirical error in the optimization process, which can easily be prone to overfitting problems [30, 81]. 

Meanwhile, the overfitting problem in modern neural networks is often attributed to high-dimensional and nonconvex loss functions, which result in complex loss landscapes containing multiple local optima. Consequently, understanding the loss surface is crucial for training robust models, and developing flat minimizers remains one of the most effective strategies [30, 33, 40, 53]. Specifically, recent studies [24, 82] demonstrate that directly minimizing empirical risk often leads to a loss landscape with many sharp minima, resulting in poor generalization to unseen data. This issue is apparently further exacerbated when optimizing multiple objectives simultaneously, as in the context of multitask learning. In fact, sharp minima of each constituent objective might appear at different locations, which can result in large generalization errors on the associated task. Therefore, finding a common flat and low-loss valued region for all tasks is desirable for improving the current methods of multi-task learning. 

**Contribution.** To address the above desideratum, we propose a novel MTL training method that enhances existing gradient manipulation strategies by promoting the learning of flat loss landscapes across all tasks. Specifically, we penalize each task’s sharpness, the gap between the largest and the empirical losses within a weight perturbation [20, 82], to improve the generalization of all tasks. This is theoretically 

2440 

supported by the generalization error in Theorem 1, which shows that our approach not only orients the model toward the joint low empirical loss value across tasks but also encourages the model to reach the task-based flat regions. Importantly, our approach is model-agnostic and compatible with current gradient-based MTL approaches (see Figure 1 for an overview of our approach). By using our proposed framework, gradient conflict across tasks is significantly mitigated, which is the goal of recent gradient-based MTL studies in alleviating negative transfer between tasks. Finally, we conduct comprehensive experiments on a variety of applications to demonstrate the merit of our approach for improving not only task performance but also model robustness and calibration. Last but not least, to the best of our knowledge, ours is the first work to improve multi-task learning by investigating the geometrical properties of the model loss landscape. 

## **2. Related Work** 

### **2.1. Multi-task learning** 

In multi-task learning (MTL), we often aim to jointly train a single model to tackle multiple different but correlated tasks. It has been proven in prior work [8, 47, 48, 70] that it can not only enhance the overall performance but also reduce the memory usage and speed up the inference process. Previous studies on MTL often employ a hard parametersharing mechanism along with lightweight, task-specific modules to handle multiple tasks. 

**Pareto multi-task learning.** Originating from MultipleGradient Descent Algorithm (MGDA), a popular line of gradient-based MTL methods aims to find Pareto stationary solutions, where it is impossible to improve model performance on any particular task without diminishing performance on another [71]. Moreover, recent studies suggest exploring the entire Pareto front by learning diverse solutions [42, 49, 54, 55, 68], or profiling the entire Pareto front with a hyper-network [43, 61]. While these methods are theoretically grounded and guaranteed to converge to Pareto-stationary points, the experimental results are often limited and lack comparisons in practical settings. 

**Loss and gradient balancing.** Another branch of preliminary work in MTL capitalizes on the idea of dynamically reweighting loss functions based on gradient magnitudes [12], task homoscedastic uncertainty [31], or difficulty prioritization [23] to balance the gradients across tasks. More recently, PCGrad [78] introduces a gradient manipulation procedure to avoid conflicts among tasks by projecting random task gradients onto the normal plane of the other. Similarly, [44] proposes a provably convergent method to minimize the average loss, and [46] calculates loss-scaling coefficients such that the combined gradient has equal-length projections onto individual task gradients. 

### **2.2. Flat minima** 

Flat minimizer has been found to improve the generalization ability of neural networks because it enables models to find wider local minima, by which they will be more robust against shifts between train and test losses [19, 29, 67]. This relationship between generalization ability and the width of minima is theoretically and empirically studied in many studies [17, 21, 26, 63], and subsequently, a variety of methods seeking flat minima have been proposed [10, 27, 32, 66]. 

Recently, SAM [20], which seeks flat regions by explicitly minimizing the worst-case loss around the current model, has received significant attention due to its effectiveness and scalability compared to previous methods. Particularly, it has been exploited in a variety of tasks and domains, such as domain generalization [9, 73, 79], federated learning [7, 69], Bayesian networks [59, 64], meta-learning [1]. In addition, SAM shows its generalization ability in vision models [11] and language models [4]. However, these studies have only focused on single-task problems. Closer to our setting are the works in [15, 77] that apply SAM to Continual Learning, but their focus is the relationship between flatness and catastrophic forgetting. In this work, we leverage SAM’s principle to develop theory and devise practical methods, allowing for seeking flat minima in gradient-based multitask learning models. 

## **3. Methodology** 

This section outlines our proposed framework for enhancing gradient-based MTL methods. We begin by recalling the goal of multi-task learning and then establish upper bounds for the general loss of each task. Based on these bounds, we develop our framework to improve model generalization by guiding it toward flatter regions for each task. 

### **3.1. Multi-task Learning and Gradient-based methods** 

In multi-task learning, we are given a data-label distribution = _D_ from which we can sample a training set _S {_ ( **_x_** _i, yi_<sup>1</sup><sup>_, ..., y_</sup> _i_<sup>_m_)</sup><sup>_n_</sup> _i_ =1<sup>_}_,where</sup><sup>**_x_**</sup><sup>_i_isadataexampleand</sup> _yi_<sup>1</sup><sup>_, ..., y_</sup> _i_<sup>_m_are the labels of the tasks 1</sup><sup>_,_2</sup><sup>_, ..., m_respectively.</sup> The model for each task **_θ_**<sup>_i_</sup> = [ **_θ_** _sh,_ **_θ_**<sup>_i_</sup> _ns_<sup>] consists of the</sup> shared part **_θ_** _sh_ and the individual non-shared part **_θ_**<sup>_i_</sup> _ns_<sup>.We</sup> denote the general loss for the task _i_ by _L_<sup>_i_</sup> _D_<sup>(</sup><sup>**_θ_**</sup><sup>_i_), while its</sup> empirical loss over the training set _S_ by _L_<sup>_i_</sup> _S_<sup>(</sup><sup>**_θ_**</sup><sup>_i_).Existing</sup> work in MTL, typically MGDA [71], PCGrad [78], CAGrad [44], and IMTL [46], aim to find a model that simultaneously minimizes the empirical losses for all tasks: min � _L_<sup>1</sup> _S_ � **_θ_**<sup>1�</sup> _, ..., L_<sup>_m_</sup> _S_<sup>(</sup><sup>**_θ_**</sup><sup>_m_)</sup> � _,_ (1) **_θ_** _sh,_ **_θ_**<sup>1:</sup> _ns_<sup>_m_</sup> 

by calculating gradient **_g_**<sup>_i_</sup> for i-th task ( _i ∈_ [ _m_ ]). The current model parameter is then updated using the unified 

2441 



Task 1 Task 2<br>Shared<br>encoder<br>Task flat gradient<br>Combined flat gradient<br>Task loss gradient<br> Combined loss gradient<br>Final gradient<br>Low loss regions<br>Input<br>
Figure 1. We demonstrate our framework in a 2-task problem. For the shared part, _task-based flat gradients_ (red dashed arrows) steer the model away from sharp areas, while _task-based loss gradients_ (orange dashed arrows) lead the model into their corresponding low-loss regions. In our method, we aggregate them to find the combined flat gradient **_g_**<sup>_flat_</sup> _sh_ and combined loss gradient **_g_**<sup>_loss_</sup> _sh_<sup>, respectively.Finally,</sup> we add these two output gradients to target the joint low-loss and flat regions across the tasks. Conversely, updating task-specific non-shared parts is straightforward and much easier as only one objective is involved. 

gradient **_g_** = gradient ~~a~~ ggregate( **_g_**<sup>1</sup> _,_ **_g_**<sup>2</sup> _, . . . ,_ **_g_**<sup>_m_</sup> ), where the generic operation gradient ~~a~~ ggregate combines multiple task gradients, as proposed in gradient-based MTL studies. Details on this operation can be found in the Appendix. 

However, prior works only focused on minimizing the empirical losses and tend to be overfitting. To alleviate this, inspired by [20, 35, 75, 82], it is desirable to develop sharpness-aware MTL approaches wherein the task models simultaneously seek low loss and flat regions, which is discussed below. 

### **3.2. Sharpness minimization for MTL** 

Intuitively, flat minima are those where neighboring points also exhibit low loss values. One effective way to find such minima is to minimize the worst-case perturbation loss, as demonstrated in [20, 35]. Here, we propose applying this concept to each task objective in MTL. Formally, the worstcase loss for each task is defined as follows: 



where _|| · ||_ 2 denotes the _l_ 2 norm, _ρsh_ and _ρns_ represent the radii of the neighborhoods for the shared and non-shared parts, respectively. 

The formulation of the worst-case loss in Eq. (2) differs from that in the single-task setting, as it involves multiple objective functions, each consisting of shared and individual non-shared parameters. This complexity makes extending the generalization error bound in [20] non-trivial. In the next sub-section, we provide such bounds for the true risks 

in the context of MTL, highlighting the concept of sharpness for the shared and non-shared parts. 

### **3.3. Theoretical development** 

We informally state our main theorem that bounds the generalization performance of individual tasks by the empirical error on the training set: 

**Theorem 1.** _For any perturbation radius ρsh, ρns >_ 0 _, under the bounded-loss and mild assumptions, with probability_ 1 _− δ (over the choice of training set S ∼D) we obtain_ 





_where f_<sup>_i_</sup> : R+ _→_ R+ _, i ∈_ [ _m_ ] _are strictly increasing functions._ 

Theorem 1 establishes the connection between the generalization error of each task and its empirical training error via worst-case perturbation on the parameter space. The formally stated theorem and proof are provided in the Appendix. We note that the worst-case shared perturbation **_ϵ_** _sh_ is common for all tasks, while the worst-case non-shared perturbation **_ϵ_**<sup>_i_</sup> _ns_<sup>istailoredforeachtask</sup><sup>_i_.Thisrequires</sup> addressing multiple objectives with both non-shared and shared components in our theory development. 

Additionally and importantly, the proof in [20] invokes the PAC-Bayesian generalization bound [58]; hence, it only 

2442 

applies to the 0-1 loss in the binary classification setting. In contrast, as a theoretical contribution, we employ a more general PAC-Bayesian generalization bound [2], which only requires the loss to be bounded, thus tackling a notably wider range of losses in MTL. Hence, our theory development is not a trivial extension of prior works due to the nature of multi-objective optimization. 

inner maximization, we now have 



Note that this relaxation gives us an upper bound for the optimal value of the maximization problem in Eq. (9) because ( **_ϵ_**<sup>_∗_</sup> _sh_<sup>)T</sup><sup>_∇_</sup><sup>**_θ_**</sup> _sh_<sup>_L_</sup> _S_<sup>_i_(</sup><sup>**_θ_**</sup><sup>_sh,_</sup><sup>**_θ_**</sup> _ns_<sup>_i_)</sup> _≤_ ( **_ϵ_**<sup>_i,_</sup> _sh_<sup>_∗_)T</sup><sup>_∇_</sup><sup>**_θ_**</sup> _sh_<sup>_L_</sup> _S_<sup>_i_(</sup><sup>**_θ_**</sup><sup>_sh,_</sup><sup>**_θ_**</sup> _ns_<sup>_i_)</sup><sup>_, ∀i_</sup> = 1 _, · · · m_ . Therefore, we aim to minimize this upper-bound multi-objectives in practice. Finally, substituting Eq. (7) and Eq. (11) back to Eq. (6), the bi-level maximization in Eq. (4) has the following approximate solution: 

### **3.4. Practical method** 

Guided by Theorem 1, we first aim to solve the bi-level maximization problem for each task loss as follows: 





The above equation shows that sharpness-aware minimization for MTL requires us to minimize the following sub-objectives for each task: (i) the conventional task-loss, (ii) the norm of gradient w.r.t the task-specific head, and (iii) the norm of gradient w.r.t the shared backbone. Minimizing the first objective leads the model toward local, but possibly sharp minima, which causes overfitting and severe task conflict. Hence, to reduce this, the last two objectives help steer the model away from such sharp minima, favoring flatter ones, i.e. gradient norm minimization seeks flat minima. However, in comparison with traditional MTL, we are tasked with more objectives, possibly leading to more conflicts between all of them. In the following, we will present how we solve this problem by separately treating each sub-objectives for all tasks. 

� 

where approximation is the first order Taylor expansion with a note that **_ϵ_** _sh_ and **_ϵ_**<sup>_i_</sup> _ns_<sup>areindependent.Nowfollowing</sup> the dual norm problem as in [20], the solution for the inner maximization is 



Next, our goal is to find **_ϵ_** _sh_ that simultaneously maximizes the following objectives 



#### **3.4.1. Update the non-shared parts** 

Since each task _i_ has its own task-specific head **_θ_**<sup>_i_</sup> _ns_<sup>, we can</sup> find flat minima for **_θ_**<sup>_i_</sup> _ns_<sup>as follows:</sup> 



where _const_ is the constant independent of **_ϵ_** _sh_ . It is nontrivial to find the closed-form solution for this problem because the worst-cased perturbation **_ϵ_** _sh_ is shared among the tasks (cf. Eq. (9)). We hence relax it by separately finding **_ϵ_**<sup>_i_</sup> _sh_<sup>foreachtask(cf.Eq.(10)).Similarlytothe</sup> 

Note that computing gradient directly on the gradient norm sub-objective requires the heavy computation of Hessian matrix. Instead, we resort to the perturbed loss (12) and approximate its gradient w.r.t **_θ_**<sup>_i_</sup> _ns_<sup>asinEq.(14).</sup> This procedure is similar to single-task SAM. 

2443 

#### **3.4.2. Update the shared part** 

This is a challenging task since as shown in Eq. (13), we have to find a common **_θ_** _sh_ to not only reduce losses of all tasks, but also to reduce their gradient norms. Specifically, define _L_<sup>_i_</sup> _loss_<sup>:=</sup><sup>_L_</sup> _S_<sup>_i_(</sup><sup>**_θ_**</sup><sup>_sh,_</sup><sup>**_θ_**</sup> _ns_<sup>_i_)and</sup><sup>_Li_</sup> _flat_<sup>:=</sup> _ρsh∥∇_ **_θ_** _shL_<sup>_i_</sup> _S_<sup>(</sup><sup>**_θ_**</sup><sup>_sh,_</sup><sup>**_θ_**</sup> _ns_<sup>_i_)</sup><sup>_∥_2+</sup><sup>_ρns∥∇_</sup> **_θ_**<sup>_i_</sup> _ns_<sup>_L_</sup> _S_<sup>_i_(</sup><sup>**_θ_**</sup><sup>_sh,_</sup><sup>**_θ_**</sup> _ns_<sup>_i_)</sup><sup>_∥_2, we</sup> have 2 _× m_ objectives in total. 

It has been shown in [73, 85] that there may exist conflict between _L_<sup>_i_</sup> _loss_<sup>and</sup><sup>_Li_</sup> _flat_<sup>,leadingtoariskofincreasing</sup> the loss when minimizing sharpness. This problem can worsen in the scope of MTL where conflicts can arise not only between task objectives (inter-conflict) but also between the two purposes of each task (intra-conflict). Inspired by this evidence and the inherently different goals between these two types of losses, we propose to consider them individually. Conceptually, we decompose the original MTL problem, MTL[ _L_<sup>_i_</sup> _loss_<sup>+</sup><sup>_Li_</sup> _flat_<sup>]</sup> _i_<sup>_m_</sup> =1<sup>, into two sub-MTLs,</sup> MTL[ _L_<sup>_i_</sup> _loss_<sup>]</sup> _i_<sup>_m_</sup> =1<sup>+ MTL[</sup><sup>_Li_</sup> _flat_<sup>]</sup> _i_<sup>_m_</sup> =1<sup>.</sup> 

To solve each sub-MTL problem, we have to compute gradients of _L_<sup>_i_</sup> _loss_<sup>and</sup><sup>_Li_</sup> _flat_<sup>w.r.t</sup><sup>**_θ_**</sup><sup>_sh_.</sup> The former is straightforward, but the latter requires heavy Hessian computation. We bypass this by noticing from Eq. (13) that the perturbed loss is the sum of _L_<sup>_i_</sup> _loss_<sup>and</sup><sup>_Li_</sup> _flat_<sup>.Hence,</sup> the gradient of _L_<sup>_i_</sup> _flat_<sup>can be approximated by the difference</sup> between the gradient of the perturbed loss and that of _L_<sup>_i_</sup> _loss_<sup>.</sup> Formally, 

**_g_**<sup>_i,_</sup> 



The purpose of the negative gradient _−_ **_g_**<sup>_i,_</sup> _sh_<sup>loss</sup> is to orient the model to minimize the loss of the task _i_ , while _−_ **_g_**<sup>_i,_</sup> _sh_<sup>flat</sup> navigates the model to the task _i_ ’s flatter region. Therefore, the gradients **_g_**<sup>_i,_</sup> _sh_<sup>loss</sup> share a similar nature, making them likely congruent. A similar relationship holds for **_g_**<sup>_i,_</sup> _sh_<sup>flat.</sup> To solve each sub-MTL, following gradient-based MTL methods that aggregate gradients such that their conflict is reduced, we aim to find a common direction that leads the joint low-valued losses for all tasks and the joint flatter region for them as: 



Finally, to combine two sub-MTL problems, we can similarly aggregate **_g_**<sup>loss</sup> _sh_<sup>and</sup><sup>**_g_**flat</sup> _sh_<sup>basedongradient-basedMTL</sup> methods. However, in practice, we find that simply adding 

them can work well in most cases (e.g., Ours vs Secondaggre in Table 6), so we adopt this strategy to save computation: **_g_**<sup>SAM</sup> _sh_ = **_g_**<sup>loss</sup> _sh_<sup>+</sup><sup>**_g_**flat</sup> _sh_<sup>;</sup><sup>**_θ_**</sup><sup>_sh_=</sup><sup>**_θ_**</sup><sup>_sh −η_</sup><sup>**_g_**SAM</sup> _sh_<sup>.</sup> 

The key steps of our proposed framework are summarized in Algorithm 1, and the overall schema of our proposed method is demonstrated in Figure 1. 

Note that one can apply gradient-based methods to remove intra-conflict between **_g_**<sup>_i,loss_</sup> _sh_ and **_g_**<sup>_i,flat_</sup> _sh_ for each task to obtain **_g_**<sup>_i_</sup> _sh_<sup>, then aggregate them once more time. This</sup> strategy, however, is extremely computationally expensive as we have to use gradient-aggregator _m_ + 1 times, and still results in similar performance compared to our method, i.e., Each-aggre vs Ours in Table 6. 

Another approach is to directly aggregate **_g_**<sup>_i,SAM_</sup> _sh_ of each task, i.e, ignoring the intra-conflict. This might still result in a higher level of gradient conflict than in our method which considers intra-conflict. We empirically demonstrate this in Figure 2 where the our strategy gains lower loss values and gradient norms than the direct strategy, and the effectiveness of our method in Table 5. 



## **4. Experiments** 

**Datasets and Baselines.** Our proposed method is evaluated on four MTL benchmarks, including Multi-MNIST [42], CelebA [51] for visual classification, and NYUv2 [72], CityScapes [14] for scene understanding. Their descriptions can be found in Appendix C. We show how our framework boosts performance of gradient-based MTL methods by comparing _vanilla_ MGDA [71], PCGrad [78], CAGrad [44], IMTL [46], NashMTL [62], FairGrad [5] to their flatbased versions F-MGDA, F-PCGrad, F-CAGrad, F-IMTL, 

2444 

|||MultiFashion|||MultiMNIST||M|ultiFashion+MNI|ST|
|---|---|---|---|---|---|---|---|---|---|
|Method||||||||||
||Task 1_↑_|Task 2_↑_|Average_↑_|Task 1_↑_|Task 2_↑_|Average_↑_|Task 1_↑_|Task 2_↑_|Average_↑_|
|STL|87_._10_±_0_._09|86_._20_±_0_._06|86_._65_±_0_._02|95_._33_±_0_._08|94_._16_±_0_._04|94_._74_±_0_._06|98_._40_±_0_._02|89_._42_±_0_._03|93_._91_±_0_._02|
|MGDA|86_._76_±_0_._09|85_._78_±_0_._36|86_._27_±_0_._22|95_._62_±_0_._02|94_._49_±_0_._10|95_._05_±_0_._06|97_._24_±_0_._04|88_._19_±_0_._13|92_._72_±_0_._07|
|F-MGDA|**88**_._**12**_±_**0**_._**11**|**87**_._**35**_±_**0**_._**11**|**87**_._**73**_±_**0**_._**09**|**96**_._**37**_±_**0**_._**06**|**94**_._**99**_±_**0**_._**06**|**95**_._**68**_±_**0**_._**00**|**97**_._**30**_±_**0**_._**09**|**89**_._**26**_±_**0**_._**14**|**93**_._**28**_±_**0**_._**03**|
|PCGrad|86_._93_±_0_._17|86_._20_±_0_._14|86_._57_±_0_._12|95_._71_±_0_._03|94_._41_±_0_._02|95_._06_±_0_._02|97_._12_±_0_._16|88_._45_±_0_._08|92_._78_±_0_._11|
|F-PCGrad|**88**_._**17**_±_**0**_._**14**|**87**_._**35**_±_**0**_._**27**|**87**_._**76**_±_**0**_._**07**|**96**_._**49**_±_**0**_._**05**|**95**_._**34**_±_**0**_._**10**|**95**_._**92**_±_**0**_._**07**|**97**_._**65**_±_**0**_._**06**|**89**_._**35**_±_**0**_._**07**<sup>_∗_</sup>|**93**_._**50**_±_**0**_._**01**|
|CAGrad|86_._99_±_0_._17|86_._04_±_0_._15|86_._51_±_0_._16|95_._62_±_0_._05|94_._39_±_0_._04|95_._01_±_0_._04|97_._19_±_0_._06|88_._18_±_0_._14|92.68_±_0_._04|
|F-CAGrad|**88**_._**19**_±_**0**_._**19**|**87**_._**45**_±_**0**_._**13**|**87**_._**82**_±_**0**_._**10**<sup>_∗_</sup>|**96**_._**54**_±_**0**_._**02**|**95**_._**36**_±_**0**_._**04**|**95**_._**95**_±_**0**_._**01**<sup>_∗_</sup>|**97**_._**82**_±_**0**_._**05**|**89**_._**26**_±_**0**_._**22**|**93**_._**54**_±_**0**_._**13**<sup>_∗_</sup>|
|IMTL|87_._35_±_0_._22|86_._45_±_0_._09|86_._90_±_0_._15|95_._93_±_0_._09|94_._63_±_0_._13|95_._28_±_0_._02|97_._47_±_0_._06|88_._46_±_0_._11|92_._97_±_0_._03|
|F-IMTL|**88**_._**1**_±_**0**_._**10**|**87**_._**50**_±_**0**_._**04**<sup>_∗_</sup>|**87**_._**80**_±_**0**_._**06**|**96**_._**55**_±_**0**_._**07**<sup>_∗_</sup>|**95**_._**16**_±_**0**_._**05**|**95**_._**85**_±_**0**_._**05**|**97**_._**59**_±_**0**_._**12**|**88**_._**99**_±_**0**_._**08**|**93**_._**29**_±_**0**_._**02**|
|NashMTL|86_._91_±_0_._09|86_._17_±_0_._03|86_._54_±_0_._04|95_._54_±_0_._00|94_._49_±_0_._09|95_._01_±_0_._05|97_._00_±_0_._18|88_._39_±_0_._16|92_._70_±_0_._02|
|F-NashMTL|**88**_._**21**_±_**0**_._**20**<sup>_∗_</sup>|**87**_._**40**_±_**0**_._**03**|**87**_._**81**_±_**0**_._**11**|**96**_._**47**_±_**0**_._**03**|**95**_._**40**_±_**0**_._**10**<sup>_∗_</sup>|**95**_._**94**_±_**0**_._**05**|**97**_._**63**_±_**0**_._**12**|**89**_._**33**_±_**0**_._**12**|**93**_._**48**_±_**0**_._**07**|
|FairGrad|86_._85_±_0_._15|86_._17_±_0_._07|86_._51_±_0_._04|95_._55_±_0_._15|94_._29_±_0_._09|94_._92_±_0_._11|97_._02_±_0_._06|88_._37_±_0_._16|92_._70_±_0_._10|
|F-FairGrad|**88**_._**05**_±_**0**_._**08**|**87**_._**41**_±_**0**_._**14**|**87**_._**73**_±_**0**_._**04**|**96**_._**48**_±_**0**_._**07**|**95**_._**34**_±_**0**_._**04**|**95**_._**91**_±_**0**_._**05**|**97**_._**94**_±_**0**_._**02**<sup>_∗_</sup>|**88**_._**99**_±_**0**_._**18**|**93**_._**46**_±_**0**_._**10**|



Table 1. Evaluation of different methods on three Multi-MNIST datasets. Rows with flat-based minimizers are shaded. Bold numbers denote higher accuracy between flat-based methods and their baselines.<sup>_∗_</sup> denotes the highest accuracy (except for STL as it unfairly exploits multiple neural networks). We also use arrows to indicate that the higher is the better ( _↑_ ) or vice-versa ( _↓_ ). 

F-NashMTL and F-FairGrad. We also add a single-task learning (STL) baseline for each dataset. 

reduced conflict, occurs when shared parameters approach a common flat region. 

### **4.1. Image classification** 

**Multi-MNIST.** Following [71], we set up three MultiMNIST experiments with ResNet18 [25], namely: MultiFashion, MultiMNIST and MultiFashion+MNIST. 

As summarized in Table 1, we can see that seeking flatter regions for all tasks can improve the performance of all the baselines across all three datasets. Especially, flat-based methods achieve the highest score for each task and for the average, outperforming STL by 1 _._ 2% on MultiFashion and MultiMNIST. We conjecture that the discrepancy between gradient update trajectories to classify digits from MNIST and fashion items from FashionMNIST has resulted in the fruitless performance of baselines, compared to STL on MultiFashion+MNIST. Even if there exists dissimilarity between tasks, our best obtained average accuracy when applying our method to CAGrad is just slightly lower than STL ( _<_ 0 _._ 4%) while employing a single model only. 

**CelebA.** CelebA [52] is a face dataset with 200K images and 40 attributes, forming a 40-class multi-label classification problem. Table 2 presents the average errors over 40 tasks, with Linear Scalarization (LS) and Uncertainty Weighting (UW) (Kendall, Gal, and Cipolla 2018) as additional baselines. The best results in each pair and overall are highlighted in bold and<sup>_∗_</sup> , respectively. Even with a large number of tasks, flat region seeking consistently shows its advantages, with F-CAGrad achieving the lowest average error. Notably, when the optimizer considers flat minima, the performance gaps between PCGrad, IMTL, and CAGrad (8.23, 8.24 vs. 8.22) are smaller than those under conventional ERM training (8.69, 8.88, and 8.52). This suggests that better aggregation of task gradients, and thus 

|Method|STL|LS|UW|MGDA|PCGrad|CAGrad|IMTL|
|---|---|---|---|---|---|---|---|
|Vanilla|8.77|9.99|9.66|9.96|8.69|8.52|8.88|
|Flat-based|-|-|-|**9.22**|**8.23**|**8.22***|**8.24**|



Table 2. Mean of error per category of MTL algorithms in multilabel classification on CelebA dataset. 

### **4.2. Scene Understanding** 

Two datasets used here are NYUv2 [72] and CityScapes [14]. For these two experiments, we additionally include several recent MTL methods, namely, scale-invariant (SI), random loss weighting (RLW), Dynamic Weight Average (DWA) [47], GradDrop [13], and Nash-MTL [62] whose results are taken from [62] and details are in Appendix C. Also following the standard protocol used in [44, 47, 62], Multi-Task Attention Network [47] is employed on top of the SegNet architecture [3], our results are averaged over the last 10 epochs to align with previous work. 

**Evaluation metric.** In this experiment, we handle different task types, each with its own metrics. We report the relative task improvement [57] to compare overall performance. Let _Mi_ and _Si_ be the metrics obtained by the main and the single-task learning (STL) model, respectively. The relative task improvement on _i_ -th task is mathematically given by: ∆ _i_ := 100 _·_ ( _−_ 1)<sup>_li_</sup> ( _Mi − Si_ ) _/Si_ , where _li_ = 1 if a lower value for the _i_ -th criterion is better and 0 otherwise. We depict our results by the average relative task improvement **∆** **_m_ %** = _m_<sup><u>1</u></sup> � _mi_ =1<sup>∆</sup><sup>_i_.</sup> 

**NYUv2.** Table 3 presents the results and relative improvements of each task over STL for different methods. Generally, the flat-based versions achieve comparable or 

2445 

||Segm|entation|De|pth||Su|rface Nor|mal|||
|---|---|---|---|---|---|---|---|---|---|---|
||mIoU_↑_|Pix Acc_↑_|Abs Err_↓_|Rel Err_↓_|Angle D<br>Mean|istance_↓_<br>Median|11.25|Within_t_<sup>_◦_</sup><br>22.5|_↑_<br>30|**∆m**%_↓_|
|STL|38_._30|63_._76|0_._6754|0_._2780|25_._01|19_._21|30_._14|57_._20|69_._15|0.00|
|LS|39_._29|65_._33|0_._5493|0_._2263|28_._15|23_._96|22_._09|47_._50|61_._08|5_._59|
|SI|38_._45|64_._27|0_._5354|0_._2201|27_._60|23_._37|22_._53|48_._57|62_._32|4_._39|
|RLW|37_._17|63_._77|0_._5759|0_._2410|28_._27|24_._18|22_._26|47_._05|60_._62|7_._78|
|DWA|39_._11|65_._31|0_._5510|0_._2285|27_._61|23_._18|24_._17|50_._18|62_._39|3_._57|
|UW|36_._87|63_._17|0_._5446|0_._2260|27_._04|22_._61|23_._54|49_._05|63_._65|4_._05|
|GradDrop|39_._39|65_._12|0_._5455|0_._2279|27_._48|22_._96|23_._38|49_._44|62_._87|3_._58|
|Nash-MTL|40_._13|65_._93|0_._5261<sup>_∗_</sup>|0_._2171|25_._26|20_._08|28_._4|55_._47|68_._15|_−_4_._04|
|MGDA|**30**_._**47**|**59**_._**90**|**0**_._**6070**|0_._2555|24_._88|19_._45|29_._18|56_._88|69_._36|1_._38|
|F-MGDA|26_._42|58_._78|0_._6078|**0**_._**2353**|**24**_._**34**<sup>_∗_</sup>|**18**_._**45**<sup>_∗_</sup>|**31**_._**64**<sup>_∗_</sup>|**58**_._**86**<sup>_∗_</sup>|**70**_._**50**<sup>_∗_</sup>|_−_**0**_._**33**|
|PCGrad|38_._06|64_._64|0_._5550|0_._2325|27_._41|**22**_._**80**|**23**_._**86**|**49**_._**83**|**63**_._**14**|3_._97|
|F-PCGrad|**40**_._**05**|**65**_._**42**|**0**_._**5429**|**0**_._**2243**|**27**_._**38**|23_._00|23_._47|49_._35|62_._74|**3**_._**14**|
|CAGrad|39_._79|65_._49|0_._5486|0_._2250|26_._31|21_._58|25_._61|52_._36|65_._58|0_._20|
|F-CAGrad|**40**_._**93**<sup>_∗_</sup>|**66**_._**68**<sup>_∗_</sup>|**0**_._**5285**|**0**_._**2162**|**25**_._**43**|**20**_._**39**|**27**_._**99**|**54**_._**82**|**67**_._**56**|_−_**3**_._**78**|
|IMTL|39_._35|65_._60|0_._5426|0_._2256|26_._02|21_._19|26_._2|53_._13|66_._24|_−_0_._76|
|F-IMTL|**40**_._**42**|**65**_._**61**|**0**_._**5389**|**0**_._**2121**<sup>_∗_</sup>|**25**_._**03**|**19**_._**75**|**28**_._**90**|**56**_._**19**|**68**_._**72**|_−_**4**_._**77**<sup>_∗_</sup>|



Table 3. Test performance for three-task NYUv2 of Segnet [3]: semantic segmentation, depth estimation, and surface normal. Using the proposed procedure with gradient-based multi-task learning methods consistently improves their overall performance. 

higher results on most metrics, except for F-MGDA in the segmentation task, where it notably decreases the _mIoU_ score. However, F-MGDA significantly boosts performance in other tasks, raising MGDA’s overall relative improvement from -1.38% to +0.33% above STL. Notably, F-CAGrad and F-IMTL outperform competitors by large margins across all tasks, with top relative improvements of 3.78% and 4.77%, respectively. 

||Segm|entation|De|pth||
|---|---|---|---|---|---|
|Method|mIoU_↑_|Pix Acc_↑_|Abs Err_↓_|Rel Err_↓_|**∆****_m_%** **_↓_**|
|STL|74_._01|93_._16|0_._0125|27_._77||
|LS|75_._18|93_._49|0_._0155|46_._77|22_._60|
|SI|70_._95|91_._73|0_._0161|33_._83|14_._11|
|RLW|74_._57|93_._41|0_._0158|47_._79|24_._38|
|DWA|75_._24|93_._52|0_._0160|44_._37|21.45|
|UW|72_._02|92_._85|0_._0140|30_._13<sup>_∗_</sup>|5.89|
|GradDrop|75_._27|93_._53|0_._0157|47_._54|23_._73|
|Nash-MTL|75_._41|93_._66|0_._0129|35_._02|6_._82|
|MGDA|68_._84|91_._54|0_._0309|33_._50|44_._14|
|F-MGDA|**73**_._**77**|**93**_._**12**|**0**_._**0129**|**27**_._**44**<sup>_∗_</sup>|**0**_._**67**<sup>_∗_</sup>|
|PCGrad|75_._13|93_._48|0_._0154|42_._07|18_._29|
|F-PCGrad|**75**_._**77**|**93**_._**67**|**0**_._**0144**|**39**_._**60**|**13**_._**65**|
|CAGrad|75_._16|93_._48|0_._0141|37_._60|11_._64|
|F-CAGrad|**76**_._**02**|**93**_._**72**|**0**_._**0134**|**34**_._**64**|**7**_._**25**|
|IMTL|75_._33|93_._49|0_._0135|38_._41|11_._10|
|F-IMTL|**76**_._**63**<sup>_∗_</sup>|**93**_._**76**<sup>_∗_</sup>|**0**_._**0124**<sup>_∗_</sup>|**31**_._**17**|**1**_._**87**|



**CityScapes.** In Table 4, the positive impact of seeking flat regions is evident across all metrics and baselines. Notably, MGDA and IMTL show significant relative improvements, achieving the highest and second-best **∆** **_m_ %** scores, respectively. PCGrad, CAGrad, and IMTL even surpass STL in segmentation scores. Interestingly, MGDA biases to the depth estimation objective, leading to the predominant performance of F-MGDA on that task, consistent with patterns observed in [46] and the NYUv2 experiment. 

### **4.3. Ablation study** 

Here, we provide experimental justification for our gradient decomposition, and our method’s impact on conventional MTL training by examining task conflict. Appendix D provides additional results for model calibration D.2, model robustness D.5, D.8, gradient norms D.9, loss landscape visualization D.4, training curves D.7, and hyper-param sensitivity D.3. 

**Directly aggregating SAM gradients neglects intraconflict.** Table 5 compares between the direct aggregation on _{_ **_g_**<sup>_i,_</sup> _sh_<sup>SAM</sup> _}_<sup>_m_</sup> _i_ =1<sup>and our individual aggregation on</sup><sup>_{_</sup><sup>**_g_**</sup><sup>_i,_</sup> _sh_<sup>flat</sup> _}_<sup>_m_</sup> _i_ =1<sup>,</sup> and _{_ **_g_**<sup>_i,_</sup> _sh_<sup>loss</sup> _}_<sup>_m_</sup> _i_ =1<sup>.</sup> 

||Segm|entation|De|pth||
|---|---|---|---|---|---|
|Method|mIoU_↑_|Pix Acc_↑_|Abs Err_↓_|Rel Err_↓_|**∆****_m_%** **_↓_**|
|ERM|68_._84|91_._54|0_._0309|33_._50|44_._14|
|Ours (**direct**)|68_._93|91_._41|0_._0130|31_._37|6_._43|
|Ours (**individual**)|**73**_._**77**|**93**_._**12**|**0**_._**0129**|**27**_._**44**<sup>_∗_</sup>|**0**_._**67**|



Table 4. Test performance for two-task CityScapes: semantic segmentation and depth estimation.<sup>_∗_</sup> denotes the best score for each task’s metrics. 

Table 5. Direct SAM gradients aggregation vs our proposed gradient aggregation strategy on CityScapes. 

2446 



600<br>individual individual<br>100<br>direct direct<br>400 75<br>50<br>200<br>25<br>0 10 20 30 40 50 60 70 80 0 10 20 30 40 50 60 70 80<br>0.8 individual 0.04 individual<br>direct direct<br>0.6 0.03<br>0.4 0.02<br>0.2<br>0 25 50 75 100 125 150 175 200 0 25 50 75 100 125 150 175 200<br>Epochs Epochs<br>Task 1 Gradnorm Task 2 Gradnorm<br>Task 1 Loss Task 2 Loss<br>
Figure 2. Evolution of gradient norms and task losses across aggregation strategies. 

Compared to the naive approach, in which per-task SAM gradients are directly aggregated, our decomposition approach consistently improves performance by a large margin across all tasks. Moreover, using our decomposed SAM yields flatter minima and lower loss values throughout the whole training process, as shown in Figure 2. These results reinforce the rationale behind separately aggregating low-loss directions and flat directions. 

**Other approaches to apply gradient aggregation.** Table 6 summarizes the performance of Second-aggre, Each-aggre and Our aggregation strategies: Second-aggre performs an additional step to aggregate **_g_**<sup>_loss_</sup> _sh_ and **_g_**<sup>_flat_</sup> _sh_<sup>;</sup> Each-aggre iteratively aggregates **_g_**<sup>_i,loss_</sup> _sh_ and **_g_**<sup>_i,flat_</sup> _sh_ to obtain **_g_**<sup>_i_</sup> _sh_<sup>for each task, then aggregate</sup><sup>**_g_**</sup><sup>_i_</sup> _sh_<sup>one more time.Our</sup> strategy requires less aggregation steps yet still achieves comparable performance. 

||Mu|ltiFashio|n|Mu|ltiMNIS|T|MultiFa|shion+M|NIST|
|---|---|---|---|---|---|---|---|---|---|
|Method|Task 1_↑_|Task 2_↑_|Avg_↑_|Task 1_↑_|Task 2_↑_|Avg_↑_|Task 1_↑_|Task 2_↑_|Avg_↑_|
|Second-aggre|88_._10|87_._70|87_._90|96_._48|95_._24|95_._86|97_._87|89_._02|93_._44|
|Each-aggre|87_._51|87_._68|87_._59|96_._45|95_._40|95_._92|97_._61|89_._17|93_._39|
|Ours|88_._19|87_._45|87_._82|96_._54|95_._36|95_._95|97_._82|89_._26|93_._54|



Table 6. Different aggregation strategies applied on CAGrad, on Multi-MNIST datasets. 

**Our improvement is not a mere result of single-task SAM.** To show this point, we provide the results of STL and linear scalarization (LS), casting MTL as a single objective, trained with and without SAM on NYUv2 in Table 7. When equipped with SAM, F-STL and F-LS improve almost all scores of their counterparts. However, they can not consistently exceed flat gradient-based MTL baselines, which take gradient conflict into account. Particularly, on Segmentation and Depth tasks, F-IMTL achieves the highest scores, while on the Surface Normal task, F-MGDA is the best method. Overall, F-IMTL obtains the best **∆** **_m_ %** . 

**Task conflict.** To empirically confirm reduced gradient conflict in flat regions, we measured the proportion of minibatches with gradient conflict at each epoch, and present 

||Segme|ntation|De|pth||Surfa|ce Norm|al|||
|---|---|---|---|---|---|---|---|---|---|---|
||mIoU_↑_|Pix Acc_↑_|Abs Err_↓_|Rel Err_↓_|Angle<br>Mean|Distance_↓_<br>Median|W<br>11.25|ithin_t_<sup>_◦_</sup><br>22.5|_↑_<br>30|∆_m_%_↓_|
|STL|38_._30|63_._76|0_._6754|0_._2780|25_._01|19_._21|30_._14|57_._20|69_._15|0.00|
|F-STL|39_._07|64_._21|0_._6183|0_._2514|25_._08|18_._72|30_._99|58_._29|69_._76|_−_3_._17|
|LS|39_._29|65_._33|0_._5493|0_._2263|28_._15|23_._96|22_._09|47_._50|61_._08|5_._59|
|F-LS|40_._28|65_._30|0_._5360|0_._2173|27_._14|22_._56|24_._45|50_._29|63_._57|1_._65|
|F-MGDA|26_._42|58_._78|0_._6078|0_._2353|24_._34|18_._45|31_._64|58_._86|70_._50|_−_0_._33|
|F-IMTL|40_._42|65_._61|0_._5389|0_._2121|25_._03|19_._75|28_._90|56_._19|68_._72|_−_4_._77|



Table 7. Test performance for three-task **NYUv2** of Segnet. 

the results in Figure 3. While ERM’s gradient conflict rises above 50%, ours decreases and approaches 0%. This reduction is also a key objective of recent gradient-based MTL methods aimed at mitigating negative transfer between tasks [74, 78, 84]. 



60<br>50<br>40<br>30<br>20<br>10<br>0 Ours ERM<br>0 25 50 75 100 125 150 175 200<br>Epochs<br>Gradient conflict (%)<br>
Figure 3. Proportion of conflict between per-task gradients ( **_g_**<sup>1</sup><sup>_,_loss</sup> _·_ **_g_**<sup>2</sup><sup>_,_loss</sup> _<_ 0) on Multi-MNIST. 

## **5. Conclusion** 

In this work, we have presented a general framework that can be incorporated into current multi-task learning methods following the gradient balancing mechanism. The core ideas of our proposed method are the employment of flat minimizers in the context of MTL and proving that they can help enhance previous works both theoretically and empirically. Concretely, our method goes beyond optimizing per-task objectives solely to yield models that have both low errors and high generalization capabilities. 

2447 

## **6. Acknowledgements** 

We thank Chau Pham for his contributions to the scene understanding experiments. This work was supported by ARC DP23 grant DP230101176 and by the Air Force Office of Scientific Research under award number FA2386-23-1-4044. 

## **References** 

- [1] Momin Abbas, Quan Xiao, Lisha Chen, Pin-Yu Chen, and Tianyi Chen. Sharp-maml: Sharpness-aware model-agnostic meta learning. _arXiv preprint arXiv:2206.03996_ , 2022. 2 

- [2] Pierre Alquier, James Ridgway, and Nicolas Chopin. On the properties of variational approximations of gibbs posteriors. _Journal of Machine Learning Research_ , 17(236):1–41, 2016. 4, 1 

- [3] Vijay Badrinarayanan, Alex Kendall, and Roberto Cipolla. Segnet: A deep convolutional encoder-decoder architecture for image segmentation. _IEEE transactions on pattern analysis and machine intelligence_ , 39(12):2481–2495, 2017. 6, 7, 5 

- [4] Dara Bahri, Hossein Mobahi, and Yi Tay. Sharpness-aware minimization improves language model generalization. In _Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)_ , pages 7360–7371, Dublin, Ireland, 2022. Association for Computational Linguistics. 2 

- [5] Hao Ban and Kaiyi Ji. Fair resource allocation in multi-task learning. In _Forty-first International Conference on Machine Learning_ , 2024. 1, 5 

- [6] Glenn W Brier et al. Verification of forecasts expressed in terms of probability. _Monthly weather review_ , 78(1):1–3, 1950. 7 

- [7] Debora Caldarola, Barbara Caputo, and Marco Ciccone. Improving generalization in federated learning by seeking flat minima. In _European Conference on Computer Vision_ , pages 654–672. Springer, 2022. 2 

- [8] Rich Caruana. Multitask learning. _Machine learning_ , 28(1): 41–75, 1997. 2 

- [9] Junbum Cha, Sanghyuk Chun, Kyungjae Lee, Han-Cheol Cho, Seunghyun Park, Yunsung Lee, and Sungrae Park. Swad: Domain generalization by seeking flat minima. _Advances in Neural Information Processing Systems_ , 34:22405–22418, 2021. 2 

- [10] Pratik Chaudhari, Anna Choromanska, Stefano Soatto, Yann´ LeCun, Carlo Baldassi, Christian Borgs, Jennifer T. Chayes, Levent Sagun, and Riccardo Zecchina. Entropy-sgd: biasing gradient descent into wide valleys. _Journal of Statistical Mechanics: Theory and Experiment_ , 2019, 2017. 2 

- [11] Xiangning Chen, Cho-Jui Hsieh, and Boqing Gong. When vision transformers outperform resnets without pre-training or strong data augmentations. _arXiv preprint arXiv:2106.01548_ , 2021. 2 

- [12] Zhao Chen, Vijay Badrinarayanan, Chen-Yu Lee, and Andrew Rabinovich. Gradnorm: Gradient normalization for adaptive loss balancing in deep multitask networks. In _International_ 

_conference on machine learning_ , pages 794–803. PMLR, 2018. 2 

- [13] Zhao Chen, Jiquan Ngiam, Yanping Huang, Thang Luong, Henrik Kretzschmar, Yuning Chai, and Dragomir Anguelov. Just pick a sign: Optimizing deep multitask models with gradient sign dropout. _Advances in Neural Information Processing Systems_ , 33:2039–2050, 2020. 6, 5 

- [14] Marius Cordts, Mohamed Omran, Sebastian Ramos, Timo Rehfeld, Markus Enzweiler, Rodrigo Benenson, Uwe Franke, Stefan Roth, and Bernt Schiele. The cityscapes dataset for semantic urban scene understanding. In _Proceedings of the IEEE conference on computer vision and pattern recognition_ , pages 3213–3223, 2016. 5, 6 

- [15] Danruo Deng, Guangyong Chen, Jianye Hao, Qiong Wang, and Pheng-Ann Heng. Flattening sharpness for dynamic gradient projection memory benefits continual learning. _Advances in Neural Information Processing Systems_ , 34: 18710–18721, 2021. 2 

- [16] Jean-Antoine Desid´ eri.´ Multiple-gradient descent algorithm (mgda) for multiobjective optimization. _Comptes Rendus Mathematique_ , 350(5-6):313–318, 2012. 4 

- [17] Laurent Dinh, Razvan Pascanu, Samy Bengio, and Yoshua Bengio. Sharp minima can generalize for deep nets. In _International Conference on Machine Learning_ , pages 1019– 1028. PMLR, 2017. 2 

- [18] Jiawei Du, Hanshu Yan, Jiashi Feng, Joey Tianyi Zhou, Liangli Zhen, R. Goh, and Vincent Y. F. Tan. Efficient sharpness-aware minimization for improved training of neural networks. _International Conference on Learning Representations_ , 2021. 11 

- [19] Gintare Karolina Dziugaite and Daniel M. Roy. Computing nonvacuous generalization bounds for deep (stochastic) neural networks with many more parameters than training data. In _UAI_ . AUAI Press, 2017. 2 

- [20] Pierre Foret, Ariel Kleiner, Hossein Mobahi, and Behnam Neyshabur. Sharpness-aware minimization for efficiently improving generalization. In _International Conference on Learning Representations_ , 2021. 1, 2, 3, 4, 5 

- [21] Stanislav Fort and Surya Ganguli. Emergent properties of the local geometry of neural loss landscapes. _arXiv preprint arXiv:1910.05929_ , 2019. 2 

- [22] Chuan Guo, Geoff Pleiss, Yu Sun, and Kilian Q Weinberger. On calibration of modern neural networks. In _International conference on machine learning_ , pages 1321–1330. PMLR, 2017. 7 

- [23] Michelle Guo, Albert Haque, De-An Huang, Serena Yeung, and Li Fei-Fei. Dynamic task prioritization for multitask learning. In _Proceedings of the European conference on computer vision (ECCV)_ , pages 270–287, 2018. 2 

- [24] Haowei He, Gao Huang, and Yang Yuan. Asymmetric valleys: Beyond sharp and flat local minima. _Advances in neural information processing systems_ , 32, 2019. 1 

- [25] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In _Proceedings of the IEEE conference on computer vision and pattern recognition_ , pages 770–778, 2016. 6, 5, 7 

2448 

- [26] Sepp Hochreiter and Jurgen Schmidhuber.¨ Simplifying neural nets by discovering flat minima. In _NIPS_ , pages 529–536. MIT Press, 1994. 2 

- [27] Pavel Izmailov, Dmitrii Podoprikhin, Timur Garipov, Dmitry P. Vetrov, and Andrew Gordon Wilson. Averaging weights leads to wider optima and better generalization. In _UAI_ , pages 876–885. AUAI Press, 2018. 2, 9 

- [28] Adrian´ Javaloy and Isabel Valera. Rotograd: Gradient homogenization in multitask learning. In _International Conference on Learning Representations_ , 2021. 1 

- [29] Yiding Jiang, Behnam Neyshabur, Hossein Mobahi, Dilip Krishnan, and Samy Bengio. Fantastic generalization measures and where to find them. In _ICLR_ . OpenReview.net, 2020. 2 

- [30] Jean Kaddour, Linqing Liu, Ricardo Silva, and Matt J Kusner. A fair comparison of two popular flat minima optimizers: Stochastic weight averaging vs. sharpness-aware minimization. _arXiv preprint arXiv:2202.00661_ , 1, 2022. 1, 9 

- [31] Alex Kendall, Yarin Gal, and Roberto Cipolla. Multi-task learning using uncertainty to weigh losses for scene geometry and semantics. In _Proceedings of the IEEE conference on computer vision and pattern recognition_ , pages 7482–7491, 2018. 2 

- [32] Nitish Shirish Keskar, Dheevatsa Mudigere, Jorge Nocedal, Mikhail Smelyanskiy, and Ping Tak Peter Tang. On largebatch training for deep learning: Generalization gap and sharp minima. In _ICLR_ . OpenReview.net, 2017. 2 

- [33] Nitish Shirish Keskar, Jorge Nocedal, Ping Tak Peter Tang, Dheevatsa Mudigere, and Mikhail Smelyanskiy. On largebatch training for deep learning: Generalization gap and sharp minima. In _5th International Conference on Learning Representations, ICLR 2017_ , 2017. 1 

- [34] Diederik P Kingma and Jimmy Ba. Adam: A method for stochastic optimization. _arXiv preprint arXiv:1412.6980_ , 2014. 5 

- [35] Jungmin Kwon, Jeongseop Kim, Hyunseo Park, and In Kwon Choi. Asam: Adaptive sharpness-aware minimization for scale-invariant learning of deep neural networks, 2021. 3, 5 

- [36] Balaji Lakshminarayanan, Alexander Pritzel, and Charles Blundell. Simple and scalable predictive uncertainty estimation using deep ensembles. _Advances in neural information processing systems_ , 30, 2017. 7 

- [37] Beatrice Laurent and Pascal Massart. Adaptive estimation of a quadratic functional by model selection. _Annals of Statistics_ , pages 1302–1338, 2000. 2, 3 

- [38] Yann LeCun, Leon´ Bottou, Yoshua Bengio, and Patrick Haffner. Gradient-based learning applied to document recognition. _Proceedings of the IEEE_ , 86(11):2278–2324, 1998. 5 

- [39] Hao Li, Zheng Xu, Gavin Taylor, Christoph Studer, and Tom Goldstein. Visualizing the loss landscape of neural nets. _Advances in neural information processing systems_ , 31, 2018. 9 

- [40] Zhouzi Li, Zixuan Wang, and Jian Li. Analyzing sharpness along gd trajectory: Progressive sharpening and edge of stability. _arXiv preprint arXiv:2207.12678_ , 2022. 1 

- [41] Baijiong Lin, Feiyang Ye, Yu Zhang, and Ivor W Tsang. Reasonable effectiveness of random weighting: A litmus test for multi-task learning. _arXiv preprint arXiv:2111.10603_ , 2021. 5 

- [42] Xi Lin, Hui-Ling Zhen, Zhenhua Li, Qing-Fu Zhang, and Sam Kwong. Pareto multi-task learning. _Advances in neural information processing systems_ , 32, 2019. 2, 5 

- [43] Xi Lin, Zhiyuan Yang, Qingfu Zhang, and Sam Kwong. Controllable pareto multi-task learning. _arXiv preprint arXiv:2010.06313_ , 2020. 2 

- [44] Bo Liu, Xingchao Liu, Xiaojie Jin, Peter Stone, and Qiang Liu. Conflict-averse gradient descent for multi-task learning. _Advances in Neural Information Processing Systems_ , 34: 18878–18890, 2021. 1, 2, 5, 6, 4 

- [45] Bo Liu, Yihao Feng, Peter Stone, and qiang liu. FAMO: Fast adaptive multitask optimization. In _Thirty-seventh Conference on Neural Information Processing Systems_ , 2023. 1 

- [46] Liyang Liu, Yi Li, Zhanghui Kuang, Jing-Hao Xue, Yimin Chen, Wenming Yang, Qingmin Liao, and Wayne Zhang. Towards impartial multi-task learning. In _International Conference on Learning Representations_ , 2020. 1, 2, 5, 7, 4 

- [47] Shikun Liu, Edward Johns, and Andrew J Davison. Endto-end multi-task learning with attention. In _Proceedings of the IEEE/CVF conference on computer vision and pattern recognition_ , pages 1871–1880, 2019. 1, 2, 6, 5 

- [48] Xiaodong Liu, Pengcheng He, Weizhu Chen, and Jianfeng Gao. Multi-task deep neural networks for natural language understanding. In _Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics_ , pages 4487– 4496. Association for Computational Linguistics, 2019. 2 

- [49] Xingchao Liu, Xin Tong, and Qiang Liu. Profiling pareto front with multi-objective stein variational gradient descent. _Advances in Neural Information Processing Systems_ , 34: 14721–14733, 2021. 2 

- [50] Yong Liu, Siqi Mai, Xiangning Chen, Cho-Jui Hsieh, and Yang You. Towards efficient and scalable sharpness-aware minimization. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)_ , pages 12360–12370, 2022. 11 

- [51] Ziwei Liu, Ping Luo, Xiaogang Wang, and Xiaoou Tang. Deep learning face attributes in the wild. In _Proceedings of the IEEE international conference on computer vision_ , pages 3730–3738, 2015. 5 

- [52] Ziwei Liu, Ping Luo, Xiaogang Wang, and Xiaoou Tang. Large-scale celebfaces attributes (celeba) dataset. _Retrieved August_ , 15(2018):11, 2018. 6, 5 

- [53] Kaifeng Lyu, Zhiyuan Li, and Sanjeev Arora. Understanding the generalization benefit of normalization layers: Sharpness reduction. _arXiv preprint arXiv:2206.07085_ , 2022. 1 

- [54] Debabrata Mahapatra and Vaibhav Rajan. Multi-task learning with user preferences: Gradient descent with controlled ascent in pareto optimization. In _International Conference on Machine Learning_ , pages 6597–6607. PMLR, 2020. 2 

- [55] Debabrata Mahapatra and Vaibhav Rajan. Exact pareto optimal search for multi-task learning: Touring the pareto front. _arXiv preprint arXiv:2108.00597_ , 2021. 2 

2449 

- [56] Andrey Malinin and Mark Gales. Predictive uncertainty estimation via prior networks. _Advances in neural information processing systems_ , 31, 2018. 7 

- [57] Kevis-Kokitsi Maninis, Ilija Radosavovic, and Iasonas Kokkinos. Attentive single-tasking of multiple tasks. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pages 1851–1860, 2019. 6 

- [58] David A McAllester. Pac-bayesian model averaging. In _Proceedings of the twelfth annual conference on Computational learning theory_ , pages 164–170, 1999. 3 

- [59] Thomas Mollenhoff and Mohammad Emtiyaz Khan.¨ SAM as an optimal relaxation of bayes. In _The Eleventh International Conference on Learning Representations_ , 2023. 2 

- [60] Mahdi Pakdaman Naeini, Gregory Cooper, and Milos Hauskrecht. Obtaining well calibrated probabilities using bayesian binning. In _Twenty-Ninth AAAI Conference on Artificial Intelligence_ , 2015. 7 

- [61] Aviv Navon, Aviv Shamsian, Gal Chechik, and Ethan Fetaya. Learning the pareto front with hypernetworks. In _International Conference on Learning Representations_ , 2021. 2 

- [62] Aviv Navon, Aviv Shamsian, Idan Achituve, Haggai Maron, Kenji Kawaguchi, Gal Chechik, and Ethan Fetaya. Multitask learning as a bargaining game. _arXiv preprint arXiv:2202.01017_ , 2022. 1, 5, 6 

- [63] Behnam Neyshabur, Srinadh Bhojanapalli, David McAllester, and Nati Srebro. Exploring generalization in deep learning. _Advances in neural information processing systems_ , 30, 2017. 2 

- [64] Van-Anh Nguyen, Tung-Long Vuong, Hoang Phan, ThanhToan Do, Dinh Phung, and Trung Le. Flat seeking bayesian neural networks. _Advances in Neural Information Processing Systems_ , 2023. 2 

- [65] Yaniv Ovadia, Emily Fertig, Jie Ren, Zachary Nado, David Sculley, Sebastian Nowozin, Joshua Dillon, Balaji Lakshminarayanan, and Jasper Snoek. Can you trust your model’s uncertainty? evaluating predictive uncertainty under dataset shift. _Advances in neural information processing systems_ , 32, 2019. 7 

- [66] Gabriel Pereyra, George Tucker, Jan Chorowski, Lukasz Kaiser, and Geoffrey E. Hinton. Regularizing neural networks by penalizing confident output distributions. In _ICLR_ 

   - _(Workshop)_ . OpenReview.net, 2017. 2 

- [67] Henning Petzka, Michael Kamp, Linara Adilova, Cristian Sminchisescu, and Mario Boley. Relative flatness and generalization. In _NeurIPS_ , pages 18420–18432, 2021. 2 

- [68] Hoang Phan, Ngoc Tran, Trung Le, Toan Tran, Nhat Ho, and Dinh Phung. Stochastic multiple target sampling gradient descent. _Advances in neural information processing systems_ , 35:22643–22655, 2022. 2 

- [69] Zhe Qu, Xingyu Li, Rui Duan, Yao Liu, Bo Tang, and Zhuo Lu. Generalized federated learning via sharpness aware minimization. _arXiv preprint arXiv:2206.02618_ , 2022. 2 

- [70] Sebastian Ruder. An overview of multi-task learning in deep neural networks. _arXiv preprint arXiv:1706.05098_ , 2017. 2 

- [71] Ozan Sener and Vladlen Koltun. Multi-task learning as multi-objective optimization. _Advances in neural information processing systems_ , 31, 2018. 1, 2, 5, 6, 4 

- [72] Nathan Silberman, Derek Hoiem, Pushmeet Kohli, and Rob Fergus. Indoor segmentation and support inference from rgbd images. In _European conference on computer vision_ , pages 746–760. Springer, 2012. 5, 6 

- [73] Pengfei Wang, Zhaoxiang Zhang, Zhen Lei, and Lei Zhang. Sharpness-aware gradient matching for domain generalization. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pages 3769– 3778, 2023. 2, 5 

- [74] Zirui Wang, Yulia Tsvetkov, Orhan Firat, and Yuan Cao. Gradient vaccine: Investigating and improving multitask optimization in massively multilingual models. In _International Conference on Learning Representations_ , 2020. 1, 8 

- [75] Dongxian Wu, Shu-Tao Xia, and Yisen Wang. Adversarial weight perturbation helps robust generalization. _Advances in neural information processing systems_ , 33:2958–2969, 2020. 3 

- [76] Han Xiao, Kashif Rasul, and Roland Vollgraf. Fashion-mnist: a novel image dataset for benchmarking machine learning algorithms. _arXiv preprint arXiv:1708.07747_ , 2017. 5 

- [77] Enneng Yang, Li Shen, Zhenyi Wang, Shiwei Liu, Guibing Guo, and Xingwei Wang. Data augmented flatness-aware gradient projection for continual learning. In _Proceedings of the IEEE/CVF International Conference on Computer Vision_ , pages 5630–5639, 2023. 2 

- [78] Tianhe Yu, Saurabh Kumar, Abhishek Gupta, Sergey Levine, Karol Hausman, and Chelsea Finn. Gradient surgery for multitask learning. _Advances in Neural Information Processing Systems_ , 33:5824–5836, 2020. 1, 2, 5, 8, 4 

- [79] Xingxuan Zhang, Renzhe Xu, Han Yu, Yancheng Dong, Pengfei Tian, and Peng Cui. Flatness-aware minimization for domain generalization. In _Proceedings of the IEEE/CVF International Conference on Computer Vision_ , pages 5189– 5202, 2023. 2 

- [80] Zhanpeng Zhang, Ping Luo, Chen Change Loy, and Xiaoou Tang. Facial landmark detection by deep multi-task learning. In _European conference on computer vision_ , pages 94–108. Springer, 2014. 1 

- [81] Yang Zhao, Hao Zhang, and Xiuyuan Hu. Penalizing gradient norm for efficiently improving generalization in deep learning. _arXiv preprint arXiv:2202.03599_ , 2022. 1 

- [82] Yaowei Zheng, Richong Zhang, and Yongyi Mao. Regularizing neural networks via adversarial model perturbation. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pages 8156–8165, 2021. 1, 3 

- [83] Zhanpeng Zhou, Mingze Wang, Yuchen Mao, Bingrui Li, and Junchi Yan. Sharpness-aware minimization efficiently selects flatter minima late in training. _arXiv preprint arXiv:2410.10373_ , 2024. 11 

- [84] Shijie Zhu, Hui Zhao, Pengjie Wang, Hongbo Deng, Jian Xu, and Bo Zheng. Gradient deconfliction via orthogonal projections onto subspaces for multi-task learning. 2022. 8 

- [85] Juntang Zhuang, Boqing Gong, Liangzhe Yuan, Yin Cui, Hartwig Adam, Nicha C Dvornek, sekhar tatikonda, James s Duncan, and Ting Liu. Surrogate gap minimization improves 

2450 

