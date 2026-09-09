# **Improving Gradient Trade-offs between Tasks in Multi-task Text Classification** 

**Heyan Chai**<sup>1</sup> **, Jinhao Cui**<sup>1</sup> **, Ye Wang**<sup>2</sup> **, Min Zhang**<sup>1</sup> **, Binxing Fang**<sup>1</sup><sup>_,_3</sup> and **Qing Liao**<sup>1</sup><sup>_,_3</sup><sup>_∗_</sup> 

1 Harbin Institute of Technology, Shenzhen, China 

2 National University of Defense Technology, China 

3 Peng Cheng Laboratory, Shenzhen, China 

{chaiheyan,cuijinhao}@stu.hit.edu.cn, ye.wang@nudt.edu.cn zhangmin2021@hit.edu.cn, fangbx@cae.cn, liaoqing@hit.edu.cn 

## **Abstract** 

Multi-task learning (MTL) has emerged as a promising approach for sharing inductive bias across multiple tasks to enable more efficient learning in text classification. However, training all tasks simultaneously often yields degraded performance of each task than learning them independently, since different tasks might conflict with each other. Existing MTL methods for alleviating this issue is to leverage heuristics or gradient-based algorithm to achieve an arbitrary Pareto optimal trade-off among different tasks. In this paper, we present a novel gradient trade-off approach to mitigate the task conflict problem, dubbed GetMTL, which can achieve a specific tradeoff among different tasks nearby the main objective of multi-task text classification (MTC), so as to improve the performance of each task simultaneously. The results of extensive experiments on two benchmark datasets back up our theoretical analysis and validate the superiority of our proposed GetMTL. 

## **1 Introduction** 

Multi-task Learning (MTL), which aims to learn a single model that can tackle multiple correlated but different tasks simultaneously, makes multiple tasks benefit from each other and obtain superior performance over learning each task independently (Caruana, 1997; Ruder, 2017; Liu et al., 2015; Mao et al., 2020). By discovering shared information/structure across the tasks, it has gained attention in many areas of research and industrial communities, such as computer vision (Misra et al., 2016; Gao et al., 2019; Yogamani et al., 2019; Sun et al., 2020) and text classification (Liu et al., 2017; Xiao et al., 2018; Mao et al., 2021, 2022). 

However, it is observed in multi-task text classification (MTC) scenarios that some tasks could conflict with each other, which may be reflected via conflicting gradients or dominating gradients (Yu 

> _∗_ Corresponding Author 



Pareto Front Pareto Front<br>Average Loss<br>Pareto Solutions<br>Our Solutions<br>Task 1 Error Task 1 Error<br>(a) Pareto Optimality (b) GetMTL (Ours)<br>Task 2 Error Task 2 Error<br>
Figure 1: Graphical interpretation of existing Pareto multi-task learning methods for a two-task learning problem. (a) Pareto optimal solutions are arbitrary and uncontrollable. (b) Our GetMTL can find the specific solutions nearby the main objective (Average loss). 

et al., 2020; Vandenhende et al., 2022), leading to the degraded performance of MTL due to poor training. How to make a proper trade-off among jointing different tasks in MTC is a difficult problem. Recently, several methods have been proposed to mitigate gradient conflicts issue via both _loss balance_ (linear weighted scalarization) such as homoscedastic uncertainty (Kendall et al., 2018) and task variance regularization (Mao et al., 2021), and _gradient balance_ like Pareto optimality (Sener and Koltun, 2018; Mao et al., 2020). Existing methods devote to finding an arbitrary Pareto optimality solution in the Pareto set, which achieve a single arbitrary trade-off among all tasks. However, they can only satisfy the improved performance on part of tasks, not all tasks simultaneously. This means that these methods can not converge to a minimum average loss of all objectives. 

To illustrate our idea, we give a two-task learning example shown in Figure 1. As shown in Figure (1a), it is observed that Pareto optimality-based methods can generate a set of Pareto solutions for a given two-task learning problem. However, some of Pareto solutions can increase the _task 1 error_ while decreasing _task 2 error_ , leading to unsatisfactory overall performance for MTL model. This im- 

2565 

_Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics Volume 1: Long Papers_ , pages 2565–2579 July 9-14, 2023 ©2023 Association for Computational Linguistics 

plies that not all Pareto solutions always satisfy the goal of mitigating the tasks conflicts in MTL, and thus failing to achieve a better trade-off between tasks. Therefore, it is necessary to find a specific trade-off between tasks that is beyond what only using Pareto optimality can achieve. 

To address this issue, inspired by multi-objective optimization (Sener and Koltun, 2018), we argue that a more efficient way to mitigate task conflicts is to find a gradient trade-off between tasks in the neighborhood of the average loss rather than exhaustively searching for a proper solution from the set of Pareto solutions. As shown in Figure 1b, the Pareto solutions nearby the average loss can achieve a better trade-off between _task 1_ and _task 2_ , leading to better performance on both tasks at the same time. Based on it, in this paper, we propose a novel gradient trade-off multi-task learning approach, named **GetMTL** , to mitigate task conflicts in multi-task text classification. Specifically, the gradients of each task are utilized to derive an update vector that can minimize the conflicts among task gradients in the neighborhood of the average gradient, so as to achieve a better trade-off performance among joint training tasks. In summary, the main contributions of our work are as follows: 

- A novel multi-task learning approach based on gradient trade-off between different tasks (GetMTL) is proposed to deal with task conflict in multi-task text classification problems, so as to improve the performance of all tasks simultaneously. 

- We give in-depth theoretical proofs and experimental analyses on establishing converge guarantees of our GetMTL. 

- We extensively verify the effectiveness of our GetMTL on two real-world text classification datasets, and the results show that our GetMTL performs competitively with a variety of state-of-the-art methods under a different number of task sets. 

## **2 Related Works** 

Multi-task Learning methods jointly minimize all task losses based on either loss balance methods (Kendall et al., 2018; Chen et al., 2018; Mao et al., 2021, 2022) or gradient balance methods (Sener and Koltun, 2018; Mao et al., 2020). The loss balance methods adaptively adjust the tasks weights during training based on various heuristic approaches, such as task uncertainty quan- 

tification (Kendall et al., 2018), gradient normalization (Chen et al., 2018), task difficulty prioritization (Guo et al., 2018), dynamic weight average (Liu et al., 2019), random loss weighting (Lin et al., 2021), task variance regularization (Mao et al., 2021), and meta learning-based approach (Mao et al., 2022). These methods are mostly heuristic and can have unstable performance while ignoring the task conflicts among all tasks, leading to the bad generalization performance of MTL models. 

Recently, some gradient balance based methods have been proposed to mitigate task conflicts for improving task performance. For example, Désidéri (2012) leverages multiple-gradient descent algorithm (MGDA) to optimize multiple objectives. Due to the guarantee of convergence to Pareto stationary point, this is an appealing approach. Sener and Koltun (2018) cast the multi-objective problem as a multi-task problem and devote to finding an arbitrary Pareto optimal solution. Mao et al. (2020) propose a novel MTL method based Tchebycheff procedure for achieving Pareto optimal without any convex assumption. However, these methods only consider achieving an arbitrary Pareto optimal solution while it is not the main objective. Unlike these methods, we propose an MTL approach based on multi-objective optimization and seek to find a set of solutions that are Pareto optimality and nearby the main MTC objective _L_ 0. 

## **3 Preliminaries** 

Consider a multi-task learning problem with _T_<sup>1</sup> tasks over an input space _X_ and a collection of task spaces _{Y_<sup>_t_</sup> _}t∈_ [ _T_ ], where each task contains a set of i.i.d. training samples _Dt_ = _{xi, yi_<sup>_t}i∈_[</sup><sup>_n_</sup> _t_<sup>],</sup> _T_ is the number of tasks, and _nt_ is the number of training samples of task _t_ . The goal of MTL is to find parameters _{θ_<sup>_sh_</sup> _, θ_<sup>1</sup> _, ..., θ_<sup>_T_</sup> _}_ of a model _F_ that can achieve high average performance across all training tasks over _X_ , defined as _F_ ( _X , θ_<sup>_sh_</sup> _, · · · , θ_<sup>_t_</sup> ) : _X →Y_ , where _θ_<sup>_sh_</sup> denotes the parameters shared between tasks and _θ_<sup>_t_</sup> denotes the task-specific parameters of task _t_ . In particular, we further consider a parametric taskspecific map as _f_<sup>_t_</sup> ( _·, θ_<sup>_sh_</sup> _, θ_<sup>_t_</sup> ) : _X →Y_<sup>_t_</sup> . We also consider task-specific loss functions _ℓt_ ( _·, ·_ ) : _Y_<sup>_t_</sup> _×Y_<sup>_t_</sup> _→_ R<sup>+</sup> . We also denote the multi-task loss as _L_ ( _θ_ ) =<sup><u>�</u></sup><sup>_T_</sup> _i_<sup>_ℓi_(</sup><sup>_θ_), and the gradients of each task</sup> 

> 1For ease of distinction, we denote the transpose of the vector as the superscript T _._ 

2566 

as _gi_ = _∇ℓi_ ( _θ_ ) for the particular _θ_ . In this paper, we choose the average loss as main objective of MTC problem, defined as _L_ 0( _θ_ ) = _T_<sup><u>1</u></sup> � _Ti_<sup>_ℓi_(</sup><sup>_θ_).</sup> 

### **3.1 MTL as Multi-objective Optimization** 

MTL can be formulated as a specific case of multiple-objective optimization (MOO), which optimizes a set of potentially conflicting objectives (Sener and Koltun, 2018; Mao et al., 2020). Given objective functions of _T_ tasks, _ℓ_ 1 _, . . . , ℓT_ , we formulate the optimization objective of MTL as the vectors of objective values : 



Since there is no natural linear ordering on vectors, it is not possible to compare solutions and thus no single solution can optimize all objectives simultaneously. In other words, there is no clear optimal value. Alternatively, we can achieve Pareto optimality to obtain different optimal trade-offs among all objectives to solve MOO problem. 

**Definition 1** (Pareto dominance) **.** _Given two points {θ, θ} in_ Ω _, a point θ Pareto dominates θ (θ_ ≼ _θ) for MTL if two conditions are satisfied:_ 

- _(i) No one strictly prefers θ to θ, that is, ∀i ∈ {_ 1 _, . . . , T }, ℓi_ ( _θ_<sup>_sh_</sup> _, θ_<sup>_i_</sup> ) _≤ ℓi_ ( _θsh, θ_ _~~i~~_ ) _._ 

- _(ii) At least one point strictly prefers θ to θ, that is, ∃j ∈{_ 1 _, ..., T }, ℓj_ ( _θ_<sup>_sh_</sup> _, θ_<sup>_j_</sup> ) _< ℓj_ ( _θ_ _~~s~~ h, θj_ ) _._ 

**Definition 2** (Pareto optimality) **.** _θ_<sup>_∗_</sup> _is a Pareto optimal point and ℓ_ ( _θ_<sup>_∗_</sup> ) _is a Pareto optimal objective vector if it does not exist θ_<sup>ˆ</sup> _∈_ Ω _such that θ_<sup>ˆ</sup> ≼ _θ_<sup>_∗_</sup> _. That is, a solution that is not dominated by any other is called Pareto optimal._ 

The set of all Pareto optimal solutions is called the Pareto set, and the image of Pareto set in the loss space is called Pareto front (Lin et al., 2019). In this paper, we focus on gradient-based multiobjective optimization to achieve an appropriate Pareto trade-off among all tasks, which can approximate the Pareto front that minimizes the average loss. 

### **3.2 Gradient-based Multi-Objective Optimization** 

Gradient-based MOO (Sener and Koltun, 2018) aims to find a direction _d_ that we can iteratively find the next solution _θ_<sup>(</sup><sup>_t_+1)</sup> that dominates the previous one _θ_<sup>(</sup><sup>_t_)</sup> ( _ℓ_ ( _θ_<sup>(</sup><sup>_t_+1)</sup> ) _≤ ℓ_ ( _θ_<sup>(</sup><sup>_t_)</sup> )) by moving 

against _d_ with step size _η_ , i.e. _θ_<sup>(</sup><sup>_t_+1)</sup> = _θ_<sup>(</sup><sup>_t_)</sup> _− ηd_ . Désidéri (2012); Sener and Koltun (2018) propose to use multiple gradient descent algorithm (MGDA) that converges to a local Pareto optimal by iteratively using the descent direction _d_ , which can be obtained as follows: 



where _d_<sup>_∗_</sup> is the direction that can improve all tasks. Essentially, gradient-based MOO methods minimize the loss by combining gradients with adaptive weights, and obtaining an arbitrary Pareto optimality solution, ignoring the true objective (the average loss) (Liu et al., 2021). In this paper, we generalize this method and propose a novel gradient-based approach to achieve a gradient trade-off among tasks for mitigating task conflicts, as well as constrain the solution that can minimize the average loss ( _L_ 0( _θ_ )). 

## **4 Gradient Trade-offs for Multi-task Text Classification** 

Following most MTL methods, as shown in Figure 2, we employ the hard parameter sharing MTL architecture, which includes _f_<sup>_sh_</sup> parameterized by heavy-weight task-shared parameters _θ_<sup>_sh_</sup> and _f_<sup>_t_</sup> parameterized by light-weight task-specific parameters _θ_<sup>_t_</sup> . All tasks take the same shared intermediate feature _z_ = _f_<sup>_sh_</sup> ( _x_ ; _θ_<sup>_sh_</sup> ) as input, and the _t_ -th taskspecific network outputs the prediction as _f_<sup>_t_</sup> ( _z_ ; _θ_<sup>_t_</sup> ). Since task-shared parameters _θ_<sup>_sh_</sup> are shared by all tasks, the different tasks may conflict with each other, leading to the degraded performance of MTL model. In this paper, we hypothesize that one of the main reasons for task conflicts arises from gradients from different tasks competing with each other in a way that is detrimental to making progress. We propose a novel gradient-based MOO optimization to find a gradient trade-off among tasks in the neighborhood of the average loss, so as to mitigate task conflicts. Note that, we omit the subscript _sh_ of task-shared parameters _θ_<sup>_sh_</sup> for the ease of notation. 

### **4.1 GetMTL** 

Given a task _i_ , we define its gradient as _gi_ = _∇ℓi_ ( _θ_ ) via back-propagation from the raw loss _ℓi_ , and _gi_ represents the optimal update direction for task _i_ . However, due to the inconsistency of the 

2567 



Multi-task Learning Gradient Trade-off 𝑔!<br>Task-specific ℓ" = 𝑓 " (𝑧;𝜃𝜃 , " )  ℓ! = 𝑓 ! (𝑧; 𝜃𝜃 . ! )  ℓ# = 𝑓 # 𝜃(𝑧;𝜃 - # )  𝑔" 𝑑  𝑔$ 𝑔#<br>Layers<br>(a) Gradient-based MOO<br>𝑧= 𝑓 *+ (𝑥; 𝜃 *+ ) 𝑔!<br>Task-shared Layers 𝜃 *+ 𝑔" 𝑑  𝑔! 𝑔#<br>𝑥 (b) GetMTL (Ours)<br>
Figure 2: Overview of GetMTL. _Left._ The left part of the figure is our MTL architecture. _Right._ We show the update direction (red) _d_ obtained by gradient-based MOO method and our GetMTL on three gradients ( _g_ 1, _g_ 2 and _g_ 3) in R<sup>3</sup> , where _gi_ denotes the gradient (black) of _i_ -th task, _g_ 0 is the average gradient, and blue arrows denote the projections of update direction to each task gradient. 

optimal update direction of task-shared parameters for each task, different task gradients may conflict with each other, leading to the training of networks being stuck in the over-training of some tasks and the under-training of other tasks. Intuitively, it is desirable to find a direction that can minimize the task conflicts among different tasks as well as achieve Pareto optimality to improve the performance of MTL model. 

We first achieve an arbitrary Pareto optimal via finding a descent direction _ddes_ by searching for a minimum-norm point in the _Convex Hull CH_ of gradients, defined by, 



where _G ∈_ R<sup>_T×m_</sup> = _{g_ 1 _, ..., gT }_ is the matrix of task gradient, _S_<sup>_T_</sup> is the _T_ -dimensional regular simplex. We use the multiple gradient descent algorithm (MGDA) (Sener and Koltun, 2018) to obtain an arbitrary Pareto optimal by iteratively using the descent direction, defined by, 



In addition, the _ddes_ can be reformulated as a linear combination of all task gradients, defined by, 



where _gi_ = _∇ℓi_ ( _θ_ ) is the _i_ -th task gradient. It implies that, when converges to an arbitrary Pareto optimal, the optimal gradient value of each task via back-propagation is _βigi_ , defined as _gβi_ = _βigi_ . 

However, moving against _ddes_ does not guarantee that the solution meets the requirements of multi-task text classification task (MTC), that is, to alleviate the gradient conflict among tasks in MTC, so as to improve the performance of all tasks. To address this issue, we seek a direction that enables us to move from a solution _θ_<sup>(</sup><sup>_t_)</sup> to _θ_<sup>(</sup><sup>_t_+1)</sup> such that both _θ_<sup>(</sup><sup>_t_+1)</sup> dominates _θ_<sup>(</sup><sup>_t_)</sup> ( _L_ ( _θ_<sup>(</sup><sup>_t_+1)</sup> ) _≤L_ ( _θ_<sup>(</sup><sup>_t_)</sup> )) and alleviate the gradient conflict among all tasks. Based on it, as shown in Figure 2(b), we propose to search for an update direction _d_ in the _Convex Hull CHβ_ of back-propagation gradients such that it can improve any worst objective and converge to an optimum of MTC objective _L_ 0( _θ_ ). We first find the worst task gradient with respect to the update direction _d_ , that is, it has a maximum angle with _d_ , which can be formulated via the following optimization problem, 



where _gβi_ is the _i_ -task gradient after optimizing by MGDA algorithm. 

To improve the worst gradient of any task and achieve a trade-off between all task gradients in a neighborhood of the average gradient (defined <u>1</u> as _g_ 0 = _T_ � _Ti_ =1<sup>_gi_),weformulatethisgradient</sup> trade-off optimization problem via the following _Maximin Optimization Problem_ (dual problem). 

### **Problem 1.** 



where _gβi_ = _βigi_ is the back-propagation gradient value of _i_ -th task via solving Eq. (5), _ε ∈_ (0 _,_ 1] is a hyper-parameter that controls the stability of MTC model. 

### **4.2 Solving Maximin Problem** 

Since the optimal direction _d_ can also be defined in the convex hull _CHβ_ of _gβi_ , we can get 



where _Gβ ∈_ R<sup>_T×m_</sup> = _{gβ_ 1 _, ..., gβT }_ is task gradient matrix, _W_<sup>_T_</sup> = _{_ **_w_** _∈_ R<sup>_T_</sup> + ��� _Tj_ =1<sup>_wj_=1</sup><sup>_}_is</sup> the _T_ -dimensional probability simplex, and **_w_** = ( _w_ 1 _, ..., wT_ ). Therefore, we can get min _i⟨gβi, d⟩_ = min _w∈W T ⟨_<sup>�</sup> _i_<sup>_wigβ_</sup> _i_<sup>_, d⟩_andProblem1canbe</sup> transformed into the following form. 

2568 

### **Algorithm 1:** GetMTL Algorithm. 

- **Input:** The number of task _T_ , loss functions _{ℓi}_<sup>_T_</sup> _i_ =1<sup>, network parameters</sup><sup>_θ_(</sup><sup>_t_) at</sup><sup>_t_step, the</sup> pre-specified hyper-parameter _ε ∈_ (0 _,_ 1] and step size _µ ∈_ R<sup>+</sup> . 

- 1: Task Gradients: _gi_ = _∇ℓi_ ( _θ_<sup>(</sup><sup>_t_)</sup> ), _i ∈_ [ _T_ ] 

- 2: Main Objective: _g_ 0 =<sup>�</sup><sup>_T_</sup> _i_ =1<sup>_gi_</sup> 

- 3: Obtain _{β_ 1 _, ...βT }_ by solving Eq.(5). 

- 4: Compute _gw_ =<sup>�</sup> _i_<sup>_wigβ_</sup> _i_<sup>, where</sup><sup>_gβ_</sup> _i_<sup>=</sup><sup>_βigi_</sup> 

|TASKS|NEWSGROUPS|
|---|---|
|COMP|GRAPHICS, OS.MS-WINDOWS.MISC,<br>SYS.MAC.HARDWARE, WINDOWS.X|
|REC|AUTOS, SPORT.BASEBALL,<br>MOTORCYCLES, SPORT.HOCKEY|
|SCI|CRYPT, SPACE,<br>MED, ELECTRONICS|
|TALK|POLITICS.MISC, POLITICS.GUNS,<br>POLITICS.MIDEAST, RELIGION.MISC|



Table 1: Tasks of topic classification dataset. 

- 5: Obtain _{w_ 1 _, ..., wT }_ by solving Eq.(14) 



### **Problem 2.** 



where _gw_ =<sup>�</sup><sup>_T_</sup> _i_ =1<sup>_wigβ_</sup> _i_<sup>istheconvexcombina-</sup> tion in _CHβ_ . For a given vector _λ ∈_ R<sup>+</sup> with non-negative components, the corresponding _Lagrangian_ associated with the Eq.(10) is defined as 



Since the objective for _d_ is concave with linear constraints and _w ∈W_<sup>_T_</sup> is a compact set<sup>2</sup> , according to the Sion’s minimax theorem (Kindler, 2005), we can switch the _max_ and _min_ without changing the solution of Problem 2. Formally, 



We get the optimal solution of primal problem (Problem 1) by solving the dual problem of Eq.(12) (See the Appendix A for a detailed derivation procedure). Then we have 



where _λ_<sup>_∗_</sup> is the optimal Lagrange multiplier, _d_<sup>_∗_</sup> is the optimal update direction of MTC model. We can reformulate the problem of Eq.(12) as following optimization problem w.r.t. _w_ . 



> 2Compact set: a set that is bounded and closed. 

where _gw_ is defined as _gw_ =<sup>�</sup><sup>_T_</sup> _i_ =1<sup>_wigβ_</sup> _i_<sup>.The</sup> detailed derivation is provided in Appendix A. Algorithm 1 shows all the steps of GetMTL algorithm in each iteration. 

### **4.3 Theoretical Analysis** 

In this section, we analyze the equivalence of solutions to dual problem and then give a theoretical analysis about convergence of GetMTL algorithm. We define the Lagrangian of problem in Eq.(10), 

_L_ ( _d, λ, w_ ) = _gw_<sup>T</sup><sup>_d −λ_</sup> 2<sup>(</sup><sup>_∥d −g_0</sup><sup>_∥_2</sup><sup>_−ε_2(</sup><sup>_g_</sup> 0<sup>T</sup><sup>_d_)2)</sup> **Theorem 4.1** (Equivalence of Optimal Value of Dual Problem) **.** _Assume that both primal problem and dual problem have optimal values, let p_<sup>_∗_</sup> = max _d_ min _λ,w L_ ( _d, λ, w_ ) _and q_<sup>_∗_</sup> = min _λ,w_ max _d L_ ( _d, λ, w_ ) _. Then, p_<sup>_∗_</sup> = max _d_ min _λ,w L_ ( _d, λ, w_ ) _≤_ min _λ,w_ max _d L_ ( _d, λ, w_ ) = _q_<sup>_∗_</sup> _._ 

_Proof._ The proof is provided in Appendix B. ■ 

**Theorem 4.2** (Convergence of GetMTL) **.** _Assume loss functions ℓi are convex and differential, and ∇ℓi_ ( _θ_<sup>(</sup><sup>_t_)</sup> ) _is L-lipschitz continuous with L>_ 0 _. The update rule is θ_<sup>(</sup><sup>_t_+1)</sup> = _θ_<sup>(</sup><sup>_t_)</sup> _− µ_<sup>(</sup><sup>_t_)</sup> _d, where d is_ _<u>∥d−g</u>_ <u>0</u> _<u>∥</u> defined in Eq.(13) and µ_<sup>(</sup><sup>_t_)</sup> = min _i∈_ [ _k_ ] _c·L·d_<sup>2</sup><sup>_.All_</sup> _the loss functions_ � _ℓ_ 1( _θ_<sup>(</sup><sup>_t_)</sup> ) _· · · ℓT_ ( _θ_<sup>(</sup><sup>_t_)</sup> )� _converges to_ ( _ℓ_ 1( _θ_<sup>_∗_</sup> ) _· · · ℓT_ ( _θ_<sup>_∗_</sup> )) _._ 

_Proof._ The proof is provided in Appendix C. ■ 

## **5 Experimental Setup** 

### **5.1 Experimental Datasets** 

We conduct experiments on two MTC benchmarks to evaluate the proposed GetMTL. 1) Amazon Review dataset (Blitzer et al., 2007) contains product reviews from 14 domains (See Details in Appendix D), including apparel, video, books, electronics, DVDs and so on. Each domain gives rise to a binary classification task and we follow Mao et al. 

2569 



(%) apparel (%) baby (%) books (%) camera_photo (%) dvd<br>91<br>89<br>90 90 88 92 88<br>89 89 87 87<br>91<br>86<br>88 88 86<br>85<br>87 90 84<br>87 85<br>(%) electronics (%) health_personal_care (%) kitchen_housewares (%) magazines (%) music<br>92 91<br>89 87<br>90<br>94<br>88 91 89 86<br>88<br>87<br>90 87 93 85<br>86<br>86 84<br>85 89 85<br>92<br>(%)94 software (%)92 sports_outdoors (%)91 toys_games (%) video (%)90 avg_acc<br>93 91 90 89<br>90<br>92 90 89 88<br>89<br>91 88<br>89 87<br>88<br>90 87<br>87 86<br>STLUniformUncertaintyGradNormTchebycheffAdvMGDAMetaWeightingBanditMTLGetMTL STLUniformUncertaintyGradNormTchebycheffAdvMGDAMetaWeightingBanditMTLGetMTL STLUniformUncertaintyGradNormTchebycheffAdvMGDAMetaWeightingBanditMTLGetMTL STLUniformUncertaintyGradNormTchebycheffAdvMGDAMetaWeightingBanditMTLGetMTL STLUniformUncertaintyGradNormTchebycheffAdvMGDAMetaWeightingBanditMTLGetMTL<br>
Figure 3: Experimental results on Amazon Review dataset. We plot the classification accuracy of all baselines for all 14 tasks and average performance. Each colored cluster illustrates the classification accuracy performance of a method over 10 runs. 



(%) comp (%) rec (%) sci (%) talk (%) avg_acc<br>90<br>87<br>92<br>89<br>96 96<br>88 86<br>91<br>95<br>87 95<br>85<br>86 90<br>94<br>STLUniformUncertaintyGradNormTchebycheffAdvMGDAMetaWeightingBanditMTLGetMTL STLUniformUncertaintyGradNormTchebycheffAdvMGDAMetaWeightingBanditMTLGetMTL STLUniformUncertaintyGradNormTchebycheffAdvMGDAMetaWeightingBanditMTLGetMTL STLUniformUncertaintyGradNormTchebycheffAdvMGDAMetaWeightingBanditMTLGetMTL STLUniformUncertaintyGradNormTchebycheffAdvMGDAMetaWeightingBanditMTLGetMTL<br>
Figure 4: Experimental results on topic classification dataset. We plot classification accuracy of all baselines for all 14 tasks and _avg_acc_ . Each colored cluster illustrates classification accuracy of a method over 10 runs. 

(2021) to treat 14 domains in the dataset as distinct tasks, creating a dataset with 14 tasks, with 22180 training instances and 5600 test instances in total. 2) Topic classification dataset, 20 Newsgroup<sup>3</sup> , consists of approximately 20,000 newsgroup documents, partitioned evenly across 20 different newsgroups. We follow Mao et al. (2021) to select 16 newsgroups from 20 Newsgroup dataset shown in Table 1 and then divide them into four groups. Each group gives rise to a 4-way classification task, creating a dataset with four 4-way classification tasks, which is a more challenging dataset than amazon review dataset. 

> 3http://qwone.com/ jason/20Newsgroups/ 

### **5.2 Experimental Implementation** 

We follow the standard MTC setting and adopt the same network architectures with the most recent baselines for fair comparisons (Mao et al., 2021). We adopt the hard parameter sharing MTL framework shown in Figure 2, where task-shared network is a TextCNN with kernel size of 3,5,7 and taskspecific network is a fully connected layer with a softmax function. Adam is utilized as the optimizer to train the model over 3000 epochs with a learning rate of 1e-3 for both sentiment analysis and topic classification. We set the batch size to 256. 

2570 



sentiment 20news<br>10<br>1.4 Uniform Uniform<br>Uncertainty 9 Uncertainty<br>GradNorm GradNorm<br>MGDA 8 MGDA<br>1.2 TchebycheffAdv TchebycheffAdv<br>BanditMTL 7 BanditMTL<br>MetaWeighting MetaWeighting<br>1.0 GetMTL 6 GetMTL<br>5<br>0.8 4<br>3<br>0.6<br>2<br>0 500 1000 1500 2000 2500 3000 0 500 1000 1500 2000 2500 3000<br>Epoch Epoch<br>Loss Loss<br>
Figure 5: Learning curve of comparison methods in both amazon review and topic classification datasets. 



sentiment 20news<br>0.12 STL STL<br>Uniform 17.5 Uniform<br>0.10 Uncertainty Uncertainty<br>GradNorm 15.0 GradNorm<br>0.08 MGDATchebycheffAdv 12.5 MGDATchebycheffAdv<br>BanditMTL BanditMTL<br>0.06 MetaWeighting 10.0 MetaWeighting<br>GetMTL GetMTL<br>7.5<br>0.04<br>5.0<br>0.02<br>2.5<br>0.00 0.0<br>0 500 1000 1500 2000 2500 3000 0 500 1000 1500 2000 2500 3000<br>Epoch Epoch<br>Loss Variance Loss Variance<br>
Figure 6: Evolution of task variance during training of baseline methods and GetMTL on the amazon review and topic classification datasets. 

### **5.3 Comparison Models** 

We compare the proposed GetMTL with a series of MTC baselines, including 

**Single-Task Learning (STL):** learning each task independently. 

**Uniform Scaling:** learning tasks simultaneously with uniform task weights. 

**Uncertainty:** using the uncertainty weighting method (Kendall et al., 2018). 

**GradNorm:** learning tasks simultaneously with gradient normalization method (Chen et al., 2018). 

**TchebycheffAdv:** using adversarial Tchebycheff procedure (Mao et al., 2020). 

**MGDA:** using gradient-based multi-objective optimization method (Sener and Koltun, 2018). 

**BanditMTL:** learning tasks simultaneously with multi-armed bandit method (Mao et al., 2021). 

**MetaWeighting:** using adaptive task weighting method (Mao et al., 2022). 

## **6 Experimental Results** 

### **6.1 Main Results** 

The main comparison results of GetMTL on two benchmark datasets are shown in Figure 3 and 4. It is clear that (See detailed numerical comparison results in Appendix D), our proposed GetMTL model performs consistently better than the all comparison methods on all tasks of both amazon review and topic classification datasets, and its average performance is superior to that of all baselines. This verifies the effectiveness of our GetMTL method in MTC problem. More concretely, in comparison with the gradient-based MOO optimization model (MGDA), our GetMTL achieves significant improvement across all datasets. This indicates that achieving a gradient trade-off nearby average loss to mitigate task conflicts can better improve all task performance and generalization ability of MTC model. 

2571 



comp rec sci talk<br>0.7 MGDA 0.5 MGDA 0.6 MGDA MGDA<br>0.6 BanditMTLGetMTL 0.4 BanditMTLGetMTL 0.5 BanditMTLGetMTL 0.5 BanditMTLGetMTL<br>0.5 0.4<br>0.4 0.3 0.4<br>0.3 0.3<br>0.3 0.2<br>0.2 0.2 0.2<br>0.1 0.1 0.1 0.1<br>0.0 0.0 0.0 0.0<br>0 600 1200 1800 2400 3000 0 600 1200 1800 2400 3000 0 600 1200 1800 2400 3000 0 600 1200 1800 2400 3000<br>Epoch Epoch Epoch Epoch<br>Weight Weight Weight Weight<br>
Figure 7: Task weights of comparison methods on four tasks (including comp, rec, sci, and talk tasks) in topic classification dataset. Task weights obtained from MGDA, BanditMTL and GetMTL throughout the optimization process. For better visualization, we plot points every 30 epochs. 



(%) 20news 90.0(%) sentiment<br>92.2 89.9<br>92.1 89.8<br>92.0 89.7<br>91.9 89.6<br>91.8 89.5<br>89.4<br>0.00250.0050.00750.010.0150.020.0250.030.040.05 0.1 0.00250.0050.00750.010.0150.020.0250.030.040.05 0.1<br>
(a) Amazon review dataset. 

(b) Topic classification datset. 

Figure 8: Impact of different values of _ε_ . 

### **6.2 Empirical Analysis on Convergence** 

### **6.3 The Evolution of Task Weight** **_w_** 

In this section, we visualize the task weights of our GetMTL and two weight adaptive MTL methods (MGDA and BanditMTL) throughout the training process using the topic classification dataset shown in Figure 7. It can be observed from these four figures that the weight adaption process of our GetMTL is different from that of MGDA and BanditMTL. GetMTL can automatically learn the task weights without pre-defined heuristic constraints. The weights adaption process of GetMTL is more stable and the search space is more compact compared with other MTL baselines. 

In Section 4.3, we theoretically prove the convergence of our proposed GetMTL. Furthermore, we conduct extensive experiments about the convergence to better demonstrate the advantages of GetMTL shown in Figure 5. It is clear that the learning curve of GetMTL is constantly decreasing as the number of iterations increases and converges to the lowest loss value compared with other baselines. It indicates that GetMTL can guarantee the convergence of the objective value and obtain better performance of all learning tasks. 

In addition, we also conduct extensive experiments to investigate how GetMTL mitigates task conflict during training. We plot the task variance (variance between the task-specific losses) of all baselines on both amazon review and topic classification datasets shown in Figure 6. It can be observed that all MTL baselines have lower task variance than STL method, which illustrates that MTL methods can indeed boost the learning of all tasks compared with STL method. Moreover, GetMTL has the lowest task variance and smoother evolution during training than other MTL baselines. This implies that our proposed GetMTL indeed mitigates task conflicts compared with other MTL methods. 

### **6.4 Impact of the Values of** _ε_ 

To investigate the impact of using different values of _ε_ on the performance of our GetMTL, we conduct experiments on two datasets, and the results are shown in Figure 8. Noting that model with _ε_ = 0 _._ 0075 and _ε_ = 0 _._ 025 perform overall better than other values on these two datasets, respectively. The model with larger value of _ε_ performs unsatisfactorily overall all tasks on two datasets, one possible reason is that larger _ε_ makes _d_ pull far away from the average loss _g_ 0 (see the conditions in Eq. (9)). That is, Pareto optimality found by GetMTL is getting further and further away from MTC objective _L_ 0, which can be quite detrimental to some tasks’ performance, leading to degraded average performance. 

## **7 Conclusion** 

In this paper, we propose a novel gradient tradeoff multi-task learning approach to mitigate the task conflict problem, which can achieve a specific trade-off among different tasks nearby the main objective of multi-task text classification problem. Moreover, we present a series of theoretical proofs to illustrate the effectiveness and superiority of our GetMTL. Experimental results on two benchmark 

2572 

datasets show that our GetMTL achieves state-ofthe-art performance in Multi-task Text Classification problem. 

## **Limitations** 

Our GetMTL needs to compute the _gi_ for each task _i_ at each iteration and requires a backwardpropagation procedure over the model parameters. Every iteration requires one forward-propagation followed by _T_ backward-propagation procedure and computation of backward-propagation is typically more expensive than the forward-propagation. Here, we define the time of one forward pass and one backward pass as _Ef_ and _Eb_ , respectively. The time of optimization process is defined as _Eo_ . Therefore, the total time _E_ of GetMTL is defined, 



For few-task learning scenario ( _T <_ 100), usually _Eo ≪ Eb_ and GetMTL still works fine. However, for large-scale task set (like _T ≫_ 100), usually _Eo ≫ Eb_ or _Eo ≫ TEb_ . Consequently, our GetMTL may get stuck in the optimization and backward-propagation process at each iteration. Therefore, the major limitation of our work is that it can not be applied to scenarios with large-scale task sets. 

## **Acknowledgements** 

This work was supported by the National Natural Science Foundation of China (No. 62076079), Guangdong Major Project of Basic and Applied Basic Research (No.2019B030302002), The Major Key Project of PCL(Grant No.PCL2022A03), and Guangdong Provincial Key Laboratory of Novel Security Intelligence Technologies (2022B1212010005). 

## **References** 

- Dimitri P Bertsekas. 1997. Nonlinear programming. _Journal of the Operational Research Society_ , 48(3):334–334. 

- John Blitzer, Mark Dredze, and Fernando Pereira. 2007. Biographies, bollywood, boom-boxes and blenders: Domain adaptation for sentiment classification. In _Proceedings of the 45th Annual Meeting of the Association for Computational Linguistics,_ . The Association for Computational Linguistics. 

- Rich Caruana. 1997. Multitask learning. _Machine learning_ , 28(1):41–75. 

- Zhao Chen, Vijay Badrinarayanan, Chen-Yu Lee, and Andrew Rabinovich. 2018. Gradnorm: Gradient normalization for adaptive loss balancing in deep multitask networks. In _Proceedings of the 35th International Conference on Machine Learning, ICML_ , volume 80 of _Proceedings of Machine Learning Research_ , pages 793–802. PMLR. 

- Jean-Antoine Désidéri. 2012. Multiple-gradient descent algorithm (mgda) for multiobjective optimization. _Comptes Rendus Mathematique_ , 350(56):313–318. 

- Yuan Gao, Jiayi Ma, Mingbo Zhao, Wei Liu, and Alan L. Yuille. 2019. NDDR-CNN: layerwise feature fusing in multi-task cnns by neural discriminative dimensionality reduction. In _IEEE Conference on Computer Vision and Pattern Recognition, CVPR_ , pages 3205–3214. 

- Michelle Guo, Albert Haque, De-An Huang, Serena Yeung, and Li Fei-Fei. 2018. Dynamic task prioritization for multitask learning. In _Proceedings of the European conference on computer vision (ECCV)_ , volume 11220 of _Lecture Notes in Computer Science_ , pages 282–299. Springer. 

- Alex Kendall, Yarin Gal, and Roberto Cipolla. 2018. Multi-task learning using uncertainty to weigh losses for scene geometry and semantics. In _IEEE Conference on Computer Vision and Pattern Recognition, CVPR_ , pages 7482–7491. Computer Vision Foundation / IEEE Computer Society. 

- Jürgen Kindler. 2005. A simple proof of sion’s minimax theorem. _The American Mathematical Monthly_ , 112(4):356–358. 

- Baijiong Lin, Feiyang Ye, and Yu Zhang. 2021. A closer look at loss weighting in multi-task learning. _CoRR_ , abs/2111.10603. 

- Xi Lin, Hui-Ling Zhen, Zhenhua Li, Qingfu Zhang, and Sam Kwong. 2019. Pareto multi-task learning. In _Advances in Neural Information Processing Systems 32: Annual Conference on Neural Information Processing Systems, NeurIPS_ , pages 12037–12047. 

- Bo Liu, Xingchao Liu, Xiaojie Jin, Peter Stone, and Qiang Liu. 2021. Conflict-averse gradient descent for multi-task learning. In _Advances in Neural Information Processing Systems 34: Annual Conference on Neural Information Processing Systems, NeurIPS_ , pages 18878–18890. 

- Pengfei Liu, Xipeng Qiu, and Xuanjing Huang. 2017. Adversarial multi-task learning for text classification. In _Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics_ , pages 1–10. Association for Computational Linguistics. 

- Shikun Liu, Edward Johns, and Andrew J. Davison. 2019. End-to-end multi-task learning with attention. In _IEEE Conference on Computer Vision and Pattern Recognition, CVPR_ , pages 1871–1880. Computer Vision Foundation / IEEE. 

2573 

- Xiaodong Liu, Jianfeng Gao, Xiaodong He, Li Deng, Kevin Duh, and Ye-Yi Wang. 2015. Representation learning using multi-task deep neural networks for semantic classification and information retrieval. In _The 2015 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies_ , pages 912– 921. The Association for Computational Linguistics. 

- Yuren Mao, Zekai Wang, Weiwei Liu, Xuemin Lin, and Wenbin Hu. 2021. Banditmtl: Bandit-based multi-task learning for text classification. In _Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing, ACL/IJCNLP_ , pages 5506–5516. Association for Computational Linguistics. 

- Yuren Mao, Zekai Wang, Weiwei Liu, Xuemin Lin, and Pengtao Xie. 2022. Metaweighting: Learning to weight tasks in multi-task learning. In _Findings of the Association for Computational Linguistics: ACL_ , pages 3436–3448. Association for Computational Linguistics. 

- Yuren Mao, Shuang Yun, Weiwei Liu, and Bo Du. 2020. Tchebycheff procedure for multi-task text classification. In _Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, ACL_ , pages 4217–4226. Association for Computational Linguistics. 

   - Rachel Ward, Xiaoxia Wu, and Leon Bottou. 2020. Adagrad stepsizes: Sharp convergence over nonconvex landscapes. _The Journal of Machine Learning Research_ , 21(1):9047–9076. 

   - Liqiang Xiao, Honglun Zhang, and Wenqing Chen. 2018. Gated multi-task network for text classification. In _Proceedings of the 2018 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, NAACL-HLT_ , pages 726–731. Association for Computational Linguistics. 

   - Senthil Kumar Yogamani, Christian Witt, Hazem Rashed, Sanjaya Nayak, Saquib Mansoor, Padraig Varley, Xavier Perrotton, Derek O’Dea, Patrick Pérez, Ciarán Hughes, Jonathan Horgan, Ganesh Sistu, Sumanth Chennupati, Michal Uricár, Stefan Milz, Martin Simon, and Karl Amende. 2019. Woodscape: A multi-task, multi-camera fisheye dataset for autonomous driving. In _IEEE/CVF International Conference on Computer Vision, ICCV_ , pages 9307–9317. 

   - Tianhe Yu, Saurabh Kumar, Abhishek Gupta, Sergey Levine, Karol Hausman, and Chelsea Finn. 2020. Gradient surgery for multi-task learning. In _Advances in Neural Information Processing Systems 33: Annual Conference on Neural Information Processing Systems, NeurIPS_ . 

- Ishan Misra, Abhinav Shrivastava, Abhinav Gupta, and Martial Hebert. 2016. Cross-stitch networks for multi-task learning. In _IEEE Conference on Computer Vision and Pattern Recognition, CVPR_ , pages 3994–4003. 

- Yurii Nesterov. 1998. Introductory lectures on convex programming volume i: Basic course. _Lecture notes_ , 3(4):5. 

- Sebastian Ruder. 2017. An overview of multitask learning in deep neural networks. _CoRR_ , abs/1706.05098. 

- Ozan Sener and Vladlen Koltun. 2018. Multi-task learning as multi-objective optimization. In _Advances in Neural Information Processing Systems 31: Annual Conference on Neural Information Processing Systems, NeurIPS_ , pages 525–536. 

- Ximeng Sun, Rameswar Panda, Rogério Feris, and Kate Saenko. 2020. Adashare: Learning what to share for efficient deep multi-task learning. In _Advances in Neural Information Processing Systems 33: Annual Conference on Neural Information Processing Systems, NeurIPS_ . 

- Simon Vandenhende, Stamatios Georgoulis, Wouter Van Gansbeke, Marc Proesmans, Dengxin Dai, and Luc Van Gool. 2022. Multi-task learning for dense prediction tasks: A survey. _IEEE Trans. Pattern Anal. Mach. Intell._ , 44(7):3614–3633. 

2574 

## **A Derivations of GetMTL Algorithm** 

**Lemma A.1.** _Let d_<sup>_∗_</sup> _be the solution of_ 

_d_ max _∈_ R<sup>_m_min</sup> _i∈_ [ _T_ ]<sup>_⟨gβi, d⟩, s.t. ∥d −g_0</sup><sup>_∥≤εg_</sup> 0<sup>T</sup><sup>_d,_</sup> (15) _where ε ∈_ (0 _,_ 1] _, {gi ∈_ R<sup>_m_</sup> _| ∀i ∈{_ 0 _,_ 1 _, ..., T }}, and gβi_ = _βigi ∈_ R<sup>_m_</sup> _. Then we have_ 



_where g_ 0 = _T_ <u>1</u> � _Ti_ =1<sup>_gi,andgw∗_=�</sup><sup>_T_</sup> _i_ =1<sup>_w_</sup> _i_<sup>_∗gβ_</sup> _i_<sup>_._</sup> _The w_<sup>_∗_</sup> _is the solution of_ 





_Proof._ We first construct Lagrange function of the objective in Eq.(10), 



According the Lagrange duality and Sion’s minimax theorem (Kindler, 2005), we can switch the _max_ and _min_ without changing the solution and then the primal problem can be reformulated as following form, 



With _λ, w_ fixing, we first solve the _max_ of _L_ ( _d, λ, w_ ) w.r.t. _d_ , 



We set the gradient of _L_ ( _d, λ, w_ ) with respect to _d_ equal to zero, 



We can get the optimal _d_<sup>_∗_</sup> , 



and we plug the solution _d_<sup>_∗_</sup> in _L_ ( _d, w, λ_ ) to obtain _L_ ˆ( _d, λ, w_ ), 



Then, we set the gradient of _L_<sup>ˆ</sup> ( _λ, w_ ) with respect to _λ_ equal to zero, 



We can get the optimal _λ_<sup>_∗_</sup> , 



We then plug the _λ_<sup>_∗_</sup> in _d_<sup>_∗_</sup> to obtain, 



Finally, plugging _d_<sup>_∗_</sup> and _λ_<sup>_∗_</sup> into the objective in Eq.(20), we can obtain the following optimization problem _J_ ( _w_ ), 



We can obtain _w_<sup>_∗_</sup> by solving following optimization problem _J_ ( _w_ ) w.r.t. _w_ , formally, 



## **B Proof of Theorem 4.1** 

Following the proof of Lemma A, we use same Lagrangian function in Eq.(19) for simplicity, 



= _Proof._ Let _PD_ ( _λ, w_ ) max _d L_ ( _d, λ, w_ ) and _PP_ ( _d_ ) = min _λ,w L_ ( _d, λ, w_ ). Then we can get, 



Thus, we have, 



Since both primal problem and dual problem have optimal solutions, we have, 



2575 

Finally, we get 



Since the dual problem is a convex programming and the solutions _d_<sup>_∗_</sup> , _λ_ , and _w_ meet Karush-KuhnTucker (KKT) (Bertsekas, 1997; Désidéri, 2012) conditions, we can get, 



That is, the optimal value defined by Eq. (14) is equal to optimal value defined by Eq. (9). Therefore, we can solve complex _Maximin Optimization Problem_ in Eq.(9) by solving its dual problem. ■ 

## **C Proof of Theorem 4.2** 

**Lemma C.1.** _If ℓ is differential and L-smooth, ∇ℓ is L-Lipschitz continuous, then_ 



_Proof._ Using the fundamental theorem of calculus with the continuous function _∇ℓ_ , we can get, 







### **Proof of Theorem 4.2** 

_Proof._ Let _{θ_<sup>(</sup><sup>_t_)</sup> _}_<sup>_∞_</sup> _t_ =1<sup>bemodelparametersse-</sup> quence generated by using update rule _θ_<sup>(</sup><sup>_t_+1)</sup> = _θ_<sup>(</sup><sup>_t_)</sup> _− µ_<sup>(</sup><sup>_t_)</sup> _d_ where _d_ is defined in Eq.(13). Since all _∇ℓi_ are Lipschitz continuous, for each loss 







This inequality implies that the objective function value of all tasks strictly decreases with each iteration when using the GetMTL algorithm. We next analyze the rationality of step size _µ_<sup>(</sup><sup>_t_)</sup> in Lemma C.2. 

■ 

**Lemma C.2.** _The convergence of Gradient Descent with step size µ is guaranteed only if the step size µ >_ 0 _is carefully chosen such that µ <_ 1 _/L (Nesterov, 1998; Ward et al., 2020) where L >_ 0 _is the e Lipschitz smoothness constant. Then we have,_ 



_Proof._ (1) Proof of left part of inequality. 



Therefore, we can get _µ >_ 0. 

(2) Proof of right part of inequality. 



where _ϕ ∈_ [0<sup>_◦_</sup> _,_ 90<sup>_◦_</sup> ) denotes the angle of _d_ and _g_ 0. In general, we all penalize gradient norm for improving the generalization and stability. We thus can get _∥d∥_<sup>2</sup> _−∥g_ 0 _∥_<sup>2</sup> _>_ 0 when _ε ∈_ (0 _,_ 1]. Then, 



Then, we can get 0 _< µ <_ 1 _/L_ . 



2576 

|Tasks|STL|Uniform|Uncertainty|GradNorm|MGDA T|chebycheffAd|v BanditMTL|MetaWeighting|GetMTL(Ours)|
|---|---|---|---|---|---|---|---|---|---|
|COMP|87.36|86.84|86.76|86.26|87.88|87.36|88.06|87.99|**89.67**|
|REC|94.48|96.21|96.02|95.63|96.25|95.84|96.16|95.9|**96.39**|
|SCI|94.45|96.26|96.35|96.08|95.78|95.82|95.66|96.08|**96.56**|
|TALK|85.04|86.08|86.27|85.94|86.56|85.96|85.93|85.82|**86.84**|
|AVG|90.43|90.93|90.87|90.7|91.2|90.87|91.26|91.25|**92.09**|



Table 2: The complete performance of 4 tasks in topic classification dataset with our GetMTL and other MTL baselines. 

|Tasks|STL|Uniform|Uncertainty|GradNorm|MGDA T|chebycheffAd|v BanditMTL|MetaWeighting|GetMTL(Ours)|
|---|---|---|---|---|---|---|---|---|---|
|Apparel|87.57|89.18|89.59|88.69|88.63|87.98|88.95|89.83|**90.03**|
|Baby|87.14|89.91|89.96|89.33|89.05|88.65|90.02|90.01|**90.32**|
|Books|87.02|87.64|87.09|87.14|85.66|86.65|87.09|86.82|**87.77**|
|Camera|90.54|91.49|91.54|90.84|91.05|91.44|91.54|91.54|**92.26**|
|Dvd|84.61|88.17|87.35|87.32|87.65|87.24|87.08|88.02|**89.30**|
|Electronics|85.42|88.09|88.68|88.88|87.94|86.80|87.60|86.99|**89.49**|
|Health|89.07|90.82|91.50|90.59|90.86|90.55|91.81|**91.85**|**91.85**|
|Kitchen|85.16|89.51|89.65|89.33|88.69|87.67|90.07|89.25|**90.81**|
|Magazines|93.32|93.61|92.54|93.35|93.21|93.40|93.36|94.30|**94.43**|
|Music|83.92|84.27|86.25|84.97|85.01|83.90|86.37|86.88|**87.04**|
|Software|89.97|92.44|92.59|93.24|92.82|92.77|92.95|92.71|**93.93**|
|Sports|87.52|90.52|90.42|90.88|90.65|89.85|89.72|89.96|**91.81**|
|Toys|87.02|88.73|89.89|88.10|88.30|88.49|88.47|89.11|**90.62**|
|Video|88.8|89.65|89.28|88.92|89.33|89.06|89.62|**89.88**|89.55|
|Avg|86.52|88.47|88.74|88.01|88.30|87.71|88.78|89.14|**89.80**|



Table 3: The complete performance of 14 tasks in amazon review dataset with our GetMTL and other MTL baselines. 

## **D Complete Performance of Each Task for Amazon Dataset** 

Amazon review dataset includes 14 domains, such as _Apparel_ , _Baby_ , _Books_ , _Camera_ , _Dvd_ , _Electronics_ , _Health_ , _Kitchen_ , _Magazines_ , _Music_ , _Software_ , _Sports_ , _Toys_ , and _Video_ . Each domain is treated as a 14 binary classification task. 

We provide the full comparison on the amazon review and topic classification datasets in Table 3 and Table 2 respectively. Table 2 shows that our GetMTL can achieve the best average classification accuracy of 92.09%, outperforming the second-best model BanditMTL by a margin of 0.83%. Moreover, our GetMTL can also beat other baselines on each individual tasks. Table 3 reports the performance of all 14 tasks on amazon review dataset. Our proposed GetMTL achieves the best performance on 13 out of 14 tasks and obtain best average classification accuracy. 

2577 

### **<u>ACL 2023 Responsible NLP Checklist</u>** 

- **A For every submission:** 

- A1. Did you describe the limitations of your work? _Section of Limitations_ 

- A2. Did you discuss any potential risks of your work? _Not applicable. Left blank._ 

- A3. Do the abstract and introduction summarize the paper’s main claims? _Abstract and Introduction_ 

- A4. Have you used AI writing assistants when working on this paper? _Left blank._ 

- **B** □ **Did you use or create scientific artifacts?** 

_Section of GetMTL, Experimental datasets_ 

- B1. Did you cite the creators of artifacts you used? _Experimental datasets_ 

- B2. Did you discuss the license or terms for use and / or distribution of any artifacts? _It is published by the authors._ 

- B3. Did you discuss if your use of existing artifact(s) was consistent with their intended use, provided that it was specified? For the artifacts you create, do you specify intended use and whether that is compatible with the original access conditions (in particular, derivatives of data accessed for research purposes should not be used outside of research contexts)? 

_Section of Experimental Implementation_ 

- B4. Did you discuss the steps taken to check whether the data that was collected / used contains any information that names or uniquely identifies individual people or offensive content, and the steps taken to protect / anonymize it? 

   - _Not applicable. Left blank._ 

- B5. Did you provide documentation of the artifacts, e.g., coverage of domains, languages, and linguistic phenomena, demographic groups represented, etc.? _Not applicable. Left blank._ 

- B6. Did you report relevant statistics like the number of examples, details of train / test / dev splits, etc. for the data that you used / created? Even for commonly-used benchmark datasets, include the number of examples in train / validation / test splits, as these provide necessary context for a reader to understand experimental results. For example, small differences in accuracy on large test sets may be significant, while on small test sets they may not be. _Left blank._ 

- **C** □ **Did you run computational experiments?** 

_Left blank._ 

- C1. Did you report the number of parameters in the models used, the total computational budget (e.g., GPU hours), and computing infrastructure used? _No response._ 

_The Responsible NLP Checklist used at ACL 2023 is adopted from NAACL 2022, with the addition of a question on AI writing assistance._ 

2578 

- C2. Did you discuss the experimental setup, including hyperparameter search and best-found hyperparameter values? _No response._ 

- C3. Did you report descriptive statistics about your results (e.g., error bars around results, summary statistics from sets of experiments), and is it transparent whether you are reporting the max, mean, etc. or just a single run? 

   - _No response._ 

- C4. If you used existing packages (e.g., for preprocessing, for normalization, or for evaluation), did you report the implementation, model, and parameter settings used (e.g., NLTK, Spacy, ROUGE, etc.)? 

_No response._ 

- **D** □ **Did you use human annotators (e.g., crowdworkers) or research with human participants?** _Left blank._ 

- D1. Did you report the full text of instructions given to participants, including e.g., screenshots, disclaimers of any risks to participants or annotators, etc.? _No response._ 

- D2. Did you report information about how you recruited (e.g., crowdsourcing platform, students) and paid participants, and discuss if such payment is adequate given the participants’ demographic (e.g., country of residence)? 

   - _No response._ 

- D3. Did you discuss whether and how consent was obtained from people whose data you’re using/curating? For example, if you collected data via crowdsourcing, did your instructions to crowdworkers explain how the data would be used? _No response._ 

- D4. Was the data collection protocol approved (or determined exempt) by an ethics review board? _No response._ 

- D5. Did you report the basic demographic and geographic characteristics of the annotator population that is the source of the data? _No response._ 

2579 

