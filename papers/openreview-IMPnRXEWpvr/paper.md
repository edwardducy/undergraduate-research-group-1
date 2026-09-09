# TOWARDS IMPARTIAL MULTI-TASK LEARNING 

**Liyang Liu**<sup>1</sup> **, Yi Li**<sup>2</sup> **, Zhanghui Kuang**<sup>2</sup> **, Jing-Hao Xue**<sup>3</sup> **, Yimin Chen**<sup>2</sup> **, Wenming Yang**<sup>1</sup><sup>_∗_</sup> **, Qingmin Liao**<sup>1</sup> **, Wayne Zhang**<sup>2</sup><sup>_,_4</sup> 1Shenzhen International Graduate School/Department of Electronic Engineering, Tsinghua University 2SenseTime Research 3Department of Statistical Science, University College London 4Qing Yuan Research Institute, Shanghai Jiao Tong University _{_ liu-ly14@mails., yang.wenming@sz., liaoqm@ _}_ tsinghua.edu.cn _{_ liyi, kuangzhanghui, chenyimin, wayne.zhang _}_ @sensetime.com jinghao.xue@ucl.ac.uk 

## ABSTRACT 

Multi-task learning (MTL) has been widely used in representation learning. However, na¨ıvely training all tasks simultaneously may lead to the partial training issue, where specific tasks are trained more adequately than others. In this paper, we propose to learn multiple tasks impartially. Specifically, for the _task-shared_ parameters, we optimize the scaling factors via a closed-form solution, such that the aggregated gradient (sum of raw gradients weighted by the scaling factors) has equal projections onto individual tasks. For the _task-specific_ parameters, we dynamically weigh the task losses so that all of them are kept at a comparable scale. Further, we find the above _gradient_ balance and _loss_ balance are complementary and thus propose a hybrid balance method to further improve the performance. Our impartial multi-task learning (IMTL) can be end-to-end trained without any heuristic hyper-parameter tuning, and is general to be applied on all kinds of losses without any distribution assumption. Moreover, our IMTL can converge to similar results even when the task losses are designed to have different scales, and thus it is scale-invariant. We extensively evaluate our IMTL on the standard MTL benchmarks including Cityscapes, NYUv2 and CelebA. It outperforms existing loss weighting methods under the same experimental settings. 

## 1 INTRODUCTION 

Recent deep networks in computer vision can match or even surpass human beings on some specific tasks separately. However, in reality multiple tasks ( _e.g._ , semantic segmentation and depth estimation) must be solved simultaneously. Multi-task learning (MTL) (Caruana, 1997; Evgeniou & Pontil, 2004; Ruder, 2017; Zhang & Yang, 2017) aims at sharing the learned representation among tasks (Zamir et al., 2018) to make them benefit from each other and achieve better results and stronger robustness (Zamir et al., 2020). However, sharing the representation can lead to a partial learning issue: some specific tasks are learned well while others are overlooked, due to the different loss scales or gradient magnitudes of various tasks and the mutual competition among them. Several methods have been proposed to mitigate this issue either via _gradient balance_ such as gradient magnitude normalization (Chen et al., 2018) and Pareto optimality (Sener & Koltun, 2018), or _loss balance_ like homoscedastic uncertainty (Kendall et al., 2018). Gradient balance can evenly learn task-shared parameters while ignoring task-specific ones. Loss balance can prevent MTL from being biased in favor of tasks with large loss scales but cannot ensure the impartial learning of the shared parameters. In this work, we find that gradient balance and loss balance are complementary, and combining the two balances can further improve the results. To this end, we propose _impartial_ MTL (IMTL) via simultaneously balancing gradients and losses across tasks. 

For gradient balance, we propose IMTL-G(rad) to learn the scaling factors such that the aggregated gradient of task-shared parameters has equal projections onto the raw gradients of individual tasks 

> _∗_ Corresponding author 

1 

Published as a conference paper at ICLR 2021 



g3 g3 g3 g3<br>g g g2 g g g g2 g g g g g2 g g2<br>g1 g1 g1 g1<br>(a) GradNorm (b) MGDA (c) PCGrad (d)  IMTL-G<br>
Figure 1: Comparison of gradient balance methods. In (a) to (d), **_g_** 1, **_g_** 2 and **_g_** 3 represent the gradient computed by the raw loss of each task, respectively. The gray surface represents the plane composed by these gradients. The red arrow denotes the aggregated gradient computed by the weighted sum loss, which is ultimately used to update the model parameters. The blue arrows show the projections of **_g_** onto the raw gradients _{_ **_g_** _t}_ . **_g_** has the largest projection on **_g_** 2 ( **nearest** to the mean direction), **_g_** 3 ( **smallest** magnitude) and **_g_** 2 ( **largest** magnitude) for GradNorm, MGDA and PCGrad, respectively, while the projections are **equal** on _{_ **_g_** _t}_ in our IMTL-G. 

(see Fig. 1 (d)). We show that the scaling factor optimization problem is equivalent to finding the angle bisector of gradients from all tasks in geometry, and derive a closed-form solution to it. In contrast with previous gradient balance methods such as GradNorm (Chen et al., 2018), MGDA (Sener & Koltun, 2018) and PCGrad (Yu et al., 2020), which have learning biases in favor of tasks with gradients close to the average gradient direction, those with small gradient magnitudes, and those with large gradient magnitudes, respectively (see Fig. 1 (a), (b) and (c)), in our IMTL-G task-shared parameters can be updated without bias to any task. 

For loss balance, we propose IMTL-L(oss) to automatically learn a loss weighting parameter for each task so that the weighted losses have comparable scales and the effect of different loss scales from various tasks can be canceled-out. Compared with uncertainty weighting (Kendall et al., 2018), which has biases towards regression tasks rather than classification tasks, our IMTL-L treats all tasks equivalently without any bias. Besides, we model the loss balance problem from the optimization perspective without any distribution assumption that is required by (Kendall et al., 2018). Therefore, ours is more general and can be used in any kinds of losses. Moreover, the loss weighting parameters and the network parameters can be jointly learned in an end-to-end fashion in IMTL-L. 

Further, we find the above two balances are complementary and can be combined to improve the performance. Specifically, we apply IMTL-G on the task-shared parameters and IMTL-L on the task-specific parameters, leading to the hybrid balance method IMTL. Our IMTL is scale-invariant: the model can converge to similar results even when the same task is designed to have different loss scales, which is common in practice. For example, the scale of the cross-entropy loss in semantic segmentation may have different scales when using “average” or “sum” reduction over locations in the loss computation. We empirically validate that our IMTL is more robust against heavy loss scale changes than its competitors. Meanwhile, our IMTL only adds negligible computational overheads. 

We extensively evaluate our proposed IMTL on standard benchmarks: Cityscapes, NYUv2 and CelebA, where the experimental results show that IMTL achieves superior performances under all settings. Besides, considering there lacks a fair and practical benchmark for comparing MTL methods, we unify the experimental settings such as image resolution, data augmentation, network structure, learning rate and optimizer option. We re-implement and compare with the representative MTL methods in a unified framework, which will be publicly available. Our contributions are: 

- We propose a novel closed-form gradient balance method, which learns task-shared parameters without any task bias; and we develop a general learnable loss balance method, where no distribution assumption is required and the scale parameters can be jointly trained with the network parameters. 

- We unveil that gradient balance and loss balance are complementary and accordingly propose a hybrid balance method to simultaneously balance gradients and losses. 

- We validate that our proposed IMTL is loss scale-invariant and is more robust against loss scale changes compared with its competitors, and we give in-depth theoretical and experimental analyses on its connections and differences with previous methods. 

- We extensively verify the effectiveness of our IMTL. For fair comparisons, a unified codebase will also be publicly available, where more practical settings are adopted and stronger performances are achieved compared with existing code-bases. 

2 

Published as a conference paper at ICLR 2021 

## 2 RELATED WORK 

Recent advances in MTL mainly come from two aspects: network structure improvements and loss weighting developments. Network-structure methods based on soft parameter-sharing usually lead to high inference cost (review in Appendix A). Loss weighting methods find loss weights to be multiplied on the raw losses for model optimization. They employ a hard parameter-sharing paradigm (Ruder, 2017), where several light-weight task-specific heads are attached upon the heavy-weight task-agnostic backbone. There are also efforts that learn to group tasks and branch the network in the middle layers (Guo et al., 2020; Standley et al., 2020), which try to achieve better accuracyefficiency trade-off and can be seen as semi-hard parameter-sharing. We believe task grouping and loss weighting are orthogonal and complementary directions to facilitate multi-task learning and can benefit from each other. In this work we focus on loss weighting methods which are the most economic as almost all of the computations are shared across tasks, leading to high inference speed. Task Prioritization (Guo et al., 2018) weights task losses by their difficulties to focus on the harder tasks during training. Uncertainty weighting (Kendall et al., 2018) models the loss weights as dataagnostic task-dependent homoscedastic uncertainty. Then loss weighting is derived from maximum likelihood estimation. GradNorm (Chen et al., 2018) learns the loss weights to enforce the norm of the scaled gradient for each task to be close. MGDA (Sener & Koltun, 2018) casts multi-task learning as multi-object optimization and finds the minimum-norm point in the convex hull composed by the gradients of multiple tasks. Pareto optimality is supposed to be achieved under mild conditions. GLS (Chennupati et al., 2019) instead uses the geometric mean of task-specific losses as the target loss, we will show it actually weights the loss by its reciprocal value. PCGrad (Yu et al., 2020) avoids interferences between tasks by projecting the gradient of one task onto the normal plane of the other. DSG (Lu et al., 2020) dynamically makes a task “stop or go” by its converging state, where a task is updated only once for a while if it is stopped. Although many loss weighting methods have been proposed, they are seldom open-sourced and rarely compared thoroughly under practical settings where strong performances are achieved, which motivates us to give an in-depth analysis and a fair comparison about them. 

## 3 IMPARTIAL MULTI-TASK LEARNING 

In MTL, we map a sample **_x_** _∈_ X to its labels _{_ **_y_** _t ∈_ Y _t}t∈_ [1 _,T_ ] of all _T_ tasks through multiple taskspecific mappings _{_ **_f_** _t_ : X _→_ Y _t}_ . In most loss weighting methods, the hard parameter-sharing paradigm is employed, such that **_f_** _t_ is parameterized by heavy-weight task-shared parameters **_θ_** and light-weight task-specific parameters **_θ_** _t_ . All tasks take the same shared intermediate feature **_z_** = **_f_** ( **_x_** ; **_θ_** ) as input, and the _t_ -th task head outputs the prediction as **_f_** _t_ ( **_x_** ) = **_f_** _t_ ( **_z_** ; **_θ_** _t_ ). We aim to find the scaling factors _{αt}_ for all _T_ task losses _{Lt_ ( **_f_** _t_ ( **_x_** ) _,_ **_y_** _t_ ) _}_ , so that the weighted sum loss _L_ =<sup>�</sup> _t_<sup>_αtLt_canbeoptimizedtomakealltasksperformwell.Thisposesgreatchallenges</sup> because: 1) losses may have distinguished forms such as cross-entropy loss and cosine similarity; 2) the dynamic ranges of losses may differ by orders of magnitude. In this work, we propose a hybrid solution for both the task-shared parameters **_θ_** and the task-specific parameters _{_ **_θ_** _t}_ , as Fig. 2. 

### 3.1 GRADIENT BALANCE: IMTL-G 

For task-shared parameters **_θ_** , we can receive _T_ gradients _{_ **_g_** _t_ = _∇_ **_θ_** _Lt}_ via back-propagation from all of the _T_ raw losses _{Lt}_ , and these gradients represent optimal update directions for individual tasks. As the parameters **_θ_** can only be updated with a single gradient, we should compute an aggregated gradient **_g_** by the linear combination of _{_ **_g_** _t}_ . It also implies to find the scaling factors _{αt}_ of raw losses _{Lt}_ , since **_g_** =<sup>�</sup> _t_<sup>_αt_</sup><sup>**_g_**</sup><sup>_t_=</sup><sup>_∇_</sup><sup>**_θ_**</sup><sup>_L_=</sup><sup>_∇_</sup><sup>**_θ_**(�</sup> _t_<sup>_αtLt_).</sup> Motivated by the principle of balance among tasks, we propose to make the projections of **_g_** onto _{_ **_g_** _t}_ to be equal, as Fig. 1 (d). In this way, 



Impartial Multi-task Learning<br>Loss  𝒇𝟏( 𝒛 ;  𝜽 𝟏) 𝒇𝟐( 𝒛 ;  𝜽 𝟐) 𝒇𝟑( 𝒛 ;  𝜽 𝟑)<br>Balance<br>Heads<br>𝜽𝟏 𝜽𝟐 𝜽𝟑<br>Shared<br>Gradient 𝒛= 𝒇(𝒙; 𝜽) Feature<br>Balance 𝜽 Backbone<br>𝒙<br>
Figure 2: Overview of IMTL. 

3 

Published as a conference paper at ICLR 2021 

**Algorithm 1** Training by Impartial Multi-task Learning **Input:** input sample **_x_** , task-specific labels _{_ **_y_** _t}_ and learning rate _η_ **Output:** task-shared/-specific parameters **_θ_** / _{_ **_θ_** _t}_ , scale parameters _{st}_ 1: compute task-shared feature **_z_** = **_f_** ( **_x_** ; **_θ_** ) 2: **for** _t_ = 1 to _T_ **do** 3: compute task prediction by head network **_f_** _t_ ( **_x_** ) = **_f_** _t_<sup>net</sup> ( **_z_** ; **_θ_** _t_ ) 4: compute raw loss by loss function _Lt_<sup>raw</sup> = _L_<sup>func</sup> _t_ ( **_f_** _t_ ( **_x_** ) _,_ **_y_** _t_ ) 5: compute scaled loss _Lt_ = _ba_<sup>_st_</sup> _Lt_<sup>raw</sup> _− st_ (default _a_ = _e, b_ = 1) _▷_ loss balance 6: compute gradient of shared feature **_z_** : **_g_** _t_ = _∇_ **_z_** _Lt_ 7: compute unit-norm gradient **_u_** _t_ = _∥_ **_gg_** _tt∥_ 8: **end for** 9: compute gradient differences **_D_**<sup>_⊤_</sup> = � **_g_** 1<sup>_⊤−_</sup><sup>**_g_**</sup> 2<sup>_⊤, · · ·,_</sup><sup>**_g_**</sup> 1<sup>_⊤−_</sup><sup>**_g_**</sup> _T_<sup>_⊤_</sup> � 10: compute unit-norm gradient differences **_U_**<sup>_⊤_</sup> = � **_u_**<sup>_⊤_</sup> 1<sup>_−_</sup><sup>**_u_**</sup><sup>_⊤_</sup> 2<sup>_, · · ·,_</sup><sup>**_u_**</sup> 1<sup>_⊤−_</sup><sup>**_u_**</sup><sup>_⊤_</sup> _T_ � 11: compute scaling factors for tasks 2 to _T_ : **_α_** 2: _T_ = **_g_** 1 **_U_**<sup>_⊤_�</sup> **_DU_**<sup>_⊤_�</sup><sup>_−_1</sup> _▷_ gradient balance 12: compute scaling factors for all tasks: **_α_** = � 1 _−_ **1** **_α_**<sup>_⊤_</sup> 2: _T_<sup>_,_</sup> **_α_** 2: _T_ � 13: update task-shared parameters **_θ_** = **_θ_** _− η∇_ **_θ_** �� _t_<sup>_αtLt_</sup> � 14: **for** _t_ = 1 to _T_ **do** 15: update task-specific parameters **_θ_** _t_ = **_θ_** _t − η∇_ **_θ_** _t Lt_ 16: update loss scale parameter _st_ = _st − η_<sup>_<u>∂</u>_</sup> _∂s_<sup>_L_</sup> _t_<sup>_<u>t</u>_</sup> 17: **<u>end for</u>** 

we treat all tasks equally so that they progress in the same speed and none is left behind. Formally, let _{_ **_u_** _t_ = **_g_** _t/ ∥_ **_g_** _t∥}_ denote the unit-norm vector of _{_ **_g_** _t}_ which are row vectors, then we have: 

**_gu_**<sup>_⊤_</sup> 1<sup>=</sup><sup>**_gu_**</sup><sup>_⊤_</sup> _t_<sup>_⇔_</sup><sup>**_g_**(</sup><sup>**_u_**1</sup><sup>_−_</sup><sup>**_u_**</sup><sup>_t_)</sup><sup>_⊤_= 0</sup><sup>_,∀_2 ⩽</sup><sup>_t_⩽</sup><sup>_T._</sup> (1) The above problem is under-determined, but we can obtain the closed-form results of _{αt}_ by constraining<sup>�</sup> _t_<sup>_αt_=1.</sup> Assume **_α_** = [ _α_ 2 _, · · · , αT_ ], **_U_**<sup>_⊤_</sup> = � **_u_**<sup>_⊤_</sup> 1<sup>_−_</sup><sup>**_u_**</sup><sup>_⊤_</sup> 2<sup>_, · · ·,_</sup><sup>**_u_**</sup><sup>_⊤_</sup> 1<sup>_−_</sup><sup>**_u_**</sup><sup>_⊤_</sup> _T_ �, **_D_**<sup>_⊤_</sup> = � **_g_** 1<sup>_⊤−_</sup><sup>**_g_**</sup> 2<sup>_⊤, · · ·,_</sup><sup>**_g_**</sup> 1<sup>_⊤−_</sup><sup>**_g_**</sup> _T_<sup>_⊤_</sup> � and **1** = [1 _, · · · ,_ 1], from Eq. (1) we can obtain: **_α_** = **_g_** 1 **_U_**<sup>_⊤_�</sup> **_DU_**<sup>_⊤_�</sup><sup>_−_1</sup> _._ (IMTL-G) (2) 

The detailed derivation is in Appendix B.1. After obtaining **_α_** , the scaling factor of the first task can be computed by _α_ 1 = 1 _−_ **1** **_α_**<sup>_⊤_</sup> since<sup>�</sup> _t_<sup>_αt_=1.The optimized</sup><sup>_{αt}_are used to compute</sup><sup>_L_=</sup> � _t_<sup>_αtLt_,which is ultimately minimized by SGD to update the model.By now,back-propagation</sup> needs to be executed _T_ times to obtain the gradient of each task loss with respect to the heavy-weight task-shared parameters **_θ_** , which is time-consuming and non-scalable. We replace the parameterlevel gradients _{_ **_g_** _t_ = _∇_ **_θ_** _Lt}_ with feature-level gradients _{∇_ **_z_** _Lt}_ to compute _{αt}_ . This implies to achieve gradient balance with respect to the last shared feature **_z_** as a surrogate of task-shared parameters **_θ_** , since it is possible for the network to back-propagate this balance all the way through the task-shared backbone starting from **_z_** . This relaxation allows us to do back propagation through the backbone only once after obtaining _{αt}_ , and thus the training time can be dramatically reduced. 

### 3.2 LOSS BALANCE: IMTL-L 

For the task-specific parameters _{_ **_θ_** _t}_ , we cannot employ IMTL-G described above, because _∇_ **_θ_** _tLτ_ = **0** _, ∀t̸_ = _τ_ , and thus only the gradient of the corresponding task _∇_ **_θ_** _tLt_ can be obtained for each **_θ_** _t_ . Instead we propose to balance the losses among tasks by forcing the scaled losses _{αtLt}_ to be constant for all tasks, without loss of generality, we take the constant as 1. Then the most direct idea is to compute the scaling factors as _{αt_ = 1 _/Lt}_ , but they are sensitive to outlier samples and manifest severe oscillations, so we further propose to _learn_ to scale losses via gradient descent and thus stronger stability can be achieved. Suppose the positive losses _{Lt >_ 0 _}_ are to be balanced, we first introduce a mapping function _h_ : R _→_ R<sup>+</sup> to transform the arbitrarily-ranged learnable scale parameters _{st}_ to positive scaling factors _{h_ ( _st_ ) _>_ 0 _}_ , hereafter we abandon the subscript _t_ for brevity. Then we should construct an appropriate scaled loss _g_ ( _s_ ) so that _both_ network parameters **_θ_** and scale parameter _s_ can be optimized by _minimizing g_ ( _s_ ). On one hand, we balance different 

4 

Published as a conference paper at ICLR 2021 

tasks by encouraging the scaled losses _h_ ( _s_ ) _L_ ( **_θ_** ) to be 1 for all tasks, so the optimality _s_<sup>_⋆_</sup> of _s_ is achieved when _h_ ( _s_ ) _L_ ( **_θ_** ) = 1, or equivalently: 



One may expect to minimize _|f_ ( _s_ ) _|_ = _|h_ ( _s_ ) _L_ ( **_θ_** ) _−_ 1 _|_ to find _s_<sup>_⋆_</sup> , however when _h_ ( _s_ ) _L_ ( **_θ_** ) _<_ 1, the gradient with respect to **_θ_** , _∇_ **_θ_** _|f_ ( _s_ ) _|_ = _−h_ ( _s_ ) _∇_ **_θ_** _L_ ( **_θ_** ), is in the opposite direction. On the other hand, assume our scaled loss _g_ ( _s_ ) is a differentiable convex function with respect to _s_ , then its minimum is achieved if and only if _s_ = _s_<sup>_⋆_</sup> , where the derivative of _g_ ( _s_ ) is zero: 



From Eq. (3) and (4) we find that the values of _f_ ( _s_ ) and _g_<sup>_′_</sup> ( _s_ ) are both 0 when _s_ = _s_<sup>_⋆_</sup> , we can then regard _f_ ( _s_ ) as the derivative of _g_ ( _s_ ), which is our target scaled loss and used to optimize both the network parameters **_θ_** and loss scale parameter _s_ , then we have: 



From Eq. (3) and (5), we notice that both _h_ ( _s_ ) and � _h_ ( _s_ ) d _s_ denote loss scales, so we have � _h_ ( _s_ ) d _s_ = _Ch_ ( _s_ ), where _C >_ 0 is a constant. According to ordinary differential equation, � _h_ ( _s_ ) d _s_ must be the exponential function: � _h_ ( _s_ ) d _s_ = _ba_<sup>_s_</sup> with _a >_ 1 _, b >_ 0 (see Appendix B.2). We then have _g_<sup>_′′_</sup> ( _s_ ) = _ka_<sup>_s_</sup> _, k >_ 0, which is always positive and verifies our assumption about the convexity of _g_ ( _s_ ). Also note that the gradient of _g_ ( _s_ ) with respect to **_θ_** , _∇_ **_θ_** _g_ ( _s_ )= � _h_ ( _s_ ) d _s∇_ **_θ_** _L_ ( **_θ_** ) = _ba_<sup>_s_</sup> _∇_ **_θ_** _L_ ( **_θ_** ), is in the appropriate direction since _ba_<sup>_s_</sup> _>_ 0. As an instantiation, we set � _h_ ( _s_ ) d _s_ = _e_<sup>_s_</sup> ( _a_ = _e, b_ = 1), then 



From Eq. (6) we find that the raw loss is scaled by _e_<sup>_s_</sup> , and _−s_ acts as a regularization to avoid the trivial solution _s_ = _−∞_ while minimizing the scaled loss _g_ ( _s_ ). As for implementation, the task losses _{Lt}_ are scaled by _{e_<sup>_st_</sup> _}_ , and the scaled losses _{e_<sup>_st_</sup> _L − st}_ are used to update both the network parameters **_θ_** , _{_ **_θ_** _t}_ and the scale parameters _{st}_ . 

### 3.3 HYBRID BALANCE: IMTL 

We have introduced IMTL-G/IMTL-L to achieve gradient/loss balance, and both of them produce scaling factors to be applied on the raw losses. They can be used solely, but we find them complementary and able to be combined to improve the performance. In IMTL-G, even if the raw losses are multiplied by arbitrary (maybe different among tasks) positive factors, the direction of the aggregated gradient **_g_** stays unchanged. Because by definition **_g_** =<sup>�</sup> _t_<sup>_αt_</sup><sup>**_g_**</sup><sup>_t_istheangularbisector</sup> of the gradients _{_ **_g_** _t}_ , and positive scaling will not change the directions of _{_ **_g_** _t}_ and thus that of **_g_** (proof in Theorem 2). So we can also obtain the scale factors _{αt}_ in IMTL-G with the losses that have been scaled by _{st}_ from IMTL-L. IMTL-G and IMTL-L are combined as: 1) the taskspecific parameters _{_ **_θ_** _t}_ and scale parameters _{st}_ are updated by scaled losses _{e_<sup>_st_</sup> _Lt − st}_ ; 2) the task-shared parameters **_θ_** are updated by<sup>�</sup> _t_<sup>_αt_(</sup><sup>_estLt_) which is the weighted average of</sup><sup>_{estLt}_,</sup> with the weights _{αt}_ computed by _{∇_ **_z_** ( _e_<sup>_st_</sup> _Lt_ ) _}_ using IMTL-G. Note that the regularization terms _{−st}_ in Eq. (6) are constants with respect to **_θ_** and **_z_** , and thus can be ignored when computing gradients and updating parameters in IMTL-G. In this way, we achieve both gradient balance for task-shared parameters and loss balance for task-specific parameters, leading to our full IMTL as illustrated in Alg. 1. 

## 4 DISCUSSION 

We draw connections between our method and previous state-of-the-arts<sup>1</sup> in Fig. 3. We will show that previous methods can all be categorized as gradient or loss balance, and thus each of them can be seen as a specification of our method. However, all of them have some intrinsic biases or short-comings leading to inferior performances, which we try to overcome. 

> 1Our analysis of PCGrad (Yu et al., 2020) can be found in Appendix C.3. 

5 

Published as a conference paper at ICLR 2021 



overlook classification tasks,<br>distribution assumption GradNorm hyper-parameter<br>uncertainty ( 𝑝𝑡 ∝𝒖𝑡𝒖𝑠 ⊤ ) tuning needed<br>( 𝛼cls𝐿cls ≈1/2, perpendicular<br>𝛼reg𝐿reg ≈1 ) to convex hull<br>IMTL-L complementary IMTL-G MGDA<br>( 𝛼𝑡𝐿𝑡 ≈const ) ( 𝑝𝑡 = const ) angular bisector ( 𝑝𝑡 ∝ 𝒈𝑡 −1 )<br>of task gradients<br>GLS equal scaled loss<br>( 𝛼𝑡𝐿𝑡 = 𝐿/𝑇 ) among tasks PCGrad may degrade to<br>unstable when task ( 𝑝𝑡 ∝ 𝒈𝑡 ) uniform scaling<br>number is large<br>loss balance gradient balance<br>
Figure 3: Relationship between our IMTL and previous methods. The blue dashed arrow indicates the characteristic of each method. In the _loss balance_ methods, we annotate the scaled loss in the bracket. _L_ cls, _L_ reg and _Lt_ are the raw loss of classification, regression and individual task, respectively. _α_ cls, _α_ reg and _αt_ is the corresponding loss scale. _L_ is the geometric mean loss and _T_ is the task number. In the _gradient balance_ methods, we annotate the projections of the aggregated gradient **_g_** =<sup>�</sup> _t_<sup>_αt_</sup><sup>**_g_**</sup><sup>_t_onto the raw gradient</sup><sup>**_g_**</sup><sup>_t_of the</sup><sup>_t_-th task in the bracket.</sup><sup>**_u_**</sup><sup>_t_=</sup><sup>**_g_**</sup><sup>_t/ ∥_</sup><sup>**_g_**</sup><sup>_t∥_is the</sup> unit-norm vector, _pt_ = **_gu_**<sup>_⊤_</sup> _t_<sup>is the projection of</sup><sup>**_g_**onto</sup><sup>**_g_**</sup><sup>_t_and</sup><sup>**_u_**</sup><sup>_s_= �</sup> _t_<sup>**_u_**</sup><sup>_t_is the mean direction.</sup> 

**GradNorm** (Chen et al., 2018) balances tasks by making the norm of the scaled gradient for each task to be approximately equal. It also introduces the inverse training rate and a hyper-parameter _γ_ to control the strength of approaching the mean gradient norm, such that tasks which learn slower can receive larger gradient magnitudes. However, it does not take into account the relationship of the gradient directions. We show that when the angle between the gradients of each pair of tasks is identical, our IMTL-G leads to the equivalent solution as GradNorm. **Theorem 1.** _If the angle between any pair of_ **_u_** _t,_ **_u_** _τ stays constant:_ **_u_** _t_ **_u_**<sup>_⊤_</sup> _τ_<sup>=</sup><sup>_C_1</sup><sup>_,∀t̸_=</sup><sup>_τwith_</sup> _C_ 1 _<_ 1 _, then our IMTL-G leads to the same solution as that of GradNorm:_ **_gu_**<sup>_⊤_</sup> _t_<sup>=</sup><sup>_C_2</sup><sup>_⇔nt≡_</sup> _∥αt_ **_g_** _t∥_ = _αt ∥_ **_g_** _t∥_ = _C_ 3 _. In the above_ **_u_** _t_ = **_g_** _t/ ∥_ **_g_** _t∥, C_ 1 _, C_ 2 _and C_ 3 _are constants._ 

Proof in Appendix C.1. In GradNorm, if without the above constant-angle condition **_u_** _t_ **_u_**<sup>_⊤_</sup> _τ_<sup>=</sup><sup>_C_1,</sup> the projection of the aggregated gradient **_g_** onto task-specific gradient, **_gu_**<sup>_⊤_</sup> _t_<sup>=(�</sup> _τ_<sup>_C_3</sup><sup>**_u_**</sup><sup>_τ_)</sup><sup>**_u_**</sup> _t_<sup>_⊤_=</sup> _C_ 3 (<sup>�</sup> _τ_<sup>**_u_**</sup><sup>_τ_)</sup><sup>**_u_**</sup> _t_<sup>_⊤_,isproportionalto(�</sup> _τ_<sup>**_u_**</sup><sup>_τ_)</sup><sup>**_u_**</sup> _t_<sup>_⊤_.Ittendstooptimizethe“majoritytasks”whose</sup> gradient directions are closer to the mean direction<sup>�</sup> _t_<sup>**_u_**</sup><sup>_t_, resulting in undesired task bias.</sup> 

**MGDA** (Sener & Koltun, 2018) finds the weighted average gradient **_g_** =<sup>�</sup> _t_<sup>_αt_</sup><sup>**_g_**</sup><sup>_t_with minimum</sup> norm in the convex hull composed by _{_ **_g_** _t}_ , so that<sup>�</sup> _t_<sup>_αt_= 1 and</sup><sup>_αt_⩾0</sup><sup>_,∀t_.It adopts an iterative</sup> method based on Frank-Wolfe algorithm to solve the multi-objective optimization problem. We note the minimum-norm point has a closed-form representation if without the constraints _{αt_ ⩾ 0 _}_ . In this case, we try to minimize **_gg_**<sup>_⊤_</sup> = (<sup>�</sup> _t_<sup>_αt_</sup><sup>**_g_**</sup><sup>_t_) (�</sup> _τ_<sup>_ατ_</sup><sup>**_g_**</sup><sup>_τ_)</sup><sup>_⊤_such that �</sup> _t_<sup>_αt_=1.It implies</sup><sup>**_g_**is</sup> perpendicular to the hyper-plane composed by _{_ **_g_** _t}_ as illustrated in Fig 1 (b), and thus we have: 



and can obtain **_α_** = **_g_** 1 **_D_**<sup>_⊤_�</sup> **_DD_**<sup>_⊤_�</sup><sup>_−_1</sup> (see Appendix C.2). From Eq. (7), we note that the aggregated gradient satisfies: **_gg_** _t_<sup>_⊤_=</sup><sup>_C_.Then the projection of</sup><sup>**_g_**onto</sup><sup>**_g_**</sup><sup>_t_,</sup><sup>**_gu_**</sup><sup>_⊤_</sup> _t_<sup>=</sup><sup>_C/ ∥_</sup><sup>**_g_**</sup><sup>_t∥_, is inversely</sup> proportional to the norm of **_g_** _t_ . So it focuses on tasks with smaller gradient magnitudes, which breaks the task balance. Even with _{αt_ ⩾ 0 _}_ , the problem still exists (see Appendix C.2) in the original MGDA method. Through experiments, we note that finding the minimum-norm point without the constraints _{αt_ ⩾ 0 _}_ leads to similar performance as MGDA with the constraints _{αt_ ⩾ 0 _}_ . In our IMTL-G, although we do not constrain _{αt_ ⩾ 0 _}_ , its loss weighting scales are always positive during the training procedure as shown in Fig. 4. 

**Uncertainty weighting** (Kendall et al., 2018) regards the task uncertainty as loss weight. For regression, it can derive _L_ 1 loss from Laplace distribution: _−_ log _p_ ( _y | f_ ( **_x_** )) = _|y − f_ ( **_x_** ) _| /b_ + log _b_ , where **_x_** is the data sample, _y_ is the ground-truth label, _f_ denotes the prediction model and _b_ is the diversity of Laplace distribution. _L_ 2 loss can be found in Appendix C.4. For classification, it takes the cross-entropy loss as a scaled categorical distribution and introduces the following approximation: 



6 

Published as a conference paper at ICLR 2021 

in which softmax _y_ ( _·_ ) stands for taking the _y_ -th entry after the softmax ( _·_ ) operator. MTL corresponds to maximizing the joint likelihood of multiple targets, then the derivations yield the scaling factor _b_ / _σ_ for the regression/classification loss. (Kendall et al., 2018) learn _b_ and _σ_ as model parameters which are updated by stochastic gradient descent. However, it is applicable only if we can find appropriate correspondence between the loss and the distribution. It is difficult to be used for losses such as cosine similarity, and it is impossible to traverse all kinds of losses to obtain a unified form for them. Moreover, it sacrifices classification tasks. From Eq. (8) we can find that the scaled cross-entropy loss is approximated as _L_ = _e_<sup>2</sup><sup>_s_</sup> _L_ cls _−s_ if we set _s_ = _−_ log _σ_ . By taking the derivative we have _∂L/∂s_ = 2 _e_<sup>2</sup><sup>_s_</sup> _L_ cls _−_ 1. Then _s_ is optimized to make the scaled loss _e_<sup>2</sup><sup>_s_</sup> _L_ cls to be close to 1 _/_ 2. However, the scaled _L_ 1 loss is approximated as _L_ = _e_<sup>_s_</sup> _L_ reg _− s_ if we set _s_ = _−_ log _b_ , and taking the derivative we have _∂L/∂s_ = _e_<sup>_s_</sup> _L_ reg _−_ 1. So _s_ is optimized to make the scaled _L_ 1 loss to achieve 1, which is twice of the classification loss, and thus the classification task is overlooked. 

We would like to remark the differences between our IMTL-L and uncertainty weighting (Kendall et al., 2018). **Firstly** , our derivation is motivated by the fairness among tasks, which intrinsically differs from uncertainty weighting which is based on task uncertainty considering each task independently. **Secondly** , IMTL-L learns to balance among tasks without any biases, while uncertainty weighting may sacrifice classification tasks to favor regression tasks as derived above. **Thirdly** , IMTL-L does not depend on any distribution assumptions and thus can be generally applied to various losses including cosine similarity, which uncertainty weighting may have difficulty with. As far as we know, there is no appropriate correspondence between cosine similarity and specific distributions. **Lastly** , uncertainty weighting needs to deal with different losses case by case, it also introduces approximations in order to derive scaling factors for certain losses (such as cross-entropy loss) which may not be optimal, but our IMTL-L has a unified form for all kinds of losses. 

<u>1</u> **GLS** (Chennupati et al., 2019) calculates the target loss as the geometric mean: _L_ = (<sup>�</sup> _t_<sup>_Lt_)</sup> _T_ , then the gradient of _L_ with respect to the model parameters **_θ_** can be obtained as Appendix C.5, which can be regarded as to weigh the loss with its reciprocal value. However, as the gradient depends on the value of _L_ , so it is not scale-invariant to the loss scale changes. Moreover, we find it to be unstable when the number of tasks is large because of the geometric mean computation. 

## 5 EXPERIMENTS 

In previous methods, various experimental settings have been adopted but there are no extensive comparisons. As one contribution of our work, we re-implement representative methods and present fair comparisons among them under the unified code-base, where more practical settings are adopted and stronger performances are achieved compared with existing code-bases. The implementations exactly follow the original papers and open-sourced code to ensure the correctness. We run experiments on the Cityscapes (Cordts et al., 2016), NYUv2 (Silberman et al., 2012) and CelebA (Liu et al., 2015) dataset to extensively analyze different methods. Details can be found in Appendix D. 

**Results on Cityscapes.** From Tab. 1 we can obtain several informative conclusions. The uniform scaling baseline, which na¨ıvely adds all losses, tends to optimize tasks with larger losses and gradient magnitudes, resulting in severe task bias. Uncertainty weighting (Kendall et al., 2018) sacrifices classification tasks to aid regression ones, leading to significantly worse results on semantic segmentation compared with our IMTL-L. GradNorm (Chen et al., 2018) is very sensitive to the choice of the hyper-parameter _γ_ controlling the strength of equal gradient magnitudes, where the default _γ_ = 1 _._ 5 works well on NYUv2 but performs badly on Cityscapes. We find its best option is _γ_ = 0 which makes the scaled gradient norm to be exactly equal. MGDA (Sener & Koltun, 2018) focuses on tasks with smaller gradient magnitudes. So the performance of semantic segmentation is good but the other two tasks have difficulty in converging. In addition, we find our proposed closed-form variant without the hard constraints _{αt_ ⩾ 0 _}_ achieves similar results as the original iterative method. Through the experiments we notice the closed-form solution almost always yields _{αt_ ⩾ 0 _}_ . As for PCGrad (Yu et al., 2020), it yields slightly better performance than uniform scaling because its conflict projection will have no effect when the angles between the gradients are equal or less than _π/_ 2. In contrast, our IMTL method, in terms of both gradient balance and loss balance, yields competitive performance and achieves the best balance among tasks. Moreover, we verify that the two balances are complementary and can be combined to further improve the performance, with the visualizations in Appendix E. Surprisingly, we find our IMTL can beat the single-task baseline where 

7 

Published as a conference paper at ICLR 2021 

Table 1: Comparison between IMTL and previous methods on Cityscapes, **sem** antic segmentation, **ins** tance segmentation and **disp** arity/depth estimation are considered. The first group of columns shows the regular results of different methods. The second group shows the results by manually multiply the semantic segmentation loss with 10 before applying these methods. The subscript numbers show the absolute change after scaling the loss to demonstrate the robustness of various methods. The arrows indicate the values are the higher the better ( _↑_ ) or the lower the better ( _↓_ ). The best and runner up results for each task are bold and underlined, respectively. 

|**method**|**sem.**<br>mIoU_↑_|**ins.**<br>_L_1 _↓_|**disp.**<br>_L_1 _↓_|**sem.**<br>mIoU_↑|_∆_|↓_|**ins.**<br>_L_1 _↓|_∆_|↓_|**disp.**<br>_L_1 _↓|_∆_|↓_|**time**<br>s/iter_↓_|
|---|---|---|---|---|---|---|---|
|baselines||||||||
|single-task|76.67|21.61|4.182|-|-|-|-|
|uniform scaling|58.99|18.13|3.512|-|-|-|1.201|
|_loss balance_||||||||
|uncertainty (Kendall et al., 2018)|74.91|16.43|**2.895**|74.000_._91|16.770_._34|**2.930**0_._035|1.204|
|GLS (Chennupati et al., 2019)|75.65|17.18|2.953|66.229_._43|21.093_._91|3.3580_._405|1.202|
|**IMTL-L**|76.89|16.69|2.944|75.551_._34|17.490_._80|2.9720_._028|1.202|
|gradient_balance_||||||||
|GradNorm (_γ_ = 0)|76.27|17.99|3.195|72.963_._31|19.361_._37|3.2160_._021|1.741|
|GradNorm (Chen et al., 2018)|52.17|19.88|4.098|54.232_._06|20.530_._65|4.1080_._010|1.742|
|MGDA (w/o_{αt_ ⩾0_}_)|76.95|53.19|6.296|76.360_._59|29.0624_._13|3.3772_._919|1.777|
|MGDA (Sener & Koltun, 2018)|76.56|53.14|6.644|72.354_._21|29.3823_._76|3.3363_._308|1.732|
|PCGrad (Yu et al., 2020)|60.50|17.99|3.450|66.335_._83|17.990_._00|3.3860_._064|2.087|
|IMTL-G (exact)|76.13|17.46|2.979|-|-|-|2.769|
|**IMTL-G**|76.52|16.61|2.997|76.060_._46|17.520_._91|3.0200_._023|1.776|
|hybrid_balance_||||||||
|**IMTL**|**77.00**|**15.96**|2.905|**76.56**0_._44|**15.85**0_._11|2.9380_._033|1.795|



each task is trained with a separate model. Training multiple tasks simultaneously can learn a better representation from multiple levels of semantics, which can in turn improve individual tasks. 

In addition, we present the real-world training time of each iteration for different methods in Tab. 1. As shown, loss balance methods are the most efficient, and our gradient balance method IMTLG adds acceptable computational overhead, similar to that of GradNorm (Chen et al., 2018) and MGDA (Sener & Koltun, 2018). It benefits from computing gradients with respect to the shared feature maps instead of the shared model parameters (the row of “IMTL-G (exact)”), which brings similar performances but adds significant complexity due to multiple ( _T_ ) backward passes through the shared parameters. Our IMTL-G only needs to do backward computation on the shared parameters once after obtaining the loss weights via Eq. (2), in which the computation overhead mainly comes from the matrix multiplication rather than the matrix inverse, since the inversed matrix **_DU_**<sup>_⊤_</sup> _∈_ R<sup>(</sup><sup>_T −_1)</sup><sup>_×_(</sup><sup>_T −_1)</sup> is small compared with dimension of the shared feature **_z_** . 

As we outperform MGDA (Sener & Koltun, 2018) and PCGrad (Yu et al., 2020) significantly in terms of the objective metrics shown in Tab. 1, we further compare the qualitative results of our hybrid balance IMTL with the loss balance method uncertainty weighting (Kendall et al., 2018) and the gradient balance method GradNorm (Chen et al., 2018) considering their strong performances (see Fig. 6). For depth estimation we only show predictions at the pixels where ground truth (GT) labels exist to compare with GT, which is different from Fig. 7 where depth predictions are shown for all pixels. Consistent with results in Tab. 1, our IMTL shows visually noticeable improvements especially for the semantic and instance segmentation tasks. It is worth noting that we conduct experiments under strong baselines and practical settings which are seldom explored before, in this case changing the backbone in PSPNet (Zhao et al., 2017) from ResNet-50 to ResNet-101 can only improve mIoU of the semantic segmentation task around 0 _._ 5% according to the public code base<sup>2</sup> . 

**Scale invariance.** We are also interested in the scale invariance, which means how the results change with the loss scale. For example, in semantic segmentation, the loss scale is different if we replace the reduction method “mean” (averaged over all locations) with “sum” (summed over all locations) in the cross-entropy loss computation, or the number of the interested classes increases. The scale invariance is beneficial for model robustness. So to simulate this effect, we manually multiply the semantic segmentation loss by 10 and apply the same methods to see how the performances are affected. In the last three columns of Tab. 1 we report the absolute changes resulting from the 

> 2https://github.com/open-mmlab/mmsegmentation/tree/master/configs/pspnet 

8 

Published as a conference paper at ICLR 2021 

Table 2: Experimental results on the NYUv2 and CelebA datasets, **sem** antic segmentation, surface **norm** al estimation, **depth** estimation and multi-class **class** ification are considered. Arrows indicate the values are the higher the better ( _↑_ ) or the lower the better ( _↓_ ). The best and runner up results in each column are bold and underlined, respectively. 

|**method**|**sem.**<br>mIoU_↑_|NYUv2<br>**norm.**<br>cos_↑_|**depth**<br>_L_1 _↓_|CelebA<br>**class.**<br>acc. _↑_|
|---|---|---|---|---|
|_baselines_<br>single-task|56.82|0.8827|0.5097|-|
|uniform scaling|57.40|0.8684|0.4248|90.01|
|_loss balance_<br>uncertainty (Kendall et al., 2018)|57.20|-|0.4400|90.34|
|<br>GLS (Chennupati et al., 2019)|57.84|0.8762|0.4243|-|
|<br>**IMTL-L**|58.36|0.8864|**0.4173**|90.54|
|_gradient balance_|||||
|<br>GradNorm (_γ_ = 0)|55.96|0.8818|0.4317|90.91|
|GradNorm (Chen et al., 2018)|56.92|0.8787|0.4285|89.92|
|MGDA (w/o_{αt_ ⩾0_}_)|49.43|0.8877|0.4839|89.68|
|<br>MGDA (Sener & Koltun, 2018)|49.44|0.8875|0.4759|90.04|
|PCGrad (Yu et al., 2020)|57.48|0.8696|0.4253|89.99|
|**IMTL-G**|57.00|0.8785|0.4226|91.03|
|_hybrid balance_|||||
|**IMTL**|**58.85**|**0.8888**|0.4215|**91.12**|



multiplier. Our IMTL achieves the smallest performance fluctuations and thus the best invariance, while other methods are more or less affected by the loss scale change. 

**Results on NYUv2.** In Tab. 2 we find similar patterns as on Cityscapes, but NYUv2 is a rather small dataset, so uniform scaling can also obtain reasonable results. Note that uncertainty weighting (Kendall et al., 2018) cannot be directly used to estimate the normal surface when the cosine similarity is used as the loss, since no appropriate distribution can be found to correspond to cosine similarity. In this case, surface normal estimation owns the smallest gradient magnitude, so MGDA (Sener & Koltun, 2018) learns it best but it performs not so well for the rest two tasks. Again, our IMTL performs best taking advantage of the complementary gradient and loss balances. 

**Results on CelebA.** To compare different methods in the many-task setting, in Tab. 2 we also conduct the multi-label classification experiments on the CelebA (Liu et al., 2015) dataset. The mean accuracy of 40 tasks is used as the final metric. Our IMTL outperforms its competitors in the scenario where the task number is large, showing its superiority. Note that in this setting, GLS (Chennupati et al., 2019) has difficulty in converging and no reasonable results can be obtained. 

## 6 CONCLUSION 

We propose an impartial multi-task learning method integrating gradient balance and loss balance, which are applied on task-shared and task-specific parameters, respectively. Through our in-depth analysis, we have theoretically compared our method with previous state-of-the-arts. We have also showed that those state-of-the-arts can all be categorized as gradient or loss balance, but lead to specific bias among tasks. Through extensive experiments we verify our analysis and demonstrate the effectiveness of our method. Besides, for fair comparisons, we contribute a unified code-base, which adopts more practical settings and delivers stronger performances compared with existing code-bases, and it will be publicly available for future research. 

## ACKNOWLEDGEMENTS 

This work was supported by the Natural Science Foundation of Guangdong Province (No. 2020A1515010711), the Special Foundation for the Development of Strategic Emerging Industries of Shenzhen (No. JCYJ20200109143010272), and the Innovation and Technology Commission of the Hong Kong Special Administrative Region, China (Enterprise Support Scheme under the Innovation and Technology Fund B/E030/18). 

9 

Published as a conference paper at ICLR 2021 

## REFERENCES 

Rich Caruana. Multitask learning. _Machine learning_ , 28(1):41–75, 1997. 

- Liang-Chieh Chen, George Papandreou, Iasonas Kokkinos, Kevin Murphy, and Alan L Yuille. Deeplab: Semantic image segmentation with deep convolutional nets, atrous convolution, and fully connected crfs. _IEEE Transactions on Pattern Analysis and Machine Intelligence_ , 40(4): 834–848, 2017. 

- Zhao Chen, Vijay Badrinarayanan, Chen-Yu Lee, and Andrew Rabinovich. Gradnorm: Gradient normalization for adaptive loss balancing in deep multitask networks. In _International Conference on Machine Learning_ , pp. 794–803, 2018. 

- Sumanth Chennupati, Ganesh Sistu, Senthil Yogamani, and Samir A Rawashdeh. Multinet++: Multi-stream feature aggregation and geometric loss strategy for multi-task learning. In _Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition Workshops_ , 2019. 

- Marius Cordts, Mohamed Omran, Sebastian Ramos, Timo Rehfeld, Markus Enzweiler, Rodrigo Benenson, Uwe Franke, Stefan Roth, and Bernt Schiele. The cityscapes dataset for semantic urban scene understanding. In _Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition_ , pp. 3213–3223, 2016. 

- Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. Imagenet: A large-scale hierarchical image database. In _2009 IEEE Conference on Computer Vision and Pattern Recognition_ , pp. 248–255. IEEE, 2009. 

- Theodoros Evgeniou and Massimiliano Pontil. Regularized multi–task learning. In _Proceedings of the tenth ACM SIGKDD International Conference on Knowledge Discovery and Data Mining_ , pp. 109–117, 2004. 

- Yuan Gao, Jiayi Ma, Mingbo Zhao, Wei Liu, and Alan L Yuille. Nddr-cnn: Layerwise feature fusing in multi-task cnns by neural discriminative dimensionality reduction. In _Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition_ , pp. 3205–3214, 2019. 

- Yuan Gao, Haoping Bai, Zequn Jie, Jiayi Ma, Kui Jia, and Wei Liu. Mtl-nas: Task-agnostic neural architecture search towards general-purpose multi-task learning. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pp. 11543–11552, 2020. 

- Michelle Guo, Albert Haque, De-An Huang, Serena Yeung, and Li Fei-Fei. Dynamic task prioritization for multitask learning. In _Proceedings of the European Conference on Computer Vision (ECCV)_ , pp. 270–287, 2018. 

- Pengsheng Guo, Chen-Yu Lee, and Daniel Ulbricht. Learning to branch for multi-task learning. In _International Conference on Machine Learning_ , 2020. 

- Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In _Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition_ , pp. 770–778, 2016. 

- Jie Hu, Li Shen, and Gang Sun. Squeeze-and-excitation networks. In _Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition_ , pp. 7132–7141, 2018. 

- Sergey Ioffe and Christian Szegedy. Batch normalization: Accelerating deep network training by reducing internal covariate shift. In _Proceedings of the 32nd International Conference on Machine Learning - Volume 37_ , pp. 448–456, 2015. 

- Alex Kendall, Yarin Gal, and Roberto Cipolla. Multi-task learning using uncertainty to weigh losses for scene geometry and semantics. In _Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition_ , pp. 7482–7491, 2018. 

- Shikun Liu, Edward Johns, and Andrew J Davison. End-to-end multi-task learning with attention. In _Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition_ , pp. 1871– 1880, 2019. 

10 

Published as a conference paper at ICLR 2021 

- Ziwei Liu, Ping Luo, Xiaogang Wang, and Xiaoou Tang. Deep learning face attributes in the wild. In _Proceedings of the IEEE International Conference on Computer Vision_ , pp. 3730–3738, 2015. 

- Jiasen Lu, Vedanuj Goswami, Marcus Rohrbach, Devi Parikh, and Stefan Lee. 12-in-1: Multi-task vision and language representation learning. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pp. 10437–10446, 2020. 

- Arun Mallya, Dillon Davis, and Svetlana Lazebnik. Piggyback: Adapting a single network to multiple tasks by learning to mask weights. In _Proceedings of the European Conference on Computer Vision (ECCV)_ , pp. 67–82, 2018. 

- Kevis-Kokitsi Maninis, Ilija Radosavovic, and Iasonas Kokkinos. Attentive single-tasking of multiple tasks. In _Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition_ , pp. 1851–1860, 2019. 

- Ishan Misra, Abhinav Shrivastava, Abhinav Gupta, and Martial Hebert. Cross-stitch networks for multi-task learning. In _Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition_ , pp. 3994–4003, 2016. 

- Chao Peng, Tete Xiao, Zeming Li, Yuning Jiang, Xiangyu Zhang, Kai Jia, Gang Yu, and Jian Sun. Megdet: A large mini-batch object detector. In _Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition_ , pp. 6181–6189, 2018. 

- Sylvestre-Alvise Rebuffi, Hakan Bilen, and Andrea Vedaldi. Learning multiple visual domains with residual adapters. In _Advances in Neural Information Processing Systems_ , pp. 506–516, 2017. 

- Sebastian Ruder. An overview of multi-task learning in deep neural networks. _arXiv preprint arXiv:1706.05098_ , 2017. 

- Sebastian Ruder, Joachim Bingel, Isabelle Augenstein, and Anders Søgaard. Latent multi-task architecture learning. In _Proceedings of the AAAI Conference on Artificial Intelligence_ , volume 33, pp. 4822–4829, 2019. 

- Ozan Sener and Vladlen Koltun. Multi-task learning as multi-objective optimization. In _Advances in Neural Information Processing Systems_ , pp. 527–538, 2018. 

- Nathan Silberman, Derek Hoiem, Pushmeet Kohli, and Rob Fergus. Indoor segmentation and support inference from rgbd images. In _European Conference on Computer Vision_ , pp. 746–760. Springer, 2012. 

- Trevor Standley, Amir R Zamir, Dawn Chen, Leonidas Guibas, Jitendra Malik, and Silvio Savarese. Which tasks should be learned together in multi-task learning? In _International Conference on Machine Learning_ , 2020. 

- Gjorgji Strezoski, Nanne van Noord, and Marcel Worring. Many task learning with task routing. In _Proceedings of the IEEE International Conference on Computer Vision_ , pp. 1375–1384, 2019. 

- Tianhe Yu, Saurabh Kumar, Abhishek Gupta, Sergey Levine, Karol Hausman, and Chelsea Finn. Gradient surgery for multi-task learning. _arXiv preprint arXiv:2001.06782_ , 2020. 

- Amir R Zamir, Alexander Sax, William Shen, Leonidas J Guibas, Jitendra Malik, and Silvio Savarese. Taskonomy: Disentangling task transfer learning. In _Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition_ , pp. 3712–3722, 2018. 

- Amir R Zamir, Alexander Sax, Nikhil Cheerla, Rohan Suri, Zhangjie Cao, Jitendra Malik, and Leonidas J Guibas. Robust learning through cross-task consistency. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pp. 11197–11206, 2020. 

- Yu Zhang and Qiang Yang. A survey on multi-task learning. _arXiv preprint arXiv:1707.08114_ , 2017. 

- Hengshuang Zhao, Jianping Shi, Xiaojuan Qi, Xiaogang Wang, and Jiaya Jia. Pyramid scene parsing network. In _Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition_ , pp. 2881–2890, 2017. 

11 

Published as a conference paper at ICLR 2021 

Barret Zoph and Quoc V. Le. Neural architecture search with reinforcement learning. In _Proceedings of the International Conference on Learning Representations_ , 2017. 

## A RELATED WORK OF NETWORK STRUCTURE 

Cross-stitch Networks (Misra et al., 2016) learn coefficients to linearly combine activations from multiple tasks to construct better task-specific representations. To break the limitation of channelwise cross-task feature fusion only, NDDR-CNN (Gao et al., 2019) proposes the layer-wise crosschannel feature aggregation as 1 _×_ 1 convolutions on the concatenated feature maps from multiple tasks. More generally, MTL-NAS (Gao et al., 2020) introduces cross-layer connections among tasks to fully exploit the feature sharing from both low and high layers, extending the idea in Sluice Networks (Ruder et al., 2019) by leveraging neural architecture search (Zoph & Le, 2017). The parameters of these methods increase linearly with the number of tasks. To improve the model compactness, Residual Adapters (Rebuffi et al., 2017) introduce a small amount of task-specific parameters for each layer and convolve them with the task-agnostic representations to form the taskrelated ones. MTAN (Liu et al., 2019) generates data-dependent attention tensors by task-specific parameters to attend to the task-shared features. Single-tasking (Maninis et al., 2019) instead applies squeeze-and-excitation (Hu et al., 2018) module to generate attentive vectors for each task. In Task Routing (Strezoski et al., 2019), the attentive vectors are randomly sampled before training and are fixed for each image. Piggyback (Mallya et al., 2018) opts to mask parameter weights in place of activation maps, dealing with task-sharing from another point-of-view. The above methods can share parameters among tasks to a large extent, however, they are not memory-efficient because each task still needs to compute all of its own intermediate feature maps, which also leads to inferior inference speed compared with loss weighting methods. 

## B DETAILED DERIVATION 

### B.1 GRADIENT BALANCE: IMTL-G 

Here we give the detailed derivation of the closed-form solution of our IMTL-G, we also demonstrate the scale-invariance property of our IMTL-G, which is invariant to the scale changes of losses. 

**Solution.** As we want to achieve: 



where **_u_** _t_ = **_g_** _t/ ∥_ **_g_** _t∥_ , recall that we have **_g_** =<sup>�</sup> _t_<sup>_αt_</sup><sup>**_g_**</sup><sup>_t_and �</sup> _t_<sup>_αt_= 1, if we set</sup><sup>**_α_**= [</sup><sup>_α_2</sup><sup>_, · · ·, αT_]</sup> and **_G_**<sup>_⊤_</sup> = � **_g_** 2<sup>_⊤, · · ·,_</sup><sup>**_g_**</sup> _T_<sup>_⊤_</sup> �, then _α_ 1 = 1 _−_ **1** **_α_**<sup>_⊤_</sup> and Eq. (9) can be expanded as: 



where **_U_**<sup>_⊤_</sup> = � **_u_**<sup>_⊤_</sup> 1<sup>_−_</sup><sup>**_u_**</sup><sup>_⊤_</sup> 2<sup>_, · · ·,_</sup><sup>**_u_**</sup><sup>_⊤_</sup> 1<sup>_−_</sup><sup>**_u_**</sup><sup>_⊤_</sup> _T_ �, **1** and **0** indicate the all-one and all-zero row vector, respectively. Eq. (10) can be solved by: ��1 _−_ **1** **_α_**<sup>_⊤_�</sup> **_g_** 1 + **_αG_** � **_U_**<sup>_⊤_</sup> = **0** _⇔_ **_α_** � **1**<sup>_⊤_</sup> **_g_** 1 _−_ **_G_** � **_U_**<sup>_⊤_</sup> = **_g_** 1 **_U_**<sup>_⊤_</sup> _._ (11) Assume **_D_**<sup>_⊤_</sup> = **_g_** 1<sup>_⊤_</sup><sup>**1**</sup><sup>_−_</sup><sup>**_G_**</sup><sup>_⊤_=</sup> � **_g_** 1<sup>_⊤−_</sup><sup>**_g_**</sup> 2<sup>_⊤, · · ·,_</sup><sup>**_g_**</sup> 1<sup>_⊤−_</sup><sup>**_g_**</sup> _T_<sup>_⊤_</sup> �, then we reach: 



**Property.** We can also prove the aggregated gradient **_g_** =<sup>�</sup> _t_<sup>_αt_</sup><sup>**_g_**</sup><sup>_t_with</sup><sup>_{αt}_given in Eq.(12) is</sup> invariant to the scale changes of losses _{Lt}_ (or gradients _{_ **_g_** _t_ = _∇_ **_θ_** _Lt}_ ), as the following theorem. **Theorem 2.** _Given_ **_g_** =<sup>�</sup> _t_<sup>_αt_</sup><sup>**_g_**</sup><sup>_t,_�</sup> _t_<sup>_αt_=1</sup><sup>_satisfying_</sup><sup>**_gu_**</sup> _t_<sup>_⊤_</sup> = _C, when {Lt} are scaled by {kt >_ 0 _} (equivalently, {_ **_g_** _t} are scaled by {kt}), if_ **_g_**<sup>_′_</sup> =<sup><u>�</u></sup> _t_<sup>_α_</sup> _t_<sup>_′_(</sup><sup>_kt_</sup><sup>**_g_**</sup><sup>_t_)</sup><sup>_,_�</sup> _t_<sup>_α_</sup> _t_<sup>_′_= 1</sup><sup>_satisfies_</sup><sup>**_g_**</sup><sup>_′_</sup><sup>**_u_**</sup><sup>_⊤_</sup> _t_<sup>=</sup> _C_<sup>_′_</sup> _, then_ **_g_**<sup>_′_</sup> = _λ_ **_g_** _. In the above we have_ **_u_** _t_ = **_<u>g</u>_** _<u>t</u> kt_ **_<u>g</u>_** _<u>t</u> ∥_ **_g_** _t∥_<sup>=</sup> _∥kt_ **_g_** _t∥_<sup>_, λ, Cand C′are constants._</sup> 

12 

Published as a conference paper at ICLR 2021 



Semantic Segmentation Instance Segmentation Disparity Estimation<br>
Figure 4: Loss scales of IMTL-G for different tasks when training on the Cityscapes dataset. 

_Proof._ As we have: 



by constructing: 



we have: 



From Eq. (12) we know that _{αt}_ has a unique solution, and thus **_g_**<sup>_′_</sup> satisfying IMTL-G is unique, so it must be the one given by Eq. (14), then we can prove that **_g_**<sup>_′_</sup> and **_g_** are linearly correlated. 

### B.2 LOSS BALANCE: IMTL-L 

With the ordinary differential equation, we can derive that the form of the scale function � _h_ ( _s_ ) d _s_ in our IMTL-L must be exponential function. As we have: 



If we set _y_ = � _h_ ( _s_ ) d _s_ , then: 



By taking the antiderivative: 



Then we have: 



## C DETAILED DISCUSSION 

C.1 CONDITIONAL EQUIVALENCE OF IMTL-G AND GRADNORM 

First we introduce the following lemma. **Lemma 3.** _If_ **_u_** _t_ **_u_**<sup>_⊤_</sup> _τ_<sup>=</sup><sup>_C_1</sup><sup>_,∀t̸_=</sup><sup>_τ, then the solution {αt} of IMTL-G satisfies {αt>_0</sup><sup>_}._</sup> 

13 

Published as a conference paper at ICLR 2021 



then we have<sup>�</sup> _t_<sup>_αt_= 1 and:</sup> 



From Eq. (12) we know the solution _{αt}_ of IMTL-G is unique, so it must be the one given by Eq. (20) where _{αt >_ 0 _}_ , so the lemma is proved. 

Then we prove Theorem 1 which states that IMTL-G leads to the same solution as GradNorm when the angle between any pair of gradients _{_ **_g_** _t}_ is identical: **_u_** _t_ **_u_**<sup>_⊤_</sup> _τ_<sup>=</sup><sup>_C_1</sup><sup>_,∀t̸_=</sup><sup>_τ_.</sup> 

_Proof._ ( _⇒_ Necessity) Given constant projections in IMTL-G, we have: 



Recall that **_u_** _t_ = **_g_** _t/ ∥_ **_g_** _t∥_ and **_u_** _t_ **_u_**<sup>_⊤_</sup> _τ_<sup>=</sup><sup>_C_1</sup><sup>_,∀t̸_=</sup><sup>_τ_.From Lemma 3 we know that</sup><sup>_{αt}_given by</sup> IMTL-G must satisfy _{αt >_ 0 _}_ . If we assume _nt_ = _∥αt_ **_g_** _t∥_ , then we know _αt_ **_g_** _t_ = _nt_ **_u_** _t_ and: 



Now we obtain: 



As _C_ 1 _<_ 1, we can then prove _nt_ = _C_ 3 _, ∀t_ . It implies the norm of the scaled gradient is constant, which is requested by GradNorm (Chen et al., 2018). Moreover, we can obtain the relationship among constants from Eq. (24): 



( _⇐_ Sufficiency) In GradNorm, _{αt}_ are always chosen to satisfy _{αt >_ 0 _}_ , so if we assume _nt_ = _∥αt_ **_g_** _t∥_ , then given the constant norm of the scaled gradient in GradNorm, we have: 



where **_u_** _t_ = **_g_** _t/ ∥_ **_g_** _t∥_ . As we have **_g_** =<sup>�</sup> _t_<sup>_αt_</sup><sup>**_g_**</sup><sup>_t_and</sup><sup>**_u_**</sup><sup>_t_</sup><sup>**_u_**</sup> _τ_<sup>_⊤_=</sup><sup>_C_1</sup><sup>_,∀t̸_=</sup><sup>_τ_, then we obtain:</sup> 



It means the projections of **_g_** onto _{_ **_g_** _t}_ are constant, which is requested by our IMTL-G. 

**Corollary 4.** _In GradNorm, if the solution {αt} satisfies_<sup>�</sup> _t_<sup>_αt_=1</sup><sup>_, then its constants are given_</sup> _by C_ 3 = 1 _/_<sup>�</sup> _t_<sup>_∥_</sup><sup>**_g_**</sup><sup>_t∥−_1</sup><sup>_and C_2=[(</sup><sup>_T−_1)</sup><sup>_C_1 + 1]</sup><sup>_/_�</sup> _t_<sup>_∥_</sup><sup>**_g_**</sup><sup>_t∥−_1</sup><sup>_, and its scaling factors are given_</sup> _by_ � _αt_ = _∥_ **_g_** _t∥_<sup>_−_1</sup> _/_<sup>�</sup> _τ_<sup>_∥_</sup><sup>**_g_**</sup><sup>_τ∥−_1�</sup> _._ 

_Proof._ By using _αt_ = _C_ 3 _/ ∥_ **_g_** _t∥_ from Eq. (26), we have<sup>�</sup> _t_<sup>_C_3</sup><sup>_/ ∥_</sup><sup>**_g_**</sup><sup>_t∥_=1,then</sup><sup>_C_3=</sup> 1 _/_<sup>�</sup> _t_<sup>_∥_</sup><sup>**_g_**</sup><sup>_t∥−_1,andalsowehave</sup><sup>_αt_=</sup><sup>_∥_</sup><sup>**_g_**</sup><sup>_t∥−_1</sup><sup>_/_�</sup> _τ_<sup>_∥_</sup><sup>**_g_**</sup><sup>_τ∥−_1.Astherelationshipof</sup><sup>_C_2and</sup><sup>_C_3</sup> from Eq. (27) is given by _C_ 3 [( _T −_ 1) _C_ 1 + 1] = _C_ 2, so _C_ 2 = [( _T −_ 1) _C_ 1 + 1] _/_<sup>�</sup> _t_<sup>_∥_</sup><sup>**_g_**</sup><sup>_t∥−_1.</sup> 

14 

Published as a conference paper at ICLR 2021 

### C.2 CLOSED-FORM SOLUTION OF MGDA 

In our relaxed MGDA (Sener & Koltun, 2018) without _{αt_ ⩾ 0 _}_ , finding **_g_** =<sup>�</sup> _t_<sup>_αt_</sup><sup>**_g_**</sup><sup>_t_with</sup> � _t_<sup>_αt_=1suchthat</sup><sup>**_g_**hasminimumnormisequivalenttofindthenormalvectorofthehyper-</sup> plane composed by _{_ **_g_** _t}_ . So we let **_g_** to be perpendicular to all of _{_ **_g_** 1 _−_ **_g_** _t}_ on the hyper-plane: 





where **_D_**<sup>_⊤_</sup> = � **_g_** 1<sup>_⊤−_</sup><sup>**_g_**</sup> 2<sup>_⊤, · · ·,_</sup><sup>**_g_**</sup> 1<sup>_⊤−_</sup><sup>**_g_**</sup> _T_<sup>_⊤_</sup> �, **1** and **0** indicates the all-one and all-zero row vector. Eq. (29) can be represented as: 





**Bias of MGDA.** In the main text we state that MGDA focuses on tasks with small gradient magnitudes, where we relaxed MGDA by not constraining _{αt_ ⩾ 0 _}_ . However, even with these constraints, the problem still exists. For example in the context of two tasks, assume _∥_ **_g_** 1 _∥ < ∥_ **_g_** 2 _∥_ , if the minimum-norm point of **_g_** satisfying **_g_** = _α_ **_g_** 1 +(1 _− α_ ) **_g_** 2 is outside the convex hull composed by _{_ **_g_** 1 _,_ **_g_** 2 _}_ , or equivalently _α >_ 1, MGDA clamps _α_ to _α_ = 1 and the optimal **_g_**<sup>_⋆_</sup> = **_g_** 1. Then the projections of **_g_**<sup>_⋆_</sup> onto **_g_** 1 and **_g_** 2 will be _∥_ **_g_** 1 _∥_ and **_g_** 1 **_u_**<sup>_⊤_</sup> 2<sup>(</sup><sup>**_u_**2=</sup><sup>**_g_**2</sup><sup>_/∥_</sup><sup>**_g_**2</sup><sup>_∥_),respectively.As</sup> _∥_ **_g_** 1 _∥ >_ �� **_g_** 1 **_u_** 2 _⊤_ ��, so MGDA still focuses on tasks with smaller gradient magnitudes. 

### C.3 ANALYSIS OF PCGRAD 

PCGrad (Yu et al., 2020) mitigates the gradient conflicts by projecting the gradient of one task to the orthogonal direction of the others, and the aggregated gradient can be written as: 



with **_u_** _t_ = **_g_** _t/ ∥_ **_g_** _t∥_ and the coefficients: 



where [ _·_ ]+ means the ReLU operator. Note that the tasks have been shuffled before calculating the aggregated gradient **_g_** to achieve expected symmetry with respect to the task order. Eq. (31) can be represented more compactly in the matrix form: 



where **_I_** _T_ is the identity matrix, **_C_** = _{Ctτ }_ is the coefficient matrix whose entries are given in Eq. (32) and **_N_** = diag (1 _/ ∥_ **_g_** 1 _∥ , · · · ,_ 1 _/ ∥_ **_g_** _T ∥_ ) is the diagonal normalization matrix. In Eq. (33) we use **_G_** and **_α_** to denote the raw gradients and scaling factors of all tasks. We find that PCGrad can also be regarded as loss weighting, with the loss weights given by **_α_** = **1** ( **_I_** _T_ + **_CN_** ). However, it still may break the balance among tasks. For example with two tasks, assume the angle between 

15 

Published as a conference paper at ICLR 2021 



projections onto the two raw gradients are _∥_ **_g_** 1 _∥_ sin<sup>2</sup> _φ_ and _∥_ **_g_** 2 _∥_ sin<sup>2</sup> _φ_ ; 2) when 0 _< φ < π/_ 2, then **_C_** = **0** and the projections are _∥_ **_g_** 1 _∥_ + _∥_ **_g_** 2 _∥_ cos _φ_ and _∥_ **_g_** 2 _∥_ + _∥_ **_g_** 1 _∥_ cos _φ_ . In both cases, the projections are equal if and only if _∥_ **_g_** 1 _∥_ = _∥_ **_g_** 2 _∥_ . Otherwise, the task with larger gradient magnitude will be trained more sufficiently, which may encounter the same problem as uniform scaling that na¨ıvely adds all the losses despite that the loss scales are highly different. 

### C.4 _L_ 2 LOSS IN UNCERTAINTY WEIGHTING 

For regression, uncertainty weighting (Kendall et al., 2018) regards the _L_ 2 loss as likelihood estimation on the sample target which follows the Gaussian distribution: 



where **_x_** is the data sample, _y_ is the ground-truth label, _f_ denotes the prediction model and _σ_ is the standard deviation of Gaussian distribution. By setting _s_ = _−_ log _σ_<sup>2</sup> , the scaled _L_ 2 loss is _L_ =<sup><u>1</u></sup> 2<sup>(</sup><sup>_esL_reg</sup><sup>_−s_), which has a similar form as the scaled</sup><sup>_L_1loss except the front factor 1</sup><sup>_/_2.So</sup> uncertainty weighting has difficulty in reaching a unified form for all kinds of losses, which is less general than our IMTL-L. 

### C.5 GRADIENT OF GEOMETRIC MEAN 

GLS (Chennupati et al., 2019) computes the loss as the geometric mean, its gradient with respect to model parameters are: 



where _L_ is the geometric mean loss and _T_ is the task number. It is equivalent to weigh the taskspecific loss with its reciprocal value, except that there exists another term _L/T_ in the front where <u>1</u> _L_ = (<sup>�</sup> _t_<sup>_Lt_)</sup> _T_ , so GLS is sensitive to the loss scale changes of _{Lt}_ and not scale-invariant. 

## D IMPLEMENTATION DETAILS 

To solely compare the loss weighting methods, we fix the network structure and choose ResNet50 (He et al., 2016) with dilation (Chen et al., 2017) and synchronized (Peng et al., 2018) batch normalization (Ioffe & Szegedy, 2015) as the shared backbone and PSPNet (Zhao et al., 2017) as the task-specific head, and the backbone model weights are pretrained on ImageNet (Deng et al., 2009). Following the common practice of semantic segmentation, in training we adopt augmentations as random resize (between 0.5 to 2), random rotate (between -10 to 10 degrees), Gaussian blur (with a radius of 5) and random horizontal flip. Besides, we apply strided cropping and horizontal flipping as testing augmentations. The predicted results in the overlapped region of different crops are averaged to obtain the aggregated prediction of the whole image. Only pixels with ground truth labels are included in loss and metric computation, while others are ignored. Semantic segmentation, instance segmentation, surface normal estimation and disparity/depth estimation are considered. As for the losses/metrics, semantic segmentation uses cross-entropy/mIoU, surface normal estimation adopts (1 _−_ cos)/cosine similarity and both instance segmentation and disparity/depth estimation use _L_ 1 loss. We use polynomial learning rate with a power of 0.9, SGD with a momentum of 0.9 and weight decay of 10<sup>_−_4</sup> as the optimizer, with the model trained for 200 epochs. After passing through the shared backbone where strided convolutions exist, the feature maps have 1 _/_ 8 size as that of the 

16 

Published as a conference paper at ICLR 2021 



semantic segmentation foreground extraction<br>offset regression centroid computation pixel assignment<br>disparity estimation<br>foreground<br>masking<br>
Figure 5: Pipeline used in the Cityscapes visual understanding experiment. The centroids are computed from the offset regression results. Each pixel is assigned to its nearest candidate centroid. 

input image. Then the results predicted by PSPNet (Zhao et al., 2017) heads are up-sampled to the original image size for loss and metric computation. 

For the **Cityscapes** dataset, the batch size is 32 (2 _×_ 16 GPUs) with the initial learning rate 0.02. We train on the 2975 training images and validate on the 500 validation images (1024 _×_ 2048 full resolution) where ground truth labels are provided. Three tasks are considered, namely semantic segmentation, instance segmentation and disparity/depth estimation. Training and testing are done on 713 _×_ 713 crops. Semantic segmentation is to differentiate among the commonly used 19 classes. Instance segmentation is taken as offset regression, where each pixel **_p_** _i_ = ( _xi, yi_ ) approximates the relative offset **_o_** _i_ = (d _xi,_ d _yi_ ) with respect to the centroid **_c_** id( **_p_** _i_ ) of its belonging instance id ( **_p_** _i_ ). To conduct inference, we abandon the time-consuming and complicated clustering methods adopted by the previous method (Kendall et al., 2018). Instead, we directly use the offset vectors _{_ **_o_** _i}_ predicted by the model to find the centroids of instances. By definition, the norm of a centroid’s offset vector should be 0, so we can transform the offset vector norm _∥_ **_o_** _i∥_ to the probability _qi_ of being a centroid with the exponential function _qi_ = _e_<sup>_−∥_</sup><sup>**_o_**</sup><sup>_i∥_</sup> . Next a 7 _×_ 7 edge filter is applied on the centroid probability map to filter out the spurious centroids on object edges resulting from the regression target ambiguity. The locations with centroid probability _qi <_ 0 _._ 1 are also manually suppressed. Then 7 _×_ 7 max-pooling on the filtered probability map is used to produce candidate centroids and filter out duplicate ones. With the predicted centroids _{_ **_c_** _i}_ , we can then assign each pixel **_p_** _i_ to its belonging instance id ( **_p_** _i_ ) by the distance between its approximated centroids **_p_** _i_ + **_o_** _i_ and the candidate centroids _{_ **_c_** _i}_ : id ( **_p_** _i_ ) = arg min _j ∥_ **_p_** _i_ + **_o_** _i −_ **_c_** _j∥_ . Depth is measured in pixels by the disparity between the left and right images. Fig. 5 shows the whole process. Note that we need to carefully deal with label transformation during data augmentation. For example, disparity ground truth needs to be up-scaled by _s_ times if the image is up-sampled by _s_ times. Also, the predicted offset vectors of the flipped input should be mirrored to comply with the normal one. 

On the **NYUv2** dataset, the batch size is 48 (6 _×_ 8 GPUs) with the initial learning rate 0.03. We use the 795 training images for training and the 654 validation images for testing with 480 _×_ 640 full resolution. 401 _×_ 401 crops are used for training and testing. 13 coarse-grain classes are considered in semantic segmentation. The surface normal is represented by the unit normal vector of the corresponding surface. When doing data augmentation, surface normal ground truth **_n_** = ( _x, y, z_ ) should be processed accordingly. If we resize the image by _s_ times, the _z_ coordinate of the normal vector should be scaled by _s_ and renormalized: **_n_**<sup>_′_</sup> = ( _x, y, sz_ ) _/ ∥_ ( _x, y, sz_ ) _∥_ . If the image is rotated by the rotation matrix **_R_** , the normal vector should also be in-plane rotated ( _x_<sup>_′_</sup> _, y_<sup>_′_</sup> ) = ( _x, y_ ) **_R_**<sup>_⊤_</sup> with _z_ unchanged. Moreover, the left-right flip should be applied on the normal vector **_n_**<sup>_′_</sup> = ( _−x, y, z_ ) when mirroring the image horizontally. During testing, the normal vectors in the overlapped region of crops are averaged and renormalized to produce the aggregated results. Depth is the absolute distance to the camera and measured by meters, which is inverse-proportional to the disparity measurement adopted by Cityscapes. So the depth in meters needs to be scaled by 1 _/s_ when the image is scaled by _s_ times, which is the reciprocal of disparity transformation. 

**CelebA** contains 202,599 face images from 10,177 identities, where each image has 40 binary attribute annotations. We train on the 162,770 training images and test on the 19,867 validation 

17 

Published as a conference paper at ICLR 2021 

images. Most of the implementation details are the same as those on the Cityscapes dataset, except that: 1) we employ the ResNet-18 as the backbone and linear classifiers as the task-specific heads, so totally 40 heads are attached on the backbone ; 2) the binary-cross entropy is used as the classification loss for each attribute; 3) the batch size is 256 (32 _×_ 8 GPUs) and the model is trained from scratch for 100 epochs; 4) the input image has been aligned with the annotated 5 landmarks and cropped to 218 _×_ 178. 

## E QUALITATIVE RESULTS 

18 

Published as a conference paper at ICLR 2021 



Uncertainty<br>GradNorm<br>IMTL (Ours)<br>Ground Truth<br>Uncertainty<br>GradNorm<br>IMTL (Ours)<br>Ground Truth<br>Uncertainty<br>GradNorm<br>IMTL (Ours)<br>Ground Truth<br>
Figure 6: Qualitative comparisons between our IMTL and previous methods on Cityscapes. 19 

Published as a conference paper at ICLR 2021 























































Figure 7: Qualitative results of our IMTL on Cityscapes. Semantic segmentation, instance segmentation and disparity estimation predictions are produced by a single network. The task-shared backbone is ResNet-50 and the task-specific heads are PSPNet. The image resolution is 1024 _×_ 2048. 

20 

Published as a conference paper at ICLR 2021 





































Figure 8: Qualitative results of our IMTL on NYUv2. Semantic segmentation, surface normal estimation and depth estimation predictions are produced by a single network. The task-shared backbone is ResNet-50 and the task-specific heads are PSPNet. The image resolution is 480 _×_ 640. 

21 

