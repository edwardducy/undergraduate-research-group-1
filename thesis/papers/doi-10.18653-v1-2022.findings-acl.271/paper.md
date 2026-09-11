# **MetaWeighting: Learning to Weight Tasks in Multi-Task Learning** 

**Yuren Mao**<sup>1</sup> _,_ **Zekai Wang**<sup>2</sup> _,_ **Weiwei Liu**<sup>2</sup><sup>_∗_</sup> _,_ **Xuemin Lin**<sup>1</sup> _,_ **Pengtao Xie**<sup>3</sup> 

1School of Computer Science and Engineering, University of New South Wales 2School of Computer Science, Wuhan University 

- 3Department of Electrical and Computer Engineering, University of California San Diego yuren.mao@unsw.edu.au, {wzekai99,liuweiwei863}@gmail.com lxue@cse.unsw.edu.au, pengtaoxie2008@gmail.com 

## **Abstract** 

Task weighting, which assigns weights on the including tasks during training, significantly matters the performance of Multi-task Learning (MTL); thus, recently, there has been an explosive interest in it. However, existing task weighting methods assign weights only based on the training loss, while ignoring the gap between the training loss and generalization loss. It degenerates MTL’s performance. To address this issue, the present paper proposes a novel task weighting algorithm, which automatically weights the tasks via a learning-to-learn paradigm, referred to as MetaWeighting. Extensive experiments are conducted to validate the superiority of our proposed method in multitask text classification. 

## **1 Introduction** 

Multi-task Learning (MTL) simultaneously learns multiple related tasks and aims to achieve better performance than learning each task independently (Caruana, 1993; Baxter, 2000). It has achieved great success in various applications; especially, in the text classification context, MTL can significantly outperform single task learning (Liu et al., 2017; Mao et al., 2021). 

In MTL, it is common for the including tasks to be competing. If we cannot properly balance these tasks, some tasks might dominate the training process and hurt the performance of other tasks, a phenomenon known as task imbalance. To address the task imbalance, the most widely used method is task weighting, which adaptively assigns weights on the tasks during training to balance their impacts. Various task weighting methods have been proposed and can be used in multi-task text classification, such as (Kendall et al., 2018; Sener and Koltun, 2018; Chen et al., 2018). 

However, existing task weighting methods compute the task weights only based on training losses or corresponding gradients. They ignore the gap 



Figure 1: Illustration of the gap between training loss and generalization loss in the training process of a fourtask topic classification experiment (500<sup>_th_</sup> , 1000<sup>_th_</sup> , 1500<sup>_th_</sup> epochs respectively). 

between the training loss and generalization loss. To illustrate this gap, we report observations of our four-task topic classification experiment in Figure 1. The detailed experimental settings are introduced in the experiment section. Figure 1 demonstrates that the training losses and generalization losses (estimated by the test losses) have different magnitudes; moreover, they have different patterns, such as a task might have the largest training loss but the lowest generalization loss among the tasks. 

This gap causes a mismatch between the task weights and tasks’ generalization performance, which reduces effectiveness of the task weighting. To tackle this issue, this paper proposes a novel task weighting method based on a bi-level optimization problem, which aims to find task weights that explicitly optimize the generalization performance. Our proposed method computes task weights by solving this bi-level optimization problem and performs in a learning-to-learn manner; thus, dubbed MetaWeighting. MetaWeighting can improve the performance of multi-task text classification. 

To verify our theoretical analysis and validate the superiority of MetaWeighting, we conduct experiments on two classical text classification problems: sentiment analysis (on reviews) and topic classification (on news). The results demonstrate that MetaWeighting outperforms several state-of-the-art multi-task text classification methods. 

3436 

_Findings of the Association for Computational Linguistics: ACL 2022_ , pages 3436 - 3448 May 22-27, 2022 _⃝_ c 2022 Association for Computational Linguistics 

## **2 Related Works** 

Existing task weighting strategies can be divided into two categories: _weight adaptation methods_ and _Pareto Optimization (PO)-based methods_ . The _weight adaptation methods_ adaptively adjust the tasks’ weights during training based on pre-defined heuristic, such as uncertainty (Kendall et al., 2018), task difficulty prioritization (Guo et al., 2018), gradient normalization (Chen et al., 2018), weight average (Liu et al., 2019) and task variance regularization (Mao et al., 2021). These methods only use training losses or their gradients to compute task weights while ignores the gap between the training loss and generalization loss. 

Besides, the _PO-based methods_ formulate MTL as a multi-objective optimization problem and aim to find an arbitrary Pareto stationary solution (Sener and Koltun, 2018; Lin et al., 2019; Mahapatra and Rajan, 2020; Lin et al., 2020; Ma et al., 2020; Mao et al., 2020). However, in these methods, the learning objectives only involve training losses; thus, they can only achieve Pareto stationary points w.r.t training losses. They also ignore the gap between the training loss and generalization loss. Moreover, (Lin et al., 2019) proposes that the _PO-based methods_ can be also regarded as weight adaptation methods for they optimize the weighted sum of training losses as well. 

Overlooking the gap between the training loss and generalization loss would degenerate the performance of MTL. This paper proposes a novel meta learning-based task weighting method to solve this issue. There are some works adopt meta learning-based weighting methods in multilingual learning, e.g., (Wang et al., 2020) and (Tarunesh et al., 2021). However, these works cannot solve multi-objective optimization problems. By contrast, this paper proposes a novel method which can solve multi-objective optimization problems. 

## **3 Preliminaries** 

Consider a multi-task learning problem with _T_ tasks over an input space _X_ and a collection of task spaces _{Yt}_<sup>_T_</sup> _t_ =1<sup>.Foreachtask,wehavea</sup> set of i.i.d. training samples _Dt_ = _{x_<sup>_i_</sup> _t_<sup>_, y_</sup> _t_<sup>_i}n_</sup> _i_ =1<sup>.</sup> The training samples are sampled from an identical distribution _Pt_ . Based on the training sets _{Dt}_<sup>_T_</sup> _t_ =1<sup>, we learn an MTL model from a param-</sup> eterized hypothesis class _H_ , which shares some parameters across tasks. Let _θs_ represent the parameters shared between tasks (task-sharing param- 

eters), while _θt_ represent the task-specific parameters. _h_ ( _·, θs, θ_ 1 _, ..., θT_ ) : _X →{Yt}_<sup>_T_</sup> _t_ =1<sup>_∈H_</sup> denotes an MTL model that learns from _H_ , while _h_ ( _·, θs, θt_ ) : _X →Yt_ denotes the task-specific module in the MTL model. 

The loss function is represented by _l_ ( _·, ·_ ) : _Y_<sup>_t_</sup> _× Y_<sup>_t_</sup> _→_ [0 _,_ 1]<sup>_T_</sup> . For each task, the generalization loss is _Lt_ ( _θ_ ) = E( _xt,yt_ ) _∼Ptl_ ( _h_ ( _xt, θs, θt_ ) _, yt_ ), and the training loss is defined as _L_<sup>_tr_</sup> _t_<sup>(</sup><sup>_θ, Dt_)=</sup> _|D_ <u>1</u> _t|_ �( _xt,yt_ ) _∈Dt_<sup>_l_(</sup><sup>_h_(</sup><sup>_xt, θs, θt_)</sup><sup>_, yt_).Inthispaper,</sup> each training set _Dt_ is randomly divided into two subsets: support set _Dt_<sup>_s_and query set</sup><sup>_D_</sup> _t_<sup>_q_.Corre-</sup> spondingly; moreover, the support loss is defined as _L_<sup>_s_</sup> _t_<sup>(</sup><sup>_θ, D_</sup> _t_<sup>_s_)=</sup> _|D_ <u>1</u> _t_<sup>_<u>s|</u>_</sup> �( _xt,yt_ ) _∈Dt_<sup>_sl_(</sup><sup>_h_(</sup><sup>_xt, θs, θt_)</sup><sup>_, yt_),</sup> and the query loss is defined as _L_<sup>_q_</sup> _t_<sup>(</sup><sup>_θ, D_</sup> _t_<sup>_q_)=</sup> _|D_ <u>1</u> _t_<sup>_~~q~~|_</sup> �( _xt,yt_ ) _∈Dt_<sup>_ql_(</sup><sup>_h_(</sup><sup>_xt, θs, θt_)</sup><sup>_, yt_).</sup> 

### **3.1 Hypergradient Descent** 

Hypergradient Descent (HD) (Almeida et al., 1998; Baydin et al., 2018) provides an efficient way to apply gradient descent on hyper-parameters. Here, we take learning rate’s HD as an example to introduce the basic form of HD. Given an objective function _f_ ( _θ_ ) and previous parameters _θ_<sup>_k−_1</sup> , gradient descent-based learning typically evaluates the gradient _∇f_ ( _θ_<sup>_k−_1</sup> ) and moves against it to arrive at updated parameters 



where _η_ is the learning rate. HD derives an update rule for the learning rate _η_ itself. Based on Eq. (1) and the chain rule, we have 



with which we construct a update rule for _η_ : 



introducing _β_ as the hypergradient step size. In this paper, we extend HD to a bi-level multi-objective optimization problem. 

### **3.2 Common Descent Direction for Multiple Objectives** 

When using gradient descent to jointly optimize multiple optimization objectives, we need to find a descent direction common to all the objectives. Based on the descent direction for each objective, (Désidéri, 2012) proposes a way to obtain the common descent direction, as in Theorem 1. This paper 

3437 

proposes a method to simultaneously optimize the tasks’ generalization loss based on Theorem 1. 

**Theorem 1** ((Désidéri, 2012)) **.** _Let A be a Hilbert space of finite or infinite dimension N . Let fi_ ( _z_ ) (1 _≤ i ≤ n ≤ N_ ) _be n smooth functions of the vector z ∈A. and z_<sup>0</sup> _a particular admissible designpoint, at which the gradient-vectors are denoted gi_ = _∇fi_ ( _z_<sup>0</sup> ) _, and_ 



_Let a_<sup>_∗_</sup> = arg min _a∈U_ ¯ _∥ a ∥, where U consists of the convex hull and closure of U. Then, if a_<sup>_∗̸_</sup> = 0 _, a_<sup>_∗_</sup> _is a descent direction common to all the objectives._ 

## **4 MetaWeighting for MTL** 

In this section, we demonstrate the gap between existing task weighting strategies and the generalization performance of MTL in Section 4.1. This gap motivates us to proposed a MetaWeighting problem, which aims to automatically learn a task weighting strategy that can narrow this gap, in Section 4.2. Moreover, we propose an algorithm to solve the MetaWeighting problem in Section 4.3. 

### **4.1 Gap Between Task Weighting and Generalization Performance** 

MTL aims to improve the generalization performance of all the including tasks, which can be formulated via the following optimization problem. 



By contrast, existing task weighting strategies train an MTL model via the following objective. 



where the _wt_ is adaptive during training and only depends on the training losses or their gradients. As the neural networks are usually heavily overparameterized (Allen-Zhu et al., 2019), the training losses cannot properly estimate the generalization losses. Thus, existing task weighting strategies, which tunes weights only based on the training losses, overlook the generalization losses. Obviously, there is a gap between these task weighting strategies and the generalization performance of MTL. 

### **4.2 MetaWeighting Problem** 

To narrow the gap between task weighting strategies and generalization performance, we propose to automatically learn task weights that can reduce the generalization losses, namely _learning to weight_ . This _learning to weight_ problem is formlated via the following bi-level optimization problem, dubbed MetaWeighting. 

### **Problem 1.** 



where **w** = ( _w_ 1 _, w_ 2 _, ..., wT_ ). This bi-level optimization problem combines (5) and (6) together, by solving which we can obtain task weights that benefit the generalization performance of MTL. 

However, the generalization loss is agnostic. To properly estimate the generalization loss, we randomly divide the training set _Dt_ into two subsets: support set _Dt_<sup>_s_and query set</sup><sup>_D_</sup> _t_<sup>_q_, where</sup><sup>_D_</sup> _t_<sup>_s_is used</sup> to train an MTL model, and _Dt_<sup>_q_is used to estimate</sup> generalization loss of the MTL model. In Section 5, we theoretically demonstrate that query loss is a good estimator for the generalization loss; besides, in Section 6.7, experimental analysis also supports that query loss is a good estimator. 

Based on the support-query split, the MetaWeighting problem is transformed into the following form. 

### **Problem 2.** 



### **4.3 MetaWeighting Algorithm** 

In the MetaWeighting problem, the inner optimization objective is embedded within the outer optimization objective. In MTL, the inner optimization objective is to minimize the weighted sum of task-specific training losses, which is typically optimized by means of iterative gradient descent; thus, Problem 2 can be formulated by the following problem in the _k_<sup>_th_</sup> learning iteration. 

3438 

**Problem 3.** 



To solve Problem 3, we adopt the Hypergradient Descent (HD) method. However, the original HD method (Almeida et al., 1998; Baydin et al., 2018) is proposed for single objective optimization, which can not used in our problem where a multiobjective optimization problem involves. In this section, this paper proposes a novel HD method for the multi-objective optimization setting, as in the following sections. 

### **4.3.1 Task-Specific Descent Direction** 

The learning objective of Problem 3 involves _T_ objectives. We aim to find a gradient direction, moving against which all the objective can be optimized. To find this gradient direction, we first find the hypergradient direction w.r.t **w** (denoted as _dt_ ) for each task. _dt_ is computed by the following equation. 



### **4.3.2 Common Descent Direction** 

Base on _dt_ , in this section, we find a common gradient direction, moving against which all the objective can be optimized. Let **d** = ( _d_<sup>_⊤_</sup> 1<sup>_, d⊤_</sup> 2<sup>_, ..., d⊤_</sup> _T_<sup>)</sup> and _dc_ be the common gradient direction. Theorem 1 presents that the following Eq. (11) is a common descent direction. 



where 





where **1** = (1 _,_ 1 _, ...,_ 1). Eq. (12) is a typical minimum Euclidean-norm point problem. We here adopt the widely used Frank-Wolfe optimization algorithm (Jaggi, 2013), a minimum-norm-point algorithm, to solve it. The Frank-Wolfe optimization algorithm is presented in Algorithm 2. 

**Algorithm 1:** MetaWeighting Algorithm 



### **4.3.3 MetaWeighting** 

Moving against _dc_ , all the objective can be optimized; thus, the update rule of **w** is 



where _α_ is the step size. Based on this update rule, the task weights are automatically learnt oriented by optimizing the generalization losses. 

Overall, we propose the MetaWeighting algorithm, which is presented in algorithmic form in Algorithm 1. Our proposed method bridges the gap between task weighting and generalization performance of MTL. 

## **5 Theoretical Analysis** 

In this section, we study the generalization error bound for MTL; furthermore, we compare the bound w.r.t training loss and the bound w.r.t the 

3439 

query loss. The comparison presents that the query loss is a more accurate estimation of the generalization loss than the training loss. 

Firstly, we derive the generalization error bound w.r.t training loss for MTL. 

**Theorem 2.** _Assume we have n training samples for each task. Let σ_ = _{{σi_<sup>_t}n_</sup> _i_ =1<sup>_}_</sup> _t_<sup>_T_</sup> =1<sup>_bease-_</sup> _quence of binary random variables such that each σi_<sup>_t_=</sup><sup>_±_1</sup><sup>_is independent with probability_1</sup><sup>_/_2</sup><sup>_. Then,_</sup> _∀δ ∈_ [0 _,_ 1] _, for all h_ ( _·, θ_<sup>_s_</sup> _, θ_<sup>1</sup> _, ..., θ_<sup>_T_</sup> ) _∈H, with probability of at least_ 1 _− δ:_ 



_where_ 



_is the Rademacher complexity for MTL._ 

_Proof._ The proof is provided in Appendix A. 

Next, we derive the generalization error bound w.r.t query loss for MTL. 

**Theorem 3.** _Assume we have m training samples for each task. ∀δ ∈_ [0 _,_ 1] _, with probability of at least_ 1 _− δ, for all h_ ( _·, θ_<sup>_s_</sup> _, θ_<sup>1</sup> _, ..., θ_<sup>_T_</sup> ) _∈H, we have_ 



_Proof._ The proof is provided in Appendix A. 

Comparing the bound (14) and (16), we can find that the upper bound for the query loss is tighter than that for the training loss. Taking _m_ to be order of _n_ , the query loss is a more accurate estimate of the generalization loss than the training loss by a factor that depends on the Rademacher complexity. 

## **6 Experiments** 

In this section, we perform experimental studies on sentiment analysis to evaluate the performance of our proposed MetaWeighting and verify our theoretical analysis. 

### **6.1 Datasets** 

**Sentiment Analysis**<sup>1</sup> . We evaluate our algorithm on product reviews from Amazon. The dataset (Blitzer et al., 2007) contains product reviews from 14 domains, including books, DVDs, electronics, kitchen appliances and so on. We consider each domain as a binary classification task. Reviews with rating _>_ 3 were labeled positive, those with rating _<_ 3 were labeled negative, reviews with rating = 3 are discarded as the sentiments were ambiguous and hard to predict. 

**Topic Classification**<sup>2</sup> . We select 16 newsgroups from the 20 Newsgroup dataset, which is a collection of approximately 20,000 newsgroup documents that is partitioned (nearly) evenly across 20 different newsgroups, then formulate them into four 4-class classification tasks (as shown in Table 1) to evaluate the performance of our algorithm on topic classification. 

Table 1: Data Allocation for Topic Classification Tasks. 

|TASKS|NEWSGROUPS|
|---|---|
|COMP|OS.MS-WINDOWS.MISC, SYS.MAC.HARDWARE,<br>GRAPHICS, WINDOWS.X|
|REC|SPORT.BASEBALL, SPORT.HOCKEY<br>AUTOS, MOTORCYCLES|
|SCI|CRYPT, ELECTRONICS,<br>MED, SPACE|
|TALK|POLITICS.MIDEAST, RELIGION.MISC,<br>POLITICS.MISC, POLITICS.GUNS|



### **6.2 Baselines** 

We compare MetaWeighting with methods: 

**Single-Task Learning (STL):** learning each task independently. 

**Uniform:** learning tasks simultaneously using uniform task weights. 

**Uncertainty:** using the uncertainty weighting method proposed by (Kendall et al., 2018). 

**GradNorm:** using the gradient normalization method proposed by (Chen et al., 2018). 

**MGDA:** using the MGDA-UB method proposed by (Sener and Koltun, 2018). 

**AdvMTL:** using the adversarial Multi-task Learning method proposed by (Liu et al., 2017). 

**TchebycheffAdv:** using the Adversarial Tchebycheff procedure proposed by (Mao et al., 2020). 

**BanditMTL:** using the BanditMTL method proposed by (Mao et al., 2021). 

> 1https://www.cs.jhu.edu/~mdredze/ datasets/sentiment/ 

> 2http://qwone.com/~jason/20Newsgroups/ 

3440 



Figure 2: Classification accuracy of Single Task Learning, Uniform Scaling, AdvMTL, MGDA, GradNorm, Uncertainty, TchebycheffAdv, BanditMTL and MetaWeighting on TextCNN for the sentiment analysis dataset. Each colored cluster illustrates the classification accuracy performance of a method over 10 runs. Our proposed MetaWeighting outperforms all baselines on ten of the fourteen tasks; besides, its average performance is superior to that of all baselines. 



Figure 3: Classification accuracy of Single Task Learning, Uniform Scaling, AdvMTL, MGDA, GradNorm, Uncertainty, TchebycheffAdv, BanditMTL and MetaWeighting on TextCNN for the topic classification dataset. Each colored cluster illustrates the classification accuracy performance of a method over 10 runs. Our proposed MetaWeighting outperforms all baselines in all tasks. 

### **6.3 Experimental Settings 6.4 Classification Performance** 

We adopt the hard parameter-sharing MTL framework (Mao et al., 2021), where the shared representation extractor is built with TextCNN or BERT; besides, the task-specific module is formulated by means of one fully connected layer ending with a softmax function. The detailed experimental settings are introduced in the Appendix B. 

We compare the proposed MetaWeighting with the baselines and report the results over 10 runs by plotting the classification accuracy of each task for both sentiment analysis and topic classification. The results on TextCNN are shown in Fig. 2 and 3. Due to space limitations, we provide the results for BERT in the Appendix C. All experimental 

3441 



Figure 4: Task-average classification accuracy w.r.t different value of _ρ_ (query-split radio) for sentiment analysis and topic classification. 



Figure 5: Task-average classification accuracy w.r.t different value of _α_ (step size) for sentiment analysis and topic classification. 

results show that our proposed MetaWeighting outperforms all baselines and achieves state-of-the-art performance. 



Figure 6: Illustration of the gap between training loss, query loss and generalization loss in the training process of sentiment analysis (500<sup>_th_</sup> , 1000<sup>_th_</sup> , 1500<sup>_th_</sup> epochs respectively). 



Figure 7: Illustration of the gap between training loss, query loss and generalization loss in the training process of topic classification (500<sup>_th_</sup> , 1000<sup>_th_</sup> , 1500<sup>_th_</sup> epochs respectively). 

timent analysis and topic classification, setting _ρ_ = 0 _._ 1 provides satisfactory results. 

### **6.6 Sensitive Study on** _α_ 

### **6.5 The Impact of Query-Split Radio** 

Let _n_ be the size of the entire training set and _m_ be the size of the query set. We define the querysplit radio as _ρ_ =<sup>_<u>m</u>_</sup> _n_<sup>to indicate the radio of query</sup> samples to the entire training samples. From the theoretical analysis of Section 5, we can see that the query loss can estimate generalization loss more accurately when _ρ_ increases, but increasing _ρ_ would hurt the training process for the size of support set decreases. Therefore, _ρ_ faces a trade-off between the performance estimation of generalization loss and training performance. 

To investigate the impact of _ρ_ , we record the changes in MetaWeighting’s average classification accuracy w.r.t different values of _ρ_ in Fig. 4, where each boxplot visually illustrates the distribution of results over ten runs through displaying the data quartiles (first quartile and third quartile), minimum/maximum value and median. These experiments are conducted based on TextCNN. In this figure, as _ρ_ increases, the average accuracy of MetaWeighting first increases and then decreases. It verifies our theoretical analysis. For both sen- 

In MetaWeighting, the step size _α_ is a hyperparameter. To determine whether the performance of MetaWeighting is sensitive to _α_ , we conduct experiments on the classification accuracy performance of MetaWeighting w.r.t different values of _α_ based on the TextCNN model. The results of these experiments are presented in Figure 5 (boxplots over ten runs). As the figure shows, the performance of our proposed method is not very sensitive to _α_ when _α_ is within the range of 0 _._ 05 to 0 _._ 1 for sentiment analysis and 0 _._ 1 to 0 _._ 5 for topic classification. The results demonstrate that MetaWeighting can work well in a wide range of _α_ values. 

### **6.7 The Gap between the Training Loss, Query Loss and Generalization Loss** 

To experimentally verify that the query loss is a good estimator for generalization loss, we record the generalization loss (estimated by test loss), query loss and training loss for each task during training and report the results in Fig. 6 and 7 for sentiment analysis and topic classification respectively. From these figures, we can see that there 

3442 



Figure 8: Comparison of task weight adaption processes between MetaWeighting, Uncertainty, Gradnorm, MGDA and BanditMTL for sentiment analysis. 



Figure 9: Comparison of task weight adaption processes between MetaWeighting, Uncertainty, Gradnorm, MGDA and BanditMTL for topic classification. 

is a large gap between the training and generalization loss, while the gap between the query and generalization loss is smaller than that between the training and generalization loss. The results verify our theoretical analysis in Section 5; furthermore, they experimentally support our motivation for MetaWeighting. 

In this section, TextCNN is used, and tasks have uniform weights during training. Fig. 1 is obtained under this setting as well. 

### **6.8 The Evolution of Task Weights** 

In this section, we observe the changes in task weights in the training process of MetaWeighting and compare these changes with four baselines (Uncertainty, Gradnorm, MGDA and BanditMTL). The results for sentiment analysis and topic classification are reported in Fig. 8 and 9 respectively. Due to space limitations, for sentiment analysis, we only report the results of the first four tasks here, and the results of the other ten tasks are presented in the Appendix D. 

From these figures, we can see that the weight 

adaption process of MetaWeighting is different with that of Uncertainty, Gradnorm, MGDA and BanditMTL. In MetaWeighting, the task weights are automatically learnt, and there is no pre-defined heuristic involved. It is verified by the evolution curves of task weights for MetaWeighting illustrated in Fig. 8 and 9, which fluctuate without any regular patterns. 

## **7 Conclusion** 

This paper presents that the gap between the training loss and the generalization loss, which is overlooked by existing task weighting methods, is nonnegligible; furthermore, to narrow this gap, a novel task weighting method (dubbed MetaWeighting) is proposed. In MetaWeighting, multi-task text classification is formulated as a multi-objective bilevel programming problem, and then solved in a learning-to-learn manner. MetaWeighting automatically learns the task weights without any predefined heuristic and achieves state-of-the-art performance. It has the potential to forge new trends in task weighting research. 

3443 

## **References** 

- Jon Wellner Aad van der Vaart. 1996. _Weak convergence and empirical processes_ . Springer. 

- Zeyuan Allen-Zhu, Yuanzhi Li, and Yingyu Liang. 2019. Learning and generalization in overparameterized neural networks, going beyond two layers. In _NeurIPS_ . 

- Luís B Almeida, Thibault Langlois, José D Amaral, and Alexander Plakhov. 1998. Parameter adaptation in stochastic optimization. In _On-Line Learning in Neural Networks_ , pages 111–134. Cambridge University Press. 

- Jonathan Baxter. 2000. A model of inductive bias learning. _Journal of artificial intelligence research_ , 12:149–198. 

- Atilim Gunes Baydin, Robert Cornish, David MartínezRubio, Mark Schmidt, and Frank Wood. 2018. Online learning rate adaptation with hypergradient descent. In _ICLR_ . 

- John Blitzer, Mark Dredze, and Fernando Pereira. 2007. Biographies, bollywood, boom-boxes and blenders: Domain adaptation for sentiment classification. In _ACL_ . 

- Rich Caruana. 1993. Multitask learning: A knowledgebased source of inductive bias. In _ICML_ . 

- Zhao Chen, Vijay Badrinarayanan, Chen-Yu Lee, and Andrew Rabinovich. 2018. Gradnorm: Gradient normalization for adaptive loss balancing in deep multitask networks. In _ICML_ . 

- Jean-Antoine Désidéri. 2012. Multiple-gradient descent algorithm (mgda) for multiobjective optimization. _Comptes Rendus Mathematique_ , 350(5-6):313–318. 

- Michelle Guo, Albert Haque, De-An Huang, Serena Yeung, and Li Fei-Fei. 2018. Dynamic task prioritization for multitask learning. In _ECCV_ . 

- Martin Jaggi. 2013. Revisiting frank-wolfe: Projectionfree sparse convex optimization. In _ICML_ . 

   - Shikun Liu, Edward Johns, and Andrew J. Davison. 2019. End-to-end multi-task learning with attention. In _CVPR_ . 

   - Pingchuan Ma, Tao Du, and Wojciech Matusik. 2020. Efficient continuous pareto exploration in multi-task learning. In _ICML_ . 

   - Debabrata Mahapatra and Vaibhav Rajan. 2020. Multitask learning with user preferences: Gradient descent with controlled ascent in pareto optimization. In _ICML_ . 

   - Yuren Mao, Zekai Wang, Weiwei Liu, Xuemin Lin, and Wenbin Hu. 2021. Banditmtl: Bandit-based multitask learning for text classification. In _ACL_ . 

   - Yuren Mao, Shuang Yun, Weiwei Liu, and Bo Du. 2020. Tchebycheff procedure for multi-task text classification. In _ACL_ . 

   - Jeffrey Pennington, Richard Socher, and Christopher D. Manning. 2014. Glove: Global vectors for word representation. In _EMNLP_ . 

   - Ozan Sener and Vladlen Koltun. 2018. Multi-task learning as multi-objective optimization. In _NeurIPS_ . 

   - Ishan Tarunesh, Sushil Khyalia, Vishwajeet Kumar, Ganesh Ramakrishnan, and Preethi Jyothi. 2021. Meta-learning for effective multi-task and multilingual modelling. In _EACL_ . 

   - Xinyi Wang, Yulia Tsvetkov, and Graham Neubig. 2020. Balancing training for multilingual neural machine translation. In _ACL_ . 

   - Thomas Wolf, Lysandre Debut, Victor Sanh, Julien Chaumond, Clement Delangue, Anthony Moi, Pierric Cistac, Tim Rault, Rémi Louf, Morgan Funtowicz, Joe Davison, Sam Shleifer, Patrick von Platen, Clara Ma, Yacine Jernite, Julien Plu, Canwen Xu, Teven Le Scao, Sylvain Gugger, Mariama Drame, Quentin Lhoest, and Alexander M. Rush. 2020. Transformers: State-of-the-art natural language processing. In _EMNLP_ . 

- Alex Kendall, Yarin Gal, and Roberto Cipolla. 2018. Multi-task learning using uncertainty to weigh losses for scene geometry and semantics. In _CVPR_ . 

- Diederik P. Kingma and Jimmy Ba. 2015. Adam: A method for stochastic optimization. In _ICLR_ . 

- Xi Lin, Zhiyuan Yang, Qingfu Zhang, and Sam Kwong. 2020. Controllable pareto multi-task learning. _CoRR_ . 

- Xi Lin, Hui-Ling Zhen, Zhenhua Li, Qingfu Zhang, and Sam Kwong. 2019. Pareto multi-task learning. In _NIPS_ . 

- Pengfei Liu, Xipeng Qiu, and Xuanjing Huang. 2017. Adversarial multi-task learning for text classification. In _ACL_ . 

3444 

## **A Proof of the Theorem 2 and Theorem 3** 

Using the McDiarmid’s Inequality (Lemma 1), we have 

**Lemma 1** (McDiarmid’s Inequality) **.** _Let V be some set and let f_ : _V_<sup>_n_</sup> _→_ R _be a function of n variables such that for some c >_ 0 _, for all i ∈_ [ _n_ ] _and for all z_ 1 _, ..., zn, zi_<sup>_′∈Vwe have_</sup> 



We conclude our proof. 



### **Proof of Theorem 2** . 

_Let Z_ 1 _, ..., Zn be n independent random variables taking values in V . Then, with probability of at least_ 1 _− δ we have_ 

_Proof._ Using the standard symmetrization argument (for example, see Lemma 2.3.1 of (Aad van der Vaart, 1996) ), we have 





Combining Eq. (21) and Eq. (24), with probability 1 _− δ/_ 2: 

**Lemma 2** (Hoeffding’s Inequality) **.** _Let z_ 1 _, ..., zm be a a sequence of i.i.d. random variables and assume that for all i,_ E( _zi_ ) = _μ and P_ ( _a ≤ zi ≤ b_ ) = 1 _. Then, for any ϵ >_ 0 





Obviously, with probability of at least 1 _− δ/_ 2, for all _h ∈H_ , we have 



Let _s_<sup>_i_</sup> _t_ = ( _x_<sup>_i_</sup> _t_<sup>_, y_</sup> _t_<sup>_i_).</sup> The training set for MTL is _D_ = _{{_ ( _s_<sup>1</sup> 1<sup>_, ..., sn_</sup> 1<sup>_}, ..., {s_</sup> _t_<sup>1</sup><sup>_, ..., sn_</sup> _t_<sup>_}, ..., {s_1</sup> _T_<sup>_, ..., sn_</sup> _T_<sup>_}}_.</sup> For _∀t, i_ , replace _s_<sup>_i_</sup> _t_<sup>with</sup><sup>_ui_</sup> _t_<sup>=(</sup><sup>_x∗_</sup> _t_<sup>_, y_</sup> _t_<sup>_∗_)</sup><sup>_∈Dt_</sup> and create a new dataset _D_ = _{{_ ( _s_<sup>1</sup> 1<sup>_, ..., sn_</sup> 1<sup>_}, ...,_</sup> _{s_<sup>1</sup> _t_<sup>_, ..., ui_</sup> _t_<sup>_, ..., sn_</sup> _t_<sup>_}, ..., {s_1</sup> _T_<sup>_, ..., sn_</sup> _T_<sup>_}}_.</sup> Let _ht_ ( _·_ ) = _h_ ( _·, θ_<sup>_s_</sup> _, θ_<sup>_t_</sup> ). As _∀_ ( _x_<sup>_i_</sup> _t_<sup>_, y_</sup> _t_<sup>_i_)</sup><sup>_,_(</sup><sup>_xj_</sup> _t_<sup>_, y_</sup> _t_<sup>_j_):</sup> _|l_ ( _h_ ( _x_<sup>_i_</sup> _t_<sup>_, θs, θt_)</sup><sup>_, y_</sup> _t_<sup>_i_)</sup><sup>_−l_(</sup><sup>_h_(</sup><sup>_xj_</sup> _t_<sup>_, θs, θt_)</sup><sup>_, y_</sup> _t_<sup>_j_)</sup><sup>_|≤c_, we</sup> have 



_then ∀δ ∈_ [0 _,_ 1] _, with probability of at least_ 1 _− δ:_ 



_Rep_ ( _H, D_ ) _− Rep_ ( _H, D_ ) _≤ Proof._ Let _s_<sup>_i_</sup> _t_ = ( _x_<sup>_i_</sup> _t_<sup>_, y_</sup> _t_<sup>_i_).</sup> The <u>1</u> _<u>c</u>_ sup _Tn_<sup>_|l_(</sup><sup>_ht_(</sup><sup>_x_</sup> _t_<sup>_n_)</sup><sup>_, y_</sup> _t_<sup>_n_)</sup><sup>_−l_(</sup><sup>_ht_(</sup><sup>_x∗_</sup> _t_<sup>)</sup><sup>_, y_</sup> _t_<sup>_∗_))</sup><sup>_| ≤_</sup> _Tn_ training set for MTL is _D_ = _h∈H {{_ ( _s_<sup>1</sup> 1<sup>_, ..., sn_</sup> 1<sup>_}, ..., {s_</sup> _t_<sup>1</sup><sup>_, ..., sn_</sup> _t_<sup>_}, ..., {s_1</sup> _T_<sup>_, ..., sn_</sup> _T_<sup>_}}_.</sup> (27) For _∀t, i_ , replace _s_<sup>_i_</sup> _t_<sup>with</sup><sup>_ui_</sup> _t_ = ( _x_<sup>_∗_</sup> _t_<sup>_<u>,y</u>_</sup> _t_<sup>_∗_)</sup> _∈_ Using the McDiarmid’s Inequality (Lemma 1), we 1), we), we _Dt_ and create a new dataset _D_ = have that: with probability of at least 1 1 _− δ/_ 2:: _{{_ ( _s_<sup>1</sup> 1<sup>_, ..., sn_</sup> 1<sup>_}, ..., {s_</sup> _t_<sup>1</sup><sup>_, ..., ui_</sup> _t_<sup>_, ..., sn_</sup> _t_<sup>_}, ..., {s_1</sup> _T_<sup>_, ..., sn_</sup> _T_<sup>_}}_.</sup> 2 log(4 _<u>/δ</u>_ <u>)</u> Let _ht_ ( _·_ ) = _h_ ( _·, θ_<sup>_s_</sup> _, θ_<sup>_t_</sup> ). As _∀_ ( _x_<sup>_i_</sup> _t_<sup>_, y_</sup> _t_<sup>_i_)</sup><sup>_,_(</sup><sup>_xj_</sup> _t_<sup>_, y_</sup> _t_<sup>_j_):</sup> E _DR_ ( _l◦H◦D_ ) _≤ R_ ( _l◦H◦D_ )+2 _c_ � _Tn . |l_ ( _h_ ( _x_<sup>_i_</sup> _t_<sup>_, θs, θt_)</sup><sup>_, y_</sup> _t_<sup>_i_)</sup><sup>_−l_(</sup><sup>_h_(</sup><sup>_xj_</sup> _t_<sup>_, θs, θt_)</sup><sup>_, y_</sup> _t_<sup>_j_)</sup><sup>_|≤c_, we</sup> (28) have Based on Eq. (28) and the union bound, we have28) and the union bound, we have) and the union bound, we have 

Using the McDiarmid’s Inequality (Lemma 1), we 1), we), we have that: with probability of at least 1 1 _− δ/_ 2:: 

Based on Eq. (28) and the union bound, we have28) and the union bound, we have) and the union bound, we have that - with probability of at least 1 _− δ_ : 





In our setting, _l_ ( _·, ·_ ) : _Y_<sup>_t_</sup> _×Y_<sup>_t_</sup> _→_ [0 _,_ 1], then _c_ = 1. We have 



We conclude our proof. 

Based on the Hoeffding’s Inequality (Lemma 2), we have the following theorem. 

### **Proof of Theorem 3** . 

_Proof._ Based on the Hoeffding’s Inequality (Lemma 2) and _l_ ( _·, ·_ ) : _Y_<sup>_t_</sup> _× Y_<sup>_t_</sup> _→_ [0 _,_ 1], for each _h_ ( _·, θ_<sup>_s_</sup> _, θ_<sup>_t_</sup> ) _∈H_<sup>_t_</sup> , we have 



Then, with probability of at least 1 _−_ 2 _exp_ ( _−_ 2 _mϵ_<sup>2</sup> ), we have 



Let _δ_ = 2 _exp_ ( _−_ 2 _mϵ_<sup>2</sup> ), we have that with probability of at least 1 _− δ_ , 



Thus, for each task, 



Since the bound for each task are independent, we have 



We conclude our proof. 

with a softmax function. The TextCNN module is structured with three parallel convolutional layers with kernels size of 3, 5, 7 respectively. For TextCNN, we adopt Pre-trained GloVe (Pennington et al., 2014) word embeddings. By contrast, the BERT module is formulated via a pre-trained BERT-base model provided by Hugging Face(Wolf et al., 2020), with a hidden size of 768, 12 Transformer blocks and 12 self-attention heads. 

We train the deep MTL network model in line with Algorithm 1. We set _α_ to be 0.1 and 0.5 for sentiment analysis and topic classification respectively, and the query-split radio (radio of query samples to entire training samples) to be 0 _._ 1 for both sentiment analysis and topic classification. We use the Adam optimizer (Kingma and Ba, 2015). We train over 3000 epochs for TextCNN and finetune over 50 epochs for BERT. For TextCNN, the learning rate is 1 _e −_ 3 and the batch size is 256. For BERT, the learning rate is 2 _e −_ 5 , the batch size is 32, and the max sequence length is 256. For the baselines, we search over the set _{_ 1 _e−_ 5 _,_ 2 _e−_ 5 _,_ 5 _e−_ 5 _,_ 1 _e−_ 4 _,_ 5 _e−_ 4 _,_ 1 _e−_ 3 _,_ 5 _e−_ 3 _}_ learning rates and choose the model with best performance. 

## **C Classification Performance on BERT** 

For the BERT-based MTL model, we compare the proposed MetaWeighting with the baselines and report the results over 10 runs by plotting the classification accuracy of each task for both sentiment analysis and topic classification in Fig. 10 and 11. AdvMTL and TchebycheffAdv are not available for BERT; thus, we do not compare with AdvMTL and compare with Tchebycheff which is TchebycheffAdv without aversarial module (Mao et al., 2021). From these figures, we can see that our proposed MetaWeighting outperforms all baselines and achieves state-of-the-art performance. 

## **B Detailed Experimental Settings** 

We adopt the hard parameter-sharing MTL framework (Mao et al., 2021), where the shared representation extractor is built with TextCNN or BERT; besides, the task-specific module is formulated by means of one fully connected layer ending 

## **D The Evolution of Task Weights for Sentiment Analysis** 

Fig. 12 illustrates the changes in task weights in the training process of MetaWeighting for all the tasks of sentiment analysis. 

3446 



Figure 10: Classification accuracy of Single Task Learning, Uniform Scaling, MGDA, TchebycheffAdv, Uncertainty, GradNorm, BanditMTL and MetaWeighting on BERT for the sentiment analysis dataset. Each colored cluster illustrates the classification accuracy performance of a method over 10 runs. Our proposed MetaWeighting outperforms all baselines on eleven of the fourteen tasks; besides, its average performance is superior to that of all baselines. 



Figure 11: Classification accuracy of Single Task Learning, Uniform Scaling, MGDA, TchebycheffAdv, Uncertainty, GradNorm, BanditMTL and MetaWeighting on BERT for the topic classification dataset. Each colored cluster illustrates the classification accuracy performance of a method over 10 runs. Our proposed MetaWeighting outperforms all baselines on three of the four tasks; besides, its average performance is superior to that of all baselines. 

3447 



Figure 12: Comparison of task weight adaption processes between MetaWeighting, Uncertainty, Gradnorm, MGDA and BanditMTL for sentiment analysis. 

3448 

