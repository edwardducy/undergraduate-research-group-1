# Uniform Loss vs. Specialized Optimization: A Comparative Analysis in Multi-Task Learning

Gabriel S. Gama

University of São Paulo

gabriel\_gama@usp.br

Valdir Grassi Jr.

University of São Paulo

vgrassi@usp.br

## Abstract

Specialized Multi-Task Optimizers (SMTOs) balance task learning in Multi-Task Learning by addressing issues like conflicting gradients and differing gradient norms, which hinder equal-weighted task training. However, recent critiques suggest that equally weighted tasks can achieve competitive results compared to SMTOs, arguing that previous SMTO results were influenced by poor hyperparameter optimization and lack of regularization. In this work, we evaluate these claims through an extensive empirical evaluation of SMTOs, including some of the latest methods, on more complex multi-task problems to clarify this behavior. More specifically, we start our analysis by evaluating all SMTOs on a simple MNIST problem to identify the promising optimizers and assess them further on progressively more complex multi-task problems. Our findings indicate that SMTOs perform well compared to uniform loss and that fixed weights can achieve competitive performance compared to SMTOs. Furthermore, we demonstrate why uniform loss performs similarly to SMTOs in some instances. The source code is available at https://github.com/Gabriel-SGama/UnitScal\_vs\_SMTOs.

## 1 Introduction

When tackling complex real-world challenges in machine learning, one must contemplate the utility of Multi-Task Learning (MTL) [1]. Scenarios such as scene understanding, multilingual translation, and predicting chemical properties require multiple outputs, possibly making them promising candidates for MTL approaches due to their inherent inductive bias [2].

Classical methods approach this problem by utilizing one model for each task, but most MTL scenarios have the potential to share information across tasks. For example, autonomous driving relies on multiple pieces of information extracted from the same visual input, such as semantic segmentation, object detection, instance segmentation, and others. Therefore, sharing information across those tasks is desirable to increase performance.

To efficiently share feature information across tasks, various approaches have been proposed, includ ing model architecture designs [3, 4, 5], task grouping strategies [6, 7], Meta-Learning techniques [8], and the focus of this work, Specialized Multi-Task Optimizers (SMTOs) [9, 10].

SMTOs are implemented together with a shared encoder architecture, where one common encoder outputs a feature space that is shared with the specific decoders used for each task [9, 10]. Consequently, this reduces the computational cost when compared to the individual models approach. This unified model can be optimized by minimizing the sum of the losses, known as uniform loss. However, the simultaneous learning of multiple tasks introduces challenges such as conflicting directions and significant gradient norm disparities [11, 12]. SMTOs aim to address these challenges by applying mathematical tools to jointly and equally optimize all objectives.

SMTOs can be classified into two types: gradient-based and loss-based. The former compute the gradients of each task and combine them to find a suitable optimization direction [9, 13, 14, 15, 11, 12, 10, 16]. This process adds complexity and computational cost to the learning problem as multiple backpropagation steps are needed. The loss-based methods only take into consideration the loss values [17, 18, 19, 20, 21], reaching computational times comparable to those of the uniform loss [22, 21].

One notable issue in the SMTO field is the lack of a standardized evaluation framework and procedure, leading to concerns about reproducibility [23]. Understandably, some works have begun questioning the efficacy of SMTOs [24, 23]. These critiques attribute reported gains in SMTO papers to suboptimal hyperparameter tuning or insufficient regularization. They provide various experiments, including natural language processing (NLP), computer vision, and reinforcement learning, as well as a theoretical hypothesis on why SMTOs apply a regularization effect and a comparison to a properly optimized baseline.

In this context, we propose to address these concerns by hypothesizing that proper hyperparameter optimization is crucial for fair performance assessment of SMTOs. Nevertheless, we also emphasize the significance of considering the inherent complexity of the MTL balancing problem. Our contributions include:

• A comprehensive evaluation of current SMTOs throughout Section 3, including recent ones that have not been covered in previous critiques, using the MNIST dataset [25] as a starting point to identify promising SMTOs, followed by other commonly used datasets such as Cityscapes [26] and Quantum Chemistry and Molecular Physics (QM9) [27].  
• An investigation into why equally weighted tasks can match the results of SMTOs. More specifically, the two similar tasks scenario in Section 3.2, and a complexity analysis in Section 3.3.  
• A re-evaluation of the hypothesis brought up by [23] that fixed weights can match the same results achieved by SMTOs in Section 3.5.

## 2 Related Work

In recent years, MTL and SMTOs have witnessed substantial research activity. Notable works, such as [17, 9], have garnered interest in MTL due to their performance improvements compared to the uniform loss and single task baselines. This surge has also given rise to numerous variations and novel SMTO methods, each improving on or proposing solutions to unaddressed concerns of early SMTOs [14, 28, 12, 16].

However, despite the growing body of research, the SMTO field faces a significant challenge: the absence of a formal benchmark. Consequently, reported results may vary between articles [9, 23, 10], raising concerns about the improvements. Specifically, recent critiques have revealed potential issues, suggesting that the reported performance gains from SMTOs stem from limited regularization [24] and under-optimized baseline hyperparameters [23].

Kurin et al. [24] show that all evaluated SMTOs are linked to a larger solution space, underoptimization or stochastic behavior, all of which induce a regularization effect [29, 30, 31]. They also evaluated their hypothesis by adding proper regularization such as weight decay, batch normalization [32], dropout [33], and early stopping to the uniform loss, referenced as Unitary Scalarization (Unit. Scal.) in this paper, and compared it to SMTOs. After the changes, they achieved competitive results to SMTOs at a lower computational cost.

Similarly, Xin et al. [23] argue that reported SMTO gains may be attributed to poor hyperparameter optimization, with particular interest in the impact of learning rates compared to random seeds. Their work demonstrates that exploring multiple fixed weight values can construct a Pareto curve, and SMTOs can only obtain one of the possible solutions. Due to computational restrictions, the discussion regarding fixed weights will be partially addressed as, to the best of our knowledge, there is no method to predict those fixed weights accurately, and grid search becomes impractical for a higher number of tasks.

Our work distinguishes itself in two key ways. Firstly, we consider most of the recent SMTOs in our evaluation [28, 12, 10, 18, 16, 20]. Secondly, we adopt some variations to the commonly used problems to introduce more complexity, as detailed in Section 3. Our objective is to demonstrate that SMTOs, mainly the recent ones, provide significant improvements over Unit. Scal. as the complexity increases. Additionally, we aim to elucidate the scenarios in which Unit. Scal. can match SMTOs and provide a complete evaluation of these promising SMTOs in multiple challenging MTL configurations.

## 3 Experimental Evaluation

Although numerous works utilize similar datasets and training configurations, a broadly accepted evaluation framework remains elusive. We have chosen specific problems from a selection of SMTO papers to address this challenge. More specifically, we restrict our evaluation to the supervised learning setting using datasets such as MNIST [25], Cityscapes [26], and QM9 [27]. To address recent critiques to SMTOs [24, 23] and ensure robustness, the models were trained with at least one form of regularization, except for the QM9 dataset, and the parameters were optimized by applying grid search. We determine the best hyperparameter configuration by training once with each value (excluding Multi-MNIST, which was trained five times for each value) and selecting the best validation result. The best combination is trained at least two more times, and the metrics reported here are from the test split. We consider the epoch in which the model achieved the highest validation metric to mitigate overfitting.

For comparative evaluation of different SMTOs, we used two widely employed metrics, the mean relative percentage gain metric $\Delta _ { \mathrm { m t m } }$ and the mean rank MR [34, 10, 20]. $\Delta _ { \mathrm { m t m } }$ measures the improvement achieved by the MTL model over single-task references, and the MR evaluates how equally all tasks were optimized. Since tasks like semantic segmentation and disparity estimation use multiple metrics for evaluation, we adapt the equations to account for that. For each set of metrics $M _ { i }$ for task $i \in \mathcal T$ , the single task reference $M _ { b , i , j } \enspace \mid j \in M _ { i }$ is compared to its multi-task counterpart $M _ { \mathrm { s m t o } , i , j } \enspace \vert \enspace j \in M _ { i } .$ , so $\Delta _ { \mathrm { m t m } }$ is given by:

$$
\Delta_ {\mathrm{mtm}} = \frac {1}{N} \sum_ {i = 1} ^ {N} \frac {1}{M _ {i}} \sum_ {j = 1} ^ {M _ {i}} (- 1) ^ {l _ {i, j}} \frac {M _ {\mathrm{smto} , i , j} - M _ {b , i , j}}{M _ {b , i , j}}, \tag {1}
$$

where $l _ { i , j }$ is 1 if a lower value is better and 0 otherwise. Likewise, the mean rank is given by:

$$
M R = \frac {1}{N} \sum_ {i = 1} ^ {N} \frac {1}{M _ {i}} \sum_ {j = 1} ^ {M _ {i}} R _ {i, j}, \tag {2}
$$

where $R _ { i , j }$ is the rank of the SMTO on task i with respect to the metric j. As a result, both metrics compute the mean within the same task and then the mean of every task, ensuring a fair evaluation in a multi-task setting.

From a wide array of publicly available SMTOs, we selected a subset for evaluation: UW [17], MGDA [9], PCGrad [35], GradDrop [15], EDM [14], IMTL [11], CAGrad [28], RotoGrad [12], Nash-MTL [10], SI [10], RLW [19], Auto-Lambda [18], CDTT [16] and FAMO [20]. We highlight that there are still some SMTOs that could be evaluated in future works [36, 13, 37, 38, 22, 21].

Including in Unit. Scal. and two RLW variations (Normal and Dirichlet distributions) as done in previous works [24], we assessed a total of 16 SMTOs. To reduce computational cost, the SMTOs considered in this paper are filtered according to their performance in a regression and classification problem. Even though evaluating an SMTO with just one example is not a definitive quality measure, this two-task problem tests the method’s ability to handle varying gradient norms. This concern motivated several SMTOs [13, 14, 11, 10]. Based on the results and contributions from each SMTO, we select the most promising ones and proceed to test them further.

In this section, the chosen SMTOs are first evaluated in the specific case of two similar task problems to demonstrate that, despite being an MTL problem, Unit. Scal. approximates the behavior of the SMTOs and achieves competitive results. We also consider some factors that dictate the need for SMTOs by comparing Unit. Scal. to one of the SMTOs while evaluating different configurations with increasing multi-task learning balancing complexity, using the Cityscapes dataset. All promising SMTOs are then evaluated in the most challenging format. Furthermore, we provide a comparison on the QM9 dataset. Finally, we evaluate the performance of fixed weights. All hyperparameter combi nations, SMTOs-specific hyperparameter optimization, and their best configuration are summarized in Appendix A.6 for reproducibility.

## 3.1 Selecting promising SMTOs for further evaluation

In this paper, we tackle multiple MTL challenges to evaluate the performance of many SMTOs. However, due to the multitude of combinations and the necessity for hyperparameter optimization to ensure a fair evaluation [24, 23], we apply a preliminary filter for SMTO selection based on the performance on the Multi-MNIST dataset [9].

The Multi-MNIST dataset is a variation of the MNIST dataset for MTL where the digits are overlapped in the same image, one on the top left and one on the bottom right. This combined image is then used to predict information regarding the left and right digits. The most common variation is predicting the class of the left and right digits, as done in multiple works [9, 35, 12, 24, 23]. Instead, we opt for the alternative suggested by Nakamura et al. [16], where the model is trained to classify the top left digit and to reconstruct the bottom right digit. This choice was made because this variation adds more complexity, involving tasks with varying levels of difficulty and loss functions that result in gradients with different norms, an important factor when considering the specific case of two tasks, as shown in Section 3.2. Those tasks will be referenced as CL, CR, RL, and RR for the classification (C) or reconstruction (R) of the left (L) or right (R) digit. When tasks are represented using the notation {task 1}\_{task 2}...{task N}, it indicates a multi-task scenario that simultaneously addresses tasks 1 through N.

For our model architecture, we use a variation of the Lenet model [25] as Nakamura et al. [16]. We then add dropout layers to the encoder and decoder architectures for regularization. The hyperparameters are optimized by grid search, varying the learning rate $l r \in \{ 0 . 0 \bar { 1 } , 0 . 0 7 5 , 0 . 0 0 5 , 0 . 0 \bar { 0 } 2 5 , 0 . 0 0 1$ 0.00075, 0.0005} and dropout $p \in \{ 0 . 0 , 0 . 1 , 0 . 2 , 0 . 3 , 0 . 4 , 0 . 5 \}$ . Specific SMTO hyperparameters are also included in the search if necessary. We train each configuration five times and the best validation configuration five more times, giving a total of ten models for each SMTO.

![](images/92f02c8a37e3e33968d6ec8fe1b3681310cf48341ba1f2b341262fa576dc8655.jpg)

<details>
<summary>boxplot</summary>

| Algorithm | Q1 | Q2 (Median) | Q3 | IQR |
| --- | --- | --- | --- | --- |
| Unit.Scal. | ~-9.5 | ~-8.8 | ~-8.5 | ~1.0 |
| Auto-Lambda | ~-5.0 | ~-3.0 | ~-2.8 | ~2.8 |
| CAGrad | ~-2.0 | ~-1.8 | ~-1.6 | ~0.6 |
| CDTT | ~-2.8 | ~-2.0 | ~-1.4 | ~1.6 |
| EDM | ~-2.8 | ~-1.8 | ~-1.2 | ~1.4 |
| FAMO | ~-4.5 | ~-1.5 | ~-1.2 | ~2.7 |
| GradDrop | ~-11.8 | ~-11.0 | ~-10.8 | ~1.0 |
| IMTL | ~-2.5 | ~-1.5 | ~-1.0 | ~1.5 |
| MGDA-UB | ~-6.5 | ~-6.0 | ~-5.8 | ~0.3 |
| Nash-MTL | ~-3.2 | ~-2.5 | ~-2.0 | ~1.2 |
| PCGrad | ~-9.0 | ~-8.0 | ~-7.5 | ~1.5 |
| RLW-Dirichlet | ~-10.2 | ~-10.0 | ~-9.8 | ~0.6 |
| RLW-Normal | ~-9.8 | ~-9.5 | ~-9.2 | ~0.4 |
| RotoGrad | ~-4.0 | ~-1.0 | ~-0.8 | ~1.8 |
| SI | ~-14.5 | ~-13.8 | ~-13.5 | ~0.8 |
| UW | ~-14.0 | ~-13.8 | ~-13.7 | ~0.7 |
</details>

Figure 1: Box plot illustrating $\Delta _ { \mathrm { m t m } }$ scores on the Multi-MNIST classification and regression problem. The red line indicates the median value for Unit. Scal.. Notably, multiple SMTOs obtained substantial improvements over the Unit. Scal. baseline.

The results, as presented in Figure 1, demonstrate that Unit. Scal. achieves competitive results compared to some SMTOs. However, the majority outperform it significantly, primarily recent ones. Based on these results, we select EDM [14], IMTL [11], CAGrad [28], RotoGrad [12], Nash-MTL [10], Auto-Lambda [18], CDTT [16] and FAMO [20] as promising SMTOs for further investigation.

Despite the Multi-MNIST dataset’s simplicity, it effectively demonstrated the selected SMTOs’ ability to provide a meaningful gradient direction even with varying gradient norms, enabling their distinction from other methods. Specifically, PCGrad and MGDA are known to emphasize gradients with the smallest and largest magnitudes, respectively [11]. GradDrop randomly drops gradients based on sign-purity, without considering magnitude balancing. Furthermore, methods like RLW, UW, and SI operate directly on loss values without estimating gradients, unlike FAMO and Auto-Lambda, potentially making them sensitive to differences in gradient magnitude. Ultimately, most of the selected SMTOs significantly outperformed Unit. Scal. in our evaluations.

## 3.2 Two similar tasks

The two-task setting is commonly used as a computationally efficient way of evaluating SMTOs, widely adopted in multiple MTL papers [9, 24, 23]. However, when considering the two-task setting, it is possible that Unit. Scal. approximates the behavior of an SMTO given a few conditions.

When considering a problem with a shared encoder, the same decoder architecture, the same loss function, task labels with similar distributions, and learning speeds, the gradient norm and the loss value of both tasks are expected to behave in remarkably similar ways. Additionally, if we restrict this assumption to equal gradient norm or equal loss values, it is possible to theoretically show that most SMTOs converge to equal weights (see Appendix A.4).

To provide empirical evidence for the above statement, we consider two problems from Multi-MNIST: dual classification and dual reconstruction of the digits. In both cases, the architecture initially described in the previous section was reused with the corresponding changes to the decoders. We follow the same hyperparameter optimization previously described for Multi-MNIST.

![](images/d43cfd265f47356f2e8935178d33e8f0fb3a57cb71f232c9b7e4956158b74250.jpg)  
Figure 2: Box plot of $\Delta _ { \mathrm { m t m } }$ (left) for all SMTOs not evaluated on [24] and mean error to equal weights (right). The top plot corresponds to CL\_CR, and the bottom one represents RL\_RR. All SMTOs had similar performance to Unit. Scal. and the computed weights varied near 0.5, excluding FAMO on the CL\_CR problem, exemplifying how similar equal weights are to the selected SMTOs solution in this setting.

The results presented in Figure 2 show that all SMTOs performed closely to the Unit. Scal. baseline, even on the dual reconstruction case where $\Delta _ { m t m } \approx - \bar { 2 } 0 \% .$ meaning that there was high interference between tasks. We can also observe that the learned weights from the SMTOs predominantly clustered around 0.5, besides FAMO, which was a notable exception, exhibiting instability on the dual classification problem, likely attributable to the small scale of the loss values.

Consequently, employing equal weights approximates the average behavior of the SMTOs, resulting in comparable performance. This observation suggests that a straightforward averaging of task gradients can serve as a surprisingly effective baseline without significant performance degradation in scenarios with similar task characteristics, such as the dual translation setting examined in the recent critique [23].

## 3.3 Complexity analysis

Cityscapes [26] is a street scene dataset containing high-resolution images 2048x1024 and labels for semantic segmentation, disparity estimation, and instance segmentation. It is commonly adopted for MTL [17, 9, 24, 10, 23, 20] under two variations, the two-task problem with 7 class semantic segmentation and disparity estimation, and the three-task problem that uses 19 semantic classes and adds instance segmentation. Some similarities arise when comparing the first version to the Multi-MNIST classification-reconstruction problem. Both problems generate gradients with different norms, and the tasks differ in complexity, yet there is no significant difference between SMTOs and Unit. Scal. with optimized hyperparameters [24, 23].

![](images/306bd740aa4ddfe10e130ff3cc0fa4d7e201ed22f04f1c27f45e04580c120b50.jpg)

<details>
<summary>line</summary>

| Training (%) | QM9 | MNIST CL_RR | City R50 | City 19C |
| --- | --- | --- | --- | --- |
| 0 | ~0.48 | ~0.62 | ~0.72 | ~0.73 |
| 20 | ~0.42 | ~0.60 | ~0.68 | ~0.45 |
| 40 | ~0.35 | ~0.58 | ~0.68 | ~0.45 |
| 60 | ~0.28 | ~0.58 | ~0.68 | ~0.45 |
| 80 | ~0.28 | ~0.58 | ~0.68 | ~0.45 |
| 100 | ~0.28 | ~0.60 | ~0.68 | ~0.45 |
</details>

(a) Mean cosine similarity.

![](images/2da2f861d24e7bf74cdfe42398c61609d2df15f88c60e60020839257453b5032.jpg)

<details>
<summary>line</summary>

| Category | Unit. Scal. (%) | IMTL (%) | Difference |
| --- | --- | --- | --- |
| R50 | ~1.2 | ~1.2 | 0 |
| R18 | 0 | 0 | 0 |
| Inst. Seg. | ~-7.5 | ~-0.5 | ~6.8 |
| HR | ~-3.7 | ~1.2 | ~4.8 |
| 19C | ~-5.4 | ~0.8 | ~6.2 |
</details>

(b) IMTL and Unit. Scal. performance.  
Figure 3: (a) Mean cosine similarity between each task gradient and the average direction. City R50 exhibits significantly higher similarity than Multi-MNIST, potentially explaining Unit. Scal.’s comparable performance to SMTOs. Conversely, the lower similarity for City 19C and QM9 correlates with Unit. Scal.’s relative underperformance. See Appendix A.2.1 for the full version with all configurations. (b) Shows the relative performance of Unit. Scal. and IMTL. Adding a third task and increasing the amount of information needed by increasing the number of classes are the relevant steps that dictate the relative performance in this problem.

The reason for this unexpected behavior is related to the lower amount of interference between each task compared to the Multi-MNIST problem, as seen in Figure 3a. Figure 3a shows the mean cosine similarity between each task gradient and the average direction, which can be seen as a measure of interference, as it correctly ranks the problems with respect to the relative performance between Unit. Scal. and SMTOs. Even after significantly reducing the parameter count in step R18, the level of interference remains about the same (see Appendix A.2.1), as reflected in the relative performance between Unit. Scal. and IMTL. However, after increasing the complexity of the problem, there is a significant difference between the two methods.

To vary the complexity in a controlled manner, we vary model size, number of tasks, and information available to learn to identify which characteristic is relevant. We first trained the model on the same initial two-task configuration of [24] and applied small cumulative steps until the most complex configuration. We opted to test only IMTL [11] and compare it to Unit. Scal. in the transitional steps as it was one of the best performing SMTOs, and it does not require any specific hyperparameter optimization. All Cityscapes configurations are optimized varying the learning rate at $l r \in \{ 0 . 0 0 5$ 0.001, 0.0005, 0.0001} and the weight decay $\dot { \in } \ \{ 1 0 ^ { - 4 } , \ 1 0 ^ { - 5 } ,$ , 0.0} with 32 batch size. Each configuration is trained once, and then the best in the validation dataset is trained a total of 3 times. The selected steps are:

• Reference (R50): A Resnet-50 [39] is used as shared encoder and the DeepLab decoder [40] is used as task specific decoder. The input resolution is 256x128, and the objective is set to 7 class segmentation and disparity estimation. This is the same configuration of Kurin et al. [24] and very similar to the one of Xin et al. [23];  
• Resnet-18 as decoder (R18): Opting for a smaller architecture should encourage competition for parameters;  
• Added instance segmentation (Inst. Seg.): Adding more tasks increases the complexity of the balancing problem, as well as the amount of information available to learn;  
• Increased resolution (HR): The resolution is increased from 256x128 to 512x256 as an intermediate step to allow better learning of objects from smaller classes;  
• 19 semantic classes (19C): By increasing the number of learnable classes and keeping the model size fixed, the amount of information needed by the semantic decoder is increased, resulting in more competition for the shared encoder parameters and feature space.

The results in Figure 3b reveal that the benefit of employing an SMTO only becomes apparent upon the inclusion of the instance segmentation task. Furthermore, a substantial relative performance gain is observed after incorporating all 19 classes, suggesting that the volume of information significantly impacts the efficacy of SMTOs. This observation aligns with prior research on information transfer [41], which highlights the necessity of carefully calibrating the shared model’s capacity in multi-task learning. Insufficient or excessive capacity can lead to negative or negligible transfer. Therefore, by increasing the number of tasks and classes, we inherently increase the demands on the shared feature space, potentially increasing inter-task conflicts.

![](images/c9a6106e26dfb4375505c0d1ba8466320c6e5b12c37da4f26379838d13f89905.jpg)

<details>
<summary>boxplot</summary>

| Algorithm | Q1 | Q2 (Median) | Q3 |
| --- | --- | --- | --- |
| Unit.Scal. | ~-5.5 | ~-5.0 | ~-4.9 |
| Auto-Lambda | ~-4.8 | ~-4.5 | ~-4.4 |
| CAGrad | ~-3.3 | ~-3.0 | ~-2.9 |
| CDTT | ~-1.2 | ~-1.1 | ~-1.0 |
| EDM | ~0.2 | ~0.4 | ~0.6 |
| FAMO | ~-0.2 | ~-0.1 | ~0.1 |
| IMTL | ~0.6 | ~0.7 | ~0.8 |
| Nash-MTL | ~-0.3 | ~-0.1 | ~0.2 |
| RotoGrad | ~-2.0 | ~-1.9 | ~-1.8 |
</details>

Figure 4: Muiti task metric on Cityscapes dataset. Most SMTOs significantly surpass the Unit. Scal. baseline on the most complex version of the Cityscapes MTL problem.

Similar behavior was also noticed in the NLP field [42], where they compared the interference caused by different languages and data size, concluding that a bigger model could mitigate the interference issue, though their evaluation was restricted to Unit. Scal.. On the other hand, we investigate the effect of those parameters on the relative performance of different SMTOs to the baseline.

This exemplifies that SMTOs are not needed on some occasions when the problem is simple enough that the conflicting gradients do not interfere with the final performance. Therefore, the best way to evaluate SMTOs is to use the most complex configuration to measure their capabilities in dealing with significant interference. Accordingly, the other promising SMTOs are trained and evaluated with the same procedures on the last configuration.

Figure 4 shows that SMTOs can indeed achieve better results when compared to Unit. Scal. in more complex balancing problems. This conclusion differs from Unit. Scal. [24] primarily because of the increased complexity. Furthermore, we evaluate some of the newer SMTOs, showing the field’s progress in the last years.

## 3.4 QM9

Finally, the last dataset we use to evaluate SMTOs is the QM9 dataset [27]. The QM9 dataset is a collection of quantum chemistry data for small organic molecules. It contains around 130,000 data points, and it was recently adopted as a problem for the MTL benchmarks [10, 20].

We use the same configuration from [10]. The example from PyTorch Geometric [43] was adapted, and we used the model from [44]. The model was used to predict 11 properties from the molecules. We use the commonly adopted distribution of 110K molecules for training, 10K for validation, and 10K for testing, with a batch size of 120. The learning rate was optimized in $l r \in \{ 5 \times 1 0 ^ { - 3 } , 1 0 ^ { - 3 } , 5 \times 1 0 ^ { - 4 } \}$ , and the ground truth values were normalized. We opted not to apply regularization because Unit. Scal. did not show significant improvement, and omitting this step reduced the overall computational cost.

Comparing the results from Table 1 to the previous experiments, it is evident that this is the problem with the most amount of interference between tasks, with

Table 1: Multi-task metric on QM9 dataset. Almost all SMTOs significantly surpass Unit. Scal.

<table><tr><td>SMTO</td><td> $\Delta_{MTM}(\%) \uparrow$ </td><td>MR ↓</td></tr><tr><td>Unit. Scal.</td><td>-135.7</td><td>4.45</td></tr><tr><td>Auto-Lambda</td><td>-186.9</td><td>5.82</td></tr><tr><td>CAGrad</td><td>-70.13</td><td>3.82</td></tr><tr><td>CDTT</td><td>-121.2</td><td>5.45</td></tr><tr><td>EDM</td><td>-95.42</td><td>4.00</td></tr><tr><td>FAMO</td><td>-56.86</td><td>2.36</td></tr><tr><td>Nash-MTL</td><td>-53.62</td><td>2.09</td></tr></table>

all SMTOs obtaining results more than 50% worse on average compared to the single task problem. This also correlates with the lower cosine similarity shown in Figure 3a.

During the experiment, IMTL was too unstable, and most runs crashed due to numerical overflow, as happened on the RL benchmark from Unit. Scal. [24], which also had a higher number of tasks. We agree with the hypothesis that this was caused by a lack of bounds on the scaling coefficients obtained from the IMTL’s optimization problem. RotoGrad also failed to converge, and our hypothesis is that RotoGrad was not able to find a proper rotation that minimizes the gradient conflict probably due to the increased difficulty of the problem. Further analysis can be seen in Appendix A.3. Both were removed from Table 1.

Table 2: Comparison between Unit. Scal, fixed weights, and the best performing SMTO. Well chosen fixed weights can achieve comparable performance to SMTOs.

<table><tr><td>Dataset</td><td>Unit. Scal.</td><td>Fixed Weights $\Delta_{mtm}(\%) \uparrow$ </td><td>Best SMTO</td></tr><tr><td>MNIST</td><td>-8.952</td><td>-3.251</td><td>-2.137</td></tr><tr><td>City 19C</td><td>-5.312</td><td>0.145</td><td>0.570</td></tr><tr><td>QM9</td><td>-135.7</td><td>-53.66</td><td>-53.62</td></tr></table>

## 3.5 Fixed weights

Though fixed weights are the most efficient and straightforward solution, choosing the correct values for the weights of each task is a challenging problem. Xin et al. [23] finds the ideal, or close to ideal, weights by performing a grid search. This process may be viable for a more limited scenario, but for a large number of tasks it is not feasible. Very recently, Royer et al. [45] proposed a population-based training to find the ideal weights efficiently, though this method was not evaluated in this work due to time constraints.

One recent critique of SMTOs raises the question of whether optimized scalar weights can compete with SMTOs [23]. However, as discussed previously, the problems used in the critique might not adequately represent the performance of SMTOs, so we evaluate this claim on our experiments.

Finding well optimized weights for all problems using grid search would be unfeasible, so instead, we opt for a simpler strategy, using the weights output from an SMTO to weight each task loss. More specifically, the weights from CAGrad, EDM and Nash-MTL optimizers are used for the MNIST, Cityscapes and QM9 datasets, respectively. As we are using the Adam optimizer, there is a normalizing effect on the gradients, so the gradients in each decoder are not severely affected by the different norms caused by the loss weighting.

To select the weights, we computed the mean value in each epoch, applied an exponential moving average with $\beta \stackrel { - } { = } 0 . 9$ , and used the result from the last epoch to define the weights. To extract and train with the fixed weights, we used the optimal hyperparameter configuration from the respective SMTO. However, for the QM9 dataset the fixed weights demonstrated to be unstable so we used the ones from $l r = 0 . 0 0 0 1$ , instead of the optimal $l r = 0 . 0 0 1$ . The model is trained the same number of times as its SMTO counterpart.

The results obtained from fixed weights, summarized in Table 2, indicate that it is indeed possible to get competitive performance using fixed weights. However, fixed weights can be more unstable, as shown in Figure 5 by the fixed weights extracted from $l r = 0 . 0 0 1$ , with results ranging from $\approx - 5 0 \%$ to $\approx - 9 5 \%$ . Surprisingly, the optimal learning rate weights were not the best option for the QM9 dataset.

This raises the question of why fixed weights can achieve comparable performance to SMTOs, as generating dynamic weights is a major benefit of SMTOs. Analyzing the weight value behavior in Figure 6, it is clear that all weights converge to a fixed value later in training. Additionally, due to the over-parameterized nature of deep learning models [46] the limitations typically associated with under-parameterized settings, such as the claim that scalarization is generally incapable of tracing out the Pareto front, do not necessarily apply [47]. We believe that further exploring the interaction between over-parameterization and multi-task learning could be a promising direction for future work.

![](images/c2c160a12fa559640437f9bafc370febf05926ed692f1b05cc40b1b63799f27d.jpg)

<details>
<summary>boxplot</summary>

| Category | Q1 | Q2 (Median) | Q3 | Min | Max |
| --- | --- | --- | --- | --- | --- |
| Nash-MTL | ~-58 | ~-57 | ~-50 | ~-60 | ~-45 |
| 0.005 | ~-76 | ~-75 | ~-69 | ~-81 | ~-67 |
| 0.001 | ~-76 | ~-58 | ~-54 | ~-93 | ~-51 |
| 0.0005 | ~-70 | ~-59 | ~-57 | ~-81 | ~-54 |
| 0.0001 | ~-57 | ~-54 | ~-50 | ~-61 | ~-46 |
</details>

Figure 5: Box plot comparing $\Delta _ { \mathrm { m t m } }$ between the original Nash-MTL optimizer and their extracted weights from each learning configuration. Fixed weights can achieve results comparable to those of an SMTO. However, they can be significantly more unstable than a dynamic SMTO, as was the case for the weights from $l r = 0 . 0 0 1$

![](images/5447d292b226a36167288c1ddace6c01a1321f11b197c4726611d61dd3d04a93.jpg)

<details>
<summary>line</summary>

| Epoch | \(\mu (\)Weight value) | \(\epsilon_\)homo (Weight value) | \(R^{2} (\)Weight value) | U₀ (Weight value) | H (Weight value) | Cv (Weight value) | \(\alpha (\)Weight value) | \(\epsilon_\)lumo (Weight value) | ZPVE (Weight value) | U (Weight value) | G (Weight value) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | ~0.12 | ~0.12 | ~0.12 | ~0.12 | ~0.12 | ~0.12 | ~0.12 | ~0.12 | ~0.12 | ~0.12 | ~0.12 |
| 100 | ~0.12 | ~0.12 | ~0.12 | ~0.12 | ~0.12 | ~0.12 | ~0.12 | ~0.12 | ~0.28 | ~0.12 | ~0.12 |
| 250 | ~0.12 | ~0.12 | ~0.12 | ~0.12 | ~0.12 | ~0.12 | ~0.12 | ~0.12 | ~0.30 | ~0.12 | ~0.12 |
| 500 | ~0.12 | ~0.12 | ~0.12 | ~0.12 | ~0.12 | ~0.12 | ~0.12 | ~0.12 | ~0.32 | ~0.12 | ~0.12 |
| 750 | ~0.12 | ~0.12 | ~0.12 | ~0.12 | ~0.12 | ~0.12 | ~0.12 | ~0.12 | ~0.33 | ~0.12 | ~0.12 |
| 1000 | ~0.12 | ~0.12 | ~0.12 | ~0.12 | ~0.12 | ~0.12 | ~0.12 | ~0.12 | ~0.34 | ~0.12 | ~0.12 |
| 1250 | ~0.12 | ~0.12 | ~0.12 | ~0.12 | ~0.12 | ~0.12 | ~0.12 | ~0.12 | ~0.35 | ~0.12 | ~0.12 |
</details>

Figure 6: Normalized weights from SMTOs. The shaded regions represent the mean value at each epoch, and the dotted lines show the results of the exponential moving average. The top left, top right, and bottom plots correspond to the MNIST, Cityscapes, and QM9 datasets, respectively. All datasets exhibit similar behavior, with the weights gradually converging to a stable value on average.

This shows that using fixed weights is a valid alternative for complex MTL problems, corroborating the findings of Xin et al. [23]. However, finding those weights is highly costly, and with the advancement of loss-based SMTOs such as FAMO [20] and GO4Align [21], it is no longer a desirable choice.

## 4 Limitations

In this work, we extensively tested multiple SMTOs on multiple configurations and datasets. However, it is important to note that we only focused on the supervised setting and did not include problems from NLP or reinforcement learning, fields in which MTL is commonly evaluated [24, 23]. Also, our initial selection process involved filtering certain SMTOs based on their norm balancing performance on the MNIST classification-reconstruction problem. Despite this strong preliminary evidence, a more exhaustive analysis of the excluded methods may reveal unknown advantages. Furthermore, some SMTOs were reported to provide better results when combining them with other SMTOs [12, 18], but because of time constraints, we limited the experiments to only the vanilla version. Finally, we only considered the MTL problem, where all tasks are equally weighted, and did not consider scenarios designed for auxiliary tasks.

## 5 Conclusion

This work aimed to address recent critiques of SMTOs by conducting a comprehensive empirical analysis under the context of supervised learning. First, we examined why Unit. Scal. can replicate the performance of SMTOs in two-task scenarios, particularly when the tasks are similar. Our findings suggest that SMTOs converge to equal weights, so Unit. Scal. approximates their behavior on average. Furthermore, we explored the conditions under which Unit. Scal. matches SMTOs, attributing its success to factors such as the degree of task interference, the number of tasks, and the amount of information to be learned.

We also investigated the relative performance of fixed weights compared to SMTOs. By deriving fixed weights from SMTOs, we demonstrated that fixed weights can perform competitively in complex scenarios. However, using fixed weights is less appealing due to the computational overhead of identifying optimal values and their comparable performance to recent loss-based SMTOs.

In conclusion, this study provides a robust evaluation of current SMTOs, underscoring their significance in addressing complex multi-task problems. Additionally, we aim to establish points to consider when evaluating SMTOs and to guide the development of future methods and applications.

## Acknowledgments

This work was partially funded by CNPq under grants 465755/2014-3, 309532/2023-0, and 130961/2024-8, and CAPES under finance code 001.

## References

[1] Rich Caruana. Multitask learning. Machine Learning, 28(1):41–75, 1997.  
[2] Jonathan Baxter. A model of inductive bias learning. Journal ofartificial intelligence research, 12:149–198, 2000.  
[3] Ishan Misra, Abhinav Shrivastava, Abhinav Gupta, and Martial Hebert. Cross-stitch networks for multi-task learning. In Proceedings ofthe IEEE conference on computer vision and pattern recognition, pages 3994–4003, 2016.  
[4] Yuan Gao, Jiayi Ma, Mingbo Zhao, Wei Liu, and Alan L Yuille. Nddr-cnn: Layerwise feature fusing in multi-task cnns by neural discriminative dimensionality reduction. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 3205–3214, 2019.  
[5] Shikun Liu, Edward Johns, and Andrew J Davison. End-to-end multi-task learning with attention. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 1871–1880, 2019.  
[6] Trevor Standley, Amir Zamir, Dawn Chen, Leonidas Guibas, Jitendra Malik, and Silvio Savarese. Which tasks should be learned together in multi-task learning? In International Conference on Machine Learning, pages 9120–9132. PMLR, 2020.  
[7] Chris Fifty, Ehsan Amid, Zhe Zhao, Tianhe Yu, Rohan Anil, and Chelsea Finn. Efficiently identifying task groupings for multi-task learning. Advances in Neural Information Processing Systems, 34:27503–27516, 2021.  
[8] Timothy Hospedales, Antreas Antoniou, Paul Micaelli, and Amos Storkey. Meta-learning in neural networks: A survey. IEEE transactions on pattern analysis and machine intelligence, 44 (9):5149–5169, 2021.  
[9] Ozan Sener and Vladlen Koltun. Multi-task learning as multi-objective optimization. Advances in neural information processing systems, 31, 2018.  
[10] Aviv Navon, Aviv Shamsian, Idan Achituve, Haggai Maron, Kenji Kawaguchi, Gal Chechik, and Ethan Fetaya. Multi-task learning as a bargaining game. In Kamalika Chaudhuri, Stefanie Jegelka, Le Song, Csaba Szepesvari, Gang Niu, and Sivan Sabato, editors, Proceedings ofthe 39th International Conference on Machine Learning, volume 162 of Proceedings ofMachine Learning Research, pages 16428–16446. PMLR, 17–23 Jul 2022.  
[11] Liyang Liu, Yi Li, Zhanghui Kuang, Jing-Hao Xue, Yimin Chen, Wenming Yang, Qingmin Liao, and Wayne Zhang. Towards impartial multi-task learning. In International Conference on Learn ing Representations, 2021. URL https://openreview.net/forum?id=IMPnRXEWpvr.  
[12] Adrián Javaloy and Isabel Valera. Rotograd: Gradient homogenization in multitask learning. In International Conference on Learning Representations, 2022.  
[13] Zhao Chen, Vijay Badrinarayanan, Chen-Yu Lee, and Andrew Rabinovich. Gradnorm: Gradient normalization for adaptive loss balancing in deep multitask networks. In International conference on machine learning, pages 794–803. PMLR, 2018.  
[14] Alexandr Katrutsa, Daniil Merkulov, Nurislam Tursynbek, and Ivan Oseledets. Follow the bisector: a simple method for multi-objective optimization. arXiv preprint arXiv:2007.06937, 2020.  
[15] Zhao Chen, Jiquan Ngiam, Yanping Huang, Thang Luong, Henrik Kretzschmar, Yuning Chai, and Dragomir Anguelov. Just pick a sign: Optimizing deep multitask models with gradient sign dropout. Advances in Neural Information Processing Systems, 33:2039–2050, 2020.  
[16] Angelica Tiemi Mizuno Nakamura, Valdir Grassi Jr, and Denis Fernando Wolf. Leveraging convergence behavior to balance conflicting tasks in multi-task learning. Neurocomputing, 511: 43–53, 2022.  
[17] Roberto Cipolla, Yarin Gal, and Alex Kendall. Multi-task learning using uncertainty to weigh losses for scene geometry and semantics. In 2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 7482–7491, 2018. doi: 10.1109/CVPR.2018.00781.  
[18] Shikun Liu, Stephen James, Andrew Davison, and Edward Johns. Auto-lambda: Disentangling dynamic task relationships. Transactions on Machine Learning Research, 2022. ISSN 2835- 8856. URL https://openreview.net/forum?id=KKeCMim5VN.  
[19] Baijiong Lin, Feiyang YE, Yu Zhang, and Ivor Tsang. Reasonable effectiveness of random weighting: A litmus test for multi-task learning. Transactions on Machine Learning Research, 2022. ISSN 2835-8856. URL https://openreview.net/forum?id=jjtFD8A1Wx.  
[20] Bo Liu, Yihao Feng, Peter Stone, and Qiang Liu. Famo: Fast adaptive multitask optimization. Advances in Neural Information Processing Systems, 36:57226–57243, 2023.  
[21] Jiayi Shen, Cheems Wang, Zehao Xiao, Nanne Van Noord, and Marcel Worring. GO4align: Group optimization for multi-task alignment. In The Thirty-eighth Annual Conference on Neural Information Processing Systems, 2024.  
[22] Idan Achituve, Idit Diamant, Arnon Netzer, Gal Chechik, and Ethan Fetaya. Bayesian uncertainty for gradient aggregation in multi-task learning. In Forty-first International Conference on Machine Learning, 2024.  
[23] Derrick Xin, Behrooz Ghorbani, Justin Gilmer, Ankush Garg, and Orhan Firat. Do current multi-task optimization methods in deep learning even help? Advances in Neural Information Processing Systems, 35:13597–13609, 2022.  
[24] Vitaly Kurin, Alessandro De Palma, Ilya Kostrikov, Shimon Whiteson, and Pawan K Mudigonda. In defense of the unitary scalarization for deep multi-task learning. Advances in Neural Information Processing Systems, 35:12169–12183, 2022.  
[25] Yann LeCun, Léon Bottou, Yoshua Bengio, and Patrick Haffner. Gradient-based learning applied to document recognition. Proceedings ofthe IEEE, 86(11):2278–2324, 1998.  
[26] Marius Cordts, Mohamed Omran, Sebastian Ramos, Timo Rehfeld, Markus Enzweiler, Rodrigo Benenson, Uwe Franke, Stefan Roth, and Bernt Schiele. The cityscapes dataset for semantic urban scene understanding. In Proc. ofthe IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2016.  
[27] Zhenqin Wu, Bharath Ramsundar, Evan N Feinberg, Joseph Gomes, Caleb Geniesse, Aneesh S Pappu, Karl Leswing, and Vijay Pande. Moleculenet: a benchmark for molecular machine learning. Chemical science, 9(2):513–530, 2018.  
[28] Bo Liu, Xingchao Liu, Xiaojie Jin, Peter Stone, and Qiang Liu. Conflict-averse gradient descent for multi-task learning. Advances in Neural Information Processing Systems, 34:18878–18890, 2021.  
[29] Tom Dietterich. Overfitting and undercomputing in machine learning. ACM computing surveys (CSUR), 27(3):326–327, 1995.  
[30] Nitish Shirish Keskar, Dheevatsa Mudigere, Jorge Nocedal, Mikhail Smelyanskiy, and Ping Tak Peter Tang. On large-batch training for deep learning: Generalization gap and sharp minima. International Conference on Learning Representations, 2016.  
[31] Bobby Kleinberg, Yuanzhi Li, and Yang Yuan. An alternative view: When does sgd escape local minima? In International conference on machine learning, pages 2698–2707. PMLR, 2018.  
[32] Sergey Ioffe and Christian Szegedy. Batch normalization: Accelerating deep network training by reducing internal covariate shift. In International conference on machine learning, pages 448–456. pmlr, 2015.  
[33] Nitish Srivastava, Geoffrey Hinton, Alex Krizhevsky, Ilya Sutskever, and Ruslan Salakhutdinov. Dropout: a simple way to prevent neural networks from overfitting. The journal of machine learning research, 15(1):1929–1958, 2014.  
[34] Simon Vandenhende, Stamatios Georgoulis, Wouter Van Gansbeke, Marc Proesmans, Dengxin Dai, and Luc Van Gool. Multi-task learning for dense prediction tasks: A survey. IEEE transactions on pattern analysis and machine intelligence, 44(7):3614–3633, 2021.  
[35] Tianhe Yu, Saurabh Kumar, Abhishek Gupta, Sergey Levine, Karol Hausman, and Chelsea Finn. Gradient surgery for multi-task learning. Advances in Neural Information Processing Systems, 33:5824–5836, 2020.  
[36] Michelle Guo, Albert Haque, De-An Huang, Serena Yeung, and Li Fei-Fei. Dynamic task prioritization for multitask learning. In Proceedings of the European conference on computer vision (ECCV), pages 270–287, 2018.  
[37] Baijiong Lin, Weisen Jiang, Feiyang Ye, Yu Zhang, Pengguang Chen, Ying-Cong Chen, and Shu Liu. A scale-invariant task balancing approach for multi-task learning. CoRR, abs/2308.12029, 2023.  
[38] Baijiong Lin, Weisen Jiang, Feiyang Ye, Yu Zhang, Pengguang Chen, Ying-Cong Chen, Shu Liu, and James T. Kwok. Dual-balancing for multi-task learning, 2023.  
[39] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 770–778, 2016.  
[40] Liang-Chieh Chen, Yukun Zhu, George Papandreou, Florian Schroff, and Hartwig Adam. Encoder-decoder with atrous separable convolution for semantic image segmentation. In Proceedings of the European conference on computer vision (ECCV), pages 801–818, 2018.  
[41] Sen Wu, Hongyang R. Zhang, and Christopher Ré. Understanding and improving information transfer in multi-task learning. In International Conference on Learning Representations, 2020. URL https://openreview.net/forum?id=SylzhkBtDB.  
[42] Uri Shaham, Maha Elbayad, Vedanuj Goswami, Omer Levy, and Shruti Bhosale. Causes and cures for interference in multilingual translation. arXiv preprint arXiv:2212.07530, 2022.  
[43] Matthias Fey and Jan E. Lenssen. Fast graph representation learning with PyTorch Geometric. In ICLR Workshop on Representation Learning on Graphs and Manifolds, 2019.  
[44] Justin Gilmer, Samuel S Schoenholz, Patrick F Riley, Oriol Vinyals, and George E Dahl. Neural message passing for quantum chemistry. In International conference on machine learning, pages 1263–1272. PMLR, 2017.  
[45] Amelie Royer, Tijmen Blankevoort, and Babak Ehteshami Bejnordi. Scalarization for multi-task and multi-domain learning at scale. Advances in Neural Information Processing Systems, 36, 2024.  
[46] Jonathan Frankle and Michael Carbin. The lottery ticket hypothesis: Finding sparse, trainable neural networks. In International Conference on Learning Representations, 2019. URL https://openreview.net/forum?id=rJl-b3RcF7.  
[47] Yuzheng Hu, Ruicheng Xian, Qilong Wu, Qiuling Fan, Lang Yin, and Han Zhao. Revisiting scalarization in multi-task learning: A theoretical perspective. In Thirty-seventh Conference on Neural Information Processing Systems, 2023. URL https://openreview.net/forum?id= 6EqUpqMnwl.  
[48] Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory Chanan, Trevor Killeen, Zeming Lin, Natalia Gimelshein, Luca Antiga, Alban Desmaison, Andreas Kopf, Edward Yang, Zachary DeVito, Martin Raison, Alykhan Tejani, Sasank Chilamkurthy, Benoit Steiner, Lu Fang, Junjie Bai, and Soumith Chintala. Pytorch: An imperative style, highperformance deep learning library. In H. Wallach, H. Larochelle, A. Beygelzimer, F. d'Alché- Buc, E. Fox, and R. Garnett, editors, Advances in Neural Information Processing Systems, volume 32. Curran Associates, Inc., 2019. URL https://proceedings.neurips.cc/paper\_ files/paper/2019/file/bdbca288fee7f92f2bfa9f7012727740-Paper.pdf.

![](images/a5e0f8af78ddf7b40855aea7ac5d27322306db370c2b9c4d085d50b3ad7657e5.jpg)

<details>
<summary>line</summary>

| Training (%) | QM9 | MNIST CL_RR | City R50 | City Inst. Seg. | City 19C | MNIST CL_CR | MNIST RL_RR | City R18 | City HR |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | ~0.47 | ~0.62 | ~0.73 | ~0.62 | ~0.62 | ~0.68 | ~0.80 | ~0.73 | ~0.73 |
| 20 | ~0.43 | ~0.58 | ~0.68 | ~0.45 | ~0.45 | ~0.73 | ~0.71 | ~0.71 | ~0.45 |
| 40 | ~0.35 | ~0.58 | ~0.68 | ~0.45 | ~0.45 | ~0.68 | ~0.68 | ~0.68 | ~0.45 |
| 60 | ~0.28 | ~0.58 | ~0.68 | ~0.45 | ~0.45 | ~0.68 | ~0.68 | ~0.68 | ~0.45 |
| 80 | ~0.28 | ~0.58 | ~0.68 | ~0.45 | ~0.45 | ~0.68 | ~0.68 | ~0.68 | ~0.45 |
| 100 | ~0.32 | ~0.58 | ~0.68 | ~0.45 | ~0.45 | ~0.68 | ~0.68 | ~0.68 | ~0.45 |
</details>

Figure 7: Mean cosine similarity between each task gradient and the average direction for all problems. It was able to correctly rank all the different problems in terms of relative performance between Unit. Scal. and SMTOs. (i.e. higher cosine similarity relates to lower difference between Unit. Scal. and SMTOs.)

## A Technical Appendices and Supplementary Material

## A.1 Experimental Setup

All experiments were implemented using PyTorch [48] and executed on a machine equipped with an AMD Ryzen 9 5950X CPU and two NVIDIA RTX 3090 GPUs (24 GB each). However, experiments were run independently on each GPU. The random seed was set based on the corresponding run index.

## A.2 Additional results

## A.2.1 Cosine similarity

Figure 7 demonstrates the strong correlation between the mean cosine similarity of task gradients and the relative performance of Unit. Scal. and SMTOs. Notably, the tasks where Unit. Scal. achieved comparable results to SMTOs (MNIST CL\_CR, MNIST RL\_RR, City R50, and City R18) exhibit high mean cosine similarities (≈0.7) and form a distinct cluster. Conversely, the more challenging Cityscapes tasks (City Inst. Seg., City HR, City 19C) show a lower, yet consistent, similarity (≈0.45). Finally, the most difficult task from the QM9 dataset, is characterized by the lowest mean cosine similarity (≈0.3).

## A.2.2 Statistics

Similarly to [12], we provide additional statistics in Tables 3, 4, and 5 and the missing visual result in Figure 8. Most notably, we see a clear preference for the classification task on the MNIST dataset from the underperforming SMTOs, as well as a lower minimum value and higher standard deviation in Table 5 for Unit. Scal. when compared to the top performing SMTOs, FAMO and Nash-MTL.

Table 3: Mean $\Delta _ { m t m } ( \% )$ ↑ of different SMTOs on the MNIST dataset.

<table><tr><td>SMTO</td><td>CL</td><td>RR</td><td>avg</td></tr><tr><td>Unit. Scal.</td><td>-1.043</td><td>-16.86</td><td>-8.952</td></tr><tr><td>Auto-Lambda</td><td>-1.232</td><td>-7.186</td><td>-4.209</td></tr><tr><td>CAGrad</td><td>-1.685</td><td>-2.590</td><td>-2.137</td></tr><tr><td>CDTT</td><td>-1.546</td><td>-3.596</td><td>-2.571</td></tr><tr><td>EDM</td><td>-1.613</td><td>-2.993</td><td>-2.303</td></tr><tr><td>FAMO</td><td>-4.587</td><td>-0.745</td><td>-2.666</td></tr><tr><td>GradDrop</td><td>-1.076</td><td>-22.04</td><td>-11.56</td></tr><tr><td>IMTL</td><td>-2.513</td><td>-1.975</td><td>-2.244</td></tr><tr><td>MGDA-UB</td><td>-1.448</td><td>-11.26</td><td>-6.352</td></tr><tr><td>Nash-MTL</td><td>-1.695</td><td>-4.430</td><td>-3.063</td></tr><tr><td>PCGrad</td><td>-1.039</td><td>-15.36</td><td>-8.201</td></tr><tr><td>RLW-Dirichlet</td><td>-1.145</td><td>-19.48</td><td>-10.31</td></tr><tr><td>RLW-Normal</td><td>-1.100</td><td>-18.53</td><td>-9.817</td></tr><tr><td>RotoGrad</td><td>-2.705</td><td>-1.572</td><td>-2.139</td></tr><tr><td>SI</td><td>-0.962</td><td>-27.47</td><td>-14.21</td></tr><tr><td>UW</td><td>-1.109</td><td>-26.30</td><td>-13.70</td></tr></table>

Table 4: Mean $\Delta _ { m t m } ( \% )$ ↑ of different SMTOs on the Cityscapes dataset.

<table><tr><td rowspan="2">SMTO</td><td colspan="2">Segmentation</td><td colspan="2">Disparity</td><td>Instance Seg.</td><td rowspan="2">avg</td></tr><tr><td>Acc</td><td>mIoU</td><td>L1 abs</td><td>L1 rel</td><td>L1 abs</td></tr><tr><td>Unit. Scal.</td><td>-2.120</td><td>-16.78</td><td>-8.542</td><td>-11.41</td><td>3.492</td><td>-5.312</td></tr><tr><td>Auto-Lambda</td><td>-1.922</td><td>-15.02</td><td>-8.546</td><td>-9.317</td><td>3.657</td><td>-4.582</td></tr><tr><td>CAGrad</td><td>-0.977</td><td>-7.647</td><td>-6.709</td><td>-10.06</td><td>3.325</td><td>-3.124</td></tr><tr><td>CDTT</td><td>-0.948</td><td>-8.954</td><td>2.043</td><td>-0.224</td><td>0.472</td><td>-1.190</td></tr><tr><td>EDM</td><td>-0.225</td><td>-3.174</td><td>0.990</td><td>-1.051</td><td>2.468</td><td>0.246</td></tr><tr><td>FAMO</td><td>-0.274</td><td>-3.268</td><td>0.755</td><td>-2.212</td><td>2.339</td><td>-0.054</td></tr><tr><td>IMTL</td><td>-0.224</td><td>-2.759</td><td>3.017</td><td>0.993</td><td>1.195</td><td>0.570</td></tr><tr><td>Nash-MTL</td><td>-0.290</td><td>-3.275</td><td>0.297</td><td>-0.227</td><td>1.409</td><td>-0.113</td></tr><tr><td>RotoGrad</td><td>-0.453</td><td>-4.484</td><td>0.303</td><td>-3.493</td><td>-1.659</td><td>-1.908</td></tr></table>

Table 5: Performance of different SMTOs on the QM9 dataset based on $\Delta _ { m t m }$ (%) statistics.

<table><tr><td>SMTO</td><td>min ↑</td><td>max ↑</td><td>med ↑</td><td>std ↓</td><td>avg ↑</td></tr><tr><td>Unit. Scal.</td><td>-849.3</td><td>-8.280</td><td>-61.62</td><td>229.4</td><td>-135.7</td></tr><tr><td>Auto-Lambda</td><td>-974.5</td><td>-18.32</td><td>-115.4</td><td>255.5</td><td>-186.9</td></tr><tr><td>CAGrad</td><td>-408.0</td><td>-5.451</td><td>-32.08</td><td>108.7</td><td>-70.13</td></tr><tr><td>CDTT</td><td>-652.9</td><td>-30.90</td><td>-70.34</td><td>169.7</td><td>-121.2</td></tr><tr><td>EDM</td><td>-526.0</td><td>-24.14</td><td>-56.26</td><td>137.0</td><td>-95.42</td></tr><tr><td>FAMO</td><td>-351.5</td><td>6.590</td><td>-16.95</td><td>97.78</td><td>-56.86</td></tr><tr><td>Nash-MTL</td><td>-390.2</td><td>15.10</td><td>-21.41</td><td>108.8</td><td>-53.62</td></tr></table>

![](images/e72f4c4b86dbb73d75df9595f4e4a9fa893c2bcc8b5d2729275df7ec58447454.jpg)

<details>
<summary>boxplot</summary>

| Method | Q1 | Q2 (Median) | Q3 | Min | Max |
| --- | --- | --- | --- | --- | --- |
| Unit. Scal. | ~-140 | ~-138 | ~-135 | ~-142 | ~-125 |
| Auto-Lambda | ~-215 | ~-200 | ~-160 | ~-235 | ~-130 |
| CAGrad | ~-72 | ~-70 | ~-68 | ~-73 | ~-65 |
| CDTT | ~-128 | ~-120 | ~-110 | ~-138 | ~-105 |
| EDM | ~-95 | ~-92 | ~-90 | ~-102 | ~-88 |
| FAMO | ~-55 | ~-53 | ~-52 | ~-62 | ~-50 |
| Nash-MTL | ~-55 | ~-53 | ~-50 | ~-58 | ~-45 |
</details>

Figure 8: Box plot comparing $\Delta _ { m t m }$ between Unit. Scal. and SMTOs on the QM9 dataset.

## A.3 Rotograd’s instability on the QM9 dataset

The RotoGrad optimizer aims to mitigate conflicts in gradient directions by rotating the feature space so that the gradient of each task’s gradient aligns with the direction formed by the equally weighted sum of the normalized gradients: $\begin{array} { r } { v = \frac { 1 } { N } \sum _ { i } ^ { N } u _ { s h , i } . } \end{array}$ , where $\begin{array} { r } { u _ { s h , i } = \frac { g _ { s h , i } } { \left\| g _ { s h , i } \right\| } } \end{array}$ represents the normalized gradient of task i for the shared parameters.

In the specific case of the QM9 experiment, RotoGrad failed to converge. We hypothesize that the relatively small final shared feature space, combined with the high number of tasks, significantly increased the difficulty of the balancing problem. As shown in Figure 9, the rotation loss $\mathcal { L } _ { \mathit { r o t } }$ is orders of magnitude higher compared to the MNIST and Cityscapes problems. Since $\mathcal { L } _ { \mathit { r o t } }$ does not directly enforce a reduction in task loss, this instability may have negatively affected the model optimization.

For this specific experiment, we used $l r = 0 . 0 0 1$ as it was the optimal learning rate for most SMTOs, and $l r r = 0 .$ 1 since it was the smallest value used on the grid search, so ideally it should be the most stable.

![](images/36f796af493f69f797656aa00f6cca225d42a9e245ea655b46d6968103fb4180.jpg)

<details>
<summary>line</summary>

| Training (%) | MNIST CL_RR (Rotation Loss) | City 19C (Rotation Loss) | QM9 (Rotation Loss) |
| --- | --- | --- | --- |
| 0 | ~3e-4 | ~7e-7 | ~5e-3 |
| 20 | ~4e-4 | ~1.5e-6 | ~7e-3 |
| 40 | ~4e-4 | ~1.5e-6 | ~3e-2 |
| 60 | ~4e-4 | ~1.5e-6 | ~5e-2 |
| 80 | ~4e-4 | ~1.5e-6 | ~4e-2 |
| 100 | ~4e-4 | ~1.5e-6 | ~5e-2 |
</details>

Figure 9: Mean rotation loss $( \mathcal { L } _ { r o t } )$ from RotoGrad. The $\mathcal { L } _ { \mathit { r o t } }$ is at least 10× higher than the other cases, which could explain the training instability.

## A.4 Proofs

Assumption 1. Consider a model $f _ { \theta }$ composed ofa shared encoder $f _ { \theta _ { s h } }$ and task-specific modules $f _ { \theta _ { t } }$ for each task $t \in \{ 1 , \ldots , N \}$ . In the case of two tasks $( N = 2 ) ,$ we assume the following symmetry conditions: (i) both tasks use the same loss function, (ii) the task-specific modules have the same architecture, (iii) the tasks are of similar difficulty, and (iv) the tasks have labels with similar distributions. Under these conditions, we assume that the gradients with respect to the shared parameters have equal norm, $i . e . , \| g _ { 1 } \| = \| g _ { 2 } \|$

Assumption 2. Under the same conditions as Assumption 1, assume that the loss values ofeach task are equal, i.e., $\ell _ { 1 } = \ell _ { 2 }$

Although Assumptions 1, 2 impose a strong symmetry requirement, our experiments on MNIST revealed that in both the dual classification and dual reconstruction tasks, the resulting weights from all SMTOs were consistently close to $0 . 5 ,$ with the sole exception of FAMO in the dual classification scenario.

Proposition 1. Under Assumption $^ { l , }$ the optimal weights computed by CAGrad correspond to equal weighting.

Proof. We consider the CAGrad optimization for two tasks:

$$
w ^ {*} = \underset {w \in \mathcal {W}} {\arg \min} g _ {w} ^ {\top} g _ {0} + \sqrt {\phi} \| g _ {w} \|,
$$

where:

$$
\begin{array}{l} g _ {w} = w _ {1} g _ {1} + w _ {2} g _ {2}, \\ g _ {0} = \frac {1}{2} (g _ {1} + g _ {2}), \\ \phi = c ^ {2} \| g _ {0} \| ^ {2}, \quad c \in [ 0, 1), \\ \mathcal {W} = \{w \in \mathbb {R} ^ {2} \mid w _ {1} + w _ {2} = 1, w _ {1}, w _ {2} \geq 0 \}. \\ \end{array}
$$

We also define:

$$
A := \| g _ {1} \| = \| g _ {2} \|,
$$

$$
p := g _ {1} ^ {\top} g _ {2}.
$$

We begin by expanding the inner product:

$$
\begin{array}{l} g _ {w} ^ {\top} g _ {0} = (w _ {1} g _ {1} + w _ {2} g _ {2}) ^ {\top} \frac {1}{2} (g _ {1} + g _ {2}) \\ = \frac {1}{2} \left[ w _ {1} (g _ {1} ^ {\top} g _ {1} + g _ {1} ^ {\top} g _ {2}) + w _ {2} (g _ {2} ^ {\top} g _ {1} + g _ {2} ^ {\top} g _ {2}) \right] \\ = \frac {1}{2} \left[ w _ {1} (A ^ {2} + p) + w _ {2} (p + A ^ {2}) \right] \\ = \frac {1}{2} (A ^ {2} + p) (w _ {1} + w _ {2}) = \frac {1}{2} (A ^ {2} + p). \\ \end{array}
$$

This term is independent of $w ,$ as well as ${ \sqrt { \phi } } ,$ so the optimization reduces to:

$$
\underset {w \in \mathcal {W}} {\arg \min} \| w _ {1} g _ {1} + w _ {2} g _ {2} \|.
$$

Let $w ^ { * }$ be the optimal solution, which is known to be equal weights if $\| g _ { 1 } \| = \| g _ { 2 } \|$ . The CAGrad update direction is then given by:

$$
d ^ {*} = g _ {0} + \frac {\sqrt {\phi}}{\| g _ {w ^ {*}} \|}   g _ {w ^ {*}}.
$$

If $\begin{array} { r } { w ^ { * } = ( \frac { 1 } { 2 } , \frac { 1 } { 2 } ) } \end{array}$ , then $\begin{array} { r } { g _ { w ^ { * } } = \frac { 1 } { 2 } ( g _ { 1 } + g _ { 2 } ) = g _ { 0 } } \end{array}$ , and:

$$
d ^ {*} = g _ {0} + \frac {c \| g _ {0} \|}{\| g _ {0} \|}   g _ {0} = (1 + c)   g _ {0} = \sum_ {i = 1} ^ {2} g _ {i} \left(\frac {1 + c}{2}\right).
$$

Thus, $d ^ { * }$ is an equal-weight combination of the gradients.

![](images/1c677beac365a46dcdd14e9203d2436c95c2ad064879a530535b3a305da764df.jpg)

Proposition 2. Under Assumption 1, the optimal weights computed by Nash-MTL [10] are equal weights in the two-task scenario.

Proof. The weights are computed by approximating the solution of:

$$
G ^ {\top} G \alpha = \frac {1}{\alpha},
$$

where G is the matrix whose columns are the shared gradients and $\alpha _ { i } > 0 , i \in \{ 1 , . . . , N \}$ the tasks weights. In a two-task scenario, the problem can be simplified to:

$$
\left\{ \begin{array}{l} \alpha_ {1} g _ {1} ^ {\top} g _ {1} + \alpha_ {2} g _ {2} ^ {\top} g _ {1} = \frac {1}{\alpha_ {1}} \\ \alpha_ {2} g _ {2} ^ {\top} g _ {2} + \alpha_ {1} g _ {1} ^ {\top} g _ {2} = \frac {1}{\alpha_ {2}} \end{array} \right.
$$

Defining $A : = \| g _ { 1 } \| = \| g _ { 2 } \|$ and $\begin{array} { r } { p : = \frac { g _ { 1 } ^ { \top } g _ { 2 } } { A ^ { 2 } } } \end{array}$

$$
\left\{ \begin{array}{l} \alpha_ {1} A ^ {2} + \alpha_ {2} A ^ {2} p = \frac {1}{\alpha_ {1}} \\ \alpha_ {2} A ^ {2} + \alpha_ {1} A ^ {2} p = \frac {1}{\alpha_ {2}} \end{array} \right.
$$

Rearrange the equations:

$$
\left\{ \begin{array}{l} \alpha_ {1} ^ {2} A ^ {2} + \alpha_ {1} \alpha_ {2} A ^ {2} p = 1 \\ \alpha_ {2} ^ {2} A ^ {2} + \alpha_ {1} \alpha_ {2} A ^ {2} p = 1 \end{array} \right.
$$

Notice that by symmetry of the two equations, the solution should satisfy $\alpha _ { 1 } ^ { 2 } = \alpha _ { 2 } ^ { 2 }$ . Since both weights are positive, this immediately implies:

$$
\alpha_ {1} = \alpha_ {2}.
$$

![](images/abc1f6f59d4c7c8aa3c1e3a9520fe09815cf4065955392000c5862b350f323f6.jpg)

Proposition 3. Under Assumption 1, the optimal weights computed by IMTL-G [11] are equal weights in the two-task scenario.

Proof. The weights are obtained by solving:

$$
\alpha = g _ {1} U ^ {\top} (D U ^ {\top}) ^ {- 1},
$$

where $\alpha = \{ \alpha _ { 2 } , . . . , \alpha _ { N } \}$ is constrained to $\begin{array} { r } { \sum _ { i } \alpha _ { i } \ = \ 1 , \ U ^ { \top } \ = \ [ u _ { 1 } ^ { \top } \ - u _ { 2 } ^ { \top } , . . . , u _ { 1 } ^ { \top } \ - u _ { N } ^ { \top } ] } \end{array}$ , with $u _ { i } = g _ { i } / \lVert g _ { i } \rVert$ , and $D ^ { \top } = [ g _ { 1 } ^ { \top } - g _ { 2 } ^ { \top } , . . . , g _ { 1 } ^ { \top } - g _ { N } ^ { \top } ]$ . Define $A : = \| g _ { 1 } \| = \| g _ { 2 } \|$ and $p = g _ { 1 } ^ { \top } g _ { 2 }$ . In the two-task scenario, the problem reduces to:

$$
\alpha_ {2} = g _ {1} \left(\frac {g _ {1} ^ {\top} - g _ {2} ^ {\top}}{A}\right) \left((g _ {1} - g _ {2}) \frac {(g _ {1} ^ {\top} - g _ {2} ^ {\top})}{A}\right) ^ {- 1}
$$

$$
\alpha_ {2} = g _ {1} \left(\frac {g _ {1} ^ {\top} - g _ {2} ^ {\top}}{A}\right) \left(\frac {\| g _ {1} - g _ {2} \| ^ {2}}{A}\right) ^ {- 1}
$$

$$
\alpha_ {2} = g _ {1} \left(\frac {g _ {1} ^ {\top} - g _ {2} ^ {\top}}{\| g _ {1} - g _ {2} \| ^ {2}}\right) = \frac {A ^ {2} - p}{2 (A ^ {2} - p)} = \frac {1}{2}.
$$

Since $\alpha _ { 1 } + \alpha _ { 2 } = 1$ , this completes the proof.

![](images/85519fb6bade854c1fa265bf9025a3ed308e74f98a1e0cfd04a98148eb324eef.jpg)

Proposition 4. Under assumption 1, the optimal weights computed by EDM [14] are equal weights in the two-task scenario.

Proof. EDM’s original paper [14] already showed that for the two task scenario the optimal direction is computed as follows:

$$
d ^ {*} = \left(\frac {1}{\| g _ {1} \|} + \frac {1}{\| g _ {2} \|}\right) ^ {- 1} \left(\frac {g _ {1}}{\| g _ {1} \|} + \frac {g _ {2}}{\| g _ {2} \|}\right).
$$

Define $A : = \| g _ { 1 } \| = \| g _ { 2 } \| \colon$

$$
d ^ {*} = \left(\frac {1}{A} + \frac {1}{A}\right) ^ {- 1} \left(\frac {g _ {1}}{A} + \frac {g _ {2}}{A}\right) = \frac {1}{2} (g _ {1} + g _ {2}).
$$

![](images/0f94b3143e236882acc996a9f53d6a37f8f462d879966453eab847eede4494ef.jpg)

Proposition 5. Under Assumptions 1 and 2, the optimal weights computed by CDTT [16] are equal weights.

Proof. CDTT applies a tension vector to the optimal direction $d ^ { * }$ computed by EDM, which is equal weights in this case (see Proposition 4). The final vector is computed by:

$$
\begin{array}{l} \zeta_ {i} = \frac {1}{N} \sum_ {i = 1} ^ {N} \| g _ {i} (t - i) \| \\ \delta_ {i} = \frac {\| \zeta_ {i} (t) \|}{\| \zeta_ {i} (t - 1) \|} + \log_ {1 0} (\ell_ {i}) \\ c _ {i} = \frac {\alpha}{1 + e ^ {(- \delta_ {i} e + e)}} + 1 - \alpha \\ d _ {n} ^ {*} = d ^ {*} + \sum_ {i = 1} ^ {N} c _ {i} \left(\frac {g _ {i} - d ^ {*}}{\| g _ {i} - d ^ {*} \|}\right), \\ \end{array}
$$

where t refers to the iteration index and $\alpha \in [ 0 , 1 ]$ is a constant to regulate the tension factor’s sensitivity.

It follows directly from Assumptions 1, 2 and the component’s equations that $c = c _ { 1 } = c _ { 2 }$ . Analyzing the $\| g _ { i } - d ^ { * } \|$ separately:

$$
\| g _ {i} - d ^ {*} \| ^ {2} = \| g _ {i} \| ^ {2} + \| d ^ {*} \| ^ {2} - 2 \langle g _ {i}, d ^ {*} \rangle .
$$

The direction computed by EDM $d ^ { * }$ is guaranteed to have the same angle with each of the shared gradients [14]. Therefore, if $\| g _ { 1 } \| = \ \| g _ { 2 } \|$ , then $\langle g _ { 1 } , d ^ { * } \rangle = \langle g _ { 2 } , d ^ { * } \rangle$ . Then we can consider ${ \dot { B } } : = \| g _ { 1 } - d ^ { * } \| = \| g _ { 2 } - d ^ { * } \|$ and remove it together with c from the sum:

$$
d _ {n} ^ {*} = d ^ {*} + \frac {c}{B} \sum_ {i = 1} ^ {2} (g _ {i} - d ^ {*})
$$

$$
d _ {n} ^ {*} = d ^ {*} + \frac {c}{B} (g _ {1} + g _ {2} - 2 d ^ {*})
$$

$$
d _ {n} ^ {*} = d ^ {*} = \frac {1}{2} (g _ {1} + g _ {2}).
$$

![](images/3ff10f8a0dd77e5166ae4f3ea6b8eb1d2b5d9d566d039f704ac9b529fc96c4fe.jpg)

Proposition 6. Under Assumption 2 (extended for N tasks), if the logits ${ \xi } _ { t } = ( { \xi } _ { 1 , t } , \dots , { \xi } _ { N , t } )$ are initialized equally, then the FAMO algorithm maintains equal task weights:

$$
w _ {t} = w _ {i, t} = c _ {t} \frac {z _ {i , t}}{\ell_ {i , t}},
$$

where $c _ { t } = \left( \sum _ { i = 1 } ^ { N } \frac { z _ { i , t } } { \ell _ { i , t } } \right)$ , for all $i = 1 , \ldots , N$ and for all $t .$

Proof. We prove the result by induction on the iteration t.

Base Case (t = 0): Assume that $\xi _ { i , 0 } = c$ for all $i ,$ for some constant c. Then the softmax yields

$$
z _ {i, 0} = \frac {\exp (\xi_ {i , 0})}{\sum_ {j = 1} ^ {N} \exp (\xi_ {j , 0})} = \frac {\exp (c)}{N \exp (c)} = \frac {1}{N}.
$$

The model parameters are updated as follows:

$$
\theta_ {1} = \theta_ {0} - \alpha \sum_ {i = 1} ^ {N} \left(c _ {0} \frac {z _ {i , 0}}{\ell_ {i , 0}}\right) \nabla \ell_ {i, 0},
$$

where $c _ { 0 } = \left( \sum _ { i = 1 } ^ { N } \frac { z _ { i , 0 } } { \ell _ { i , 0 } } \right) ^ { - 1 }$

Under the assumption $\ell _ { i , t } = \ell _ { t }$ for all i, each term $\frac { z _ { i , t } } { \ell _ { i , t } }$ is identical across tasks. Therefore, this update does not introduce any asymmetry among tasks. Thus, the base case holds.

Inductive Step: Suppose that at iteration t we have

$$
\xi_ {t} = (c _ {t}, c _ {t}, \dots , c _ {t}),
$$

which implies

$$
z _ {i, t} = \frac {\exp (c _ {t})}{N \exp (c _ {t})} = \frac {1}{N} \quad \text {for all} i.
$$

Next, the logits are updated via

$$
\xi_ {t + 1} = \xi_ {t} - \beta (\delta_ {t} + \gamma \xi_ {t}),
$$

with

$$
\delta_ {t} = \left[ \begin{array}{c} \nabla^ {\top} z _ {1, t} (\xi_ {t}) \\ \vdots \\ \nabla^ {\top} z _ {n, t} (\xi_ {t}) \end{array} \right] \left[ \begin{array}{c} \log \ell_ {1, t} - \log \ell_ {1, t + 1} \\ \vdots \\ \log \ell_ {n, t} - \log \ell_ {n, t + 1} \end{array} \right].
$$

Since the losses satisfy $\ell _ { i , t } = \ell _ { t }$ for every i (and similarly for $t + 1$ by the symmetry of the update), the differences log $\ell _ { i , t } - \log \ell _ { i , t + 1 }$ are the same for all tasks.

Moreover, the softmax function is given by

$$
z _ {i, t} = \frac {\exp (\xi_ {i , t})}{\sum_ {j = 1} ^ {N} \exp (\xi_ {j , t})}.
$$

The derivative of $z _ { i , t }$ with respect to $\xi _ { j }$ is

$$
\frac {\partial z _ {i}}{\partial \xi_ {j}} = z _ {i} \left(\delta_ {i j} - z _ {j}\right).
$$

where $\delta _ { i j }$ is the Kronecker delta, defined as $\delta _ { i j } = 1 \ : \mathrm { i f } \ : i = j$ and 0 otherwise. At the symmetric point $\xi _ { t } = ( c _ { t } , \ldots , c _ { t } )$ , we have $\begin{array} { r } { z _ { i } = \frac { 1 } { N } } \end{array}$ for all i, hence

$$
\frac {\partial z _ {i}}{\partial \xi_ {j}} = \frac {1}{N} \left(\delta_ {i j} - \frac {1}{N}\right).
$$

This shows that the Jacobian of the softmax is the same for every coordinate, ensuring that the scalar products involved in the definition of $\delta _ { t }$ yield the same value for every task. Let us denote this common value by D, i.e.,

$$
[ \delta_ {t} ] _ {i} = D \quad \text { for   all   } i.
$$

Since the regularization term $\gamma \xi _ { t }$ is also applied uniformly, the update for each coordinate becomes

$$
\xi_ {i, t + 1} = c _ {t} - \beta (D + \gamma c _ {t}) \quad \text { for   all   } i.
$$

Thus, $\xi _ { t + 1 } = ( c _ { t + 1 } , c _ { t + 1 } , \ldots , c _ { t + 1 } )$ remains symmetric. Evidently, the induction holds for the specific case $N = 2$

![](images/1faebd61f1603f8f49d0f4a43d0ad971de7c536e1ba4f6a30de5a5902f0b684e.jpg)

## A.5 Dataset Usage and Licensing

• MNIST: Creative Commons Attribution-Share Alike 3.0  
• Cityscapes: Custom, allows for academic and non-commercial use, https://www. cityscapes-dataset.com/license/  
• QM9: We were not able to find a license for this dataset. However the GitHub repository that manages the dataset is under an MIT license https://github.com/pyg-team/pytorch\_ geometric.

## A.6 Hyperparameters

Tables 6, 7, 8, 9, and 10 summarize the best hyperparameters for each SMTO in each scenario. Also, it shows the best SMTO specific parameters in bold. In the MNIST configuration, these parameters were searched in a broader range to better understand their impact on optimization and limit the search space in the more complex dataset. To determine which specific hyperparameters of each SMTO to optimize, we selected those explicitly optimized in the respective articles. Consequently, we kept Auto-Lambda’s initial weight, FAMO’s learning rate for task logits, Nash-MTL’s optimizer iterations, and GradDrop’s parameters fixed throughout the experiments. Additionally, for the Cityscapes dataset, RotoGrad’s feature size was set to the maximum possible value or capped at 512, the same used for the ResNet18 model trained on the original Rotograd paper.

For the hyperparameters that were optimized without a defined range, such as Auto-Lambda’s and RotoGrad’s auxiliary learning rate, we opted to define it as a scale of the learning rate (aux\_lr = lr ∗ lr\_scale).

For the specific case of the CDTT optimizer in the QM9 dataset, we opted to use the same optimal learning rate from EDM and only optimized specific hyperparameters due to time constraints. We justify this choice because CDTT was proposed as a direct improvement from EDM.

Table 6: Hyperparameters for MNIST - CL & CR.

<table><tr><td rowspan="2">Dataset</td><td rowspan="2">Tasks</td><td rowspan="2">SMTO</td><td colspan="3">Hyperparameters</td><td rowspan="2">SMTO specific parameters</td></tr><tr><td>LR</td><td>P</td><td>WD</td></tr><tr><td rowspan="2">MNIST</td><td rowspan="2">CLCR</td><td rowspan="2">Single Task</td><td>0.0025</td><td>0.5</td><td>0</td><td rowspan="2"></td></tr><tr><td>0.005</td><td>0.3</td><td>0</td></tr><tr><td rowspan="9"></td><td rowspan="9"></td><td>Unit. Scal.</td><td>0.0025</td><td>0.3</td><td>0</td><td></td></tr><tr><td>Auto-Lambda</td><td>0.005</td><td>0.3</td><td>0</td><td rowspan="4">auxiliary learning rate scale: $\{1000, 100, 10\ 1.0\ \mathbf{0.1}\ 0.01\ 0.001\}$ initial weight:  $\mathbf{0.1}$  $c:\ \{0.2,\ 0.5,\ 0.8\}$  $\alpha:\ \{0.2,\ 0.4,\ \mathbf{0.6},\ 0.8,\ 1.0\}$ </td></tr><tr><td>CAGrad</td><td>0.0025</td><td>0.3</td><td>0</td></tr><tr><td>CDTT</td><td>0.005</td><td>0.2</td><td>0</td></tr><tr><td>EDM</td><td>0.005</td><td>0.2</td><td>0</td></tr><tr><td>FAMO</td><td>0.0025</td><td>0.2</td><td>0</td><td rowspan="2"> $\gamma:\ \{0.01,\ \mathbf{0.001},\ 0.0001\}$ learning rate of the task logits:  $\mathbf{0.025}$ </td></tr><tr><td>IMTL</td><td>0.0025</td><td>0.3</td><td>0</td></tr><tr><td>Nash-MTL</td><td>0.0025</td><td>0.3</td><td>0</td><td rowspan="2">optimizer iterations:  $\mathbf{20}$ auxiliary learning rate scale: $\{5.0\ 1.0\ 0.5\ \mathbf{0.1}\}$ feature size:  $\mathbf{50}$ </td></tr><tr><td>RotoGrad</td><td>0.005</td><td>0.3</td><td>0</td></tr></table>

Table 7: Hyperparameters for MNIST - CL & RR.

<table><tr><td rowspan="2">Dataset</td><td rowspan="2">Tasks</td><td rowspan="2">SMTO</td><td colspan="3">Hyperparameters</td><td rowspan="2">SMTO specific parameters</td></tr><tr><td>LR</td><td>P</td><td>WD</td></tr><tr><td rowspan="2">MNIST</td><td rowspan="2">CLRR</td><td rowspan="2">Single Task</td><td>0.0025</td><td>0.5</td><td>0</td><td rowspan="2"></td></tr><tr><td>0.0025</td><td>0.0</td><td>0</td></tr><tr><td rowspan="16"></td><td rowspan="16"></td><td>Unit. Scal.</td><td>0.001</td><td>0</td><td>0</td><td></td></tr><tr><td>Auto-Lambda</td><td>0.001</td><td>0</td><td>0</td><td rowspan="4">auxiliary learning rate scale: $\{1000\ 100\ 10\ 1\ 0.1\ 0.01\ 0.001\}$ initial weight: 0.1c:  $\{0.2, 0.5, 0.8\}$  $\alpha: \{0.2, 0.4, 0.6, 0.8, 1.0\}$ </td></tr><tr><td>CAGrad</td><td>0.001</td><td>0</td><td>0</td></tr><tr><td>CDTT</td><td>0.0025</td><td>0</td><td>0</td></tr><tr><td>EDM</td><td>0.001</td><td>0</td><td>0</td></tr><tr><td>FAMO</td><td>0.0025</td><td>0</td><td>0</td><td rowspan="4"> $\gamma: \{0.01, 0.001, 0.0001\}$ learning rate of the task logits: 0.025k: 1.0p: 0.5</td></tr><tr><td>GradDrop</td><td>0.001</td><td>0</td><td>0</td></tr><tr><td>IMTL</td><td>0.001</td><td>0</td><td>0</td></tr><tr><td>MGDA-UB</td><td>0.001</td><td>0</td><td>0</td></tr><tr><td>Nash-MTL</td><td>0.001</td><td>0</td><td>0</td><td rowspan="4">optimizer iterations: 20</td></tr><tr><td>PCGrad</td><td>0.001</td><td>0</td><td>0</td></tr><tr><td>RLW-Dirichlet</td><td>0.001</td><td>0</td><td>0</td></tr><tr><td>RLW-Normal</td><td>0.001</td><td>0</td><td>0</td></tr><tr><td>RotoGrad</td><td>0.001</td><td>0</td><td>0</td><td rowspan="3">auxiliary learning rate scale: $\{5.0\ 1.0\ 0.5\ 0.1\}$ feature size: 50</td></tr><tr><td>SI</td><td>0.001</td><td>0</td><td>0</td></tr><tr><td>UW</td><td>0.001</td><td>0</td><td>0</td></tr></table>

Table 8: Hyperparameters for MNIST - RL & RR.

<table><tr><td rowspan="2">Dataset</td><td rowspan="2">Tasks</td><td rowspan="2">SMTO</td><td colspan="3">Hyperparameters</td><td rowspan="2">SMTO specific parameters</td></tr><tr><td>LR</td><td>P</td><td>WD</td></tr><tr><td>MNIST</td><td>RLRR</td><td>Single Task</td><td>0.00250.0025</td><td>0</td><td>0</td><td></td></tr><tr><td rowspan="9"></td><td rowspan="9"></td><td>Unit. Scal.</td><td>0.001</td><td>0</td><td>0</td><td rowspan="7">auxiliary learning rate scale: $\{1000\ 100\ 10\ 1\ 0.01\ 0.001\ 0.0001\}$ initial weight:  $\mathbf{0.1}$  $c:\{0.2,\ \mathbf{0.5},\ 0.8\}$  $\alpha:\{0.2,\ 0.4,\ 0.6,\ 0.8,\ \mathbf{1.0}\}$  $\gamma:\{0.01,\ 0.001,\ \mathbf{0.0001}\}$ learning rate of the task logits:  $\mathbf{0.025}$ </td></tr><tr><td>Auto-Lambda</td><td>0.001</td><td>0</td><td>0</td></tr><tr><td>CAGrad</td><td>0.0025</td><td>0</td><td>0</td></tr><tr><td>CDTT</td><td>0.001</td><td>0</td><td>0</td></tr><tr><td>EDM</td><td>0.001</td><td>0</td><td>0</td></tr><tr><td>FAMO</td><td>0.001</td><td>0</td><td>0</td></tr><tr><td>IMTL</td><td>0.001</td><td>0</td><td>0</td></tr><tr><td>Nash-MTL</td><td>0.001</td><td>0</td><td>0</td><td rowspan="2">optimizer iterations:  $\mathbf{20}$ auxiliary learning rate scale: $\{5.0\ 1.0\ 0.5\ \mathbf{0.1}\}$ feature size:  $\mathbf{50}$ </td></tr><tr><td>RotoGrad</td><td>0.001</td><td>0</td><td>0</td></tr></table>

Table 9: Hyperparameters for Cityscapes.

<table><tr><td rowspan="2">Dataset</td><td rowspan="2">Tasks</td><td rowspan="2">SMTO</td><td colspan="3">Hyperparameters</td><td rowspan="2">SMTO specific parameters</td></tr><tr><td>LR</td><td>P</td><td>WD</td></tr><tr><td rowspan="5">City.</td><td rowspan="2">S</td><td rowspan="5">Single Task</td><td>ACC: 0.001</td><td>0</td><td> $1e-4$ </td><td rowspan="5"></td></tr><tr><td>mIoU: 0.001</td><td>0</td><td> $1e-4$ </td></tr><tr><td rowspan="2">D</td><td>L1 abs: 0.001</td><td>0</td><td> $1e-4$ </td></tr><tr><td>L1 rel: 0.0005</td><td>0</td><td> $1e-4$ </td></tr><tr><td>I</td><td>L1 abs: 0.001</td><td>0</td><td> $1e-5$ </td></tr><tr><td rowspan="9"></td><td rowspan="9"></td><td>Unit. Scal.</td><td>0.001</td><td>0</td><td> $1e-4$ </td><td rowspan="7">auxiliary learning rate scale: $\{100\ 10\ 1\ 0.1\ 0.01\ 0.001\ \mathbf{0.0001}\}$ initial weight:  $\mathbf{0.1}$  $c:\ \{\mathbf{0.2},\ 0.5,\ 0.8\}$  $\alpha:\ \{\mathbf{0.2},\ 0.4,\ 0.6,\ 0.8,\ 1.0\}$  $\gamma:\ \{\mathbf{0.01},\ 0.001,\ 0.0001\}$ learning rate of the task logits:  $\mathbf{0.025}$ </td></tr><tr><td>Auto-Lambda</td><td>0.001</td><td>0</td><td> $1e-4$ </td></tr><tr><td>CAGrad</td><td>0.001</td><td>0</td><td> $1e-4$ </td></tr><tr><td>CDTT</td><td>0.001</td><td>0</td><td> $1e-4$ </td></tr><tr><td>EDM</td><td>0.001</td><td>0</td><td> $1e-4$ </td></tr><tr><td>FAMO</td><td>0.001</td><td>0</td><td> $1e-4$ </td></tr><tr><td>IMTL</td><td>0.001</td><td>0</td><td> $1e-4$ </td></tr><tr><td>Nash-MTL</td><td>0.001</td><td>0</td><td> $1e-4$ </td><td rowspan="2">optimizer iterations:  $\mathbf{20}$ auxiliary learning rate scale: $\{5.0\ \mathbf{1.0}\ 0.5\ 0.1\}$ feature size:  $\mathbf{512}$ </td></tr><tr><td>RotoGrad</td><td>0.001</td><td>0</td><td>0</td></tr></table>

Table 10: Hyperparameters for QM9.

<table><tr><td rowspan="2">Dataset</td><td rowspan="2">Tasks</td><td rowspan="2">SMTO</td><td colspan="3">Hyperparameters</td><td rowspan="2">SMTO specific parameters</td></tr><tr><td>LR</td><td>P</td><td>WD</td></tr><tr><td rowspan="11">QM9</td><td> $C_v$ </td><td></td><td>0.001</td><td>0</td><td>0</td><td></td></tr><tr><td>G</td><td></td><td>0.001</td><td>0</td><td>0</td><td></td></tr><tr><td>H</td><td></td><td>0.001</td><td>0</td><td>0</td><td></td></tr><tr><td> $U_0$ </td><td></td><td>0.0005</td><td>0</td><td>0</td><td></td></tr><tr><td>U</td><td></td><td>0.001</td><td>0</td><td>0</td><td></td></tr><tr><td> $\alpha$ </td><td>Single Task</td><td>0.005</td><td>0</td><td>0</td><td></td></tr><tr><td> $\epsilon_{homo}$ </td><td></td><td>0.001</td><td>0</td><td>0</td><td></td></tr><tr><td> $\epsilon_{lumo}$ </td><td></td><td>0.001</td><td>0</td><td>0</td><td></td></tr><tr><td> $\mu$ </td><td></td><td>0.001</td><td>0</td><td>0</td><td></td></tr><tr><td> $R^2$ </td><td></td><td>0.001</td><td>0</td><td>0</td><td></td></tr><tr><td>ZPVE</td><td></td><td>0.001</td><td>0</td><td>0</td><td></td></tr><tr><td rowspan="7"></td><td rowspan="7"></td><td>Unit. Scal.</td><td>0.005</td><td>0</td><td>0</td><td></td></tr><tr><td>Auto-Lambda</td><td>0.001</td><td>0</td><td>0</td><td rowspan="6">auxiliary learning rate scale: $\{100, 10, 1.0\}$ initial weight: 0.1c:  $\{0.2, 0.5, 0.8\}$  $\alpha: \{0.2, 0.4, 0.6, 0.8, 1.0\}$  $\gamma: \{0.01, 0.001, 0.0001\}$ learning rate of the task logits: 0.025optimizer iterations: 20</td></tr><tr><td>CAGrad</td><td>0.001</td><td>0</td><td>0</td></tr><tr><td>CDTT</td><td>0.001</td><td>0</td><td>0</td></tr><tr><td>EDM</td><td>0.001</td><td>0</td><td>0</td></tr><tr><td>FAMO</td><td>0.005</td><td>0</td><td>0</td></tr><tr><td>Nash-MTL</td><td>0.001</td><td>0</td><td>0</td></tr></table>