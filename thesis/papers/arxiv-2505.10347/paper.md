# Uniform Loss vs. Specialized Optimization: A Comparative Analysis in Multi-Task Learning

 Gabriel S. Gama Affiliation: University of São Paulo Email: [gabriel\_gama@usp.br](mailto:)    Valdir Grassi Jr Affiliation: University of São Paulo Email: [vgrassi@usp.br](mailto:vgrassi@usp.br) 

###### Abstract

Specialized Multi-Task Optimizers (SMTOs) balance task learning in Multi-Task Learning by addressing issues like conflicting gradients and differing gradient norms, which hinder equal-weighted task training. However, recent critiques suggest that equally weighted tasks can achieve competitive results compared to SMTOs, arguing that previous SMTO results were influenced by poor hyperparameter optimization and lack of regularization. In this work, we evaluate these claims through an extensive empirical evaluation of SMTOs, including some of the latest methods, on more complex multi-task problems to clarify this behavior. More specifically, we start our analysis by evaluating all SMTOs on a simple MNIST problem to identify the promising optimizers and assess them further on progressively more complex multi-task problems. Our findings indicate that SMTOs perform well compared to uniform loss and that fixed weights can achieve competitive performance compared to SMTOs. Furthermore, we demonstrate why uniform loss performs similarly to SMTOs in some instances. The source code is available at [https://github.com/Gabriel-SGama/UnitScal\_vs\_SMTOs](https://github.com/Gabriel-SGama/UnitScal_vs_SMTOs "").

## 1 Introduction

When tackling complex real-world challenges in machine learning, one must contemplate the utility of Multi-Task Learning (MTL) \[[1](#bib.bib30 "")\]. Scenarios such as scene understanding, multilingual translation, and predicting chemical properties require multiple outputs, possibly making them promising candidates for MTL approaches due to their inherent inductive bias \[[2](#bib.bib31 "")\].

Classical methods approach this problem by utilizing one model for each task, but most MTL scenarios have the potential to share information across tasks. For example, autonomous driving relies on multiple pieces of information extracted from the same visual input, such as semantic segmentation, object detection, instance segmentation, and others. Therefore, sharing information across those tasks is desirable to increase performance.

To efficiently share feature information across tasks, various approaches have been proposed, including model architecture designs \[[3](#bib.bib38 ""), [4](#bib.bib40 ""), [5](#bib.bib39 "")\], task grouping strategies \[[6](#bib.bib41 ""), [7](#bib.bib42 "")\], Meta-Learning techniques \[[8](#bib.bib43 "")\], and the focus of this work, Specialized Multi-Task Optimizers (SMTOs) \[[9](#bib.bib9 ""), [10](#bib.bib15 "")\].

SMTOs are implemented together with a shared encoder architecture, where one common encoder outputs a feature space that is shared with the specific decoders used for each task \[[9](#bib.bib9 ""), [10](#bib.bib15 "")\]. Consequently, this reduces the computational cost when compared to the individual models approach. This unified model can be optimized by minimizing the sum of the losses, known as uniform loss. However, the simultaneous learning of multiple tasks introduces challenges such as conflicting directions and significant gradient norm disparities \[[11](#bib.bib16 ""), [12](#bib.bib17 "")\]. SMTOs aim to address these challenges by applying mathematical tools to jointly and equally optimize all objectives.

SMTOs can be classified into two types: gradient-based and loss-based. The former compute the gradients of each task and combine them to find a suitable optimization direction \[[9](#bib.bib9 ""), [13](#bib.bib10 ""), [14](#bib.bib11 ""), [15](#bib.bib12 ""), [11](#bib.bib16 ""), [12](#bib.bib17 ""), [10](#bib.bib15 ""), [16](#bib.bib18 "")\]. This process adds complexity and computational cost to the learning problem as multiple backpropagation steps are needed. The loss-based methods only take into consideration the loss values \[[17](#bib.bib8 ""), [18](#bib.bib19 ""), [19](#bib.bib14 ""), [20](#bib.bib20 ""), [21](#bib.bib27 "")\], reaching computational times comparable to those of the uniform loss \[[22](#bib.bib25 ""), [21](#bib.bib27 "")\].

One notable issue in the SMTO field is the lack of a standardized evaluation framework and procedure, leading to concerns about reproducibility \[[23](#bib.bib24 "")\]. Understandably, some works have begun questioning the efficacy of SMTOs \[[24](#bib.bib23 ""), [23](#bib.bib24 "")\]. These critiques attribute reported gains in SMTO papers to sub-optimal hyperparameter tuning or insufficient regularization. They provide various experiments, including natural language processing (NLP), computer vision, and reinforcement learning, as well as a theoretical hypothesis on why SMTOs apply a regularization effect and a comparison to a properly optimized baseline.

In this context, we propose to address these concerns by hypothesizing that proper hyperparameter optimization is crucial for fair performance assessment of SMTOs. Nevertheless, we also emphasize the significance of considering the inherent complexity of the MTL balancing problem. Our contributions include:

*   •

```
A comprehensive evaluation of current SMTOs throughout Section [3](#S3 "3 Experimental Evaluation ‣ Uniform Loss vs. Specialized Optimization: A Comparative Analysis in Multi-Task Learning"), including recent ones that have not been covered in previous critiques, using the MNIST dataset \[[25](#bib.bib3 "")\] as a starting point to identify promising SMTOs, followed by other commonly used datasets such as Cityscapes \[[26](#bib.bib5 "")\] and Quantum Chemistry and Molecular Physics (QM9) \[[27](#bib.bib6 "")\].
```
*   •

```
An investigation into why equally weighted tasks can match the results of SMTOs. More specifically, the two similar tasks scenario in Section [3.2](#S3.SS2 "3.2 Two similar tasks ‣ 3 Experimental Evaluation ‣ Uniform Loss vs. Specialized Optimization: A Comparative Analysis in Multi-Task Learning"), and a complexity analysis in Section [3.3](#S3.SS3 "3.3 Complexity analysis ‣ 3 Experimental Evaluation ‣ Uniform Loss vs. Specialized Optimization: A Comparative Analysis in Multi-Task Learning").
```
*   •

```
A re-evaluation of the hypothesis brought up by \[[23](#bib.bib24 "")\] that fixed weights can match the same results achieved by SMTOs in Section [3.5](#S3.SS5 "3.5 Fixed weights ‣ 3 Experimental Evaluation ‣ Uniform Loss vs. Specialized Optimization: A Comparative Analysis in Multi-Task Learning").
```
## 2 Related Work

In recent years, MTL and SMTOs have witnessed substantial research activity. Notable works, such as \[[17](#bib.bib8 ""), [9](#bib.bib9 "")\], have garnered interest in MTL due to their performance improvements compared to the uniform loss and single task baselines. This surge has also given rise to numerous variations and novel SMTO methods, each improving on or proposing solutions to unaddressed concerns of early SMTOs \[[14](#bib.bib11 ""), [28](#bib.bib21 ""), [12](#bib.bib17 ""), [16](#bib.bib18 "")\].

However, despite the growing body of research, the SMTO field faces a significant challenge: the absence of a formal benchmark. Consequently, reported results may vary between articles \[[9](#bib.bib9 ""), [23](#bib.bib24 ""), [10](#bib.bib15 "")\], raising concerns about the improvements. Specifically, recent critiques have revealed potential issues, suggesting that the reported performance gains from SMTOs stem from limited regularization \[[24](#bib.bib23 "")\] and under-optimized baseline hyperparameters \[[23](#bib.bib24 "")\].

[Kurin et al. \[24\]](#bib.bib23 "") show that all evaluated SMTOs are linked to a larger solution space, under-optimization or stochastic behavior, all of which induce a regularization effect \[[29](#bib.bib33 ""), [30](#bib.bib35 ""), [31](#bib.bib34 "")\]. They also evaluated their hypothesis by adding proper regularization such as weight decay, batch normalization \[[32](#bib.bib36 "")\], dropout \[[33](#bib.bib37 "")\], and early stopping to the uniform loss, referenced as Unitary Scalarization (Unit. Scal.) in this paper, and compared it to SMTOs. After the changes, they achieved competitive results to SMTOs at a lower computational cost.

Similarly, [Xin et al. \[23\]](#bib.bib24 "") argue that reported SMTO gains may be attributed to poor hyperparameter optimization, with particular interest in the impact of learning rates compared to random seeds. Their work demonstrates that exploring multiple fixed weight values can construct a Pareto curve, and SMTOs can only obtain one of the possible solutions. Due to computational restrictions, the discussion regarding fixed weights will be partially addressed as, to the best of our knowledge, there is no method to predict those fixed weights accurately, and grid search becomes impractical for a higher number of tasks.

Our work distinguishes itself in two key ways. Firstly, we consider most of the recent SMTOs in our evaluation \[[28](#bib.bib21 ""), [12](#bib.bib17 ""), [10](#bib.bib15 ""), [18](#bib.bib19 ""), [16](#bib.bib18 ""), [20](#bib.bib20 "")\]. Secondly, we adopt some variations to the commonly used problems to introduce more complexity, as detailed in Section [3](#S3 "3 Experimental Evaluation ‣ Uniform Loss vs. Specialized Optimization: A Comparative Analysis in Multi-Task Learning"). Our objective is to demonstrate that SMTOs, mainly the recent ones, provide significant improvements over Unit. Scal. as the complexity increases. Additionally, we aim to elucidate the scenarios in which Unit. Scal. can match SMTOs and provide a complete evaluation of these promising SMTOs in multiple challenging MTL configurations.

## 3 Experimental Evaluation

Although numerous works utilize similar datasets and training configurations, a broadly accepted evaluation framework remains elusive. We have chosen specific problems from a selection of SMTO papers to address this challenge. More specifically, we restrict our evaluation to the supervised learning setting using datasets such as MNIST \[[25](#bib.bib3 "")\], Cityscapes \[[26](#bib.bib5 "")\], and QM9 \[[27](#bib.bib6 "")\]. To address recent critiques to SMTOs \[[24](#bib.bib23 ""), [23](#bib.bib24 "")\] and ensure robustness, the models were trained with at least one form of regularization, except for the QM9 dataset, and the parameters were optimized by applying grid search. We determine the best hyperparameter configuration by training once with each value (excluding Multi-MNIST, which was trained five times for each value) and selecting the best validation result. The best combination is trained at least two more times, and the metrics reported here are from the test split. We consider the epoch in which the model achieved the highest validation metric to mitigate overfitting.

For comparative evaluation of different SMTOs, we used two widely employed metrics, the mean relative percentage gain metric Δmtm\\Delta\_{\\text{mtm}} and the mean rank M​RMR \[[34](#bib.bib32 ""), [10](#bib.bib15 ""), [20](#bib.bib20 "")\]. Δmtm\\Delta\_{\\text{mtm}} measures the improvement achieved by the MTL model over single-task references, and the M​RMR evaluates how equally all tasks were optimized. Since tasks like semantic segmentation and disparity estimation use multiple metrics for evaluation, we adapt the equations to account for that. For each set of metrics MiM\_{i} for task i∈𝒯i\\in\\mathcal{T}, the single task reference Mb,i,j|j∈MiM\_{b,i,j}\\;\\mid j\\in M\_{i} is compared to its multi-task counterpart Msmto,i,j|j∈MiM\_{\\text{smto},i,j}\\;\\mid j\\in M\_{i}, so Δmtm\\Delta\_{\\text{mtm}} is given by:

Δmtm\=1N​∑i\=1N1Mi​∑j\=1Mi(−1)li,j​Msmto,i,j−Mb,i,jMb,i,j,\\Delta\_{\\text{mtm}}=\\frac{1}{N}\\sum\_{i=1}^{N}\\frac{1}{M\_{i}}\\sum\_{j=1}^{M\_{i}}(-1)^{l\_{i,j}}\\frac{M\_{\\text{smto},i,j}-M\_{b,i,j}}{M\_{b,i,j}},

(1)

where li,jl\_{i,j} is 11 if a lower value is better and 00 otherwise. Likewise, the mean rank is given by:

M​R\=1N​∑i\=1N1Mi​∑j\=1MiRi,j,MR=\\frac{1}{N}\\sum\_{i=1}^{N}\\frac{1}{M\_{i}}\\sum\_{j=1}^{M\_{i}}R\_{i,j},

(2)

where Ri,jR\_{i,j} is the rank of the SMTO on task ii with respect to the metric jj. As a result, both metrics compute the mean within the same task and then the mean of every task, ensuring a fair evaluation in a multi-task setting.

From a wide array of publicly available SMTOs, we selected a subset for evaluation: UW \[[17](#bib.bib8 "")\], MGDA \[[9](#bib.bib9 "")\], PCGrad \[[35](#bib.bib13 "")\], GradDrop \[[15](#bib.bib12 "")\], EDM \[[14](#bib.bib11 "")\], IMTL \[[11](#bib.bib16 "")\], CAGrad \[[28](#bib.bib21 "")\], RotoGrad \[[12](#bib.bib17 "")\], Nash-MTL \[[10](#bib.bib15 "")\], SI \[[10](#bib.bib15 "")\], RLW \[[19](#bib.bib14 "")\], Auto-Lambda \[[18](#bib.bib19 "")\], CDTT \[[16](#bib.bib18 "")\] and FAMO \[[20](#bib.bib20 "")\]. We highlight that there are still some SMTOs that could be evaluated in future works \[[36](#bib.bib7 ""), [13](#bib.bib10 ""), [37](#bib.bib22 ""), [38](#bib.bib26 ""), [22](#bib.bib25 ""), [21](#bib.bib27 "")\].

Including in Unit. Scal. and two RLW variations (Normal and Dirichlet distributions) as done in previous works \[[24](#bib.bib23 "")\], we assessed a total of 16 SMTOs. To reduce computational cost, the SMTOs considered in this paper are filtered according to their performance in a regression and classification problem. Even though evaluating an SMTO with just one example is not a definitive quality measure, this two-task problem tests the method’s ability to handle varying gradient norms. This concern motivated several SMTOs \[[13](#bib.bib10 ""), [14](#bib.bib11 ""), [11](#bib.bib16 ""), [10](#bib.bib15 "")\]. Based on the results and contributions from each SMTO, we select the most promising ones and proceed to test them further.

In this section, the chosen SMTOs are first evaluated in the specific case of two similar task problems to demonstrate that, despite being an MTL problem, Unit. Scal. approximates the behavior of the SMTOs and achieves competitive results. We also consider some factors that dictate the need for SMTOs by comparing Unit. Scal. to one of the SMTOs while evaluating different configurations with increasing multi-task learning balancing complexity, using the Cityscapes dataset. All promising SMTOs are then evaluated in the most challenging format. Furthermore, we provide a comparison on the QM9 dataset. Finally, we evaluate the performance of fixed weights. All hyperparameter combinations, SMTOs-specific hyperparameter optimization, and their best configuration are summarized in Appendix [A.6](#A1.SS6 "A.6 Hyperparameters ‣ Appendix A Technical Appendices and Supplementary Material ‣ Uniform Loss vs. Specialized Optimization: A Comparative Analysis in Multi-Task Learning") for reproducibility.

### 3.1 Selecting promising SMTOs for further evaluation

In this paper, we tackle multiple MTL challenges to evaluate the performance of many SMTOs. However, due to the multitude of combinations and the necessity for hyperparameter optimization to ensure a fair evaluation \[[24](#bib.bib23 ""), [23](#bib.bib24 "")\], we apply a preliminary filter for SMTO selection based on the performance on the Multi-MNIST dataset \[[9](#bib.bib9 "")\].

The Multi-MNIST dataset is a variation of the MNIST dataset for MTL where the digits are overlapped in the same image, one on the top left and one on the bottom right. This combined image is then used to predict information regarding the left and right digits. The most common variation is predicting the class of the left and right digits, as done in multiple works \[[9](#bib.bib9 ""), [35](#bib.bib13 ""), [12](#bib.bib17 ""), [24](#bib.bib23 ""), [23](#bib.bib24 "")\]. Instead, we opt for the alternative suggested by [Nakamura et al. \[16\]](#bib.bib18 ""), where the model is trained to classify the top left digit and to reconstruct the bottom right digit. This choice was made because this variation adds more complexity, involving tasks with varying levels of difficulty and loss functions that result in gradients with different norms, an important factor when considering the specific case of two tasks, as shown in Section [3.2](#S3.SS2 "3.2 Two similar tasks ‣ 3 Experimental Evaluation ‣ Uniform Loss vs. Specialized Optimization: A Comparative Analysis in Multi-Task Learning"). Those tasks will be referenced as CL, CR, RL, and RR for the classification (C) or reconstruction (R) of the left (L) or right (R) digit. When tasks are represented using the notation {task 1}\_{task 2}…{task N}, it indicates a multi-task scenario that simultaneously addresses tasks 1 through N.

For our model architecture, we use a variation of the Lenet model \[[25](#bib.bib3 "")\] as [Nakamura et al. \[16\]](#bib.bib18 ""). We then add dropout layers to the encoder and decoder architectures for regularization. The hyperparameters are optimized by grid search, varying the learning rate l​r∈lr\\in {0.01, 0.075, 0.005, 0.0025, 0.001, 0.00075, 0.0005} and dropout p∈p\\in {0.0, 0.1, 0.2, 0.3, 0.4, 0.5}. Specific SMTO hyperparameters are also included in the search if necessary. We train each configuration five times and the best validation configuration five more times, giving a total of ten models for each SMTO.

Figure 1: Box plot illustrating Δmtm\\Delta\_{\\text{mtm}} scores on the Multi-MNIST classification and regression problem. The red line indicates the median value for Unit. Scal.. Notably, multiple SMTOs obtained substantial improvements over the Unit. Scal. baseline.

The results, as presented in Figure [1](#S3.F1 "Figure 1 ‣ 3.1 Selecting promising SMTOs for further evaluation ‣ 3 Experimental Evaluation ‣ Uniform Loss vs. Specialized Optimization: A Comparative Analysis in Multi-Task Learning"), demonstrate that Unit. Scal. achieves competitive results compared to some SMTOs. However, the majority outperform it significantly, primarily recent ones. Based on these results, we select EDM \[[14](#bib.bib11 "")\], IMTL \[[11](#bib.bib16 "")\], CAGrad \[[28](#bib.bib21 "")\], RotoGrad \[[12](#bib.bib17 "")\], Nash-MTL \[[10](#bib.bib15 "")\], Auto-Lambda \[[18](#bib.bib19 "")\], CDTT \[[16](#bib.bib18 "")\] and FAMO \[[20](#bib.bib20 "")\] as promising SMTOs for further investigation.

Despite the Multi-MNIST dataset’s simplicity, it effectively demonstrated the selected SMTOs’ ability to provide a meaningful gradient direction even with varying gradient norms, enabling their distinction from other methods. Specifically, PCGrad and MGDA are known to emphasize gradients with the smallest and largest magnitudes, respectively \[[11](#bib.bib16 "")\]. GradDrop randomly drops gradients based on sign-purity, without considering magnitude balancing. Furthermore, methods like RLW, UW, and SI operate directly on loss values without estimating gradients, unlike FAMO and Auto-Lambda, potentially making them sensitive to differences in gradient magnitude. Ultimately, most of the selected SMTOs significantly outperformed Unit. Scal. in our evaluations.

### 3.2 Two similar tasks

The two-task setting is commonly used as a computationally efficient way of evaluating SMTOs, widely adopted in multiple MTL papers \[[9](#bib.bib9 ""), [24](#bib.bib23 ""), [23](#bib.bib24 "")\]. However, when considering the two-task setting, it is possible that Unit. Scal. approximates the behavior of an SMTO given a few conditions.

When considering a problem with a shared encoder, the same decoder architecture, the same loss function, task labels with similar distributions, and learning speeds, the gradient norm and the loss value of both tasks are expected to behave in remarkably similar ways. Additionally, if we restrict this assumption to equal gradient norm or equal loss values, it is possible to theoretically show that most SMTOs converge to equal weights (see Appendix [A.4](#A1.SS4 "A.4 Proofs ‣ Appendix A Technical Appendices and Supplementary Material ‣ Uniform Loss vs. Specialized Optimization: A Comparative Analysis in Multi-Task Learning")).

To provide empirical evidence for the above statement, we consider two problems from Multi-MNIST: dual classification and dual reconstruction of the digits. In both cases, the architecture initially described in the previous section was reused with the corresponding changes to the decoders. We follow the same hyperparameter optimization previously described for Multi-MNIST.

Figure 2: Box plot of Δmtm\\Delta\_{\\text{mtm}} (left) for all SMTOs not evaluated on \[[24](#bib.bib23 "")\] and mean error to equal weights (right). The top plot corresponds to CL\_CR, and the bottom one represents RL\_RR. All SMTOs had similar performance to Unit. Scal. and the computed weights varied near 0.5, excluding FAMO on the CL\_CR problem, exemplifying how similar equal weights are to the selected SMTOs’ solution in this setting.

The results presented in Figure [2](#S3.F2 "Figure 2 ‣ 3.2 Two similar tasks ‣ 3 Experimental Evaluation ‣ Uniform Loss vs. Specialized Optimization: A Comparative Analysis in Multi-Task Learning") show that all SMTOs performed closely to the Unit. Scal. baseline, even on the dual reconstruction case where Δm​t​m≈−20%\\Delta\_{mtm}\\approx-20\\%, meaning that there was high interference between tasks. We can also observe that the learned weights from the SMTOs predominantly clustered around 0.5, besides FAMO, which was a notable exception, exhibiting instability on the dual classification problem, likely attributable to the small scale of the loss values.

Consequently, employing equal weights approximates the average behavior of the SMTOs, resulting in comparable performance. This observation suggests that a straightforward averaging of task gradients can serve as a surprisingly effective baseline without significant performance degradation in scenarios with similar task characteristics, such as the dual translation setting examined in the recent critique \[[23](#bib.bib24 "")\].

### 3.3 Complexity analysis

Cityscapes \[[26](#bib.bib5 "")\] is a street scene dataset containing high-resolution images 2048x1024 and labels for semantic segmentation, disparity estimation, and instance segmentation. It is commonly adopted for MTL \[[17](#bib.bib8 ""), [9](#bib.bib9 ""), [24](#bib.bib23 ""), [10](#bib.bib15 ""), [23](#bib.bib24 ""), [20](#bib.bib20 "")\] under two variations, the two-task problem with 7 class semantic segmentation and disparity estimation, and the three-task problem that uses 19 semantic classes and adds instance segmentation. Some similarities arise when comparing the first version to the Multi-MNIST classification-reconstruction problem. Both problems generate gradients with different norms, and the tasks differ in complexity, yet there is no significant difference between SMTOs and Unit. Scal. with optimized hyperparameters \[[24](#bib.bib23 ""), [23](#bib.bib24 "")\].

(a) Mean cosine similarity.

(b) IMTL and Unit. Scal. performance.

Figure 3: (a) Mean cosine similarity between each task gradient and the average direction. City R50 exhibits significantly higher similarity than Multi-MNIST, potentially explaining Unit. Scal.’s comparable performance to SMTOs. Conversely, the lower similarity for City 19C and QM9 correlates with Unit. Scal.’s relative underperformance. See Appendix [A.2.1](#A1.SS2.SSS1 "A.2.1 Cosine similarity ‣ A.2 Additional results ‣ Appendix A Technical Appendices and Supplementary Material ‣ Uniform Loss vs. Specialized Optimization: A Comparative Analysis in Multi-Task Learning") for the full version with all configurations. (b) Shows the relative performance of Unit. Scal. and IMTL. Adding a third task and increasing the amount of information needed by increasing the number of classes are the relevant steps that dictate the relative performance in this problem.

The reason for this unexpected behavior is related to the lower amount of interference between each task compared to the Multi-MNIST problem, as seen in Figure [3(a)](#S3.F3.sf1 "In Figure 3 ‣ 3.3 Complexity analysis ‣ 3 Experimental Evaluation ‣ Uniform Loss vs. Specialized Optimization: A Comparative Analysis in Multi-Task Learning"). Figure [3(a)](#S3.F3.sf1 "In Figure 3 ‣ 3.3 Complexity analysis ‣ 3 Experimental Evaluation ‣ Uniform Loss vs. Specialized Optimization: A Comparative Analysis in Multi-Task Learning") shows the mean cosine similarity between each task gradient and the average direction, which can be seen as a measure of interference, as it correctly ranks the problems with respect to the relative performance between Unit. Scal. and SMTOs. Even after significantly reducing the parameter count in step R18, the level of interference remains about the same (see Appendix [A.2.1](#A1.SS2.SSS1 "A.2.1 Cosine similarity ‣ A.2 Additional results ‣ Appendix A Technical Appendices and Supplementary Material ‣ Uniform Loss vs. Specialized Optimization: A Comparative Analysis in Multi-Task Learning")), as reflected in the relative performance between Unit. Scal. and IMTL. However, after increasing the complexity of the problem, there is a significant difference between the two methods.

To vary the complexity in a controlled manner, we vary model size, number of tasks, and information available to learn to identify which characteristic is relevant. We first trained the model on the same initial two-task configuration of \[[24](#bib.bib23 "")\] and applied small cumulative steps until the most complex configuration. We opted to test only IMTL \[[11](#bib.bib16 "")\] and compare it to Unit. Scal. in the transitional steps as it was one of the best performing SMTOs, and it does not require any specific hyperparameter optimization. All Cityscapes configurations are optimized varying the learning rate at l​rlr ∈\\in {0.005, 0.001, 0.0005, 0.0001} and the weight decay ∈\\in {10−410^{-4}, 10−510^{-5}, 0.0} with 32 batch size. Each configuration is trained once, and then the best in the validation dataset is trained a total of 3 times. The selected steps are:

*   •

```
Reference (R50): A Resnet-50 \[[39](#bib.bib2 "")\] is used as shared encoder and the DeepLab decoder \[[40](#bib.bib4 "")\] is used as task specific decoder. The input resolution is 256x128, and the objective is set to 7 class segmentation and disparity estimation. This is the same configuration of [Kurin et al. \[24\]](#bib.bib23 "") and very similar to the one of [Xin et al. \[23\]](#bib.bib24 "");
```
*   •

```
Resnet-18 as decoder (R18): Opting for a smaller architecture should encourage competition for parameters;
```
*   •

```
Added instance segmentation (Inst. Seg.): Adding more tasks increases the complexity of the balancing problem, as well as the amount of information available to learn;
```
*   •

```
Increased resolution (HR): The resolution is increased from 256x128 to 512x256 as an intermediate step to allow better learning of objects from smaller classes;
```
*   •

```
19 semantic classes (19C): By increasing the number of learnable classes and keeping the model size fixed, the amount of information needed by the semantic decoder is increased, resulting in more competition for the shared encoder parameters and feature space.
```
The results in Figure [3(b)](#S3.F3.sf2 "In Figure 3 ‣ 3.3 Complexity analysis ‣ 3 Experimental Evaluation ‣ Uniform Loss vs. Specialized Optimization: A Comparative Analysis in Multi-Task Learning") reveal that the benefit of employing an SMTO only becomes apparent upon the inclusion of the instance segmentation task. Furthermore, a substantial relative performance gain is observed after incorporating all 19 classes, suggesting that the volume of information significantly impacts the efficacy of SMTOs. This observation aligns with prior research on information transfer \[[41](#bib.bib46 "")\], which highlights the necessity of carefully calibrating the shared model’s capacity in multi-task learning. Insufficient or excessive capacity can lead to negative or negligible transfer. Therefore, by increasing the number of tasks and classes, we inherently increase the demands on the shared feature space, potentially increasing inter-task conflicts.

Similar behavior was also noticed in the NLP field \[[42](#bib.bib29 "")\], where they compared the interference caused by different languages and data size, concluding that a bigger model could mitigate the interference issue, though their evaluation was restricted to Unit. Scal.. On the other hand, we investigate the effect of those parameters on the relative performance of different SMTOs to the baseline.

This exemplifies that SMTOs are not needed on some occasions when the problem is simple enough that the conflicting gradients do not interfere with the final performance. Therefore, the best way to evaluate SMTOs is to use the most complex configuration to measure their capabilities in dealing with significant interference. Accordingly, the other promising SMTOs are trained and evaluated with the same procedures on the last configuration.

Figure [4](#S3.F4 "Figure 4 ‣ 3.3 Complexity analysis ‣ 3 Experimental Evaluation ‣ Uniform Loss vs. Specialized Optimization: A Comparative Analysis in Multi-Task Learning") shows that SMTOs can indeed achieve better results when compared to Unit. Scal. in more complex balancing problems. This conclusion differs from Unit. Scal. \[[24](#bib.bib23 "")\] primarily because of the increased complexity. Furthermore, we evaluate some of the newer SMTOs, showing the field’s progress in the last years.

Figure 4: Muiti task metric on Cityscapes dataset. Most SMTOs significantly surpass the Unit. Scal. baseline on the most complex version of the Cityscapes MTL problem.

### 3.4 QM9

Finally, the last dataset we use to evaluate SMTOs is the QM9 dataset \[[27](#bib.bib6 "")\]. The QM9 dataset is a collection of quantum chemistry data for small organic molecules. It contains around 130,000 data points, and it was recently adopted as a problem for the MTL benchmarks \[[10](#bib.bib15 ""), [20](#bib.bib20 "")\].

Table 1: Multi-task metric on QM9 dataset. Almost all SMTOs significantly surpass Unit. Scal.

SMTO

ΔM​T​M\\Delta\_{MTM}(%) ↑\\uparrow

MR ↓\\downarrow

Unit. Scal.

-135.7

4.45

Auto-Lambda

-186.9

5.82

CAGrad

-70.13

3.82

CDTT

-121.2

5.45

EDM

-95.42

4.00

FAMO

-56.86

2.36

Nash-MTL

-53.62

2.09

We use the same configuration from \[[10](#bib.bib15 "")\]. The example from PyTorch Geometric \[[43](#bib.bib44 "")\] was adapted, and we used the model from \[[44](#bib.bib1 "")\]. The model was used to predict 11 properties from the molecules. We use the commonly adopted distribution of 110K molecules for training, 10K for validation, and 10K for testing, with a batch size of 120. The learning rate was optimized in l​rlr ∈\\in {5×10−35\\times 10^{-3}, 10−310^{-3}, 5×10−45\\times 10^{-4}}, and the ground truth values were normalized. We opted not to apply regularization because Unit. Scal. did not show significant improvement, and omitting this step reduced the overall computational cost.

Comparing the results from Table [1](#S3.T1 "Table 1 ‣ 3.4 QM9 ‣ 3 Experimental Evaluation ‣ Uniform Loss vs. Specialized Optimization: A Comparative Analysis in Multi-Task Learning") to the previous experiments, it is evident that this is the problem with the most amount of interference between tasks, with all SMTOs obtaining results more than 50% worse on average compared to the single task problem. This also correlates with the lower cosine similarity shown in Figure [3(a)](#S3.F3.sf1 "In Figure 3 ‣ 3.3 Complexity analysis ‣ 3 Experimental Evaluation ‣ Uniform Loss vs. Specialized Optimization: A Comparative Analysis in Multi-Task Learning").

During the experiment, IMTL was too unstable, and most runs crashed due to numerical overflow, as happened on the RL benchmark from Unit. Scal. \[[24](#bib.bib23 "")\], which also had a higher number of tasks. We agree with the hypothesis that this was caused by a lack of bounds on the scaling coefficients obtained from the IMTL’s optimization problem. RotoGrad also failed to converge, and our hypothesis is that RotoGrad was not able to find a proper rotation that minimizes the gradient conflict probably due to the increased difficulty of the problem. Further analysis can be seen in Appendix [A.3](#A1.SS3 "A.3 Rotograd’s instability on the QM9 dataset ‣ Appendix A Technical Appendices and Supplementary Material ‣ Uniform Loss vs. Specialized Optimization: A Comparative Analysis in Multi-Task Learning"). Both were removed from Table [1](#S3.T1 "Table 1 ‣ 3.4 QM9 ‣ 3 Experimental Evaluation ‣ Uniform Loss vs. Specialized Optimization: A Comparative Analysis in Multi-Task Learning").

### 3.5 Fixed weights

Though fixed weights are the most efficient and straightforward solution, choosing the correct values for the weights of each task is a challenging problem. [Xin et al. \[23\]](#bib.bib24 "") finds the ideal, or close to ideal, weights by performing a grid search. This process may be viable for a more limited scenario, but for a large number of tasks it is not feasible. Very recently, [Royer et al. \[45\]](#bib.bib28 "") proposed a population-based training to find the ideal weights efficiently, though this method was not evaluated in this work due to time constraints.

One recent critique of SMTOs raises the question of whether optimized scalar weights can compete with SMTOs \[[23](#bib.bib24 "")\]. However, as discussed previously, the problems used in the critique might not adequately represent the performance of SMTOs, so we evaluate this claim on our experiments.

Finding well optimized weights for all problems using grid search would be unfeasible, so instead, we opt for a simpler strategy, using the weights output from an SMTO to weight each task loss. More specifically, the weights from CAGrad, EDM and Nash-MTL optimizers are used for the MNIST, Cityscapes and QM9 datasets, respectively. As we are using the Adam optimizer, there is a normalizing effect on the gradients, so the gradients in each decoder are not severely affected by the different norms caused by the loss weighting.

To select the weights, we computed the mean value in each epoch, applied an exponential moving average with β\=0.9\\beta=0.9, and used the result from the last epoch to define the weights. To extract and train with the fixed weights, we used the optimal hyperparameter configuration from the respective SMTO. However, for the QM9 dataset the fixed weights demonstrated to be unstable so we used the ones from l​r\=0.0001lr=0.0001, instead of the optimal l​r\=0.001lr=0.001. The model is trained the same number of times as its SMTO counterpart.

Figure 5: Box plot comparing Δmtm\\Delta\_{\\text{mtm}} between the original Nash-MTL optimizer and their extracted weights from each learning configuration. Fixed weights can achieve results comparable to those of an SMTO. However, they can be significantly more unstable than a dynamic SMTO, as was the case for the weights from l​r\=0.001lr=0.001.

The results obtained from fixed weights, summarized in Table [2](#S3.T2 "Table 2 ‣ 3.5 Fixed weights ‣ 3 Experimental Evaluation ‣ Uniform Loss vs. Specialized Optimization: A Comparative Analysis in Multi-Task Learning"), indicate that it is indeed possible to get competitive performance using fixed weights. However, fixed weights can be more unstable, as shown in Figure [5](#S3.F5 "Figure 5 ‣ 3.5 Fixed weights ‣ 3 Experimental Evaluation ‣ Uniform Loss vs. Specialized Optimization: A Comparative Analysis in Multi-Task Learning") by the fixed weights extracted from l​r\=0.001lr=0.001, with results ranging from ≈−50%\\approx-50\\% to ≈−95%\\approx-95\\%. Surprisingly, the optimal learning rate weights were not the best option for the QM9 dataset.

Table 2: Comparison between Unit. Scal, fixed weights, and the best performing SMTO. Well chosen fixed weights can achieve comparable performance to SMTOs.

Dataset

Unit. Scal.

Fixed Weights

Best SMTO

Δm​t​m(%)↑\\Delta\_{mtm}(\\%)\\uparrow

MNIST

-8.952

-3.251

-2.137

City 19C

-5.312

0.145

0.570

QM9

-135.7

-53.66

-53.62

This raises the question of why fixed weights can achieve comparable performance to SMTOs, as generating dynamic weights is a major benefit of SMTOs. Analyzing the weight value behavior in Figure [6](#S3.F6 "Figure 6 ‣ 3.5 Fixed weights ‣ 3 Experimental Evaluation ‣ Uniform Loss vs. Specialized Optimization: A Comparative Analysis in Multi-Task Learning"), it is clear that all weights converge to a fixed value later in training. Additionally, due to the over-parameterized nature of deep learning models \[[46](#bib.bib47 "")\] the limitations typically associated with under-parameterized settings, such as the claim that scalarization is generally incapable of tracing out the Pareto front, do not necessarily apply \[[47](#bib.bib48 "")\]. We believe that further exploring the interaction between over-parameterization and multi-task learning could be a promising direction for future work.

This shows that using fixed weights is a valid alternative for complex MTL problems, corroborating the findings of [Xin et al. \[23\]](#bib.bib24 ""). However, finding those weights is highly costly, and with the advancement of loss-based SMTOs such as FAMO \[[20](#bib.bib20 "")\] and GO4Align \[[21](#bib.bib27 "")\], it is no longer a desirable choice.

Figure 6: Normalized weights from SMTOs. The shaded regions represent the mean value at each epoch, and the dotted lines show the results of the exponential moving average. The top left, top right, and bottom plots correspond to the MNIST, Cityscapes, and QM9 datasets, respectively. All datasets exhibit similar behavior, with the weights gradually converging to a stable value on average.

## 4 Limitations

In this work, we extensively tested multiple SMTOs on multiple configurations and datasets. However, it is important to note that we only focused on the supervised setting and did not include problems from NLP or reinforcement learning, fields in which MTL is commonly evaluated \[[24](#bib.bib23 ""), [23](#bib.bib24 "")\]. Also, our initial selection process involved filtering certain SMTOs based on their norm balancing performance on the MNIST classification-reconstruction problem. Despite this strong preliminary evidence, a more exhaustive analysis of the excluded methods may reveal unknown advantages. Furthermore, some SMTOs were reported to provide better results when combining them with other SMTOs \[[12](#bib.bib17 ""), [18](#bib.bib19 "")\], but because of time constraints, we limited the experiments to only the vanilla version. Finally, we only considered the MTL problem, where all tasks are equally weighted, and did not consider scenarios designed for auxiliary tasks.

## 5 Conclusion

This work aimed to address recent critiques of SMTOs by conducting a comprehensive empirical analysis under the context of supervised learning. First, we examined why Unit. Scal. can replicate the performance of SMTOs in two-task scenarios, particularly when the tasks are similar. Our findings suggest that SMTOs converge to equal weights, so Unit. Scal. approximates their behavior on average. Furthermore, we explored the conditions under which Unit. Scal. matches SMTOs, attributing its success to factors such as the degree of task interference, the number of tasks, and the amount of information to be learned.

We also investigated the relative performance of fixed weights compared to SMTOs. By deriving fixed weights from SMTOs, we demonstrated that fixed weights can perform competitively in complex scenarios. However, using fixed weights is less appealing due to the computational overhead of identifying optimal values and their comparable performance to recent loss-based SMTOs.

In conclusion, this study provides a robust evaluation of current SMTOs, underscoring their significance in addressing complex multi-task problems. Additionally, we aim to establish points to consider when evaluating SMTOs and to guide the development of future methods and applications.

## Acknowledgments

This work was partially funded by CNPq under grants 465755/2014-3, 309532/2023-0, and 130961/2024-8, and CAPES under finance code 001.

## References

*   \[1\] R. Caruana (1997) Multitask learning. Machine Learning 28 (1), pp. 41–75. Cited by: [§1](#S1.p1.1 "1 Introduction ‣ Uniform Loss vs. Specialized Optimization: A Comparative Analysis in Multi-Task Learning").
*   \[2\] J. Baxter (2000) A model of inductive bias learning. Journal of artificial intelligence research 12, pp. 149–198. Cited by: [§1](#S1.p1.1 "1 Introduction ‣ Uniform Loss vs. Specialized Optimization: A Comparative Analysis in Multi-Task Learning").
*   \[3\] I. Misra, A. Shrivastava, A. Gupta, and M. Hebert (2016) Cross-stitch networks for multi-task learning. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 3994–4003. Cited by: [§1](#S1.p3.1 "1 Introduction ‣ Uniform Loss vs. Specialized Optimization: A Comparative Analysis in Multi-Task Learning").
*   \[4\] Y. Gao, J. Ma, M. Zhao, W. Liu, and A. L. Yuille (2019) Nddr-cnn: layerwise feature fusing in multi-task cnns by neural discriminative dimensionality reduction. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 3205–3214. Cited by: [§1](#S1.p3.1 "1 Introduction ‣ Uniform Loss vs. Specialized Optimization: A Comparative Analysis in Multi-Task Learning").
*   \[5\] S. Liu, E. Johns, and A. J. Davison (2019) End-to-end multi-task learning with attention. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 1871–1880. Cited by: [§1](#S1.p3.1 "1 Introduction ‣ Uniform Loss vs. Specialized Optimization: A Comparative Analysis in Multi-Task Learning").
*   \[6\] T. Standley, A. Zamir, D. Chen, L. Guibas, J. Malik, and S. Savarese (2020) Which tasks should be learned together in multi-task learning?. In International Conference on Machine Learning, pp. 9120–9132. Cited by: [§1](#S1.p3.1 "1 Introduction ‣ Uniform Loss vs. Specialized Optimization: A Comparative Analysis in Multi-Task Learning").
*   \[7\] C. Fifty, E. Amid, Z. Zhao, T. Yu, R. Anil, and C. Finn (2021) Efficiently identifying task groupings for multi-task learning. Advances in Neural Information Processing Systems 34, pp. 27503–27516. Cited by: [§1](#S1.p3.1 "1 Introduction ‣ Uniform Loss vs. Specialized Optimization: A Comparative Analysis in Multi-Task Learning").
*   \[8\] T. Hospedales, A. Antoniou, P. Micaelli, and A. Storkey (2021) Meta-learning in neural networks: a survey. IEEE transactions on pattern analysis and machine intelligence 44 (9), pp. 5149–5169. Cited by: [§1](#S1.p3.1 "1 Introduction ‣ Uniform Loss vs. Specialized Optimization: A Comparative Analysis in Multi-Task Learning").
*   \[9\] O. Sener and V. Koltun (2018) Multi-task learning as multi-objective optimization. Advances in neural information processing systems 31. Cited by: [§1](#S1.p3.1 "1 Introduction ‣ Uniform Loss vs. Specialized Optimization: A Comparative Analysis in Multi-Task Learning"), [§1](#S1.p4.1 "1 Introduction ‣ Uniform Loss vs. Specialized Optimization: A Comparative Analysis in Multi-Task Learning"), [§1](#S1.p5.1 "1 Introduction ‣ Uniform Loss vs. Specialized Optimization: A Comparative Analysis in Multi-Task Learning"), [§2](#S2.p1.1 "2 Related Work ‣ Uniform Loss vs. Specialized Optimization: A Comparative Analysis in Multi-Task Learning"), [§2](#S2.p2.1 "2 Related Work ‣ Uniform Loss vs. Specialized Optimization: A Comparative Analysis in Multi-Task Learning"), [§3.1](#S3.SS1.p1.1 "3.1 Selecting promising SMTOs for further evaluation ‣ 3 Experimental Evaluation ‣ Uniform Loss vs. Specialized Optimization: A Comparative Analysis in Multi-Task Learning"), [§3.1](#S3.SS1.p2.1 "3.1 Selecting promising SMTOs for further evaluation ‣ 3 Experimental Evaluation ‣ Uniform Loss vs. Specialized Optimization: A Comparative Analysis in Multi-Task Learning"), [§3.2](#S3.SS2.p1.1 "3.2 Two similar tasks ‣ 3 Experimental Evaluation ‣ Uniform Loss vs. Specialized Optimization: A Comparative Analysis in Multi-Task Learning"), [§3.3](#S3.SS3.p1.1 "3.3 Complexity analysis ‣ 3 Experimental Evaluation ‣ Uniform Loss vs. Specialized Optimization: A Comparative Analysis in Multi-Task Learning"), [§3](#S3.p3.1 "3 Experimental Evaluation ‣ Uniform Loss vs. Specialized Optimization: A Comparative Analysis in Multi-Task Learning").
*   \[10\] A. Navon, A. Shamsian, I. Achituve, H. Maron, K. Kawaguchi, G. Chechik, and E. Fetaya (2022) Multi-task learning as a bargaining game. In Proceedings of the 39th International Conference on Machine Learning, K. Chaudhuri, S. Jegelka, L. Song, C. Szepesvari, G. Niu, and S. Sabato (Eds.), Proceedings of Machine Learning Research, Vol. 162, pp. 16428–16446. Cited by: [§1](#S1.p3.1 "1 Introduction ‣ Uniform Loss vs. Specialized Optimization: A Comparative Analysis in Multi-Task Learning"), [§1](#S1.p4.1 "1 Introduction ‣ Uniform Loss vs. Specialized Optimization: A Comparative Analysis in Multi-Task Learning"), [§1](#S1.p5.1 "1 Introduction ‣ Uniform Loss vs. Specialized Optimization: A Comparative Analysis in Multi-Task Learning"), [§2](#S2.p2.1 "2 Related Work ‣ Uniform Loss vs. Specialized Optimization: A Comparative Analysis in Multi-Task Learning"), [§2](#S2.p5.1 "2 Related Work ‣ Uniform Loss vs. Specialized Optimization: A Comparative Analysis in Multi-Task Learning"), [§3.1](#S3.SS1.p4.1 "3.1 Selecting promising SMTOs for further evaluation ‣ 3 Experimental Evaluation ‣ Uniform Loss vs. Specialized Optimization: A Comparative Analysis in Multi-Task Learning"), [§3.3](#S3.SS3.p1.1 "3.3 Complexity analysis ‣ 3 Experimental Evaluation ‣ Uniform Loss vs. Specialized Optimization: A Comparative Analysis in Multi-Task Learning"), [§3.4](#S3.SS4.p1.1 "3.4 QM9 ‣ 3 Experimental Evaluation ‣ Uniform Loss vs. Specialized Optimization: A Comparative Analysis in Multi-Task Learning"), [§3.4](#S3.SS4.p2.1 "3.4 QM9 ‣ 3 Experimental Evaluation ‣ Uniform Loss vs. Specialized Optimization: A Comparative Analysis in Multi-Task Learning"), [§3](#S3.p2.1 "3 Experimental Evaluation ‣ Uniform Loss vs. Specialized Optimization: A Comparative Analysis in Multi-Task Learning"), [§3](#S3.p3.1 "3 Experimental Evaluation ‣ Uniform Loss vs. Specialized Optimization: A Comparative Analysis in Multi-Task Learning"), [§3](#S3.p4.1 "3 Experimental Evaluation ‣ Uniform Loss vs. Specialized Optimization: A Comparative Analysis in Multi-Task Learning"), [Proposition 2](#Thmproposition2.p1.1.1 "Proposition 2. ‣ A.4 Proofs ‣ Appendix A Technical Appendices and Supplementary Material ‣ Uniform Loss vs. Specialized Optimization: A Comparative Analysis in Multi-Task Learning").
*   \[11\] L. Liu, Y. Li, Z. Kuang, J. Xue, Y. Chen, W. Yang, Q. Liao, and W. Zhang (2021) Towards impartial multi-task learning. In International Conference on Learning Representations, External Links: [Link](https://openreview.net/forum?id=IMPnRXEWpvr "") Cited by: [§1](#S1.p4.1 "1 Introduction ‣ Uniform Loss vs. Specialized Optimization: A Comparative Analysis in Multi-Task Learning"), [§1](#S1.p5.1 "1 Introduction ‣ Uniform Loss vs. Specialized Optimization: A Comparative Analysis in Multi-Task Learning"), [§3.1](#S3.SS1.p4.1 "3.1 Selecting promising SMTOs for further evaluation ‣ 3 Experimental Evaluation ‣ Uniform Loss vs. Specialized Optimization: A Comparative Analysis in Multi-Task Learning"), [§3.1](#S3.SS1.p5.1 "3.1 Selecting promising SMTOs for further evaluation ‣ 3 Experimental Evaluation ‣ Uniform Loss vs. Specialized Optimization: A Comparative Analysis in Multi-Task Learning"), [§3.3](#S3.SS3.p3.1 "3.3 Complexity analysis ‣ 3 Experimental Evaluation ‣ Uniform Loss vs. Specialized Optimization: A Comparative Analysis in Multi-Task Learning"), [§3](#S3.p3.1 "3 Experimental Evaluation ‣ Uniform Loss vs. Specialized Optimization: A Comparative Analysis in Multi-Task Learning"), [§3](#S3.p4.1 "3 Experimental Evaluation ‣ Uniform Loss vs. Specialized Optimization: A Comparative Analysis in Multi-Task Learning"), [Proposition 3](#Thmproposition3.p1.1.1 "Proposition 3. ‣ A.4 Proofs ‣ Appendix A Technical Appendices and Supplementary Material ‣ Uniform Loss vs. Specialized Optimization: A Comparative Analysis in Multi-Task Learning").
*   \[12\] A. Javaloy and I. Valera (2022) RotoGrad: gradient homogenization in multitask learning. In International Conference on Learning Representations, Cited by: [§A.2.2](#A1.SS2.SSS2.p1.1 "A.2.2 Statistics ‣ A.2 Additional results ‣ Appendix A Technical Appendices and Supplementary Material ‣ Uniform Loss vs. Specialized Optimization: A Comparative Analysis in Multi-Task Learning"), [§1](#S1.p4.1 "1 Introduction ‣ Uniform Loss vs. Specialized Optimization: A Comparative Analysis in Multi-Task Learning"), [§1](#S1.p5.1 "1 Introduction ‣ Uniform Loss vs. Specialized Optimization: A Comparative Analysis in Multi-Task Learning"), [§2](#S2.p1.1 "2 Related Work ‣ Uniform Loss vs. Specialized Optimization: A Comparative Analysis in Multi-Task Learning"), [§2](#S2.p5.1 "2 Related Work ‣ Uniform Loss vs. Specialized Optimization: A Comparative Analysis in Multi-Task Learning"), [§3.1](#S3.SS1.p2.1 "3.1 Selecting promising SMTOs for further evaluation ‣ 3 Experimental Evaluation ‣ Uniform Loss vs. Specialized Optimization: A Comparative Analysis in Multi-Task Learning"), [§3.1](#S3.SS1.p4.1 "3.1 Selecting promising SMTOs for further evaluation ‣ 3 Experimental Evaluation ‣ Uniform Loss vs. Specialized Optimization: A Comparative Analysis in Multi-Task Learning"), [§3](#S3.p3.1 "3 Experimental Evaluation ‣ Uniform Loss vs. Specialized Optimization: A Comparative Analysis in Multi-Task Learning"), [§4](#S4.p1.1 "4 Limitations ‣ Uniform Loss vs. Specialized Optimization: A Comparative Analysis in Multi-Task Learning").
*   \[13\] Z. Chen, V. Badrinarayanan, C. Lee, and A. Rabinovich (2018) Gradnorm: gradient normalization for adaptive loss balancing in deep multitask networks. In International conference on machine learning, pp. 794–803. Cited by: [§1](#S1.p5.1 "1 Introduction ‣ Uniform Loss vs. Specialized Optimization: A Comparative Analysis in Multi-Task Learning"), [§3](#S3.p3.1 "3 Experimental Evaluation ‣ Uniform Loss vs. Specialized Optimization: A Comparative Analysis in Multi-Task Learning"), [§3](#S3.p4.1 "3 Experimental Evaluation ‣ Uniform Loss vs. Specialized Optimization: A Comparative Analysis in Multi-Task Learning").
*   \[14\] A. Katrutsa, D. Merkulov, N. Tursynbek, and I. Oseledets (2020) Follow the bisector: a simple method for multi-objective optimization. arXiv preprint arXiv:2007.06937. Cited by: [§A.4](#A1.SS4.p11.1.1 "Proof. ‣ A.4 Proofs ‣ Appendix A Technical Appendices and Supplementary Material ‣ Uniform Loss vs. Specialized Optimization: A Comparative Analysis in Multi-Task Learning"), [§A.4](#A1.SS4.p15.1.1 "Proof. ‣ A.4 Proofs ‣ Appendix A Technical Appendices and Supplementary Material ‣ Uniform Loss vs. Specialized Optimization: A Comparative Analysis in Multi-Task Learning"), [§1](#S1.p5.1 "1 Introduction ‣ Uniform Loss vs. Specialized Optimization: A Comparative Analysis in Multi-Task Learning"), [§2](#S2.p1.1 "2 Related Work ‣ Uniform Loss vs. Specialized Optimization: A Comparative Analysis in Multi-Task Learning"), [§3.1](#S3.SS1.p4.1 "3.1 Selecting promising SMTOs for further evaluation ‣ 3 Experimental Evaluation ‣ Uniform Loss vs. Specialized Optimization: A Comparative Analysis in Multi-Task Learning"), [§3](#S3.p3.1 "3 Experimental Evaluation ‣ Uniform Loss vs. Specialized Optimization: A Comparative Analysis in Multi-Task Learning"), [§3](#S3.p4.1 "3 Experimental Evaluation ‣ Uniform Loss vs. Specialized Optimization: A Comparative Analysis in Multi-Task Learning"), [Proposition 4](#Thmproposition4.p1.1.1 "Proposition 4. ‣ A.4 Proofs ‣ Appendix A Technical Appendices and Supplementary Material ‣ Uniform Loss vs. Specialized Optimization: A Comparative Analysis in Multi-Task Learning").
*   \[15\] Z. Chen, J. Ngiam, Y. Huang, T. Luong, H. Kretzschmar, Y. Chai, and D. Anguelov (2020) Just pick a sign: optimizing deep multitask models with gradient sign dropout. Advances in Neural Information Processing Systems 33, pp. 2039–2050. Cited by: [§1](#S1.p5.1 "1 Introduction ‣ Uniform Loss vs. Specialized Optimization: A Comparative Analysis in Multi-Task Learning"), [§3](#S3.p3.1 "3 Experimental Evaluation ‣ Uniform Loss vs. Specialized Optimization: A Comparative Analysis in Multi-Task Learning").
*   \[16\] A. T. M. Nakamura, V. Grassi Jr, and D. F. Wolf (2022) Leveraging convergence behavior to balance conflicting tasks in multi-task learning. Neurocomputing 511, pp. 43–53. Cited by: [§1](#S1.p5.1 "1 Introduction ‣ Uniform Loss vs. Specialized Optimization: A Comparative Analysis in Multi-Task Learning"), [§2](#S2.p1.1 "2 Related Work ‣ Uniform Loss vs. Specialized Optimization: A Comparative Analysis in Multi-Task Learning"), [§2](#S2.p5.1 "2 Related Work ‣ Uniform Loss vs. Specialized Optimization: A Comparative Analysis in Multi-Task Learning"), [§3.1](#S3.SS1.p2.1 "3.1 Selecting promising SMTOs for further evaluation ‣ 3 Experimental Evaluation ‣ Uniform Loss vs. Specialized Optimization: A Comparative Analysis in Multi-Task Learning"), [§3.1](#S3.SS1.p3.1 "3.1 Selecting promising SMTOs for further evaluation ‣ 3 Experimental Evaluation ‣ Uniform Loss vs. Specialized Optimization: A Comparative Analysis in Multi-Task Learning"), [§3.1](#S3.SS1.p4.1 "3.1 Selecting promising SMTOs for further evaluation ‣ 3 Experimental Evaluation ‣ Uniform Loss vs. Specialized Optimization: A Comparative Analysis in Multi-Task Learning"), [§3](#S3.p3.1 "3 Experimental Evaluation ‣ Uniform Loss vs. Specialized Optimization: A Comparative Analysis in Multi-Task Learning"), [Proposition 5](#Thmproposition5.p1.1.1 "Proposition 5. ‣ A.4 Proofs ‣ Appendix A Technical Appendices and Supplementary Material ‣ Uniform Loss vs. Specialized Optimization: A Comparative Analysis in Multi-Task Learning").
*   \[17\] R. Cipolla, Y. Gal, and A. Kendall (2018) Multi-task learning using uncertainty to weigh losses for scene geometry and semantics. In 2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition, Vol. , pp. 7482–7491. External Links: [Document](https://dx.doi.org/10.1109/CVPR.2018.00781 "") Cited by: [§1](#S1.p5.1 "1 Introduction ‣ Uniform Loss vs. Specialized Optimization: A Comparative Analysis in Multi-Task Learning"), [§2](#S2.p1.1 "2 Related Work ‣ Uniform Loss vs. Specialized Optimization: A Comparative Analysis in Multi-Task Learning"), [§3.3](#S3.SS3.p1.1 "3.3 Complexity analysis ‣ 3 Experimental Evaluation ‣ Uniform Loss vs. Specialized Optimization: A Comparative Analysis in Multi-Task Learning"), [§3](#S3.p3.1 "3 Experimental Evaluation ‣ Uniform Loss vs. Specialized Optimization: A Comparative Analysis in Multi-Task Learning").
*   \[18\] S. Liu, S. James, A. Davison, and E. Johns (2022) Auto-lambda: disentangling dynamic task relationships. Transactions on Machine Learning Research. Note: External Links: ISSN 2835-8856, [Link](https://openreview.net/forum?id=KKeCMim5VN "") Cited by: [§1](#S1.p5.1 "1 Introduction ‣ Uniform Loss vs. Specialized Optimization: A Comparative Analysis in Multi-Task Learning"), [§2](#S2.p5.1 "2 Related Work ‣ Uniform Loss vs. Specialized Optimization: A Comparative Analysis in Multi-Task Learning"), [§3.1](#S3.SS1.p4.1 "3.1 Selecting promising SMTOs for further evaluation ‣ 3 Experimental Evaluation ‣ Uniform Loss vs. Specialized Optimization: A Comparative Analysis in Multi-Task Learning"), [§3](#S3.p3.1 "3 Experimental Evaluation ‣ Uniform Loss vs. Specialized Optimization: A Comparative Analysis in Multi-Task Learning"), [§4](#S4.p1.1 "4 Limitations ‣ Uniform Loss vs. Specialized Optimization: A Comparative Analysis in Multi-Task Learning").
*   \[19\] B. Lin, F. YE, Y. Zhang, and I. Tsang (2022) Reasonable effectiveness of random weighting: a litmus test for multi-task learning. Transactions on Machine Learning Research. Note: External Links: ISSN 2835-8856, [Link](https://openreview.net/forum?id=jjtFD8A1Wx "") Cited by: [§1](#S1.p5.1 "1 Introduction ‣ Uniform Loss vs. Specialized Optimization: A Comparative Analysis in Multi-Task Learning"), [§3](#S3.p3.1 "3 Experimental Evaluation ‣ Uniform Loss vs. Specialized Optimization: A Comparative Analysis in Multi-Task Learning").
*   \[20\] B. Liu, Y. Feng, P. Stone, and Q. Liu (2023) Famo: fast adaptive multitask optimization. Advances in Neural Information Processing Systems 36, pp. 57226–57243. Cited by: [§1](#S1.p5.1 "1 Introduction ‣ Uniform Loss vs. Specialized Optimization: A Comparative Analysis in Multi-Task Learning"), [§2](#S2.p5.1 "2 Related Work ‣ Uniform Loss vs. Specialized Optimization: A Comparative Analysis in Multi-Task Learning"), [§3.1](#S3.SS1.p4.1 "3.1 Selecting promising SMTOs for further evaluation ‣ 3 Experimental Evaluation ‣ Uniform Loss vs. Specialized Optimization: A Comparative Analysis in Multi-Task Learning"), [§3.3](#S3.SS3.p1.1 "3.3 Complexity analysis ‣ 3 Experimental Evaluation ‣ Uniform Loss vs. Specialized Optimization: A Comparative Analysis in Multi-Task Learning"), [§3.4](#S3.SS4.p1.1 "3.4 QM9 ‣ 3 Experimental Evaluation ‣ Uniform Loss vs. Specialized Optimization: A Comparative Analysis in Multi-Task Learning"), [§3.5](#S3.SS5.p7.1 "3.5 Fixed weights ‣ 3 Experimental Evaluation ‣ Uniform Loss vs. Specialized Optimization: A Comparative Analysis in Multi-Task Learning"), [§3](#S3.p2.1 "3 Experimental Evaluation ‣ Uniform Loss vs. Specialized Optimization: A Comparative Analysis in Multi-Task Learning"), [§3](#S3.p3.1 "3 Experimental Evaluation ‣ Uniform Loss vs. Specialized Optimization: A Comparative Analysis in Multi-Task Learning").
*   \[21\] J. Shen, C. Wang, Z. Xiao, N. V. Noord, and M. Worring (2024) GO4align: group optimization for multi-task alignment. In The Thirty-eighth Annual Conference on Neural Information Processing Systems, Cited by: [§1](#S1.p5.1 "1 Introduction ‣ Uniform Loss vs. Specialized Optimization: A Comparative Analysis in Multi-Task Learning"), [§3.5](#S3.SS5.p7.1 "3.5 Fixed weights ‣ 3 Experimental Evaluation ‣ Uniform Loss vs. Specialized Optimization: A Comparative Analysis in Multi-Task Learning"), [§3](#S3.p3.1 "3 Experimental Evaluation ‣ Uniform Loss vs. Specialized Optimization: A Comparative Analysis in Multi-Task Learning").
*   \[22\] I. Achituve, I. Diamant, A. Netzer, G. Chechik, and E. Fetaya (2024) Bayesian uncertainty for gradient aggregation in multi-task learning. In Forty-first International Conference on Machine Learning, Cited by: [§1](#S1.p5.1 "1 Introduction ‣ Uniform Loss vs. Specialized Optimization: A Comparative Analysis in Multi-Task Learning"), [§3](#S3.p3.1 "3 Experimental Evaluation ‣ Uniform Loss vs. Specialized Optimization: A Comparative Analysis in Multi-Task Learning").
*   \[23\] D. Xin, B. Ghorbani, J. Gilmer, A. Garg, and O. Firat (2022) Do current multi-task optimization methods in deep learning even help?. Advances in Neural Information Processing Systems 35, pp. 13597–13609. Cited by: [3rd item](#S1.I1.i3.p1.1 "In 1 Introduction ‣ Uniform Loss vs. Specialized Optimization: A Comparative Analysis in Multi-Task Learning"), [§1](#S1.p6.1 "1 Introduction ‣ Uniform Loss vs. Specialized Optimization: A Comparative Analysis in Multi-Task Learning"), [§2](#S2.p2.1 "2 Related Work ‣ Uniform Loss vs. Specialized Optimization: A Comparative Analysis in Multi-Task Learning"), [§2](#S2.p4.1 "2 Related Work ‣ Uniform Loss vs. Specialized Optimization: A Comparative Analysis in Multi-Task Learning"), [1st item](#S3.I1.i1.p1.1 "In 3.3 Complexity analysis ‣ 3 Experimental Evaluation ‣ Uniform Loss vs. Specialized Optimization: A Comparative Analysis in Multi-Task Learning"), [§3.1](#S3.SS1.p1.1 "3.1 Selecting promising SMTOs for further evaluation ‣ 3 Experimental Evaluation ‣ Uniform Loss vs. Specialized Optimization: A Comparative Analysis in Multi-Task Learning"), [§3.1](#S3.SS1.p2.1 "3.1 Selecting promising SMTOs for further evaluation ‣ 3 Experimental Evaluation ‣ Uniform Loss vs. Specialized Optimization: A Comparative Analysis in Multi-Task Learning"), [§3.2](#S3.SS2.p1.1 "3.2 Two similar tasks ‣ 3 Experimental Evaluation ‣ Uniform Loss vs. Specialized Optimization: A Comparative Analysis in Multi-Task Learning"), [§3.2](#S3.SS2.p5.1 "3.2 Two similar tasks ‣ 3 Experimental Evaluation ‣ Uniform Loss vs. Specialized Optimization: A Comparative Analysis in Multi-Task Learning"), [§3.3](#S3.SS3.p1.1 "3.3 Complexity analysis ‣ 3 Experimental Evaluation ‣ Uniform Loss vs. Specialized Optimization: A Comparative Analysis in Multi-Task Learning"), [§3.5](#S3.SS5.p1.1 "3.5 Fixed weights ‣ 3 Experimental Evaluation ‣ Uniform Loss vs. Specialized Optimization: A Comparative Analysis in Multi-Task Learning"), [§3.5](#S3.SS5.p2.1 "3.5 Fixed weights ‣ 3 Experimental Evaluation ‣ Uniform Loss vs. Specialized Optimization: A Comparative Analysis in Multi-Task Learning"), [§3.5](#S3.SS5.p7.1 "3.5 Fixed weights ‣ 3 Experimental Evaluation ‣ Uniform Loss vs. Specialized Optimization: A Comparative Analysis in Multi-Task Learning"), [§3](#S3.p1.1 "3 Experimental Evaluation ‣ Uniform Loss vs. Specialized Optimization: A Comparative Analysis in Multi-Task Learning"), [§4](#S4.p1.1 "4 Limitations ‣ Uniform Loss vs. Specialized Optimization: A Comparative Analysis in Multi-Task Learning").
*   \[24\] V. Kurin, A. De Palma, I. Kostrikov, S. Whiteson, and P. K. Mudigonda (2022) In defense of the unitary scalarization for deep multi-task learning. Advances in Neural Information Processing Systems 35, pp. 12169–12183. Cited by: [§1](#S1.p6.1 "1 Introduction ‣ Uniform Loss vs. Specialized Optimization: A Comparative Analysis in Multi-Task Learning"), [§2](#S2.p2.1 "2 Related Work ‣ Uniform Loss vs. Specialized Optimization: A Comparative Analysis in Multi-Task Learning"), [§2](#S2.p3.1 "2 Related Work ‣ Uniform Loss vs. Specialized Optimization: A Comparative Analysis in Multi-Task Learning"), [Figure 2](#S3.F2 "In 3.2 Two similar tasks ‣ 3 Experimental Evaluation ‣ Uniform Loss vs. Specialized Optimization: A Comparative Analysis in Multi-Task Learning"), [Figure 2](#S3.F2.6 "In 3.2 Two similar tasks ‣ 3 Experimental Evaluation ‣ Uniform Loss vs. Specialized Optimization: A Comparative Analysis in Multi-Task Learning"), [1st item](#S3.I1.i1.p1.1 "In 3.3 Complexity analysis ‣ 3 Experimental Evaluation ‣ Uniform Loss vs. Specialized Optimization: A Comparative Analysis in Multi-Task Learning"), [§3.1](#S3.SS1.p1.1 "3.1 Selecting promising SMTOs for further evaluation ‣ 3 Experimental Evaluation ‣ Uniform Loss vs. Specialized Optimization: A Comparative Analysis in Multi-Task Learning"), [§3.1](#S3.SS1.p2.1 "3.1 Selecting promising SMTOs for further evaluation ‣ 3 Experimental Evaluation ‣ Uniform Loss vs. Specialized Optimization: A Comparative Analysis in Multi-Task Learning"), [§3.2](#S3.SS2.p1.1 "3.2 Two similar tasks ‣ 3 Experimental Evaluation ‣ Uniform Loss vs. Specialized Optimization: A Comparative Analysis in Multi-Task Learning"), [§3.3](#S3.SS3.p1.1 "3.3 Complexity analysis ‣ 3 Experimental Evaluation ‣ Uniform Loss vs. Specialized Optimization: A Comparative Analysis in Multi-Task Learning"), [§3.3](#S3.SS3.p3.1 "3.3 Complexity analysis ‣ 3 Experimental Evaluation ‣ Uniform Loss vs. Specialized Optimization: A Comparative Analysis in Multi-Task Learning"), [§3.3](#S3.SS3.p8.1 "3.3 Complexity analysis ‣ 3 Experimental Evaluation ‣ Uniform Loss vs. Specialized Optimization: A Comparative Analysis in Multi-Task Learning"), [§3.4](#S3.SS4.p4.1 "3.4 QM9 ‣ 3 Experimental Evaluation ‣ Uniform Loss vs. Specialized Optimization: A Comparative Analysis in Multi-Task Learning"), [§3](#S3.p1.1 "3 Experimental Evaluation ‣ Uniform Loss vs. Specialized Optimization: A Comparative Analysis in Multi-Task Learning"), [§3](#S3.p4.1 "3 Experimental Evaluation ‣ Uniform Loss vs. Specialized Optimization: A Comparative Analysis in Multi-Task Learning"), [§4](#S4.p1.1 "4 Limitations ‣ Uniform Loss vs. Specialized Optimization: A Comparative Analysis in Multi-Task Learning").
*   \[25\] Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner (1998) Gradient-based learning applied to document recognition. Proceedings of the IEEE 86 (11), pp. 2278–2324. Cited by: [1st item](#S1.I1.i1.p1.1 "In 1 Introduction ‣ Uniform Loss vs. Specialized Optimization: A Comparative Analysis in Multi-Task Learning"), [§3.1](#S3.SS1.p3.1 "3.1 Selecting promising SMTOs for further evaluation ‣ 3 Experimental Evaluation ‣ Uniform Loss vs. Specialized Optimization: A Comparative Analysis in Multi-Task Learning"), [§3](#S3.p1.1 "3 Experimental Evaluation ‣ Uniform Loss vs. Specialized Optimization: A Comparative Analysis in Multi-Task Learning").
*   \[26\] M. Cordts, M. Omran, S. Ramos, T. Rehfeld, M. Enzweiler, R. Benenson, U. Franke, S. Roth, and B. Schiele (2016) The cityscapes dataset for semantic urban scene understanding. In Proc. of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), Cited by: [1st item](#S1.I1.i1.p1.1 "In 1 Introduction ‣ Uniform Loss vs. Specialized Optimization: A Comparative Analysis in Multi-Task Learning"), [§3.3](#S3.SS3.p1.1 "3.3 Complexity analysis ‣ 3 Experimental Evaluation ‣ Uniform Loss vs. Specialized Optimization: A Comparative Analysis in Multi-Task Learning"), [§3](#S3.p1.1 "3 Experimental Evaluation ‣ Uniform Loss vs. Specialized Optimization: A Comparative Analysis in Multi-Task Learning").
*   \[27\] Z. Wu, B. Ramsundar, E. N. Feinberg, J. Gomes, C. Geniesse, A. S. Pappu, K. Leswing, and V. Pande (2018) MoleculeNet: a benchmark for molecular machine learning. Chemical science 9 (2), pp. 513–530. Cited by: [1st item](#S1.I1.i1.p1.1 "In 1 Introduction ‣ Uniform Loss vs. Specialized Optimization: A Comparative Analysis in Multi-Task Learning"), [§3.4](#S3.SS4.p1.1 "3.4 QM9 ‣ 3 Experimental Evaluation ‣ Uniform Loss vs. Specialized Optimization: A Comparative Analysis in Multi-Task Learning"), [§3](#S3.p1.1 "3 Experimental Evaluation ‣ Uniform Loss vs. Specialized Optimization: A Comparative Analysis in Multi-Task Learning").
*   \[28\] B. Liu, X. Liu, X. Jin, P. Stone, and Q. Liu (2021) Conflict-averse gradient descent for multi-task learning. Advances in Neural Information Processing Systems 34, pp. 18878–18890. Cited by: [§2](#S2.p1.1 "2 Related Work ‣ Uniform Loss vs. Specialized Optimization: A Comparative Analysis in Multi-Task Learning"), [§2](#S2.p5.1 "2 Related Work ‣ Uniform Loss vs. Specialized Optimization: A Comparative Analysis in Multi-Task Learning"), [§3.1](#S3.SS1.p4.1 "3.1 Selecting promising SMTOs for further evaluation ‣ 3 Experimental Evaluation ‣ Uniform Loss vs. Specialized Optimization: A Comparative Analysis in Multi-Task Learning"), [§3](#S3.p3.1 "3 Experimental Evaluation ‣ Uniform Loss vs. Specialized Optimization: A Comparative Analysis in Multi-Task Learning").
*   \[29\] T. Dietterich (1995) Overfitting and undercomputing in machine learning. ACM computing surveys (CSUR) 27 (3), pp. 326–327. Cited by: [§2](#S2.p3.1 "2 Related Work ‣ Uniform Loss vs. Specialized Optimization: A Comparative Analysis in Multi-Task Learning").
*   \[30\] N. S. Keskar, D. Mudigere, J. Nocedal, M. Smelyanskiy, and P. T. P. Tang (2016) On large-batch training for deep learning: generalization gap and sharp minima. International Conference on Learning Representations. Cited by: [§2](#S2.p3.1 "2 Related Work ‣ Uniform Loss vs. Specialized Optimization: A Comparative Analysis in Multi-Task Learning").
*   \[31\] B. Kleinberg, Y. Li, and Y. Yuan (2018) An alternative view: when does sgd escape local minima?. In International conference on machine learning, pp. 2698–2707. Cited by: [§2](#S2.p3.1 "2 Related Work ‣ Uniform Loss vs. Specialized Optimization: A Comparative Analysis in Multi-Task Learning").
*   \[32\] S. Ioffe and C. Szegedy (2015) Batch normalization: accelerating deep network training by reducing internal covariate shift. In International conference on machine learning, pp. 448–456. Cited by: [§2](#S2.p3.1 "2 Related Work ‣ Uniform Loss vs. Specialized Optimization: A Comparative Analysis in Multi-Task Learning").
*   \[33\] N. Srivastava, G. Hinton, A. Krizhevsky, I. Sutskever, and R. Salakhutdinov (2014) Dropout: a simple way to prevent neural networks from overfitting. The journal of machine learning research 15 (1), pp. 1929–1958. Cited by: [§2](#S2.p3.1 "2 Related Work ‣ Uniform Loss vs. Specialized Optimization: A Comparative Analysis in Multi-Task Learning").
*   \[34\] S. Vandenhende, S. Georgoulis, W. Van Gansbeke, M. Proesmans, D. Dai, and L. Van Gool (2021) Multi-task learning for dense prediction tasks: a survey. IEEE transactions on pattern analysis and machine intelligence 44 (7), pp. 3614–3633. Cited by: [§3](#S3.p2.1 "3 Experimental Evaluation ‣ Uniform Loss vs. Specialized Optimization: A Comparative Analysis in Multi-Task Learning").
*   \[35\] T. Yu, S. Kumar, A. Gupta, S. Levine, K. Hausman, and C. Finn (2020) Gradient surgery for multi-task learning. Advances in Neural Information Processing Systems 33, pp. 5824–5836. Cited by: [§3.1](#S3.SS1.p2.1 "3.1 Selecting promising SMTOs for further evaluation ‣ 3 Experimental Evaluation ‣ Uniform Loss vs. Specialized Optimization: A Comparative Analysis in Multi-Task Learning"), [§3](#S3.p3.1 "3 Experimental Evaluation ‣ Uniform Loss vs. Specialized Optimization: A Comparative Analysis in Multi-Task Learning").
*   \[36\] M. Guo, A. Haque, D. Huang, S. Yeung, and L. Fei-Fei (2018) Dynamic task prioritization for multitask learning. In Proceedings of the European conference on computer vision (ECCV), pp. 270–287. Cited by: [§3](#S3.p3.1 "3 Experimental Evaluation ‣ Uniform Loss vs. Specialized Optimization: A Comparative Analysis in Multi-Task Learning").
*   \[37\] B. Lin, W. Jiang, F. Ye, Y. Zhang, P. Chen, Y. Chen, and S. Liu (2023) A scale-invariant task balancing approach for multi-task learning. CoRR abs/2308.12029. Cited by: [§3](#S3.p3.1 "3 Experimental Evaluation ‣ Uniform Loss vs. Specialized Optimization: A Comparative Analysis in Multi-Task Learning").
*   \[38\] B. Lin, W. Jiang, F. Ye, Y. Zhang, P. Chen, Y. Chen, S. Liu, and J. T. Kwok (2023) Dual-balancing for multi-task learning. External Links: 2308.12029 Cited by: [§3](#S3.p3.1 "3 Experimental Evaluation ‣ Uniform Loss vs. Specialized Optimization: A Comparative Analysis in Multi-Task Learning").
*   \[39\] K. He, X. Zhang, S. Ren, and J. Sun (2016) Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 770–778. Cited by: [1st item](#S3.I1.i1.p1.1 "In 3.3 Complexity analysis ‣ 3 Experimental Evaluation ‣ Uniform Loss vs. Specialized Optimization: A Comparative Analysis in Multi-Task Learning").
*   \[40\] L. Chen, Y. Zhu, G. Papandreou, F. Schroff, and H. Adam (2018) Encoder-decoder with atrous separable convolution for semantic image segmentation. In Proceedings of the European conference on computer vision (ECCV), pp. 801–818. Cited by: [1st item](#S3.I1.i1.p1.1 "In 3.3 Complexity analysis ‣ 3 Experimental Evaluation ‣ Uniform Loss vs. Specialized Optimization: A Comparative Analysis in Multi-Task Learning").
*   \[41\] S. Wu, H. R. Zhang, and C. Ré (2020) Understanding and improving information transfer in multi-task learning. In International Conference on Learning Representations, External Links: [Link](https://openreview.net/forum?id=SylzhkBtDB "") Cited by: [§3.3](#S3.SS3.p5.1 "3.3 Complexity analysis ‣ 3 Experimental Evaluation ‣ Uniform Loss vs. Specialized Optimization: A Comparative Analysis in Multi-Task Learning").
*   \[42\] U. Shaham, M. Elbayad, V. Goswami, O. Levy, and S. Bhosale (2022) Causes and cures for interference in multilingual translation. arXiv preprint arXiv:2212.07530. Cited by: [§3.3](#S3.SS3.p6.1 "3.3 Complexity analysis ‣ 3 Experimental Evaluation ‣ Uniform Loss vs. Specialized Optimization: A Comparative Analysis in Multi-Task Learning").
*   \[43\] M. Fey and J. E. Lenssen (2019) Fast graph representation learning with PyTorch Geometric. In ICLR Workshop on Representation Learning on Graphs and Manifolds, Cited by: [§3.4](#S3.SS4.p2.1 "3.4 QM9 ‣ 3 Experimental Evaluation ‣ Uniform Loss vs. Specialized Optimization: A Comparative Analysis in Multi-Task Learning").
*   \[44\] J. Gilmer, S. S. Schoenholz, P. F. Riley, O. Vinyals, and G. E. Dahl (2017) Neural message passing for quantum chemistry. In International conference on machine learning, pp. 1263–1272. Cited by: [§3.4](#S3.SS4.p2.1 "3.4 QM9 ‣ 3 Experimental Evaluation ‣ Uniform Loss vs. Specialized Optimization: A Comparative Analysis in Multi-Task Learning").
*   \[45\] A. Royer, T. Blankevoort, and B. Ehteshami Bejnordi (2024) Scalarization for multi-task and multi-domain learning at scale. Advances in Neural Information Processing Systems 36. Cited by: [§3.5](#S3.SS5.p1.1 "3.5 Fixed weights ‣ 3 Experimental Evaluation ‣ Uniform Loss vs. Specialized Optimization: A Comparative Analysis in Multi-Task Learning").
*   \[46\] J. Frankle and M. Carbin (2019) The lottery ticket hypothesis: finding sparse, trainable neural networks. In International Conference on Learning Representations, External Links: [Link](https://openreview.net/forum?id=rJl-b3RcF7 "") Cited by: [§3.5](#S3.SS5.p6.1 "3.5 Fixed weights ‣ 3 Experimental Evaluation ‣ Uniform Loss vs. Specialized Optimization: A Comparative Analysis in Multi-Task Learning").
*   \[47\] Y. Hu, R. Xian, Q. Wu, Q. Fan, L. Yin, and H. Zhao (2023) Revisiting scalarization in multi-task learning: a theoretical perspective. In Thirty-seventh Conference on Neural Information Processing Systems, External Links: [Link](https://openreview.net/forum?id=6EqUpqMnwl "") Cited by: [§3.5](#S3.SS5.p6.1 "3.5 Fixed weights ‣ 3 Experimental Evaluation ‣ Uniform Loss vs. Specialized Optimization: A Comparative Analysis in Multi-Task Learning").
*   \[48\] A. Paszke, S. Gross, F. Massa, A. Lerer, J. Bradbury, G. Chanan, T. Killeen, Z. Lin, N. Gimelshein, L. Antiga, A. Desmaison, A. Kopf, E. Yang, Z. DeVito, M. Raison, A. Tejani, S. Chilamkurthy, B. Steiner, L. Fang, J. Bai, and S. Chintala (2019) PyTorch: an imperative style, high-performance deep learning library. In Advances in Neural Information Processing Systems, H. Wallach, H. Larochelle, A. Beygelzimer, F. d'Alché-Buc, E. Fox, and R. Garnett (Eds.), Vol. 32, pp. . External Links: [Link](https://proceedings.neurips.cc/paper_files/paper/2019/file/bdbca288fee7f92f2bfa9f7012727740-Paper.pdf "") Cited by: [§A.1](#A1.SS1.p1.1 "A.1 Experimental Setup ‣ Appendix A Technical Appendices and Supplementary Material ‣ Uniform Loss vs. Specialized Optimization: A Comparative Analysis in Multi-Task Learning").

## Appendix A Technical Appendices and Supplementary Material

### A.1 Experimental Setup

All experiments were implemented using PyTorch \[[48](#bib.bib45 "")\] and executed on a machine equipped with an AMD Ryzen 9 5950X CPU and two NVIDIA RTX 3090 GPUs (24 GB each). However, experiments were run independently on each GPU. The random seed was set based on the corresponding run index.

### A.2 Additional results

#### A.2.1 Cosine similarity

Figure [7](#A1.F7 "Figure 7 ‣ A.2.1 Cosine similarity ‣ A.2 Additional results ‣ Appendix A Technical Appendices and Supplementary Material ‣ Uniform Loss vs. Specialized Optimization: A Comparative Analysis in Multi-Task Learning") demonstrates the strong correlation between the mean cosine similarity of task gradients and the relative performance of Unit. Scal. and SMTOs. Notably, the tasks where Unit. Scal. achieved comparable results to SMTOs (MNIST CL\_CR, MNIST RL\_RR, City R50, and City R18) exhibit high mean cosine similarities (≈\\approx0.7) and form a distinct cluster. Conversely, the more challenging Cityscapes tasks (City Inst. Seg., City HR, City 19C) show a lower, yet consistent, similarity (≈\\approx0.45). Finally, the most difficult task from the QM9 dataset, is characterized by the lowest mean cosine similarity (≈\\approx0.3).

Figure 7: Mean cosine similarity between each task gradient and the average direction for all problems. It was able to correctly rank all the different problems in terms of relative performance between Unit. Scal. and SMTOs. (i.e. higher cosine similarity relates to lower difference between Unit. Scal. and SMTOs.)

#### A.2.2 Statistics

Similarly to \[[12](#bib.bib17 "")\], we provide additional statistics in Tables [3](#A1.T3 "Table 3 ‣ A.2.2 Statistics ‣ A.2 Additional results ‣ Appendix A Technical Appendices and Supplementary Material ‣ Uniform Loss vs. Specialized Optimization: A Comparative Analysis in Multi-Task Learning"), [4](#A1.T4 "Table 4 ‣ A.2.2 Statistics ‣ A.2 Additional results ‣ Appendix A Technical Appendices and Supplementary Material ‣ Uniform Loss vs. Specialized Optimization: A Comparative Analysis in Multi-Task Learning"), and [5](#A1.T5 "Table 5 ‣ A.2.2 Statistics ‣ A.2 Additional results ‣ Appendix A Technical Appendices and Supplementary Material ‣ Uniform Loss vs. Specialized Optimization: A Comparative Analysis in Multi-Task Learning") and the missing visual result in Figure [8](#A1.F8 "Figure 8 ‣ A.2.2 Statistics ‣ A.2 Additional results ‣ Appendix A Technical Appendices and Supplementary Material ‣ Uniform Loss vs. Specialized Optimization: A Comparative Analysis in Multi-Task Learning"). Most notably, we see a clear preference for the classification task on the MNIST dataset from the underperforming SMTOs, as well as a lower minimum value and higher standard deviation in Table [5](#A1.T5 "Table 5 ‣ A.2.2 Statistics ‣ A.2 Additional results ‣ Appendix A Technical Appendices and Supplementary Material ‣ Uniform Loss vs. Specialized Optimization: A Comparative Analysis in Multi-Task Learning") for Unit. Scal. when compared to the top performing SMTOs, FAMO and Nash-MTL.

Table 3: Mean Δm​t​m(%)↑\\Delta\_{mtm}(\\%)\\uparrow of different SMTOs on the MNIST dataset.

SMTO

CL

RR

avg

Unit. Scal.

-1.043

-16.86

-8.952

Auto-Lambda

-1.232

-7.186

-4.209

CAGrad

-1.685

-2.590

-2.137

CDTT

-1.546

-3.596

-2.571

EDM

-1.613

-2.993

-2.303

FAMO

-4.587

-0.745

-2.666

GradDrop

-1.076

-22.04

-11.56

IMTL

-2.513

-1.975

-2.244

MGDA-UB

-1.448

-11.26

-6.352

Nash-MTL

-1.695

-4.430

-3.063

PCGrad

-1.039

-15.36

-8.201

RLW-Dirichlet

-1.145

-19.48

-10.31

RLW-Normal

-1.100

-18.53

-9.817

RotoGrad

-2.705

-1.572

-2.139

SI

-0.962

-27.47

-14.21

UW

-1.109

-26.30

-13.70

Table 4: Mean Δm​t​m(%)↑\\Delta\_{mtm}(\\%)\\uparrow of different SMTOs on the Cityscapes dataset.

SMTO

Segmentation

Disparity

Instance Seg.

avg

Acc

mIoU

L1 abs

L1 rel

L1 abs

Unit. Scal.

-2.120

-16.78

-8.542

-11.41

3.492

-5.312

Auto-Lambda

-1.922

-15.02

-8.546

-9.317

3.657

-4.582

CAGrad

-0.977

-7.647

-6.709

-10.06

3.325

-3.124

CDTT

-0.948

-8.954

2.043

-0.224

0.472

-1.190

EDM

-0.225

-3.174

0.990

-1.051

2.468

0.246

FAMO

-0.274

-3.268

0.755

-2.212

2.339

-0.054

IMTL

-0.224

-2.759

3.017

0.993

1.195

0.570

Nash-MTL

-0.290

-3.275

0.297

-0.227

1.409

-0.113

RotoGrad

-0.453

-4.484

0.303

-3.493

-1.659

-1.908

Table 5: Performance of different SMTOs on the QM9 dataset based on Δm​t​m(%)\\Delta\_{mtm}(\\%) statistics.

SMTO

min ↑\\uparrow

max ↑\\uparrow

med ↑\\uparrow

std ↓\\downarrow

avg ↑\\uparrow

Unit. Scal.

-849.3

-8.280

-61.62

229.4

-135.7

Auto-Lambda

-974.5

-18.32

-115.4

255.5

-186.9

CAGrad

-408.0

-5.451

-32.08

108.7

-70.13

CDTT

-652.9

-30.90

-70.34

169.7

-121.2

EDM

-526.0

-24.14

-56.26

137.0

-95.42

FAMO

-351.5

6.590

-16.95

97.78

-56.86

Nash-MTL

-390.2

15.10

-21.41

108.8

-53.62

Figure 8: Box plot comparing Δm​t​m\\Delta\_{mtm} between Unit. Scal. and SMTOs on the QM9 dataset.

### A.3 Rotograd’s instability on the QM9 dataset

The RotoGrad optimizer aims to mitigate conflicts in gradient directions by rotating the feature space so that the gradient of each task’s gradient aligns with the direction formed by the equally weighted sum of the normalized gradients: v\=1N​∑iNus​h,iv=\\frac{1}{N}\\sum\_{i}^{N}u\_{sh,i}, where us​h,i\=gs​h,i‖gs​h,i‖u\_{sh,i}=\\frac{g\_{sh,i}}{\\|g\_{sh,i}\\|} represents the normalized gradient of task ii for the shared parameters.

In the specific case of the QM9 experiment, RotoGrad failed to converge. We hypothesize that the relatively small final shared feature space, combined with the high number of tasks, significantly increased the difficulty of the balancing problem. As shown in Figure [9](#A1.F9 "Figure 9 ‣ A.3 Rotograd’s instability on the QM9 dataset ‣ Appendix A Technical Appendices and Supplementary Material ‣ Uniform Loss vs. Specialized Optimization: A Comparative Analysis in Multi-Task Learning"), the rotation loss ℒr​o​t\\mathcal{L}\_{rot} is orders of magnitude higher compared to the MNIST and Cityscapes problems. Since ℒr​o​t\\mathcal{L}\_{rot} does not directly enforce a reduction in task loss, this instability may have negatively affected the model optimization.

For this specific experiment, we used l​r\=0.001lr=0.001 as it was the optimal learning rate for most SMTOs, and l​r​r\=0.1lrr=0.1 since it was the smallest value used on the grid search, so ideally it should be the most stable.

Figure 9: Mean rotation loss (ℒr​o​t)\\left(\\mathcal{L}\_{rot}\\right) from RotoGrad. The ℒr​o​t\\mathcal{L}\_{rot} is at least 10×\\times higher than the other cases, which could explain the training instability.

### A.4 Proofs

###### Assumption 1.

Consider a model fθf\_{\\theta} composed of a shared encoder fθs​hf\_{\\theta\_{sh}} and task-specific modules fθtf\_{\\theta\_{t}} for each task t∈{1,…,N}t\\in\\{1,\\dots,N\\}. In the case of two tasks (N\=2N=2), we assume the following symmetry conditions: (i) both tasks use the same loss function, (ii) the task-specific modules have the same architecture, (iii) the tasks are of similar difficulty, and (iv) the tasks have labels with similar distributions. Under these conditions, we assume that the gradients with respect to the shared parameters have equal norm, i.e., ‖g1‖\=‖g2‖\\|g\_{1}\\|=\\|g\_{2}\\|.

###### Assumption 2.

Under the same conditions as Assumption [1](#Thmassumption1 "Assumption 1. ‣ A.4 Proofs ‣ Appendix A Technical Appendices and Supplementary Material ‣ Uniform Loss vs. Specialized Optimization: A Comparative Analysis in Multi-Task Learning"), assume that the loss values of each task are equal, i.e., ℓ1\=ℓ2\\ell\_{1}=\\ell\_{2}.

Although Assumptions [1](#Thmassumption1 "Assumption 1. ‣ A.4 Proofs ‣ Appendix A Technical Appendices and Supplementary Material ‣ Uniform Loss vs. Specialized Optimization: A Comparative Analysis in Multi-Task Learning"), [2](#Thmassumption2 "Assumption 2. ‣ A.4 Proofs ‣ Appendix A Technical Appendices and Supplementary Material ‣ Uniform Loss vs. Specialized Optimization: A Comparative Analysis in Multi-Task Learning") impose a strong symmetry requirement, our experiments on MNIST revealed that in both the dual classification and dual reconstruction tasks, the resulting weights from all SMTOs were consistently close to 0.5, with the sole exception of FAMO in the dual classification scenario.

###### Proposition 1.

Under Assumption [1](#Thmassumption1 "Assumption 1. ‣ A.4 Proofs ‣ Appendix A Technical Appendices and Supplementary Material ‣ Uniform Loss vs. Specialized Optimization: A Comparative Analysis in Multi-Task Learning"), the optimal weights computed by CAGrad correspond to equal weighting.

###### Proof.

We consider the CAGrad optimization for two tasks:

w∗\=arg⁡minw∈𝒲⁡gw⊤​g0+ϕ​‖gw‖,w^{\*}=\\mathop{\\arg\\min}\_{w\\in\\mathcal{W}}\\;g\_{w}^{\\top}g\_{0}+\\sqrt{\\phi}\\,\\|g\_{w}\\|, where:

gw\\displaystyle g\_{w}

\=w1​g1+w2​g2,\\displaystyle=w\_{1}g\_{1}+w\_{2}g\_{2}, g0\\displaystyle g\_{0}

\=12​(g1+g2),\\displaystyle=\\frac{1}{2}(g\_{1}+g\_{2}), ϕ\\displaystyle\\phi

\=c2​‖g0‖2,c∈\[0,1),\\displaystyle=c^{2}\\|g\_{0}\\|^{2},\\quad c\\in\[0,1),

𝒲\\displaystyle\\mathcal{W}

\={w∈ℝ2∣w1+w2\=1,w1,w2≥0}.\\displaystyle=\\{w\\in\\mathbb{R}^{2}\\mid w\_{1}+w\_{2}=1,\\;w\_{1},w\_{2}\\geq 0\\}.

We also define:

A\\displaystyle A

≔‖g1‖\=‖g2‖,\\displaystyle\\coloneqq\\|g\_{1}\\|=\\|g\_{2}\\|, p\\displaystyle p

≔g1⊤​g2.\\displaystyle\\coloneqq g\_{1}^{\\top}g\_{2}.

We begin by expanding the inner product:

gw⊤​g0\\displaystyle g\_{w}^{\\top}g\_{0}

\=(w1​g1+w2​g2)⊤​12​(g1+g2)\\displaystyle=\\left(w\_{1}g\_{1}+w\_{2}g\_{2}\\right)^{\\top}\\frac{1}{2}(g\_{1}+g\_{2})

\=12​\[w1​(g1⊤​g1+g1⊤​g2)+w2​(g2⊤​g1+g2⊤​g2)\]\\displaystyle=\\frac{1}{2}\\left\[w\_{1}(g\_{1}^{\\top}g\_{1}+g\_{1}^{\\top}g\_{2})+w\_{2}(g\_{2}^{\\top}g\_{1}+g\_{2}^{\\top}g\_{2})\\right\]

\=12​\[w1​(A2+p)+w2​(p+A2)\]\\displaystyle=\\frac{1}{2}\\left\[w\_{1}(A^{2}+p)+w\_{2}(p+A^{2})\\right\]

\=12​(A2+p)​(w1+w2)\=12​(A2+p).\\displaystyle=\\frac{1}{2}(A^{2}+p)(w\_{1}+w\_{2})=\\frac{1}{2}(A^{2}+p).

This term is independent of ww, as well as ϕ\\sqrt{\\phi}, so the optimization reduces to:

arg⁡minw∈𝒲⁡‖w1​g1+w2​g2‖.\\mathop{\\arg\\min}\_{w\\in\\mathcal{W}}\\|w\_{1}g\_{1}+w\_{2}g\_{2}\\|.

Let w∗w^{\*} be the optimal solution, which is known to be equal weights if ‖g1‖\=‖g2‖\\|g\_{1}\\|=\\|g\_{2}\\|. The CAGrad update direction is then given by:

d∗\=g0+ϕ‖gw∗‖​gw∗.d^{\*}=g\_{0}+\\frac{\\sqrt{\\phi}}{\\|g\_{w^{\*}}\\|}\\,g\_{w^{\*}}.

If w∗\=(12,12)w^{\*}=(\\tfrac{1}{2},\\tfrac{1}{2}), then gw∗\=12​(g1+g2)\=g0g\_{w^{\*}}=\\frac{1}{2}(g\_{1}+g\_{2})=g\_{0}, and:

d∗\=g0+c​‖g0‖‖g0‖​g0\=(1+c)​g0\=∑i\=12gi​(1+c2).d^{\*}=g\_{0}+\\frac{c\\|g\_{0}\\|}{\\|g\_{0}\\|}\\,g\_{0}=(1+c)\\,g\_{0}=\\sum\_{i=1}^{2}g\_{i}\\left(\\frac{1+c}{2}\\right).

Thus, d∗d^{\*} is an equal-weight combination of the gradients. ∎

###### Proposition 2.

Under Assumption [1](#Thmassumption1 "Assumption 1. ‣ A.4 Proofs ‣ Appendix A Technical Appendices and Supplementary Material ‣ Uniform Loss vs. Specialized Optimization: A Comparative Analysis in Multi-Task Learning"), the optimal weights computed by Nash-MTL \[[10](#bib.bib15 "")\] are equal weights in the two-task scenario.

###### Proof.

The weights are computed by approximating the solution of:

G⊤​G​α\=1α,G^{\\top}G\\alpha=\\frac{1}{\\alpha}, where GG is the matrix whose columns are the shared gradients and αi\>0,i∈{1,…,N}\\alpha\_{i}\\;>0,\\;i\\in\\{1,...,N\\} the tasks’ weights. In a two-task scenario, the problem can be simplified to:

{α1​g1⊤​g1+α2​g2⊤​g1\=1α1α2​g2⊤​g2+α1​g1⊤​g2\=1α2\\left\\{\\begin{aligned} \\alpha\_{1}g\_{1}^{\\top}g\_{1}+\\alpha\_{2}g\_{2}^{\\top}g\_{1}&=\\frac{1}{\\alpha\_{1}}\\\\ \\alpha\_{2}g\_{2}^{\\top}g\_{2}+\\alpha\_{1}g\_{1}^{\\top}g\_{2}&=\\frac{1}{\\alpha\_{2}}\\end{aligned}\\right.

Defining A≔‖g1‖\=‖g2‖A\\coloneqq\\|g\_{1}\\|=\\|g\_{2}\\| and p≔g1⊤​g2A2p\\coloneqq\\frac{g\_{1}^{\\top}g\_{2}}{A^{2}}:

{α1​A2+α2​A2​p\=1α1α2​A2+α1​A2​p\=1α2\\left\\{\\begin{aligned} \\alpha\_{1}A^{2}+\\alpha\_{2}A^{2}p&=\\frac{1}{\\alpha\_{1}}\\\\ \\alpha\_{2}A^{2}+\\alpha\_{1}A^{2}p&=\\frac{1}{\\alpha\_{2}}\\end{aligned}\\right.

Rearrange the equations:

{α12​A2+α1​α2​A2​p\=1α22​A2+α1​α2​A2​p\=1\\left\\{\\begin{aligned} \\alpha\_{1}^{2}A^{2}+\\alpha\_{1}\\alpha\_{2}A^{2}p&=1\\\\ \\alpha\_{2}^{2}A^{2}+\\alpha\_{1}\\alpha\_{2}A^{2}p&=1\\end{aligned}\\right.

Notice that by symmetry of the two equations, the solution should satisfy α12\=α22\\alpha\_{1}^{2}=\\alpha\_{2}^{2}. Since both weights are positive, this immediately implies:

α1\=α2.\\alpha\_{1}=\\alpha\_{2}.

∎

###### Proposition 3.

Under Assumption [1](#Thmassumption1 "Assumption 1. ‣ A.4 Proofs ‣ Appendix A Technical Appendices and Supplementary Material ‣ Uniform Loss vs. Specialized Optimization: A Comparative Analysis in Multi-Task Learning"), the optimal weights computed by IMTL-G \[[11](#bib.bib16 "")\] are equal weights in the two-task scenario.

###### Proof.

The weights are obtained by solving:

α\=g1​U⊤​(D​U⊤)−1,\\alpha=g\_{1}U^{\\top}(DU^{\\top})^{-1}, where α\={α2,…,αN}\\alpha=\\{\\alpha\_{2},...,\\alpha\_{N}\\} is constrained to ∑iαi\=1\\sum\_{i}\\alpha\_{i}=1, U⊤\=\[u1⊤−u2⊤,…,u1⊤−uN⊤\]U^{\\top}=\[u^{\\top}\_{1}-u^{\\top}\_{2},...,u^{\\top}\_{1}-u^{\\top}\_{N}\], with ui\=gi/‖gi‖u\_{i}=g\_{i}/\\|g\_{i}\\|, and D⊤\=\[g1⊤−g2⊤,…,g1⊤−gN⊤\]D^{\\top}=\[g\_{1}^{\\top}-g\_{2}^{\\top},...,g\_{1}^{\\top}-g\_{N}^{\\top}\]. Define A≔‖g1‖\=‖g2‖A\\coloneqq\\|g\_{1}\\|=\\|g\_{2}\\| and p\=g1⊤​g2p=g\_{1}^{\\top}g\_{2}. In the two-task scenario, the problem reduces to:

α2\\displaystyle\\alpha\_{2}

\=g1​(g1⊤−g2⊤A)​((g1−g2)​(g1⊤−g2⊤)A)−1\\displaystyle=g\_{1}\\left(\\frac{g\_{1}^{\\top}-g\_{2}^{\\top}}{A}\\right)\\left((g\_{1}-g\_{2})\\frac{(g\_{1}^{\\top}-g\_{2}^{\\top})}{A}\\right)^{-1}

α2\\displaystyle\\alpha\_{2}

\=g1​(g1⊤−g2⊤A)​(‖g1−g2‖2A)−1\\displaystyle=g\_{1}\\left(\\frac{g\_{1}^{\\top}-g\_{2}^{\\top}}{A}\\right)\\left(\\frac{\\|g\_{1}-g\_{2}\\|^{2}}{A}\\right)^{-1}

α2\\displaystyle\\alpha\_{2}

\=g1​(g1⊤−g2⊤‖g1−g2‖2)\=A2−p2​(A2−p)\=12.\\displaystyle=g\_{1}\\left(\\frac{g\_{1}^{\\top}-g\_{2}^{\\top}}{\\|g\_{1}-g\_{2}\\|^{2}}\\right)=\\frac{A^{2}-p}{2(A^{2}-p)}=\\frac{1}{2}.

Since α1+α2\=1\\alpha\_{1}+\\alpha\_{2}=1, this completes the proof.

∎

###### Proposition 4.

Under assumption [1](#Thmassumption1 "Assumption 1. ‣ A.4 Proofs ‣ Appendix A Technical Appendices and Supplementary Material ‣ Uniform Loss vs. Specialized Optimization: A Comparative Analysis in Multi-Task Learning"), the optimal weights computed by EDM \[[14](#bib.bib11 "")\] are equal weights in the two-task scenario.

###### Proof.

EDM’s original paper \[[14](#bib.bib11 "")\] already showed that for the two task scenario the optimal direction is computed as follows:

d∗\=(1‖g1‖+1‖g2‖)−1​(g1‖g1‖+g2‖g2‖).d^{\*}=\\left(\\frac{1}{\\|g\_{1}\\|}+\\frac{1}{\\|g\_{2}\\|}\\right)^{-1}\\left(\\frac{g\_{1}}{\\|g\_{1}\\|}+\\frac{g\_{2}}{\\|g\_{2}\\|}\\right).

Define A≔‖g1‖\=‖g2‖A\\coloneqq\\|g\_{1}\\|=\\|g\_{2}\\|:

d∗\=(1A+1A)−1​(g1A+g2A)\=12​(g1+g2).d^{\*}=\\left(\\frac{1}{A}+\\frac{1}{A}\\right)^{-1}\\left(\\frac{g\_{1}}{A}+\\frac{g\_{2}}{A}\\right)=\\frac{1}{2}(g\_{1}+g\_{2}).\\\\ 

∎

###### Proposition 5.

Under Assumptions [1](#Thmassumption1 "Assumption 1. ‣ A.4 Proofs ‣ Appendix A Technical Appendices and Supplementary Material ‣ Uniform Loss vs. Specialized Optimization: A Comparative Analysis in Multi-Task Learning") and [2](#Thmassumption2 "Assumption 2. ‣ A.4 Proofs ‣ Appendix A Technical Appendices and Supplementary Material ‣ Uniform Loss vs. Specialized Optimization: A Comparative Analysis in Multi-Task Learning"), the optimal weights computed by CDTT \[[16](#bib.bib18 "")\] are equal weights.

###### Proof.

CDTT applies a tension vector to the optimal direction d∗d^{\*} computed by EDM, which is equal weights in this case (see Proposition [4](#Thmproposition4 "Proposition 4. ‣ A.4 Proofs ‣ Appendix A Technical Appendices and Supplementary Material ‣ Uniform Loss vs. Specialized Optimization: A Comparative Analysis in Multi-Task Learning")). The final vector is computed by:

ζi\\displaystyle\\zeta\_{i}

\=1N​∑i\=1N‖gi​(t−i)‖\\displaystyle=\\frac{1}{N}\\sum\_{i=1}^{N}\\|g\_{i}(t-i)\\|

δi\\displaystyle\\delta\_{i}

\=‖ζi​(t)‖‖ζi​(t−1)‖+log10⁡(ℓi)\\displaystyle=\\frac{\\|\\zeta\_{i}(t)\\|}{\\|\\zeta\_{i}(t-1)\\|}+\\log\_{10}(\\ell\_{i})

ci\\displaystyle c\_{i}

\=α1+e(−δi​e+e)+1−α\\displaystyle=\\frac{\\alpha}{1+e^{(-\\delta\_{i}e+e)}}+1-\\alpha

dn∗\\displaystyle d\_{n}^{\*}

\=d∗+∑i\=1Nci​(gi−d∗‖gi−d∗‖),\\displaystyle=d^{\*}+\\sum^{N}\_{i=1}c\_{i}\\left(\\frac{g\_{i}-d^{\*}}{\\|g\_{i}-d^{\*}\\|}\\right), where tt refers to the iteration index and α∈\[0,1\]\\alpha\\in\[0,1\] is a constant to regulate the tension factor’s sensitivity.

It follows directly from Assumptions [1](#Thmassumption1 "Assumption 1. ‣ A.4 Proofs ‣ Appendix A Technical Appendices and Supplementary Material ‣ Uniform Loss vs. Specialized Optimization: A Comparative Analysis in Multi-Task Learning"), [2](#Thmassumption2 "Assumption 2. ‣ A.4 Proofs ‣ Appendix A Technical Appendices and Supplementary Material ‣ Uniform Loss vs. Specialized Optimization: A Comparative Analysis in Multi-Task Learning") and the component’s equations that c\=c1\=c2c=c\_{1}=c\_{2}. Analyzing the ‖gi−d∗‖\\|g\_{i}-d^{\*}\\| separately:

‖gi−d∗‖2\=‖gi‖2+‖d∗‖2−2​⟨gi,d∗⟩.\\|g\_{i}-d^{\*}\\|^{2}=\\|g\_{i}\\|^{2}+\\|d^{\*}\\|^{2}-2\\langle g\_{i},d^{\*}\\rangle.

The direction computed by EDM d∗d^{\*} is guaranteed to have the same angle with each of the shared gradients \[[14](#bib.bib11 "")\]. Therefore, if ‖g1‖\=‖g2‖,then​⟨g1,d∗⟩\=⟨g2,d∗⟩\\|g\_{1}\\|=\\|g\_{2}\\|,\\;\\text{then}\\;\\langle g\_{1},d^{\*}\\rangle=\\langle g\_{2},d^{\*}\\rangle. Then we can consider B≔‖g1−d∗‖\=‖g2−d∗‖B\\coloneqq\\|g\_{1}-d^{\*}\\|=\\|g\_{2}-d^{\*}\\| and remove it together with cc from the sum:

dn∗\\displaystyle d\_{n}^{\*}

\=d∗+cB​∑i\=12(gi−d∗)\\displaystyle=d^{\*}+\\frac{c}{B}\\sum^{2}\_{i=1}\\left(g\_{i}-d^{\*}\\right)

dn∗\\displaystyle d\_{n}^{\*}

\=d∗+cB​(g1+g2−2​d∗)\\displaystyle=d^{\*}+\\frac{c}{B}(g\_{1}+g\_{2}-2d^{\*})

dn∗\\displaystyle d\_{n}^{\*}

\=d∗\=12​(g1+g2).\\displaystyle=d^{\*}=\\frac{1}{2}(g\_{1}+g\_{2}).

∎

###### Proposition 6.

Under Assumption [2](#Thmassumption2 "Assumption 2. ‣ A.4 Proofs ‣ Appendix A Technical Appendices and Supplementary Material ‣ Uniform Loss vs. Specialized Optimization: A Comparative Analysis in Multi-Task Learning") (extended for N tasks), if the logits ξt\=(ξ1,t,…,ξN,t)\\xi\_{t}=(\\xi\_{1,t},\\dots,\\xi\_{N,t}) are initialized equally, then the FAMO algorithm maintains equal task weights:

wt\=wi,t\=ct​zi,tℓi,t,w\_{t}=w\_{i,t}=c\_{t}\\frac{z\_{i,t}}{\\ell\_{i,t}}, where ct\=(∑i\=1Nzi,tℓi,t)c\_{t}=\\left(\\sum\_{i=1}^{N}\\dfrac{z\_{i,t}}{\\ell\_{i,t}}\\right), for all i\=1,…,Ni=1,\\dots,N and for all tt.

###### Proof.

We prove the result by induction on the iteration tt.

Base Case (t\=0t=0): Assume that ξi,0\=c\\xi\_{i,0}=c for all ii, for some constant cc. Then the softmax yields

zi,0\=exp⁡(ξi,0)∑j\=1Nexp⁡(ξj,0)\=exp⁡(c)N​exp⁡(c)\=1N.z\_{i,0}=\\frac{\\exp(\\xi\_{i,0})}{\\sum\_{j=1}^{N}\\exp(\\xi\_{j,0})}=\\frac{\\exp(c)}{N\\exp(c)}=\\frac{1}{N}.

The model parameters are updated as follows:

θ1\=θ0−α∑i\=1N(c0zi,0ℓi,0)∇ℓi,0,\\theta\_{1}=\\theta\_{0}-\\alpha\\sum\_{i=1}^{N}\\left(c\_{0}\\frac{z\_{i,0}}{\\ell\_{i,0}}\\right)\\nabla\\ell\_{i,0}, where c0\=(∑i\=1Nzi,0ℓi,0)−1c\_{0}=\\left(\\sum\_{i=1}^{N}\\dfrac{z\_{i,0}}{\\ell\_{i,0}}\\right)^{-1}.

Under the assumption ℓi,t\=ℓt\\ell\_{i,t}=\\ell\_{t} for all ii, each term zi,tℓi,t\\frac{z\_{i,t}}{\\ell\_{i,t}} is identical across tasks. Therefore, this update does not introduce any asymmetry among tasks. Thus, the base case holds.

Inductive Step: Suppose that at iteration tt we have

ξt\=(ct,ct,…,ct),\\xi\_{t}=(c\_{t},c\_{t},\\dots,c\_{t}), which implies

zi,t\=exp⁡(ct)N​exp⁡(ct)\=1Nfor all ​i.z\_{i,t}=\\frac{\\exp(c\_{t})}{N\\exp(c\_{t})}=\\frac{1}{N}\\quad\\text{for all }i.

Next, the logits are updated via

ξt+1\=ξt−β⁡(δt+γ​ξt),\\xi\_{t+1}=\\xi\_{t}-\\beta\\left(\\delta\_{t}+\\gamma\\,\\xi\_{t}\\right), with

δt\=\[∇⊤z1,t​(ξt)∇⊤zn,t​(ξt)\]​\[log⁡ℓ1,t−log⁡ℓ1,t+1log⁡ℓn,t−log⁡ℓn,t+1\].\\delta\_{t}=\\begin{bmatrix}\\nabla^{\\top}z\_{1,t}(\\xi\_{t})\\\\ \\vdots\\\\ \\nabla^{\\top}z\_{n,t}(\\xi\_{t})\\end{bmatrix}\\begin{bmatrix}\\log\\ell\_{1,t}-\\log\\ell\_{1,t+1}\\\\ \\vdots\\\\ \\log\\ell\_{n,t}-\\log\\ell\_{n,t+1}\\end{bmatrix}.

Since the losses satisfy ℓi,t\=ℓt\\ell\_{i,t}=\\ell\_{t} for every ii (and similarly for t+1t+1 by the symmetry of the update), the differences log⁡ℓi,t−log⁡ℓi,t+1\\log\\ell\_{i,t}-\\log\\ell\_{i,t+1} are the same for all tasks.

Moreover, the softmax function is given by

zi,t\=exp⁡(ξi,t)∑j\=1Nexp⁡(ξj,t).z\_{i,t}=\\frac{\\exp(\\xi\_{i,t})}{\\sum\_{j=1}^{N}\\exp(\\xi\_{j,t})}.

The derivative of zi,tz\_{i,t} with respect to ξj\\xi\_{j} is

∂zi∂ξj\=zi​(δi​j−zj).\\frac{\\partial z\_{i}}{\\partial\\xi\_{j}}=z\_{i}\\left(\\delta\_{ij}-z\_{j}\\right).

where δi​j\\delta\_{ij} is the Kronecker delta, defined as δi​j\=1\\delta\_{ij}=1 if i\=ji=j and 0 otherwise. At the symmetric point ξt\=(ct,…,ct)\\xi\_{t}=(c\_{t},\\dots,c\_{t}), we have zi\=1Nz\_{i}=\\frac{1}{N} for all ii, hence

∂zi∂ξj\=1N​(δi​j−1N).\\frac{\\partial z\_{i}}{\\partial\\xi\_{j}}=\\frac{1}{N}\\left(\\delta\_{ij}-\\frac{1}{N}\\right).

This shows that the Jacobian of the softmax is the same for every coordinate, ensuring that the scalar products involved in the definition of δt\\delta\_{t} yield the same value for every task. Let us denote this common value by DD, i.e.,

\[δt\]i\=Dfor all ​i.\[\\delta\_{t}\]\_{i}=D\\quad\\text{for all }i.

Since the regularization term γ​ξt\\gamma\\,\\xi\_{t} is also applied uniformly, the update for each coordinate becomes

ξi,t+1\=ct−β⁡(D+γ​ct)for all ​i.\\xi\_{i,t+1}=c\_{t}-\\beta(D+\\gamma c\_{t})\\quad\\text{for all }i.

Thus, ξt+1\=(ct+1,ct+1,…,ct+1)\\xi\_{t+1}=(c\_{t+1},c\_{t+1},\\ldots,c\_{t+1}) remains symmetric. Evidently, the induction holds for the specific case N\=2N=2.

∎

### A.5 Dataset Usage and Licensing

*   •

```
MNIST: Creative Commons Attribution-Share Alike 3.0
```
*   •

```
Cityscapes: Custom, allows for academic and non-commercial use, [https://www.cityscapes-dataset.com/license/](https://www.cityscapes-dataset.com/license/ "")
```
*   •

```
QM9: We were not able to find a license for this dataset. However the GitHub repository that manages the dataset is under an MIT license [https://github.com/pyg-team/pytorch\_geometric](https://github.com/pyg-team/pytorch_geometric "").
```
### A.6 Hyperparameters

Tables [6](#A1.T6 "Table 6 ‣ A.6 Hyperparameters ‣ Appendix A Technical Appendices and Supplementary Material ‣ Uniform Loss vs. Specialized Optimization: A Comparative Analysis in Multi-Task Learning"), [7](#A1.T7 "Table 7 ‣ A.6 Hyperparameters ‣ Appendix A Technical Appendices and Supplementary Material ‣ Uniform Loss vs. Specialized Optimization: A Comparative Analysis in Multi-Task Learning"), [8](#A1.T8 "Table 8 ‣ A.6 Hyperparameters ‣ Appendix A Technical Appendices and Supplementary Material ‣ Uniform Loss vs. Specialized Optimization: A Comparative Analysis in Multi-Task Learning"), [9](#A1.T9 "Table 9 ‣ A.6 Hyperparameters ‣ Appendix A Technical Appendices and Supplementary Material ‣ Uniform Loss vs. Specialized Optimization: A Comparative Analysis in Multi-Task Learning"), and [10](#A1.T10 "Table 10 ‣ A.6 Hyperparameters ‣ Appendix A Technical Appendices and Supplementary Material ‣ Uniform Loss vs. Specialized Optimization: A Comparative Analysis in Multi-Task Learning") summarize the best hyperparameters for each SMTO in each scenario. Also, it shows the best SMTO specific parameters in bold. In the MNIST configuration, these parameters were searched in a broader range to better understand their impact on optimization and limit the search space in the more complex dataset. To determine which specific hyperparameters of each SMTO to optimize, we selected those explicitly optimized in the respective articles. Consequently, we kept Auto-Lambda’s initial weight, FAMO’s learning rate for task logits, Nash-MTL’s optimizer iterations, and GradDrop’s parameters fixed throughout the experiments. Additionally, for the Cityscapes dataset, RotoGrad’s feature size was set to the maximum possible value or capped at 512, the same used for the ResNet18 model trained on the original Rotograd paper.

For the hyperparameters that were optimized without a defined range, such as Auto-Lambda’s and RotoGrad’s auxiliary learning rate, we opted to define it as a scale of the learning rate (a​u​x​\_​l​r\=l​r∗l​r​\_​s​c​a​l​eaux\\\_lr=lr\*lr\\\_scale).

For the specific case of the CDTT optimizer in the QM9 dataset, we opted to use the same optimal learning rate from EDM and only optimized specific hyperparameters due to time constraints. We justify this choice because CDTT was proposed as a direct improvement from EDM.

Table 6: Hyperparameters for MNIST - CL & CR.

Dataset

Tasks

SMTO

Hyperparameters

SMTO specific parameters

LR

P

WD

MNIST

 CL CR 

Single Task

 0.0025 0.005 

 0.5 0.3 

 0 0 

Unit. Scal.

0.0025

0.3

0

Auto-Lambda

0.005

0.3

0

 auxiliary learning rate scale: {1000, 100, 10 1.0 0.1 0.01 0.001} initial weight: 0.1 

CAGrad

0.0025

0.3

0

cc: {0.2, 0.5, 0.8}

CDTT

0.005

0.2

0

α\\alpha: {0.2, 0.4, 0.6, 0.8, 1.0}

EDM

0.005

0.2

0

FAMO

0.0025

0.2

0

 γ\\gamma: {0.01, 0.001, 0.0001} learning rate of the task logits: 0.025 

IMTL

0.0025

0.3

0

Nash-MTL

0.0025

0.3

0

optimizer iterations: 20

RotoGrad

0.005

0.3

0

 auxiliary learning rate scale: {5.0 1.0 0.5 0.1} feature size: 50 

Table 7: Hyperparameters for MNIST - CL & RR.

Dataset

Tasks

SMTO

Hyperparameters

SMTO specific parameters

LR

P

WD

MNIST

 CL RR 

Single Task

 0.0025 0.0025 

 0.5 0.0 

 0 0 

Unit. Scal.

0.001

0

0

Auto-Lambda

0.001

0

0

 auxiliary learning rate scale: {1000 100 10 1 0.1 0.01 0.001} initial weight: 0.1 

CAGrad

0.001

0

0

cc: {0.2, 0.5, 0.8}

CDTT

0.0025

0

0

α\\alpha: {0.2, 0.4, 0.6, 0.8, 1.0}

EDM

0.001

0

0

FAMO

0.0025

0

0

 γ\\gamma: {0.01, 0.001, 0.0001} learning rate of the task logits: 0.025 

GradDrop

0.001

0

0

 k: 1.0 p: 0.5 

IMTL

0.001

0

0

MGDA-UB

0.001

0

0

Nash-MTL

0.001

0

0

optimizer iterations: 20

PCGrad

0.001

0

0

RLW-Dirichlet

0.001

0

0

RLW-Normal

0.001

0

0

RotoGrad

0.001

0

0

 auxiliary learning rate scale: {5.0 1.0 0.5 0.1} feature size: 50 

SI

0.001

0

0

UW

0.001

0

0

Table 8: Hyperparameters for MNIST - RL & RR.

Dataset

Tasks

SMTO

Hyperparameters

SMTO specific parameters

LR

P

WD

MNIST

 RL RR 

Single Task

 0.0025 0.0025 

0

0

Unit. Scal.

0.001

0

0

Auto-Lambda

0.001

0

0

 auxiliary learning rate scale: {1000 100 10 1 0.01 0.001 0.0001} initial weight: 0.1 

CAGrad

0.0025

0

0

cc: {0.2, 0.5, 0.8}

CDTT

0.001

0

0

α\\alpha: {0.2, 0.4, 0.6, 0.8, 1.0}

EDM

0.001

0

0

FAMO

0.001

0

0

 γ\\gamma: {0.01, 0.001, 0.0001} learning rate of the task logits: 0.025 

IMTL

0.001

0

0

Nash-MTL

0.001

0

0

optimizer iterations: 20

RotoGrad

0.001

0

0

 auxiliary learning rate scale: {5.0 1.0 0.5 0.1} feature size: 50 

Table 9: Hyperparameters for Cityscapes.

Dataset

Tasks

SMTO

Hyperparameters

SMTO specific parameters

LR

P

WD

City.

 S D I 

Single Task

 ACC: 0.001 mIoU: 0.001 L1 abs: 0.001 L1 rel: 0.0005 L1 abs: 0.001 

 0 0 0 0 0 

 1​e−41e-4 1​e−41e-4 1​e−41e-4 1​e−41e-4 1​e−51e-5 

Unit. Scal.

0.001

0

1​e−41e-4

Auto-Lambda

0.001

0

1​e−41e-4

 auxiliary learning rate scale: {100 10 1 0.1 0.01 0.001 0.0001} initial weight: 0.1 

CAGrad

0.001

0

1​e−41e-4

cc: {0.2, 0.5, 0.8}

CDTT

0.001

0

1​e−41e-4

α\\alpha: {0.2, 0.4, 0.6, 0.8, 1.0}

EDM

0.001

0

1​e−41e-4

FAMO

0.001

0

1​e−41e-4

 γ\\gamma: {0.01, 0.001, 0.0001} learning rate of the task logits: 0.025 

IMTL

0.001

0

1​e−41e-4

Nash-MTL

0.001

0

1​e−41e-4

optimizer iterations: 20

RotoGrad

0.001

0

0

 auxiliary learning rate scale: {5.0 1.0 0.5 0.1} feature size: 512 

Table 10: Hyperparameters for QM9.

Dataset

Tasks

SMTO

Hyperparameters

SMTO specific parameters

LR

P

WD

QM9

 CvC\_{v} GG HH U0U\_{0} UU α\\alpha ϵh​o​m​o\\epsilon\_{homo} ϵl​u​m​o\\epsilon\_{lumo} μ\\mu R2R^{2} ZPVE 

Single Task

 0.001 0.001 0.001 0.0005 0.001 0.005 0.001 0.001 0.001 0.001 0.001 

 0 0 0 0 0 0 0 0 0 0 0 

 0 0 0 0 0 0 0 0 0 0 0 

Unit. Scal.

0.005

0

0

Auto-Lambda

0.001

0

0

 auxiliary learning rate scale: {100, 10, 1.0} initial weight: 0.1 

CAGrad

0.001

0

0

cc: {0.2, 0.5, 0.8}

CDTT

0.001

0

0

α\\alpha: {0.2, 0.4, 0.6, 0.8, 1.0}

EDM

0.001

0

0

FAMO

0.005

0

0

 γ\\gamma: {0.01, 0.001, 0.0001} learning rate of the task logits: 0.025 

Nash-MTL

0.001

0

0

optimizer iterations: 20

[](javascript:toggleReadingMode\(\); "Disable reading mode, show header and footer")