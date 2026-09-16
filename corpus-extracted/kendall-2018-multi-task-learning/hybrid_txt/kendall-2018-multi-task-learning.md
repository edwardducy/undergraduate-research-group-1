# Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics

Alex Kendall University of Cambridge agk34@cam.ac.uk

Yarin Gal University of Oxford yarin@cs.ox.ac.uk

Roberto Cipolla University of Cambridge rc10001@cam.ac.uk

## Abstract

Numerous deep learning applications benefitfrom multitask learning with multiple regression and classification objectives. In this paper we make the observation that the performance of such systems is strongly dependent on the relative weighting between each task’s loss. Tuning these weights by hand is a difficult and expensive process, making multi-task learning prohibitive in practice. We propose a principled approach to multi-task deep learning which weighs multiple lossfunctions by considering the homoscedastic uncertainty of each task. This allows us to simultaneously learn various quantities with different units or scales in both classification and regression settings. We demonstrate our model learning per-pixel depth regression, semantic and instance segmentation from a monocular input image. Perhaps surprisingly, we show our model can learn multi-task weightings and outperform separate models trained individually on each task.

## 1. Introduction

Multi-task learning aims to improve learning efficiency and prediction accuracy by learning multiple objectives from a shared representation [7]. Multi-task learning is prevalent in many applications of machine learning – from computer vision [27] to natural language processing [11] to speech recognition [23].

We explore multi-task learning within the setting of visual scene understanding in computer vision. Scene understanding algorithms must understand both the geometry and semantics of the scene at the same time. This forms an interesting multi-task learning problem because scene understanding involves joint learning of various regression and classification tasks with different units and scales. Multitask learning of visual scene understanding is of crucial importance in systems where long computation run-time is prohibitive, such as the ones used in robotics. Combining all tasks into a single model reduces computation and allows these systems to run in real-time.

Prior approaches to simultaneously learning multiple tasks use a na¨ıve weighted sum of losses, where the loss weights are uniform, or manually tuned [38, 27, 15]. However, we show that performance is highly dependent on an appropriate choice of weighting between each task’s loss. Searching for an optimal weighting is prohibitively expensive and difficult to resolve with manual tuning. We observe that the optimal weighting of each task is dependent on the measurement scale (e.g. meters, centimetres or millimetres) and ultimately the magnitude of the task’s noise.

In this work we propose a principled way of combining multiple loss functions to simultaneously learn multiple objectives using homoscedastic uncertainty. We interpret homoscedastic uncertainty as task-dependent weighting and show how to derive a principled multi-task loss function which can learn to balance various regression and classification losses. Our method can learn to balance these weightings optimally, resulting in superior performance, compared with learning each task individually.

Specifically, we demonstrate our method in learning scene geometry and semantics with three tasks. Firstly, we learn to classify objects at a pixel level, also known as semantic segmentation [32, 3, 42, 8, 45]. Secondly, our model performs instance segmentation, which is the harder task of segmenting separate masks for each individual object in an image (for example, a separate, precise mask for each individual car on the road) [37, 18, 14, 4]. This is a more difficult task than semantic segmentation, as it requires not only an estimate of each pixel’s class, but also which object that pixel belongs to. It is also more complicated than object detection, which often predicts object bounding boxes alone [17]. Finally, our model predicts pixel-wise metric depth. Depth by recognition has been demonstrated using dense prediction networks with supervised [15] and unsupervised [16] deep learning. However it is very hard to estimate depth in a way which generalises well. We show that we can improve our estimation of geometry and depth by using semantic labels and multi-task deep learning.

In existing literature, separate deep learning models would be used to learn depth regression, semantic segmentation and instance segmentation to create a complete scene understanding system. Given a single monocular input image, our system is the first to produce a semantic segmentation, a dense estimate of metric depth and an instance level segmentation jointly (Figure 1). While other vision models have demonstrated multi-task learning, we show how to learn to combine semantics and geometry. Combining these tasks into a single model ensures that the model agrees between the separate task outputs while reducing computation. Finally, we show that using a shared representation with multi-task learning improves performance on various metrics, making the models more effective.

![](images/da677922d0706c451d82e5aa93474e1e72da4764e4d74b7ccf78120fb84eb43b.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph LR
  Input["Input Image"] --> Encoder["Encoder"]
  Encoder --> SemanticDecoder["Semantic Decoder"]
  Encoder --> InstanceDecoder["Instance Decoder"]
  Encoder --> DepthDecoder["Depth Decoder"]
  SemanticDecoder --> TaskUncertainty1["Semantic Task Uncertainty"]
  InstanceDecoder --> TaskUncertainty2["Instance Task Uncertainty"]
  DepthDecoder --> TaskUncertainty3["Depth Task Uncertainty"]
  TaskUncertainty1 --> Sum((Sum))
  TaskUncertainty2 --> Sum
  TaskUncertainty3 --> Sum
  Sum --> MultiTaskLoss["Multi-Task Loss"]
```
</details>

Figure 1: Multi-task deep learning. We derive a principled way of combining multiple regression and classification loss functions for multi-task learning. Our architecture takes a single monocular RGB image as input and produces a pixel-wise classification, an instance semantic segmentation and an estimate of per pixel depth. Multi-task learning can improve accuracy over separately trained models because cues from one task, such as depth, are used to regularize and improve the generalization of another domain, such as segmentation.

In summary, the key contributions of this paper are:

1. a novel and principled multi-task loss to simultaneously learn various classification and regression losses of varying quantities and units using homoscedastic task uncertainty,  
2. a unified architecture for semantic segmentation, instance segmentation and depth regression,  
3. demonstrating the importance of loss weighting in multi-task deep learning and how to obtain superior performance compared to equivalent separately trained models.

## 2. Related Work

Multi-task learning aims to improve learning efficiency and prediction accuracy for each task, when compared to training a separate model for each task [40, 5]. It can be considered an approach to inductive knowledge transfer which improves generalisation by sharing the domain information between complimentary tasks. It does this by using a shared representation to learn multiple tasks – what is learned from one task can help learn other tasks [7].

Fine-tuning [1, 36] is a basic example of multi-task learning, where we can leverage different learning tasks by considering them as a pre-training step. Other models alternate learning between each training task, for example in natural language processing [11]. Multi-task learning can also be used in a data streaming setting [40], or to prevent forgetting previously learned tasks in reinforcement learning [26]. It can also be used to learn unsupervised features from various data sources with an auto-encoder [35].

In computer vision there are many examples of methods for multi-task learning. Many focus on semantic tasks, such as classification and semantic segmentation [30] or classification and detection [38]. MultiNet [39] proposes an architecture for detection, classification and semantic segmentation. CrossStitch networks [34] explore methods to combine multi-task neural activations. Uhrig et al. [41] learn semantic and instance segmentations under a classification setting. Multi-task deep learning has also been used for geometry and regression tasks. [15] show how to learn semantic segmentation, depth and surface normals. PoseNet [25] is a model which learns camera position and orientation. UberNet [27] learns a number of different regression and classification tasks under a single architecture. In this work we are the first to propose a method for jointly learning depth regression, semantic and instance segmentation. Like the model of [15], our model learns both semantic and geometry representations, which is important for scene understanding. However, our model learns the much harder task of instance segmentation which requires knowledge of both semantics and geometry. This is because our model must determine the class and spatial relationship for each pixel in each object for instance segmentation.

![](images/d3ce6258d67d2d02f51442186fe7bba281b404dc301dbf0e9face8319f6b10cd.jpg)

<details>
<summary>line</summary>

| Depth Weight | Classification (%) | Depth Regression \((m^{-1})\) |
| --- | --- | --- |
| 0.05 | ~59.5 | ~0.648 |
| 0.1 | ~60.0 | ~0.675 |
| 0.15 | ~60.3 | ~0.685 |
| 0.2 | ~59.8 | ~0.698 |
| 0.3 | ~59.0 | ~0.705 |
| 0.5 | ~56.3 | ~0.648 |
| 0.8 | ~47.2 | ~0.618 |
| 0.9 | — | ~0.615 |
| 1.0 | — | ~0.638 |
</details>

<table><tr><td colspan="2">Task Weights</td><td rowspan="2">Class IoU [%]</td><td rowspan="2">Depth Err. [px]</td></tr><tr><td>Class</td><td>Depth</td></tr><tr><td>1.0</td><td>0.0</td><td>59.4</td><td>-</td></tr><tr><td>0.975</td><td>0.025</td><td>59.5</td><td>0.664</td></tr><tr><td>0.95</td><td>0.05</td><td>59.9</td><td>0.603</td></tr><tr><td>0.9</td><td>0.1</td><td>60.1</td><td>0.586</td></tr><tr><td>0.85</td><td>0.15</td><td>60.4</td><td>0.582</td></tr><tr><td>0.8</td><td>0.2</td><td>59.6</td><td>0.577</td></tr><tr><td>0.7</td><td>0.3</td><td>59.0</td><td>0.573</td></tr><tr><td>0.5</td><td>0.5</td><td>56.3</td><td>0.602</td></tr><tr><td>0.2</td><td>0.8</td><td>47.2</td><td>0.625</td></tr><tr><td>0.1</td><td>0.9</td><td>42.7</td><td>0.628</td></tr><tr><td>0.0</td><td>1.0</td><td>-</td><td>0.640</td></tr><tr><td colspan="2">Learned weights with task uncertainty (this work, Section 3.2)</td><td>62.7</td><td>0.533</td></tr></table>

(a) Comparing loss weightings when learning semantic classification and depth regression  
![](images/81f2b6dd56c7202d625e92f50dd21dbfacde4c285aa22409a215e631605d2763.jpg)

<details>
<summary>line</summary>

| Depth Weight | Instance Regression (RMS Instance (px)) | Depth Regression (RMS Inverse Depth Error \((m^{-1}))\) |
| --- | --- | --- |
| 0 | 4.6 | — |
| 0.25 | 4.5 | 0.71 |
| 0.5 | 4.3 | 0.66 |
| 0.6 | 4.15 | 0.64 |
| 0.7 | 4.05 | 0.62 |
| 0.8 | 3.85 | 0.61 |
| 0.9 | 4.05 | 0.60 |
| 0.95 | 4.35 | 0.62 |
| 1 | 4.4 | 0.64 |
</details>

<table><tr><td colspan="2">Task Weights</td><td rowspan="2">Instance Err. [px]</td><td rowspan="2">Depth Err. [px]</td></tr><tr><td>Instance</td><td>Depth</td></tr><tr><td>1.0</td><td>0.0</td><td>4.61</td><td></td></tr><tr><td>0.75</td><td>0.25</td><td>4.52</td><td>0.692</td></tr><tr><td>0.5</td><td>0.5</td><td>4.30</td><td>0.655</td></tr><tr><td>0.4</td><td>0.6</td><td>4.14</td><td>0.641</td></tr><tr><td>0.3</td><td>0.7</td><td>4.04</td><td>0.615</td></tr><tr><td>0.2</td><td>0.8</td><td>3.83</td><td>0.607</td></tr><tr><td>0.1</td><td>0.9</td><td>3.91</td><td>0.600</td></tr><tr><td>0.05</td><td>0.95</td><td>4.27</td><td>0.607</td></tr><tr><td>0.025</td><td>0.975</td><td>4.31</td><td>0.624</td></tr><tr><td>0.0</td><td>1.0</td><td></td><td>0.640</td></tr><tr><td colspan="2">Learned weights with task uncertainty (this work, Section 3.2)</td><td>3.54</td><td>0.539</td></tr></table>

(b) Comparing loss weightings when learning instance regression and depth regression  
Figure 2: Learning multiple tasks improves the model’s representation and individual task performance. These figures and tables illustrate the advantages of multi-task learning for (a) semantic classification and depth regression and (b) instance and depth regression. Performance of the model in individual tasks is seen at both edges of the plot where $w = 0$ and $w = 1$ . For some balance of weightings between each task, we observe improved performance for both tasks. All models were trained with a learning rate of 0.01 with the respective weightings applied to the losses using the loss function in (1). Results are shown using the Tiny CityScapes validation dataset using a down-sampled resolution of 128 × 256.

More importantly, all previous methods which learn multiple tasks simultaneously use a na¨ıve weighted sum of losses, where the loss weights are uniform, or crudely and manually tuned. In this work we propose a principled way of combining multiple loss functions to simultaneously learn multiple objectives using homoscedastic task uncertainty. We illustrate the importance of appropriately weighting each task in deep learning to achieve good performance and show that our method can learn to balance these weightings optimally.

## 3. Multi Task Learning with Homoscedastic Uncertainty

Multi-task learning concerns the problem of optimising a model with respect to multiple objectives. It is prevalent in many deep learning problems. The naive approach to combining multi objective losses would be to simply perform a weighted linear sum of the losses for each individual task:

$$
L _ {t o t a l} = \sum_ {i} w _ {i} L _ {i}. \tag {1}
$$

This is the dominant approach used by prior work [39, 38, 30, 41], for example for dense prediction tasks [27], for scene understanding tasks [15] and for rotation (in quaternions) and translation (in meters) for camera pose [25]. However, there are a number of issues with this method. Namely, model performance is extremely sensitive to weight selection, $w _ { i } .$ , as illustrated in Figure 2. These weight hyper-parameters are expensive to tune, often taking many days for each trial. Therefore, it is desirable to find a more convenient approach which is able to learn the optimal weights.

More concretely, let us consider a network which learns to predict pixel-wise depth and semantic class from an input image. In Figure 2 the two boundaries of each plot show models trained on individual tasks, with the curves showing performance for varying weights $w _ { i }$ for each task. We observe that at some optimal weighting, the joint network performs better than separate networks trained on each task individually (performance of the model in individual tasks is seen at both edges of the plot: $w = 0 \mathrm { a n d } w = 1 )$ . At nearby values to the optimal weight the network performs worse on one of the tasks. However, searching for these optimal weightings is expensive and increasingly difficult with large models with numerous tasks. Figure 2 also shows a similar result for two regression tasks; instance segmentation and depth regression. We next show how to learn optimal task weightings using ideas from probabilistic modelling.

## 3.1. Homoscedastic uncertainty as task-dependent uncertainty

In Bayesian modelling, there are two main types of uncertainty one can model [24].

• Epistemic uncertainty is uncertainty in the model, which captures what our model does not know due to lack of training data. It can be explained away with increased training data.  
• Aleatoric uncertainty captures our uncertainty with respect to information which our data cannot explain. Aleatoric uncertainty can be explained away with the ability to observe all explanatory variables with increasing precision.

Aleatoric uncertainty can again be divided into two subcategories.

• Data-dependent or Heteroscedastic uncertainty is aleatoric uncertainty which depends on the input data and is predicted as a model output.  
• Task-dependent or Homoscedastic uncertainty is aleatoric uncertainty which is not dependent on the input data. It is not a model output, rather it is a quantity which stays constant for all input data and varies between different tasks. It can therefore be described as task-dependent uncertainty.

In a multi-task setting, we show that the task uncertainty captures the relative confidence between tasks, reflecting the uncertainty inherent to the regression or classification task. It will also depend on the task’s representation or unit of measure. We propose that we can use homoscedastic uncertainty as a basis for weighting losses in a multi-task learning problem.

## 3.2. Multi-task likelihoods

In this section we derive a multi-task loss function based on maximising the Gaussian likelihood with homoscedastic uncertainty. Let $\mathbf { f } ^ { \mathbf { W } } ( \mathbf { x } )$ be the output of a neural network with weights W on input x. We define the following probabilistic model. For regression tasks we define our likelihood as a Gaussian with mean given by the model output:

$$
p (\mathbf {y} | \mathbf {f} ^ {\mathbf {W}} (\mathbf {x})) = \mathcal {N} (\mathbf {f} ^ {\mathbf {W}} (\mathbf {x}), \sigma^ {2}) \tag {2}
$$

with an observation noise scalar $\sigma$ . For classification we often squash the model output through a softmax function, and sample from the resulting probability vector:

$$
p (\mathbf {y} | \mathbf {f} ^ {\mathbf {W}} (\mathbf {x})) = \operatorname{Softmax} (\mathbf {f} ^ {\mathbf {W}} (\mathbf {x})). \tag {3}
$$

In the case of multiple model outputs, we often define the likelihood to factorise over the outputs, given some sufficient statistics. We define $\mathbf { f } ^ { \mathbf { W } } ( \mathbf { x } )$ as our sufficient statistics, and obtain the following multi-task likelihood:

$$
p (\mathbf {y} _ {1}, \dots , \mathbf {y} _ {K} | \mathbf {f} ^ {\mathbf {W}} (\mathbf {x})) = p (\mathbf {y} _ {1} | \mathbf {f} ^ {\mathbf {W}} (\mathbf {x})) \dots p (\mathbf {y} _ {K} | \mathbf {f} ^ {\mathbf {W}} (\mathbf {x})) \tag {4}
$$

with model outputs $\mathbf { y } _ { 1 } , . . . , \mathbf { y } _ { K }$ (such as semantic segmentation, depth regression, etc).

In maximum likelihood inference, we maximise the log likelihood of the model. In regression, for example, the log likelihood can be written as

$$
\log p (\mathbf {y} | \mathbf {f} ^ {\mathbf {W}} (\mathbf {x})) \propto - \frac {1}{2 \sigma^ {2}} | | \mathbf {y} - \mathbf {f} ^ {\mathbf {W}} (\mathbf {x}) | | ^ {2} - \log \sigma \tag {5}
$$

for a Gaussian likelihood (or similarly for a Laplace likelihood) with $\sigma$ the model’s observation noise parameter – capturing how much noise we have in the outputs. We then maximise the log likelihood with respect to the model parameters W and observation noise parameter $\sigma .$

Let us now assume that our model output is composed of two vectors $\mathbf { y } _ { 1 }$ and $\mathbf { y } _ { 2 } .$ , each following a Gaussian distribution:

$$
\begin{array}{l} p (\mathbf {y} _ {1}, \mathbf {y} _ {2} | \mathbf {f} ^ {\mathbf {W}} (\mathbf {x})) = p (\mathbf {y} _ {1} | \mathbf {f} ^ {\mathbf {W}} (\mathbf {x})) \cdot p (\mathbf {y} _ {2} | \mathbf {f} ^ {\mathbf {W}} (\mathbf {x})) \\ = \mathcal {N} (\mathbf {y} _ {1}; \mathbf {f} ^ {\mathbf {W}} (\mathbf {x}), \sigma_ {1} ^ {2}) \cdot \mathcal {N} (\mathbf {y} _ {2}; \mathbf {f} ^ {\mathbf {W}} (\mathbf {x}), \sigma_ {2} ^ {2}). \tag {6} \\ \end{array}
$$

$$
\begin{array}{l} p (\mathbf {y} _ {1}, \mathbf {y} _ {2} | \mathbf {f} ^ {\mathbf {W}} (\mathbf {x})) = p (\mathbf {y} _ {1} | \mathbf {f} ^ {\mathbf {W}} (\mathbf {x})) \cdot p (\mathbf {y} _ {2} | \mathbf {f} ^ {\mathbf {W}} (\mathbf {x})) \\ = \mathcal {N} (\mathbf {y} _ {1}; \mathbf {f} ^ {\mathbf {W}} (\mathbf {x}), \sigma_ {1} ^ {2}) \cdot \mathcal {N} (\mathbf {y} _ {2}; \mathbf {f} ^ {\mathbf {W}} (\mathbf {x}), \sigma_ {2} ^ {2}). \tag {6} \\ \end{array}
$$

This leads to the minimisation objective, $\mathcal { L } ( \mathbf { W } , \sigma _ { 1 } , \sigma _ { 2 } )$ ， (our loss) for our multi-output model:

$$
\begin{array}{l} = - \log p (\mathbf {y} _ {1}, \mathbf {y} _ {2} | \mathbf {f} ^ {\mathbf {W}} (\mathbf {x})) \\ \propto \frac {1}{2 \sigma_ {1} ^ {2}} | | \mathbf {y} _ {1} - \mathbf {f} ^ {\mathbf {W}} (\mathbf {x}) | | ^ {2} + \frac {1}{2 \sigma_ {2} ^ {2}} | | \mathbf {y} _ {2} - \mathbf {f} ^ {\mathbf {W}} (\mathbf {x}) | | ^ {2} + \log \sigma_ {1} \sigma_ {2} \\ = \frac {1}{2 \sigma_ {1} ^ {2}} \mathcal {L} _ {1} (\mathbf {W}) + \frac {1}{2 \sigma_ {2} ^ {2}} \mathcal {L} _ {2} (\mathbf {W}) + \log \sigma_ {1} \sigma_ {2} \tag {7} \\ \end{array}
$$

Where we wrote $\mathcal { L } _ { 1 } ( \mathbf { W } ) = | | \mathbf { y } _ { 1 } - \mathbf { f } ^ { \mathbf { W } } ( \mathbf { x } ) | | ^ { 2 }$ for the loss of the first output variable, and similarly for $\mathcal { L } _ { 2 } ( \mathbf { W } )$ .

We interpret minimising this last objective with respect to $\sigma _ { 1 }$ and $\sigma _ { 2 }$ as learning the relative weight of the losses $\mathcal { L } _ { 1 } ( \mathbf { W } )$ and $\mathcal { L } _ { 2 } ( \mathbf { W } )$ adaptively, based on the data. $\mathbf { A s } \ \sigma _ { 1 }$ – the noise parameter for the variable y – increases, we have that the weight of $\mathcal { L } _ { 1 } ( \mathbf { W } )$ decreases. On the other hand, as the noise decreases, we have that the weight of the respective objective increases. The noise is discouraged from increasing too much (effectively ignoring the data) by the last term in the objective, which acts as a regulariser for the noise terms.

This construction can be trivially extended to multiple regression outputs. However, the extension to classification likelihoods is more interesting. We adapt the classification likelihood to squash a scaled version of the model output through a softmax function:

$$
p (\mathbf {y} | \mathbf {f} ^ {\mathbf {W}} (\mathbf {x}), \sigma) = \operatorname{Softmax} \left(\frac {1}{\sigma^ {2}} \mathbf {f} ^ {\mathbf {W}} (\mathbf {x})\right) \tag {8}
$$

with a positive scalar σ. This can be interpreted as a Boltzmann distribution (also called Gibbs distribution) where the input is scaled by $\sigma ^ { 2 }$ (often referred to as temperature). This scalar is either fixed or can be learnt, where the parameter’s magnitude determines how ‘uniform’ (flat) the discrete distribution is. This relates to its uncertainty, as measured in entropy. The log likelihood for this output can then be written as

$$
\begin{array}{l} \log p (\mathbf {y} = c | \mathbf {f} ^ {\mathbf {W}} (\mathbf {x}), \sigma) = \frac {1}{\sigma^ {2}} f _ {c} ^ {\mathbf {W}} (\mathbf {x}) \\ - \log \sum_ {c ^ {\prime}} \exp \left(\frac {1}{\sigma^ {2}} f _ {c ^ {\prime}} ^ {\mathbf {W}} (\mathbf {x})\right) \tag {9} \\ \end{array}
$$

with $f _ { c } ^ { \mathbf { W } } ( \mathbf { x } )$ the c’th element of the vector $\mathbf { f } ^ { \mathbf { W } } ( \mathbf { x } )$

Next, assume that a model’s multiple outputs are composed of a continuous output $\mathbf { y } _ { 1 }$ and a discrete output $\mathbf { y } _ { 2 } ,$ , modelled with a Gaussian likelihood and a softmax likelihood, respectively. Like before, the joint loss, $\mathcal { L } ( \mathbf { W } , \sigma _ { 1 } , \sigma _ { 2 } )$ , is given as:

$$
\begin{array}{l} = - \log p (\mathbf {y} _ {1}, \mathbf {y} _ {2} = c | \mathbf {f} ^ {\mathbf {W}} (\mathbf {x})) \\ = - \log \mathcal {N} (\mathbf {y} _ {1}; \mathbf {f} ^ {\mathbf {W}} (\mathbf {x}), \sigma_ {1} ^ {2}) \cdot \operatorname{Softmax} (\mathbf {y} _ {2} = c; \mathbf {f} ^ {\mathbf {W}} (\mathbf {x}), \sigma_ {2}) \\ = \frac {1}{2 \sigma_ {1} ^ {2}} | | \mathbf {y} _ {1} - \mathbf {f} ^ {\mathbf {W}} (\mathbf {x}) | | ^ {2} + \log \sigma_ {1} - \log p (\mathbf {y} _ {2} = c | \mathbf {f} ^ {\mathbf {W}} (\mathbf {x}), \sigma_ {2}) \\ = \frac {1}{2 \sigma_ {1} ^ {2}} \mathcal {L} _ {1} (\mathbf {W}) + \frac {1}{\sigma_ {2} ^ {2}} \mathcal {L} _ {2} (\mathbf {W}) + \log \sigma_ {1} \\ + \log \frac {\sum_ {c ^ {\prime}} \exp \left(\frac {1}{\sigma_ {2} ^ {2}} f _ {c ^ {\prime}} ^ {\mathbf {W}} (\mathbf {x})\right)}{\left(\sum_ {c ^ {\prime}} \exp \left(f _ {c ^ {\prime}} ^ {\mathbf {W}} (\mathbf {x})\right)\right) ^ {\frac {1}{\sigma_ {2} ^ {2}}}} \\ \approx \frac {1}{2 \sigma_ {1} ^ {2}} \mathcal {L} _ {1} (\mathbf {W}) + \frac {1}{\sigma_ {2} ^ {2}} \mathcal {L} _ {2} (\mathbf {W}) + \log \sigma_ {1} + \log \sigma_ {2}, \tag {10} \\ \end{array}
$$

where again we write $\begin{array} { r c l } { \mathcal { L } _ { 1 } ( \mathbf { W } ) } & { = } & { | | \mathbf { y } _ { 1 } ~ - ~ \mathbf { f } ^ { \mathbf { W } } ( \mathbf { x } ) | | ^ { 2 } } \end{array}$ for the Euclidean loss of y<sub>1</sub>, write $\begin{array} { r l } { \mathcal { L } _ { 2 } ( \mathbf { W } ) } & { { } = } \end{array}$ − log Softmax $\left( \mathbf { y } _ { 2 } , \mathbf { f } ^ { \mathbf { W } } ( \mathbf { x } ) \right)$ for the cross entropy loss of $\mathbf { y } _ { 2 }$ (with $\mathbf { f } ^ { \mathbf { W } } ( \mathbf { x } )$ not scaled), and optimise with respect to W as well as $\sigma _ { 1 } , \sigma _ { 2 }$ . In the last transition we introduced the explicit simplifying assumption $\begin{array} { r } { \frac { 1 } { \sigma _ { 2 } } \sum _ { c ^ { \prime } } \exp \Big ( \frac { 1 } { \sigma _ { 2 } ^ { 2 } } f _ { c ^ { \prime } } ^ { \mathbf { W } } ( \mathbf { x } ) \Big ) } \end{array}$ ≈ $\begin{array} { r } { \left( \sum _ { c ^ { \prime } } \exp \Big ( f _ { c ^ { \prime } } ^ { \mathbf { W } } ( \mathbf { x } ) \Big ) \right) ^ { \frac { 1 } { \sigma _ { 2 } ^ { 2 } } } } \end{array}$ which becomes an equality when $\sigma _ { 2 }  1$ . This has the advantage of simplifying the optimisation objective, as well as empirically improving results.

This last objective can be seen as learning the relative weights of the losses for each output. Large scale values $\sigma _ { 2 }$ will decrease the contribution of $\mathcal { L } _ { 2 } ( \mathbf { W } )$ , whereas small scale $\sigma _ { 2 }$ will increase its contribution. The scale is regulated by the last term in the equation. The objective is penalised when setting $\sigma _ { 2 }$ too large.

This construction can be trivially extended to arbitrary combinations of discrete and continuous loss functions, allowing us to learn the relative weights of each loss in a principled and well-founded way. This loss is smoothly differentiable, and is well formed such that the task weights will not converge to zero. In contrast, directly learning the weights using a simple linear sum of losses (1) would result in weights which quickly converge to zero. In the following sections we introduce our experimental model and present empirical results.

In practice, we train the network to predict the log variance, $s : = \log \sigma ^ { 2 }$ . This is because it is more numerically stable than regressing the variance, $\sigma ^ { 2 } ,$ as the loss avoids any division by zero. The exponential mapping also allows us to regress unconstrained scalar values, where exp(−s) is resolved to the positive domain giving valid values for variance.

## 4. Scene Understanding Model

To understand semantics and geometry we first propose an architecture which can learn regression and classification outputs, at a pixel level. Our architecture is a deep convolutional encoder decoder network [3]. Our model consists of a number of convolutional encoders which produce a shared representation, followed by a corresponding number of task-specific convolutional decoders. A high level summary is shown in Figure 1.

The purpose of the encoder is to learn a deep mapping to produce rich, contextual features, using domain knowledge from a number of related tasks. Our encoder is based on DeepLabV3 [10], which is a state of the art semantic segmentation framework. We use ResNet101 [20] as the base feature encoder, followed by an Atrous Spatial Pyramid Pooling (ASPP) module [10] to increase contextual awareness. We apply dilated convolutions in this encoder, such that the resulting feature map is sub-sampled by a factor of

![](images/16e81c75e2cb1d8b545238837f04a35cf51e4f4cd7b7d843915f09e222ff3012.jpg)

<details>
<summary>natural_image</summary>

Street view with cars and buildings in motion, no visible text or signage
</details>

(a) Input Image

![](images/969e3796e3fa48b0f60d78ee9681959407f4770c21d9e7f4f8ae4b23d5f32c34.jpg)

<details>
<summary>natural_image</summary>

Colorful landscape illustration with mountains, trees, and a winding road (no text or symbols)
</details>

(b) Semantic Segmentation

![](images/c964644dac1ad7f5db49eb7b4d3645c020e6fc3ed95bf04a36dfd39f748ebcfd.jpg)

<details>
<summary>natural_image</summary>

Abstract colorful gradient background with no text or symbols
</details>

(c) Instance vector regression

![](images/f3249137f1b1a0f82e685d47b70107fed33397df2a76470d078182e6d827f733.jpg)  
(d) Instance Segmentation  
Figure 3: Instance centroid regression method. For each pixel, we regress a vector pointing to the instance’s centroid. The loss is only computed over pixels which are from instances. We visualise (c) by representing colour as the orientation of the instance vector, and intensity as the magnitude of the vector.

8 compared to the input image dimensions.

We then split the network into separate decoders (with separate weights) for each task. The purpose of the decoder is to learn a mapping from the shared features to an output. Each decoder consists of a $3 \times 3$ convolutional layer with output feature size 256, followed by a 1 × 1 layer regressing the task’s output. Further architectural details are described in Appendix A.

Semantic Segmentation. We use the cross-entropy loss to learn pixel-wise class probabilities, averaging the loss over the pixels with semantic labels in each mini-batch.

Instance Segmentation. An intuitive method for defining which instance a pixel belongs to is an association to the instance’s centroid. We use a regression approach for instance segmentation [29]. This approach is inspired by [28] which identifies instances using Hough votes from object parts. In this work we extend this idea by using votes from individual pixels using deep learning. We learn an instance vector, $\hat { x } _ { n } .$ , for each pixel coordinate, $c _ { n } ,$ , which points to the centroid of the pixel’s instance, $i _ { n } ,$ , such that $i _ { n } = { \hat { x } } _ { n } + c _ { n } .$ We train this regression with an $L _ { 1 }$ loss using ground truth labels $x _ { n }$ , averaged over all labelled pixels, $N _ { I }$ , in a minibatch: $\begin{array} { r } { \mathcal { L } _ { I n s t a n c e } = \frac { 1 } { | N _ { I } | } \sum _ { N _ { I } } \| x _ { n } - \hat { x } _ { n } \| _ { 1 } } \end{array}$

Figure 3 details the representation we use for instance segmentation. Figure 3(a) shows the input image and a mask of the pixels which are of an instance class (at test time inferred from the predicted semantic segmentation). Figure 3(b) and Figure 3(c) show the ground truth and predicted instance vectors for both x and y coordinates. We then cluster these votes using OPTICS [2], resulting in the predicted instance segmentation output in Figure 3(d).

One of the most difficult cases for instance segmentation algorithms to handle is when the instance mask is split due to occlusion. Figure 4 shows that our method can handle these situations, by allowing pixels to vote for their instance centroid with geometry. Methods which rely on watershed approaches [4], or instance edge identification approaches fail in these scenarios.

![](images/c41aab2832fe8c60ef004f3cb94e827398d3f28204f8365fb3ab7e8f153709a8.jpg)

<details>
<summary>natural_image</summary>

Street scene with parked cars and pedestrians on a paved area, surrounded by trees (no visible text or signage)
</details>

(a) Input Image

![](images/565404e74dd3be35e66bdb714fbdbee8a0b20dfe36b3140341bc707f9235ae32.jpg)  
(b) Instance Segmentation  
Figure 4: This example shows two cars which are occluded by trees and lampposts, making the instance segmentation challenging. Our instance segmentation method can handle occlusions effectively. We can correctly handle segmentation masks which are split by occlusion, yet part of the same instance, by incorporating semantics and geometry.

To obtain segmentations for each instance, we now need to estimate the instance centres, $\hat { i } _ { n }$ . We propose to consider the estimated instance vectors, ${ \hat { x } } _ { n } .$ , as votes in a Hough parameter space and use a clustering algorithm to identify these instance centres. OPTICS [2], is an efficient density based clustering algorithm. It is able to identify an unknown number of multi-scale clusters with varying density from a given set of samples. We chose OPICS for two reasons. Crucially, it does not assume knowledge of the number of clusters like algorithms such as k-means [33]. Secondly, it does not assume a canonical instance size or density like discretised binning approaches [12]. Using OPTICS, we cluster the points $c _ { n } + \hat { x } _ { n }$ into a number of estimated instances, <sup>ˆ</sup>i. We can then assign each pixel, $p _ { n }$ to the instance closest to its estimated instance vector, $c _ { n } + \hat { x } _ { n }$

Depth Regression. We train with supervised labels using pixel-wise metric inverse depth using a $L _ { 1 }$ loss function: $\begin{array} { r } { \mathcal { L } _ { D e p t h } = \frac { 1 } { | N _ { D } | } \sum _ { N _ { D } } \left\| d _ { n } - \hat { d } _ { n } \right\| _ { 1 } } \end{array}$ . Our architecture estimates inverse depth, ${ \hat { d } } _ { n } ,$ , because it can represent points at infinite distance (such as sky). We can obtain inverse depth labels, $d _ { n }$ , from a RGBD sensor or stereo imagery. Pixels which do not have an inverse depth label are ignored in the loss.

## 5. Experiments

We demonstrate the efficacy of our method on CityScapes [13], a large dataset for road scene understanding. It comprises of stereo imagery, from automotive grade stereo cameras with a 22cm baseline, labelled with instance and semantic segmentations from 20 classes. Depth images are also provided, labelled using SGM [22], which we treat as pseudo ground truth. Additionally, we assign zero inverse depth to pixels labelled as sky. The dataset was collected from a number of cities in fine weather and consists of 2,975 training and 500 validation images at $2 0 4 8 \times 1 0 2 4$ resolution. 1,525 images are withheld for testing on an online evaluation server.

<table><tr><td rowspan="2">Loss</td><td colspan="3">Task Weights</td><td rowspan="2">Segmentation IoU [%]</td><td rowspan="2">Instance Mean Error [px]</td><td rowspan="2">Inverse Depth Mean Error [px]</td></tr><tr><td>Seg.</td><td>Inst.</td><td>Depth</td></tr><tr><td>Segmentation only</td><td>1</td><td>0</td><td>0</td><td>59.4%</td><td>-</td><td>-</td></tr><tr><td>Instance only</td><td>0</td><td>1</td><td>0</td><td>-</td><td>4.61</td><td>-</td></tr><tr><td>Depth only</td><td>0</td><td>0</td><td>1</td><td>-</td><td>-</td><td>0.640</td></tr><tr><td>Unweighted sum of losses</td><td>0.333</td><td>0.333</td><td>0.333</td><td>50.1%</td><td>3.79</td><td>0.592</td></tr><tr><td>Approx. optimal weights</td><td>0.89</td><td>0.01</td><td>0.1</td><td>62.8%</td><td>3.61</td><td>0.549</td></tr><tr><td>2 task uncertainty weighting</td><td>✓</td><td>✓</td><td></td><td>61.0%</td><td>3.42</td><td>-</td></tr><tr><td>2 task uncertainty weighting</td><td>✓</td><td></td><td>✓</td><td>62.7%</td><td>-</td><td>0.533</td></tr><tr><td>2 task uncertainty weighting</td><td></td><td>✓</td><td>✓</td><td>-</td><td>3.54</td><td>0.539</td></tr><tr><td>3 task uncertainty weighting</td><td>✓</td><td>✓</td><td>✓</td><td>63.4%</td><td>3.50</td><td>0.522</td></tr></table>

Table 1: Quantitative improvement when learning semantic segmentation, instance segmentation and depth with our multi-task loss. Experiments were conducted on the Tiny CityScapes dataset (sub-sampled to a resolution of $1 2 8 \times 2 5 6 )$ . Results are shown from the validation set. We observe an improvement in performance when training with our multi-task loss, over both single-task models and weighted losses. Additionally, we observe an improvement when training on all three tasks $( 3 \times \checkmark )$ using our multi-task loss, compared with all pairs of tasks alone (denoted by $2 \times \checkmark )$ . This shows that our loss function can automatically learn a better performing weighting between the tasks than the baselines.

Further training details, and optimisation hyperparameters, are provided in Appendix A.

## 5.1. Model Analysis

In Table 1 we compare individual models to multi-task learning models using a na¨ıve weighted loss or the task uncertainty weighting we propose in this paper. To reduce the computational burden, we train each model at a reduced resolution of 128 × 256 pixels, over 50, 000 iterations. When we downsample the data by a factor of four, we also need to scale the disparity labels accordingly. Table 1 clearly illustrates the benefit of multi-task learning, which obtains significantly better performing results than individual task models. For example, using our method we improve classification results from 59.4% to 63.4%.

We also compare to a number of na¨ıve multi-task losses. We compare weighting each task equally and using approximately optimal weights. Using a uniform weighting results in poor performance, in some cases not even improving on the results from the single task model. Obtaining approximately optimal weights is difficult with increasing number of tasks as it requires an expensive grid search over parameters. However, even these weights perform worse compared with our proposed method. Figure 2 shows that using task uncertainty weights can even perform better compared to optimal weights found through fine-grained grid search. We believe that this is due to two reasons. First, grid search is restricted in accuracy by the resolution of the search.

Second, optimising the task weights using a homoscedastic noise term allows for the weights to be dynamic during training. In general, we observe that the uncertainty term decreases during training which improves the optimisation process.

In Appendix B we find that our task-uncertainty loss is robust to the initialisation chosen for the parameters. These quickly converge to a similar optima in a few hundred training iterations. We also find the resulting task weightings varies throughout the course of training. For our final model (in Table 2), at the end of training, the losses are weighted with the ratio 43 : 1 : 0.16 for semantic segmentation, depth regression and instance segmentation, respectively.

Finally, we benchmark our model using the full-size CityScapes dataset. In Table 2 we compare to a number of other state of the art methods in all three tasks. Our method is the first model which completes all three tasks with a single model. We compare favourably with other approaches, outperforming many which use comparable training data and inference tools. Figure 5 shows some qualitative examples of our model.

## 6. Conclusions

We have shown that correctly weighting loss terms is of paramount importance for multi-task learning problems. We demonstrated that homoscedastic (task) uncertainty is an effective way to weight losses. We derived a principled loss function which can learn a relative weighting automatically from the data and is robust to the weight initialization. We showed that this can improve performance for scene understanding tasks with a unified architecture for semantic segmentation, instance segmentation and per-pixel depth regression. We demonstrated modelling task-dependent homoscedastic uncertainty improves the model’s representation and each task’s performance when compared to separate models trained on each task individually.

<table><tr><td rowspan="2">Method</td><td colspan="4">Semantic Segmentation</td><td colspan="4">Instance Segmentation</td><td colspan="2">Monocular Disparity Estimation</td></tr><tr><td>IoU class</td><td>iIoU class</td><td>IoU cat</td><td>iIoU cat</td><td>AP</td><td>AP 50%</td><td>AP 100m</td><td>AP 50m</td><td>Mean Error [px]</td><td>RMS Error [px]</td></tr><tr><td colspan="11">Semantic segmentation, instance segmentation and depth regression methods (this work)</td></tr><tr><td>Multi-Task Learning</td><td>78.5</td><td>57.4</td><td>89.9</td><td>77.7</td><td>21.6</td><td>39.0</td><td>35.0</td><td>37.0</td><td>2.92</td><td>5.88</td></tr><tr><td colspan="11">Semantic segmentation and instance segmentation methods</td></tr><tr><td>Uhrig et al. [41]</td><td>64.3</td><td>41.6</td><td>85.9</td><td>73.9</td><td>8.9</td><td>21.1</td><td>15.3</td><td>16.7</td><td>-</td><td>-</td></tr><tr><td colspan="11">Instance segmentation only methods</td></tr><tr><td>Mask R-CNN [19]</td><td>-</td><td>-</td><td>-</td><td>-</td><td>26.2</td><td>49.9</td><td>37.6</td><td>40.1</td><td>-</td><td>-</td></tr><tr><td>Deep Watershed [4]</td><td>-</td><td>-</td><td>-</td><td>-</td><td>19.4</td><td>35.3</td><td>31.4</td><td>36.8</td><td>-</td><td>-</td></tr><tr><td>R-CNN + MCG [13]</td><td>-</td><td>-</td><td>-</td><td>-</td><td>4.6</td><td>12.9</td><td>7.7</td><td>10.3</td><td>-</td><td>-</td></tr><tr><td colspan="11">Semantic segmentation only methods</td></tr><tr><td>DeepLab V3 [10]</td><td>81.3</td><td>60.9</td><td>91.6</td><td>81.7</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>PSPNet [44]</td><td>81.2</td><td>59.6</td><td>91.2</td><td>79.2</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>Adelaide [31]</td><td>71.6</td><td>51.7</td><td>87.3</td><td>74.1</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr></table>

Table 2: CityScapes Benchmark [13]. We show results from the test dataset using the full resolution of 1024 × 2048 pixels. For the full leaderboard, please see www.cityscapes-dataset.com/benchmarks. The disparity (inverse depth) metrics were computed against the CityScapes depth maps, which are sparse and computed using SGM stereo [21]. Note, these comparisons are not entirely fair, as many methods use ensembles of different training datasets. Our method is the first to address all three tasks with a single model.

![](images/015d1cb5cc0ae7f20257d4547515decf78f74f32fbc72c50a6eaaaebeaec98bc.jpg)

<details>
<summary>natural_image</summary>

Four-panel street scene with cars, pedestrians, and vehicles on a city road (no visible text or signage)
</details>

(a) Input image

![](images/e16f49b626a7c82d52f53c83e20a4f5daa022e2a7d66102c4e153637860188ba.jpg)

<details>
<summary>natural_image</summary>

Four-panel illustration of a city street scene with buildings, trees, and vehicles (no text or symbols)
</details>

(b) Segmentation output

![](images/d84eb3be81466a8422510ebda9ead115e375fce27367f77b45c6adac58bbf003.jpg)

<details>
<summary>natural_image</summary>

Four abstract colorful illustrations of various human figures and objects arranged in a row (no text or symbols)
</details>

(c) Instance output

![](images/da40fb5d232bb90389d3970680f81c78d725388aabcbdc39dd66b23aa37214c5.jpg)

<details>
<summary>natural_image</summary>

Thermal imaging of four different scenes showing heat distribution patterns, no text or symbols present
</details>

(d) Depth output  
Figure 5: Qualitative results for multi-task learning of geometry and semantics for road scene understanding. Results are shown on test images from the CityScapes dataset using our multi-task approach with a single network trained on all tasks. We observe that multi-task learning improves the smoothness and accuracy for depth perception because it learns a representation that uses cues from other tasks, such as segmentation (and vice versa).

There are many interesting questions left unanswered. Firstly, our results show that there is usually not a single optimal weighting for all tasks. Therefore, what is the optimal weighting? Is multitask learning is an ill-posed optimisation problem without a single higher-level goal?

A second interesting question is where the optimal location is for splitting the shared encoder network into separate decoders for each task? And, what network depth is best for the shared multi-task representation?

Finally, why do the semantics and depth tasks outperform the semantics and instance tasks results in Table 1? Clearly the three tasks explored in this paper are complimentary and useful for learning a rich representation about the scene. It would be beneficial to be able to quantify the relationship between tasks and how useful they would be for multitask representation learning.

## References

[1] P. Agrawal, J. Carreira, and J. Malik. Learning to see by moving. In Proceedings of the IEEE International Conference on Computer Vision, pages 37–45, 2015. 2  
[2] M. Ankerst, M. M. Breunig, H.-P. Kriegel, and J. Sander. Optics: ordering points to identify the clustering structure. In ACM Sigmod Record, volume 28, pages 49–60. ACM, 1999. 6  
[3] V. Badrinarayanan, A. Kendall, and R. Cipolla. Segnet: A deep convolutional encoder-decoder architecture for scene segmentation. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2017. 1, 5  
[4] M. Bai and R. Urtasun. Deep watershed transform for instance segmentation. arXiv preprint arXiv:1611.08303, 2016. 1, 6, 8  
[5] J. Baxter et al. A model of inductive bias learning. J. Artif. Intell. Res.(JAIR), 12(149-198):3, 2000. 2  
[6] S. R. Bulo, L. Porzi, and P. Kontschieder. In-place activated\` batchnorm for memory-optimized training of dnns. arXiv preprint arXiv:1712.02616, 2017.  
[7] R. Caruana. Multitask learning. In Learning to learn, pages 95–133. Springer, 1998. 1, 2  
[8] L.-C. Chen, G. Papandreou, I. Kokkinos, K. Murphy, and A. L. Yuille. Semantic image segmentation with deep convolutional nets and fully connected crfs. In ICLR, 2015. 1  
[9] L.-C. Chen, G. Papandreou, I. Kokkinos, K. Murphy, and A. L. Yuille. Deeplab: Semantic image segmentation with deep convolutional nets, atrous convolution, and fully connected crfs. arXiv preprint arXiv:1606.00915, 2016.  
[10] L.-C. Chen, G. Papandreou, F. Schroff, and H. Adam. Rethinking atrous convolution for semantic image segmentation. arXiv preprint arXiv:1706.05587, 2017. 5, 8, 11  
[11] R. Collobert and J. Weston. A unified architecture for natural language processing: Deep neural networks with multitask learning. In Proceedings ofthe 25th international conference on Machine learning, pages 160–167. ACM, 2008. 1, 2  
[12] D. Comaniciu and P. Meer. Mean shift: A robust approach toward feature space analysis. IEEE Transactions on pattern analysis and machine intelligence, 24(5):603–619, 2002. 6  
[13] M. Cordts, M. Omran, S. Ramos, T. Rehfeld, M. Enzweiler, R. Benenson, U. Franke, S. Roth, and B. Schiele. The cityscapes dataset for semantic urban scene understanding. In In Proc. IEEE Conf. on Computer Vision and Pattern Recognition, 2016. 6, 8  
[14] J. Dai, K. He, and J. Sun. Instance-aware semantic segmentation via multi-task network cascades. In In Proc. IEEE Conf. on Computer Vision and Pattern Recognition, 2016. 1  
[15] D. Eigen and R. Fergus. Predicting depth, surface normals and semantic labels with a common multi-scale convolutional architecture. In Proceedings of the IEEE International Conference on Computer Vision, pages 2650–2658, 2015. 1, 2, 3  
[16] R. Garg and I. Reid. Unsupervised cnn for single view depth estimation: Geometry to the rescue. Computer Vision–ECCV 2016, pages 740–756, 2016. 1  
[17] R. Girshick, J. Donahue, T. Darrell, and J. Malik. Rich feature hierarchies for accurate object detection and semantic  
segmentation. In In Proc. IEEE Conf. on Computer Vision and Pattern Recognition, pages 580–587, 2014. 1  
[18] B. Hariharan, P. Arbelaez, R. Girshick, and J. Malik. Hyper-´ columns for object segmentation and fine-grained localization. In In Proc. IEEE Conf. on Computer Vision and Pattern Recognition, pages 447–456. IEEE, 2014. 1  
[19] K. He, G. Gkioxari, P. Dollar, and R. Girshick. Mask r-cnn.´ arXiv preprint arXiv:1703.06870, 2017. 8  
[20] K. He, X. Zhang, S. Ren, and J. Sun. Deep residual learning for image recognition. In In Proc. IEEE Conf. on Computer Vision and Pattern Recognition, 2016. 5, 11  
[21] H. Hirschmuller. Accurate and efficient stereo processing by semi-global matching and mutual information. In In Proc. IEEE Conf. on Computer Vision and Pattern Recognition, volume 2, pages 807–814. IEEE, 2005. 8  
[22] H. Hirschmuller. Stereo processing by semiglobal matching and mutual information. IEEE Transactions on pattern analysis and machine intelligence, 30(2):328–341, 2008. 6  
[23] J.-T. Huang, J. Li, D. Yu, L. Deng, and Y. Gong. Crosslanguage knowledge transfer using multilingual deep neural network with shared hidden layers. In Acoustics, Speech and Signal Processing (ICASSP), 2013 IEEE International Conference on, pages 7304–7308. IEEE, 2013. 1  
[24] A. Kendall and Y. Gal. What uncertainties do we need in bayesian deep learning for computer vision? arXiv preprint arXiv:1703.04977, 2017. 4  
[25] A. Kendall, M. Grimes, and R. Cipolla. Convolutional networks for real-time 6-dof camera relocalization. In Proceedings of the International Conference on Computer Vision (ICCV), 2015. 2, 3  
[26] J. Kirkpatrick, R. Pascanu, N. Rabinowitz, J. Veness, G. Desjardins, A. A. Rusu, K. Milan, J. Quan, T. Ramalho, A. Grabska-Barwinska, et al. Overcoming catastrophic forgetting in neural networks. Proceedings of the National Academy ofSciences, page 201611835, 2017. 2  
[27] I. Kokkinos. Ubernet: Training auniversal’convolutional neural network for low-, mid-, and high-level vision using diverse datasets and limited memory. arXiv preprint arXiv:1609.02132, 2016. 1, 2, 3  
[28] B. Leibe, A. Leonardis, and B. Schiele. Robust object detection with interleaved categorization and segmentation. International Journal of Computer Vision (IJCV), 77(1-3):259– 289, 2008. 6  
[29] X. Liang, Y. Wei, X. Shen, J. Yang, L. Lin, and S. Yan. Proposal-free network for instance-level object segmentation. arXiv preprint arXiv:1509.02636, 2015. 6  
[30] Y. Liao, S. Kodagoda, Y. Wang, L. Shi, and Y. Liu. Understand scene categories by objects: A semantic regularized scene classifier using convolutional neural networks. In 2016 IEEE International Conference on Robotics and Automation (ICRA), pages 2318–2325. IEEE, 2016. 2, 3  
[31] G. Lin, C. Shen, I. Reid, et al. Efficient piecewise training of deep structured models for semantic segmentation. arXiv preprint arXiv:1504.01013, 2015. 8  
[32] J. Long, E. Shelhamer, and T. Darrell. Fully convolutional networks for semantic segmentation. In Proc. IEEE Conf. on Computer Vision and Pattern Recognition, 2015. 1  
[33] J. MacQueen et al. Some methods for classification and analysis of multivariate observations. In Proceedings of the fifth Berkeley symposium on mathematical statistics and probability, volume 1, pages 281–297. Oakland, CA, USA., 1967. 6  
[34] I. Misra, A. Shrivastava, A. Gupta, and M. Hebert. Crossstitch networks for multi-task learning. In Proceedings ofthe IEEE Conference on Computer Vision and Pattern Recognition, pages 3994–4003, 2016. 2  
[35] J. Ngiam, A. Khosla, M. Kim, J. Nam, H. Lee, and A. Y. Ng. Multimodal deep learning. In Proceedings of the 28th international conference on machine learning (ICML-11), pages 689–696, 2011. 2  
[36] M. Oquab, L. Bottou, I. Laptev, and J. Sivic. Learning and transferring mid-level image representations using convolutional neural networks. In In Proc. IEEE Conf. on Computer Vision and Pattern Recognition, pages 1717–1724. IEEE, 2014. 2  
[37] P. O. Pinheiro, R. Collobert, and P. Dollar. Learning to segment object candidates. In Advances in Neural Information Processing Systems, pages 1990–1998, 2015. 1  
[38] P. Sermanet, D. Eigen, X. Zhang, M. Mathieu, R. Fergus, and Y. LeCun. Overfeat: Integrated recognition, localization and detection using convolutional networks. International Conference on Learning Representations (ICLR), 2014. 1, 2, 3  
[39] M. Teichmann, M. Weber, M. Zoellner, R. Cipolla, and R. Urtasun. Multinet: Real-time joint semantic reasoning for autonomous driving. arXiv preprint arXiv:1612.07695, 2016. 2, 3  
[40] S. Thrun. Is learning the n-th thing any easier than learning the first? In Advances in neural information processing systems, pages 640–646. MORGAN KAUFMANN PUBLISH-ERS, 1996. 2  
[41] J. Uhrig, M. Cordts, U. Franke, and T. Brox. Pixel-level encoding and depth layering for instance-level semantic labeling. arXiv preprint arXiv:1604.05096, 2016. 2, 3, 8  
[42] F. Yu and V. Koltun. Multi-scale context aggregation by dilated convolutions. In ICLR, 2016. 1  
[43] S. Zagoruyko and N. Komodakis. Wide residual networks. In E. R. H. Richard C. Wilson and W. A. P. Smith, editors, Proceedings of the British Machine Vision Conference (BMVC), pages 87.1–87.12. BMVA Press, September 2016.  
[44] H. Zhao, J. Shi, X. Qi, X. Wang, and J. Jia. Pyramid scene parsing network. arXiv preprint arXiv:1612.01105, 2016. 8  
[45] S. Zheng, S. Jayasumana, B. Romera-Paredes, V. Vineet, Z. Su, D. Du, C. Huang, and P. Torr. Conditional random fields as recurrent neural networks. In International Conference on Computer Vision (ICCV), 2015. 1

## A. Model Architecture Details

We base our model on the recently introduced DeepLabV3 [10] segmentation architecture. We use ResNet101 [20] as our base feature encoder, with dilated convolutions, resulting in a feature map which is downsampled by a factor of 8 compared with the original input image. We then append dilated (atrous) convolutional ASPP module [10]. This module is designed to improve the contextual reasoning of the network. We use an ASPP module comprised of four parallel convolutional layers, with 256 output channels and dilation rates (1, 12, 24, 36), with kernel sizes $( 1 ^ { 2 } , 3 ^ { 2 } , 3 ^ { 2 } , 3 ^ { 2 } )$ ). Additionally, we also apply global average pooling to the encoded features, and convolve them to 256 dimensions with $\textbf { a } 1 \times 1$ kernel. We apply batch normalisation to each of these layers and concatenate the resulting 1280 features together. This produces the shared representation between each task.

We then split the network, to decode this representation to a given task output. For each task, we construct a decoder consisting of two layers. First, we apply a 1×1 convolution, outputting 256 features, followed by batch normalisation and a non-linear activation. Finally, we convolve this output to the required dimensions for a given task. For classification, this will be equal to the number of semantic classes, otherwise the output will be 1 or 2 channels for depth or instance segmentation respectively. Finally, we apply bilinear upsampling to scale the output to the same resolution as the input.

The majority of the model’s parameters and depth is in the feature encoding, with very little flexibility in each task decoder. This illustrates the attraction of multitask learning; most of the compute can be shared between each task to learn a better shared representation.

## A.1. Optimisation

For all experiments, we use an initial learning rate of $2 . 5 \times 1 0 ^ { - 3 }$ and polynomial learning rate decay $( 1 \textrm { -- }$ $\textstyle \frac { i t e r } { m a x ~ i t e r } ) ^ { 0 . 9 }$ . We train using stochastic gradient descent, with Nesterov updates and momentum 0.9 and weight decay 10<sup>4</sup>. We conduct all experiments in this paper using PyTorch.

For the experiments on the Tiny CityScapes validation dataset (using a down-sampled resolution of 128 × 256) we train over 50, 000 iterations, using $2 5 6 \times 2 5 6$ crops with batch size of 8 on a single NVIDIA 1080Ti GPU. We apply random horizontal flipping to the data.

For the full-scale CityScapes benchmark experiment, we train over 100, 000 iterations with a batch size of 16. We apply random horizontal flipping (with probability 0.5) and random scaling (selected from $0 . 7 \textrm { - } 2 . 0 )$ to the data during training, before making a $5 1 2 \times 5 1 2$ crop. The training data is sampled uniformly, and is randomly shuffled for each epoch. Training takes five days on a single computer with four NVIDIA 1080Ti GPUs.

## B. Further Analysis

This task uncertainty loss is also robust to the value we use to initialise the task uncertainty values. One of the attractive properties of our approach to weighting multi-task losses is that it is robust to the initialisation choice for the homoscedastic noise parameters. Figure 6 shows that for an array of initial choices of log $\sigma ^ { 2 }$ from −2.0 to 5.0 the homoscedastic noise and task loss is able to converge to the same minima. Additionally, the homoscedastic noise terms converges after only 100 iterations, while the network requires 30, 000+ iterations to train. Therefore our model is robust to the choice of initial value for the weighting terms.

Figure 7 shows losses and uncertainty estimates for each task during training of the final model on the full-size CityScapes dataset. At a point 500 iterations into training, the model estimates task variance of 0.60, 62.5 and 13.5 for semantic segmentation, instance segmentation and depth regression, respectively. Becuase the losses are weighted by the inverse of the uncertainty estimates, this results in a task weighting ratio of approximately 23 : 0.22 : 1 between semantics, instance and depth, respectively. At the conclusion of training, the three tasks have uncertainty estimates of 0.075, 3.25 and 20.4, which results in effective weighting between the tasks of 43: 0.16 : 1. This shows how the task uncertainty estimates evolve over time, and the approximate final weightings the network learns. We observe they are far from uniform, as is often assumed in previous literature.

Interestingly, we observe that this loss allows the network to dynamically tune the weighting. Typically, the homoscedastic noise terms decrease in magnitude as training progresses. This makes sense, as during training the model becomes more effective at a task. Therefore the error, and uncertainty, will decrease. This has a side-effect of increasing the effective learning rate – because the overall uncertainty decreases, the weight for each task’s loss increases. In our experiments we compensate for this by annealing the learning rate with a power law.

Finally, a comment on the model’s failure modes. The model exhibits similar failure modes to state-of-the-art single-task models. For example, failure with objects out of the training distribution, occlusion or visually challenging situations. However, we also observe our multi-task model tends to fail with similar effect in all three modalities. Ie. an erroneous pixel’s prediction in one task will often be highly correlated with error in another modality. Some examples can be seen in Figure 8.

![](images/5845077310e8706c5cbee6a2b25ca812deab76455137497a08d8ecc9bf984694.jpg)

<details>
<summary>line</summary>

| Training Iterations | Yellow Line | Cyan Line | Gray Line | Red Line | Pink Line | Purple Line |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | ~4.9 | ~3.9 | ~2.9 | ~1.9 | ~1.0 | ~-0.3 |
| 50 | ~0.8 | ~0.6 | ~0.1 | ~0.1 | ~-0.2 | ~-0.2 |
| 100 | ~-0.3 | ~-0.3 | ~-0.4 | ~-0.4 | ~-0.4 | ~-0.4 |
| 150 | ~-0.4 | ~-0.5 | ~-0.5 | ~-0.4 | ~-0.6 | ~-0.6 |
| 200 | ~-0.6 | ~-0.6 | ~-0.6 | ~-0.5 | ~-0.6 | ~-0.6 |
| 250 | ~-0.7 | ~-0.7 | ~-0.6 | ~-0.5 | ~-0.6 | ~-0.6 |
</details>

(a) Semantic segmentation task

![](images/48d73675cc566ef2f15c9ec75da7c12f9200732aa3462472428e5ad8485a1986.jpg)

<details>
<summary>line</summary>

| Training Iterations | Purple Line | Yellow Line | Brown Line | Pink Line | Cyan Line | Grey Line |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | ~1.5 | ~4.9 | ~1.5 | ~1.5 | ~3.9 | ~2.8 |
| 25 | ~7.2 | ~3.6 | ~4.4 | ~3.6 | ~2.9 | ~3.0 |
| 50 | ~5.0 | ~2.8 | ~2.8 | ~2.8 | ~2.7 | ~2.7 |
| 100 | ~2.5 | ~2.5 | ~2.5 | ~2.5 | ~2.4 | ~2.4 |
| 150 | ~2.3 | ~2.3 | ~2.3 | ~2.3 | ~2.3 | ~2.3 |
| 200 | ~2.1 | ~2.1 | ~2.1 | ~2.1 | ~2.2 | ~2.1 |
| 250 | ~2.1 | ~2.1 | ~2.1 | ~2.1 | ~2.0 | ~2.0 |
</details>

(b) Instance segmentation task

![](images/1c7c314a4398714c63700eef24c9ae7503cb006f3af21805136f079466c8b2d2.jpg)

<details>
<summary>line</summary>

| Training Iterations | Yellow Line | Cyan Line | Gray Line | Red Line | Purple Line |
| --- | --- | --- | --- | --- | --- |
| 0 | ~4.9 | ~3.9 | ~2.9 | ~2.0 | ~0.3 |
| 10 | ~4.2 | ~3.3 | ~2.4 | ~1.6 | ~1.1 |
| 20 | ~3.4 | ~2.6 | ~1.7 | ~1.2 | ~1.0 |
| 30 | ~2.6 | ~1.9 | ~1.3 | ~0.8 | ~0.7 |
| 40 | ~1.9 | ~1.3 | ~0.8 | ~0.5 | ~0.4 |
| 50 | ~1.2 | ~0.7 | ~0.4 | ~0.2 | ~0.1 |
| 60 | ~0.6 | ~0.3 | ~0.1 | ~-0.1 | ~-0.1 |
| 70 | ~0.2 | ~0.1 | ~-0.1 | ~-0.2 | ~-0.2 |
| 80 | ~-0.1 | ~-0.1 | ~-0.2 | ~-0.2 | ~-0.2 |
| 90 | ~-0.2 | ~-0.2 | ~-0.2 | ~-0.2 | ~-0.2 |
| 100 | ~-0.2 | ~-0.2 | ~-0.2 | ~-0.2 | ~-0.2 |
| 150 | ~-0.3 | ~-0.3 | ~-0.3 | ~-0.3 | ~-0.3 |
| 200 | ~-0.4 | ~-0.4 | ~-0.4 | ~-0.4 | ~-0.4 |
| 250 | ~-0.4 | ~-0.4 | ~-0.4 | ~-0.4 | ~-0.5 |
</details>

(c) Depth regression task  
Figure 6: Training plots showing convergence of homoscedastic noise and task loss for an array of initialisation choices for the homoscedastic uncertainty terms for all three tasks. Each plot shows the the homoscedastic noise value optimises to the same solution from a variety of initialisations. Despite the network taking 10, 000+ iterations for the training loss to converge, the task uncertainty converges very rapidly after only 100 iterations.

![](images/04aad0ef4be821070d695100778c7b2bac97f6dc70abb1ba5add92d867e1b053.jpg)

<details>
<summary>line</summary>

| Training Iterations | Loss |
| --- | --- |
| 0 | ~0.92 |
| 2000 | ~0.35 |
| 4000 | ~0.28 |
| 6000 | ~0.15 |
| 8000 | ~0.12 |
| 10000 | ~0.18 |
| 12000 | ~0.12 |
| 14000 | ~0.15 |
| 16000 | ~0.12 |
| 18000 | ~0.18 |
| 20000 | ~0.12 |
| 22000 | ~0.15 |
| 24000 | ~0.12 |
| 26000 | ~0.15 |
| 28000 | ~0.12 |
| 30000 | ~0.15 |
| 32000 | ~0.12 |
| 34000 | ~0.15 |
| 36000 | ~0.12 |
| 38000 | ~0.15 |
| 40000 | ~0.12 |
| 42000 | ~0.15 |
| 44000 | ~0.12 |
| 46000 | ~0.15 |
| 48000 | ~0.12 |
| 50000 | ~0.15 |
| 52000 | ~0.12 |
| 54000 | ~0.15 |
| 56000 | ~0.12 |
| 58000 | ~0.15 |
| 60000 | ~0.12 |
| 62000 | ~0.15 |
| 64000 | ~0.12 |
| 66000 | ~0.15 |
| 68000 | ~0.12 |
| 70000 | ~0.15 |
| 72000 | ~0.33 |
| 74000 | ~0.12 |
| 76000 | ~0.15 |
| 78000 | ~0.12 |
| 80000 | ~0.15 |
| 82000 | ~0.12 |
| 84000 | ~0.15 |
| 86000 | ~0.12 |
| 88000 | ~0.15 |
| 90000 | ~0.12 |
| 92000 | ~0.15 |
| 94000 | ~0.12 |
| 96000 | ~0.15 |
| 98000 | ~0.12 |
| 100000 | ~0.15 |
</details>

![](images/087746528f37d3e2e40f5408832198fc5c1d89847d35ae301046da7005f04869.jpg)

<details>
<summary>line</summary>

| Training Iterations | Loss |
| --- | --- |
| 0 | ~75 |
| 2000 | ~80 |
| 4000 | ~65 |
| 6000 | ~75 |
| 8000 | ~92 |
| 10000 | ~35 |
| 12000 | ~45 |
| 14000 | ~30 |
| 16000 | ~75 |
| 18000 | ~118 |
| 20000 | ~35 |
| 22000 | ~95 |
| 24000 | ~102 |
| 26000 | ~35 |
| 28000 | ~55 |
| 30000 | ~35 |
| 32000 | ~55 |
| 34000 | ~35 |
| 36000 | ~70 |
| 38000 | ~35 |
| 40000 | ~45 |
| 42000 | ~90 |
| 44000 | ~145 |
| 46000 | ~92 |
| 48000 | ~35 |
| 50000 | ~45 |
| 52000 | ~35 |
| 54000 | ~58 |
| 56000 | ~35 |
| 58000 | ~68 |
| 60000 | ~35 |
| 62000 | ~75 |
| 64000 | ~35 |
| 66000 | ~75 |
| 68000 | ~35 |
| 70000 | ~58 |
| 72000 | ~35 |
| 74000 | ~35 |
| 76000 | ~35 |
| 78000 | ~35 |
| 80000 | ~35 |
| 82000 | ~92 |
| 84000 | ~35 |
| 86000 | ~35 |
| 88000 | ~45 |
| 90000 | ~35 |
| 92000 | ~35 |
| 94000 | ~35 |
| 96000 | ~35 |
| 98000 | ~35 |
| 100000 | ~35 |
</details>

![](images/2b48b3a5f45df06fb94a2df9576e3b8f4ac5d21adbfc420c88bc7cdc6b7f7e82.jpg)

<details>
<summary>line</summary>

| Training Iterations | Loss |
| --- | --- |
| 0 | ~22 |
| 5000 | ~7 |
| 10000 | ~6 |
| 15000 | ~5 |
| 20000 | ~4 |
| 25000 | ~4 |
| 30000 | ~4 |
| 35000 | ~4 |
| 40000 | ~4 |
| 45000 | ~4 |
| 50000 | ~3 |
| 55000 | ~3 |
| 60000 | ~3 |
| 65000 | ~3 |
| 70000 | ~3 |
| 75000 | ~3 |
| 80000 | ~3 |
| 85000 | ~3 |
| 90000 | ~3 |
| 95000 | ~3 |
| 100000 | ~3 |
</details>

![](images/7c1e1a12256ba5de8173b5243547a521b5fe0f24032d539830dfcbb000b93589.jpg)

<details>
<summary>line</summary>

| Training Iterations | Task Uncertainty \((\sigma^{2})\) |
| --- | --- |
| 0 | ~1.9 |
| 20000 | ~0.15 |
| 40000 | ~0.12 |
| 60000 | ~0.1 |
| 80000 | ~0.08 |
| 100000 | ~0.07 |
</details>

(a) Semantic segmentation task

![](images/0d1856b299ef8f9ad297235860f0aefc26f2de9bffab3494d1ff45bede823c0f.jpg)

<details>
<summary>line</summary>

| Training Iterations | Task Uncertainty \((\sigma^{2})\) |
| --- | --- |
| 0 | ~93 |
| 20000 | ~40 |
| 40000 | ~35 |
| 60000 | ~30 |
| 80000 | ~25 |
| 100000 | ~21 |
</details>

(b) Instance segmentation task

![](images/3d9228feab3d06433dab482cc907959b21ca8c07f26e633e7a6ee75dd7452527.jpg)

<details>
<summary>line</summary>

| Training iterations | Task Uncertainty \((\sigma^{2})\) |
| --- | --- |
| 0 | ~43 |
| 5000 | ~8 |
| 10000 | ~6 |
| 20000 | ~5.5 |
| 40000 | ~5 |
| 60000 | ~4.5 |
| 80000 | ~4 |
| 100000 | ~3 |
</details>

(c) Depth regression task  
Figure 7: Learning task uncertainty. These training plots show the losses and task uncertainty estimates for each task during training. Results are shown for the final model, trained on the fullsize CityScapes dataset.

C. Further Qualitative Results  
![](images/3b9b194846d343aa64378dc6625dea217d6c2c3d99dedcb2043d12fe276cc571.jpg)  
Figure 8: More qualitative results on test images from the CityScapes dataset.

D. Failure Examples  
![](images/44afa1946e59237708ba356993346a3fab8a278b102891d61790a885b7f5d8aa.jpg)  
Figure 9: Example where our model fails on the CityScapes test data. The first two rows show examples of challenging visual effects such as reflection, which confuse the model. Rows three and four show the model incorrectly distinguishing between road and footpath. This is a common mistake, which we believe is due to a lack of contextual reasoning. Rows five, six and seven demonstrate incorrect classification of a rare class (bus, fence and motorbike, respectively). Finally, the last two rows show failure due to occlusion and where the object is too big for the model’s receptive field. Additionally, we observe that failures are highly correlated between the modes, which makes sense as each output is conditioned on the same feature vector. For example, in the second row, the incorrect labelling of the reflection as a person causes the depth estimation to predict human geometry.