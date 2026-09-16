# End-to-End Multi-Task Learning with Attention

Shikun Liu Edward Johns Andrew J. Davison

Department of Computing, Imperial College London

{shikun.liu17, e.johns, a.davison}@imperial.ac.uk

## Abstract

We propose a novel multi-task learning architecture, which allows learning of task-specific feature-level attention. Our design, the Multi-Task Attention Network (MTAN), consists of a single shared network containing a global feature pool, together with a soft-attention module for each task. These modules allow for learning of taskspecific features from the global features, whilst simultaneously allowing for features to be shared across different tasks. The architecture can be trained end-to-end and can be built upon any feed-forward neural network, is simple to implement, and is parameter efficient. We evaluate our approach on a variety of datasets, across both image-toimage predictions and image classification tasks. We show that our architecture is state-of-the-art in multi-task learning compared to existing methods, and is also less sensitive to various weighting schemes in the multi-task loss function. Code is available at https://github.com/ lorenmt/mtan.

## 1. Introduction

Convolutional Neural Networks (CNNs) have seen great success in a range of computer vision tasks, including image classification [11], semantic segmentation [1], and style transfer [13]. However, these networks are typically designed to achieve only one particular task. For more complete vision systems in real-world applications, a network which can perform multiple tasks simultaneously is far more desirable than building a set of independent networks, one for each task. This is more efficient not only in terms of memory and inference speed, but also in terms of data, since related tasks may share informative visual features.

This type of learning is called Multi-Task Learning (MTL) [20, 14, 6], and in this paper we present a novel architecture for MTL based on feature-level attention masks, which add greater flexibility to share complementary features. Compared to standard single-task learning, training multiple tasks whilst successfully learning a shared representation poses two key challenges:

![](images/16aa4e1c9c1036d79fe91ba85f31e16ca75b245607b382137cdda85d9706d963.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph LR
  A["Shared Features"] --> B["Task-Specific Attention Modules"]
  B --> C["Visual Inputs"]
  B --> D["Task-Specific Attention Modules"]
  D --> E["Visual Inputs"]
```
</details>

Figure 1: Overview of our proposal MTAN. The shared network takes input data and learns task-shared features, whilst each attention network learns task-specific features, by applying attention modules to the shared network.

i) Network Architecture (how to share): A multi-task learning architecture should express both task-shared and task-specific features. In this way, the network is encouraged to learn a generalisable representation (to avoid over-fitting), whilst also providing the ability to learn features tailored to each task (to avoid under-fitting).  
ii) Loss Function (how to balance tasks): A multi-task loss function, which weights the relative contributions of each task, should enable learning of all tasks with equal importance, without allowing easier tasks to dominate. Manual tuning of loss weights is tedious, and it is preferable to automatically learn the weights, or design a network which is robust to different weights.

However, most prior MTL approaches focus on only one of these two challenges, whilst maintaining a standard implementation of the other. In this paper, we introduce a unified approach which addresses both challenges cohesively, by designing a novel network which (i) enables both taskshared and task-specific features to be learned automatically, and consequently (ii) learns an inherent robustness to the choice of loss weighting scheme.

The proposed network, which we call the Multi-Task Attention Network (MTAN) (see Figure 1), is composed of a single shared network, which learns a global feature pool containing features across all tasks. Then for each task, rather than learning directly from the shared feature pool, a soft attention mask is applied at each convolution block in the shared network. In this way, each attention mask automatically determines the importance of the shared features for the respective task, allowing learning of both taskshared and task-specific features in a self-supervised, endto-end manner. This flexibility enables much more expressive combinations of features to be learned for generalisation across tasks, whilst still allowing for discriminative features to be tailored for each individual task. Furthermore, automatically choosing which features to share and which to be task specific allows for a highly efficient architecture with far fewer parameters than multi-task architectures which have explicit separation of tasks [26, 20].

MTAN can be built on any feed-forward neural network depending on the type of tasks. We first evaluate MTAN with SegNet [1], an encoder-decoder network on the tasks of semantic segmentation and depth estimation on the outdoor CityScapes dataset [4], and then with an additional task of surface normal prediction on the more challenging indoor dataset NYUv2 [21]. We also test our approach with a different backbone architecture, Wide Residual Network [31], on the recently proposed Visual Decathlon Challenge [23], to solve 10 individual image classification tasks. Results show that MTAN outperforms several baselines and is competitive with the state-of-the-art for multi-task learning, whilst being more parameter efficient and therefore scaling more gracefully with the number of tasks. Furthermore, our method shows greater robustness to the choice of weighting scheme in the loss function compared to baselines. As part of our evaluation of this robustness, we also propose a novel weighting scheme, Dynamic Weight Average (DWA), which adapts the task weighting over time by considering the rate of change of the loss for each task.

## 2. Related Work

The term Multi-Task Learning (MTL) has been broadly used in machine learning [2, 8, 6, 17], with similarities to transfer learning [22, 18] and continual learning [29]. In computer vision, multi-task learning has been used to for learning similar tasks such as image classification in multiple domains [23], pose estimation and action recognition [9], and dense prediction of depth, surface normals, and semantic classes [20, 7]. In this paper, we consider two important aspects of multi-task learning: how can a good multitask network architecture be designed, and how to balance feature sharing in multi-task learning across all tasks?

Most multi-task learning network architectures for computer vision are designed based on existing CNN architectures. For example, Cross-Stitch Networks [20] contain one standard feed-forward network per task, with cross-stitch units to allow features to be shared across tasks. The selfsupervised approach of [6], based on the ResNet101 architecture [30], learns a regularised combination of features from different layers of a single shared network. UberNet [16] proposes an image pyramid approach to process images across multiple resolutions, where for each resolution, additional task-specific layers are formed top of the shared VGG-Net [27]. The Progressive Networks [26] uses a sequence of incrementally-trained networks to transfer knowledge between tasks. However, architectures such as Cross-Stitch Networks and Progressive Networks require a large number of network parameters, and scale linearly with the number of tasks. In contrast, our model requires only a rough 10% increase in parameters for per learning task.

On the balancing of feature sharing in multi-task learning, there is extensive experimental analysis in [20, 14], with both papers arguing that different amounts of sharing and weighting tend to work best for different tasks. One example of weighting tasks appropriately is with the use of weight uncertainty [14], which modifies the loss functions in multi-task learning using task uncertainty. Another method is that of GradNorm [3], which manipulates gradient norms over time to control the training dynamics. As an alternative to using task losses to determine task difficulties, Dynamic Task Prioritisation [10] encourages prioritisation of difficult tasks directly using performance metrics such as accuracy and precision.

## 3. Multi-Task Attention Network

We now introduce our novel multi-task learning architecture, the Multi-Task Attention Network (MTAN). Whilst the architecture can be incorporated into any feed-forward network, in the following section we demonstrate how to build MTAN upon an encoder-decoder network, SegNet [1]. This example configuration allows for image-to-image dense pixel-level prediction, such as semantic segmentation, depth estimation, and surface normal prediction.

## 3.1. Architecture Design

MTAN consists of two components: a single shared network, and K task-specific attention networks. The shared network can be designed based on the particular task, whilst each task-specific network consists of a set of attention modules, which link with the shared network. Each attention module applies a soft attention mask to a particular layer of the shared network, to learn task-specific features. As such, the attention masks can be considered as feature selectors from the shared network, which are automatically learned in an end-to-end manner, whilst the shared network learns a compact global feature pool across all tasks.

Figure 2 shows a detailed visualisation of our network based on VGG-16 [27], illustrating the encoder half of Seg-Net. The decoder half of SegNet is then symmetric to VGG-16. As shown, each attention module learns a soft attention mask, which itself is dependent on the features in the shared network at the corresponding layer. Therefore, the features in the shared network, and the soft attention masks, can be learned jointly to maximise the generalisation of the shared features across multiple tasks, whilst simultaneously maximising the task-specific performance due to the attention masks.

![](images/8a28b6bacfb1a5b06a9b79507174f22a610e07b8e1d945a2914439f34b1af7b7.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph LR
  subgraph Task1
  A["Image"] --> B["conv conv pool"]
  B --> C["conv conv pool"]
  C --> D["conv conv pool"]
  D --> E["conv conv pool"]
  E --> F["conv conv pool"]
  F --> G["conv conv pool"]
  G --> H["conv conv pool"]
  end

  subgraph Task2
    I["Attention Module"] -.-> J["Attention Module"]
    K["Attention Module"] -.-> L["Attention Module"]
    M["Attention Module"] -.-> N["Attention Module"]
    O["Attention Module"] -.-> P["Attention Module"]
    Q["Attention Module"] -.-> R["Attention Module"]
    S["Attention Module"] -.-> T["Attention Module"]
    U["Attention Module"] -.-> V["Attention Module"]
  end

  subgraph AttentionModule["Attention Module for Encoder & Decoder"]
  W["conv"] --> X["conv"]
  Y["conv"] --> Z["conv"]
  AA["conv"] --> AB["conv"]
  AC["conv"] --> AD["conv"]
  AE["conv"] --> AF["conv"]
  AG["conv"] --> AH["conv"]
  AI["conv"] --> AJ["conv"]
  end

  subgraph Concatenation
  AK["samp"] --> AL["conv"]
  AM["conv"] --> AN["conv"]
  AO["conv"] --> AP["conv"]
  AQ["conv"] --> AR["conv"]
  AS["conv"] --> AT["conv"]
  end

  subgraph ElementWiseMultiplication
  AU["samp"] --> AV["samp"]
  AW["samp"] --> AX["samp"]
  A --> AW
  A --> AX
  AX --> AY{"Merge"}
  AX --> AXY{"Merge"}
  AXY --> AXF{"Merge"}
  AXF --> AXG{"Merge"}
  AXG --> AXH{"Merge"}
  AXH --> AXI{"Merge"}
  AXI --> AXJ{"Merge"}
  AXJ --> AXK{"Merge"}
  AXK --> AXL{"Merge"}
  end

  subgraph ForwardDataFlow
  M --> N
  N --> O
  O --> P
  P --> Q
  Q --> R
  R --> S
  S --> T
  T --> U
  U --> V
  V --> W
  W --> X
  X --> Y
  Y --> Z
  Z --> AA
  AA --> AB
  AB --> AC
  AC --> AD
  AD --> AE
  AE --> AF
  AF --> AG
  AG --> AH
  AH --> AI
  AI --> AJ
  AJ --> AK
  AK --> AL
  AL --> AM
  AM --> AN
  AN --> AO
  AO --> AP
  AP --> AQ
  AQ --> AR
  AR --> AS
  AS --> AT
  AT --> AU
  AU --> AV
  AV --> AW
  AW --> AX
  AX --> AXY
  AXY --> AXZ{"Merge"}
  AXZ -- "u" --> W
  AXZ -- "g" --> X
  X --> Y
  Y --> Z
  Z --> AA
  AA --> AB
  AB --> AC
  AC --> AD
  AD --> AE
  AE --> AF
  AF --> AG
  AG --> AH
  AH --> AI
  AI --> AJ
  AJ --> AK
  end
```
</details>

Figure 2: Visualisation of MTAN based on VGG-16, showing the encoder half of SegNet (with the decoder half being symmetrical to the encoder). Task one (green) and task two (blue) have their own set of attention modules, which link with the shared network (grey). The middle attention module has its structure exposed for visualisation, which is further expanded in the bottom section of the figure, showing both the encoder and decoder versions of the module. All attention modules have the same design, although their weights are individually learned.

## 3.2. Task Specific Attention Module

The attention module is designed to allow the taskspecific network to learn task-related features, by applying a soft attention mask to the features in the shared network, with one attention mask per task per feature channel. We denote the shared features in the $j ^ { t h }$ block of the shared network as $p ^ { ( j ) }$ , and the learned attention mask in this layer for task i as $a _ { i } ^ { ( j ) }$ . The task-specific features $\hat { a } _ { i } ^ { ( j ) }$ in this layer, are then computed by element-wise multiplication of the attention masks with the shared features:

$$
\hat {a} _ {i} ^ {(j)} = a _ {i} ^ {(j)} \odot p ^ {(j)}, \tag {1}
$$

where  denotes element-wise multiplication.

As shown in Figure 2, the first attention module in the encoder takes as input only features in the shared network. But for subsequent attention modules in block j, the input is formed by a concatenation of the shared features $u ^ { ( j ) }$ , and the task-specific features from the previous layer $\hat { a } _ { i } ^ { ( j - 1 ) }$

$$
a _ {i} ^ {(j)} = h _ {i} ^ {(j)} \left(g _ {i} ^ {(j)} \left(\left[ u ^ {(j)}; f ^ {(j)} \left(\hat {a} _ {i} ^ {(j - 1)}\right) \right]\right)\right), j \geq 2 \tag {2}
$$

Here, $f ^ { ( j ) } , g _ { i } ^ { ( j ) } , h _ { i } ^ { ( j ) }$ are convolutional layers with batch normalisation, following a non-linear activation. Both $g _ { i } ^ { ( j ) }$ and $h _ { i } ^ { ( j ) }$ are composed of $[ 1 \times 1 ]$ kernels presenting the $i ^ { t h }$ task-specific attention mask in block $\begin{array} { r l } { j . } & { { } f ^ { ( j ) } } \end{array}$ is composed of $[ 3 \times 3 ]$ kernels representing a shared feature extractor for passing to another attention module, following by a pooling or sampling layer to match the corresponding resolution.

The attention mask, following a sigmoid activation to ensure $a _ { i } ^ { ( j ) } \in [ 0 , 1 ]$ , is learned in a self-supervised fashion with back-propagation. $\operatorname { I f } a _ { i } ^ { ( j ) }  1$ such that the mask becomes an identity map, the attended feature maps are equivalent to global feature maps and the tasks share all the features. Therefore, we expect the performance to be no worse than that of a shared multi-task network, which splits into individual tasks only at the end of the network, and we show results demonstrating this in Section 4.

## 3.3. The Model Objective

In general multi-task learning with K tasks, input X and task-specific labels $\mathbf { Y } _ { i } , i = 1 , 2 , \cdots , K$ , the loss function is defined as,

$$
\mathcal {L} _ {t o t} (\mathbf {X}, \mathbf {Y} _ {1: K}) = \sum_ {i = 1} ^ {K} \lambda_ {i} \mathcal {L} _ {i} (\mathbf {X}, \mathbf {Y} _ {i}). \tag {3}
$$

This is the linear combination of task-specific losses $\mathcal { L } _ { i }$ with task weightings $\lambda _ { i } .$ . In our experiments, we study the effect of different weighting schemes on various multi-task learning approaches.

For image-to-image prediction tasks, we consider each mapping from input data X to a set of labels $\mathbf { Y } _ { i }$ as one task with total three tasks for evaluation. In each loss function, Y<sup>ˆ</sup> represents the network’s prediction, and Y represents the ground-truth label.

• For semantic segmentation, we apply a pixel-wise crossentropy loss for each predicted class label from a depthsoftmax classifier.

$$
\mathcal {L} _ {1} (\mathbf {X}, \mathbf {Y} _ {1}) = - \frac {1}{p q} \sum_ {p, q} \mathbf {Y} _ {1} (p, q) \log \hat {\mathbf {Y}} _ {1} (p, q). \tag {4}
$$

• For depth estimation, we apply an $L _ { 1 }$ norm comparing the predicted and ground-truth depth. We use true depth for the NYUv2 indoor scene dataset, and inverse depth in CityScapes outdoor scene dataset as standard, which can more easily represent points at infinite distances, such as the sky:

$$
\mathcal {L} _ {2} (\mathbf {X}, \mathbf {Y} _ {2}) = \frac {1}{p q} \sum_ {p, q} | \mathbf {Y} _ {2} (p, q) - \hat {\mathbf {Y}} _ {2} (p, q) |. \tag {5}
$$

• For surface normals (only available in NYUv2), we apply an element-wise dot product at each normalised pixel with the ground-truth map:

$$
\mathcal {L} _ {3} (\mathbf {X}, \mathbf {Y} _ {3}) = - \frac {1}{p q} \sum_ {p, q} \mathbf {Y} _ {3} (p, q) \cdot \hat {\mathbf {Y}} _ {3} (p, q). \tag {6}
$$

For image classification tasks, we consider each dataset as one task for which each dataset represents each individual classification task for one domain. We apply standard cross-entropy loss for all classification tasks.

## 4. Experiments

In this section, we evaluate our proposed method on two types of tasks: one-to-many predictions for image-to-image regression tasks in Section 4.1 and many-to-many predictions for image classification tasks (Visual Decathlon Challenge) in Section 4.2.

## 4.1. Image-to-Image Prediction (One-to-Many)

In this section, we evaluate MTAN built upon SegNet [1] on image-to-image prediction tasks. We first introduce the datasets used for validation in Section 4.1.1, and several baselines for comparison in Section 4.1.2. In Section 4.1.3, we introduce a novel adaptive weighting method, and in Section 4.1.4 we show the effectiveness of MTAN with various weighting methods compared with single and multitask baseline methods. We explore how the performance of our method scales with task complexity in Section 4.1.5 and we show visualisations of the learned attention masks in Section 4.1.6.

## 4.1.1 Datasets

CityScapes. The CityScapes dataset [4] consists of high resolution street-view images. We use this dataset for two tasks: semantic segmentation and depth estimation. To speed up training, all training and validation images were resized to $[ 1 2 8 \times 2 5 6 ]$ . The dataset contains 19 classes for pixel-wise semantic segmentation, together with groundtruth inverse depth labels. We pair the depth estimation task with three levels of semantic segmentation using 2, 7 or 19 classes (excluding the void group in 7 and 19 classes). Labels for the 19 classes and the coarser 7 categories are defined as in the original CityScapes dataset. We then further create a 2-class dataset with only background and foreground objects. The details of these segmentation classes are presented in Table 1. We perform multi-task learning for 7-class CityScapes dataset in Section 4.1.4. We compare the 2/7/19-class results in Section 4.1.5, with visualisation of these attention maps in Section 4.1.6.

NYUv2. The NYUv2 dataset [21] is consisted with RGB-D indoor scene images. We evaluate performances on three learning tasks: 13-class semantic segmentation defined in [5], true depth data which is recorded by depth cameras from Microsoft Kinect, and surface normals which are provided in [7]. To speed up training, all training and validation images were resized to [288 × 384] resolution.

Compared to CityScapes, NYUv2 contains images of indoor scenes, which are much more complex since the viewpoints can vary significantly, changable lighting conditions are present, and the appearance for each object class shifts widely in texture and shape. We evaluate performance on different datasets, together with different numbers of tasks, and further with different class complexities, in order to attain a comprehensive understanding on how our proposed method behaves and scales under a range of scenarios.

<table><tr><td>2-class</td><td>7-class</td><td>19-class</td></tr><tr><td rowspan="6">background</td><td>void</td><td>void</td></tr><tr><td>flat</td><td>road, sidewalk</td></tr><tr><td>construction</td><td>building, wall, fence</td></tr><tr><td>object</td><td>pole, traffic light, traffic sign</td></tr><tr><td>nature</td><td>vegetation, terrain</td></tr><tr><td>sky</td><td>sky</td></tr><tr><td rowspan="2">foreground</td><td>human</td><td>person, rider</td></tr><tr><td>vehicle</td><td>carm truck, bus, caravan, trailer, train, motorcycle</td></tr></table>

Table 1: Three levels of semantic classes for the CityScapes data used in our experiments.

## 4.1.2 Baselines

Most image-to-image multi-task learning architectures are designed based on specific feed-forward neural networks, or implemented on varying network architectures, and thus they are typically not directly comparable based on published results. Our method is general and can be applied to any feed-forward neural network, and so for a fair comparison, we implemented 5 different network architectures (2 single-task + 3 multi-task) based on SegNet [1], which we consider as baselines:

• Single-Task, One Task: The vanilla SegNet for single task learning.  
• Single-Task, STAN: A Single-Task Attention Network, where we directly apply our proposed MTAN whilst only performing a single task.  
• Multi-Task, Split (Wide, Deep): The standard multitask learning, which splits at the last layer for the final prediction for each specific task. We introduce two verions of Split: Wide, where we adjusted the number of convolutional filters, and Deep, where we adjusted the number of convolutional layers, until Split had at least as many parameters as MTAN.

• Multi-Task, Dense: A shared network together with task-specific networks, where each task-specific network receives all features from the shared network, without any attention modules.  
• Multi-Task, Cross-Stitch: The Cross-Stitch Network [20], a previously proposed adaptive multi-task learning approach, which we implemented on SegNet.

Note that all the baselines were designed to have at least as many parameters than our proposed MTAN, and were tested to validate that our proposed method’s better performance is due to the attention modules, rather than simply due to the increase in network parameters.

## 4.1.3 Dynamic Weight Average

For most multi-task learning networks, training multiple tasks is difficult without finding the correct balance between those tasks, and recent approaches have attempted to address this issue [3, 14]. To test our method across a range of weighting schemes, we propose a simple yet effective adaptive weighting method, named Dynamic Weight Average (DWA). Inspired by GradNorm [3], this learns to average task weighting over time by considering the rate of change of loss for each task. But whilst GradNorm requires access to the network’s internal gradients, our DWA proposal only requires the numerical task loss, and therefore its implementation is far simpler.

With DWA, we define the weighting $\lambda _ { k }$ for task k as:

$$
\lambda_ {k} (t) := \frac {K \exp (w _ {k} (t - 1) / T)}{\sum_ {i} \exp (w _ {i} (t - 1) / T)}, w _ {k} (t - 1) = \frac {\mathcal {L} _ {k} (t - 1)}{\mathcal {L} _ {k} (t - 2)}, \tag {7}
$$

Here, $w _ { k } ( \cdot )$ calculates the relative descending rate in the range $( 0 , + \infty )$ , t is an iteration index, and $T$ represents a temperature which controls the softness of task weighting, similar to [12]. A large $T$ results in a more even distribution between different tasks. If T is large enough, we have $\lambda _ { i } \approx 1$ , and tasks are weighted equally. Finally, the softmax operator, which is multiplied by $K ,$ , ensures that $\begin{array} { r } { \sum _ { i } \lambda _ { i } ( t ) = K } \end{array}$

In our implementation, the loss value $\mathcal { L } _ { k } ( t )$ is calculated as the average loss in each epoch over several iterations. Doing so reduces the uncertainty from stochastic gradient descent and random training data selection. For t = 1, 2, we initialise $w _ { k } ( t ) = 1$ , but any non-balanced initialisation based on prior knowledge could also be introduced.

## 4.1.4 Results on Image-to-Image Predictions

We now evaluate the performance of our proposed MTAN method in image-to-image multi-task learning, based on the SegNet architecture. Using the 7-class version of the

CityScapes dataset and 13-class version of NYUv2 dataset, we compare all the baselines introduced in Section 4.1.2.

Training. For each network architecture, we ran experiments with three types of weighting methods: equal weighting, weight uncertainty [14], and our proposed DWA (with hyper-parameter temperature $T = 2 ,$ found empirically to be optimum across all architectures). We did not include GradNorm [3] because it requires a manual choice of subset network weights across all baselines, based on their specific architectures, which distracts from a fair evaluation of the architectures themselves. We trained all the models with ADAM optimiser [15] using a learning rate of $1 0 ^ { - 4 }$ , with a batch size of 2 for NYUv2 dataset and 8 for CityScapes dataset. During training, we halve the learning rate at 40k iterations, for a total of 80k iterations.

Results. Table 2 and 3 shows experimental results for CityScales and NYUv2 datasets across all architectures, and across all loss function weighting schemes. Results also show the number of network parameters for each architecture. Our MTAN method performs similarly to our baseline Dense in the CityScapes dataset, whilst only having less than half the number of parameters, and outperforms all other baselines. For the more challenging NYUv2 dataset, our method outperforms all baselines across all weighting methods and all learning tasks.

<table><tr><td rowspan="2">#P.</td><td rowspan="2">Architecture</td><td rowspan="2">Weighting</td><td colspan="2">Segmentation</td><td colspan="2">Depth</td></tr><tr><td>(Higher mIoU)</td><td>Better) Pix Acc</td><td>(Lower Abs Err)</td><td>Better) Rel Err</td></tr><tr><td>2</td><td>One Task</td><td>n.a.</td><td>51.09</td><td>90.69</td><td>0.0158</td><td>34.17</td></tr><tr><td>3.04</td><td>STAN</td><td>n.a.</td><td>51.90</td><td>90.87</td><td>0.0145</td><td>27.46</td></tr><tr><td rowspan="3">1.75</td><td rowspan="3">Split, Wide</td><td>Equal Weights</td><td>50.17</td><td>90.63</td><td>0.0167</td><td>44.73</td></tr><tr><td>Uncert. Weights [14]</td><td>51.21</td><td>90.72</td><td>0.0158</td><td>44.01</td></tr><tr><td>DWA,  $T = 2$ </td><td>50.39</td><td>90.45</td><td>0.0164</td><td>43.93</td></tr><tr><td rowspan="3">2</td><td rowspan="3">Split, Deep</td><td>Equal Weights</td><td>49.85</td><td>88.69</td><td>0.0180</td><td>43.86</td></tr><tr><td>Uncert. Weights [14]</td><td>48.12</td><td>88.68</td><td>0.0169</td><td>39.73</td></tr><tr><td>DWA,  $T = 2$ </td><td>49.67</td><td>88.81</td><td>0.0182</td><td>46.63</td></tr><tr><td rowspan="3">3.63</td><td rowspan="3">Dense</td><td>Equal Weights</td><td>51.91</td><td>90.89</td><td>0.0138</td><td>27.21</td></tr><tr><td>Uncert. Weights [14]</td><td>51.89</td><td>91.22</td><td>0.0134</td><td>25.36</td></tr><tr><td>DWA,  $T = 2$ </td><td>51.78</td><td>90.88</td><td>0.0137</td><td>26.67</td></tr><tr><td rowspan="3"> $\approx 2$ </td><td rowspan="3">Cross-Stitch [20]</td><td>Equal Weights</td><td>50.08</td><td>90.33</td><td>0.0154</td><td>34.49</td></tr><tr><td>Uncert. Weights [14]</td><td>50.31</td><td>90.43</td><td>0.0152</td><td>31.36</td></tr><tr><td>DWA,  $T = 2$ </td><td>50.33</td><td>90.55</td><td>0.0153</td><td>33.37</td></tr><tr><td rowspan="3">1.65</td><td rowspan="3">MTAN (Ours)</td><td>Equal Weights</td><td>53.04</td><td>91.11</td><td>0.0144</td><td>33.63</td></tr><tr><td>Uncert. Weights [14]</td><td>53.86</td><td>91.10</td><td>0.0144</td><td>35.72</td></tr><tr><td>DWA,  $T = 2$ </td><td>53.29</td><td>91.09</td><td>0.0144</td><td>34.14</td></tr></table>

Table 2: 7-class semantic segmentation and depth estimation results on CityScapes validation dataset. #P shows the number of network parameters, and the best performing combination of multi-task architecture and weighting is highlighted in bold. The top validation scores for each task are annotated with boxes.

In particular, our method has two key advantages. First, due to the efficiency of having a single shared feature pool with attention masks automatically learning which features to share, our method outperforms other methods without requiring extra parameters (column #P), and even with significantly fewer parameters in some cases.

Second, our method maintains high performance across different loss function weighting schemes, and is more robust to the choice of weighting scheme than other methods, avoiding the need for cumbersome tweaking of loss weights. We illustrate the robustness of our method to the weighting schemes with a comparison to the Cross-Stitch Network [20], by plotting learning curves in Figure 3 with respect to the performance of three learning tasks in NYUv2 dataset. We can clearly see that our network follows similar learning trends across various weighting schemes, compared to the Cross-Stitch Network which produces notably different behaviour across the different schemes.

![](images/1bac0249ffa7cf6dc98a6a97b3ebc180b7678df887494339bef3858e80cfcf3f.jpg)

<details>
<summary>line</summary>

| Network | Epoch | Equal Weights (Semantic Acc.) | DWA (Semantic Acc.) | Weight Uncertainty (Semantic Acc.) |
| --- | --- | --- | --- | --- |
| Cross-Stitch Network | 0 | ~0.85 | ~0.85 | ~0.85 |
| Cross-Stitch Network | 200 | ~0.65 | ~0.60 | ~0.62 |
| Multi-Task Attention Network | 0 | ~0.85 | ~0.85 | ~0.85 |
| Multi-Task Attention Network | 200 | ~0.60 | ~0.60 | ~0.60 |
</details>

Figure 3: Validation performance curves on the NYUv2 dataset, across all three tasks (semantics, depth, normals, from left to right), showing robustness to loss function weighting schemes on the Cross-Stitch Network [20] (top) and our Multi-task Attention Network (bottom).

Figure 4 then shows qualitative results on the CityScapes validation dataset. We can see the advantage of our multitask learning approach over vanilla single-task learning, where the edges of objects are clearly more pronounced.

## 4.1.5 Effect of Task Complexity

For further introspection into the benefits of multi-task learning, we evaluated our implementations on CityScapes across different numbers of semantic classes, with the depth labels the same across all experiments. We trained the networks with the same settings as in Section 4.1.4, with an additional multi-task baseline Split (the standard version), which we found to perform better than the other modified versions. All networks are trained with equal weighting.

Table 4 (left) shows the validation performance improvement across all multi-task implementations and the singletask STAN implementation, plotted relative to the performance of the vanilla single-task learning on the CityScapes dataset. Interestingly, for only a 2-class setup, the singletask attention network (STAN) performs better than all

<table><tr><td rowspan="3">Type</td><td rowspan="3">#P.</td><td rowspan="3">Architecture</td><td rowspan="3">Weighting</td><td colspan="2">Segmentation</td><td colspan="2">Depth</td><td colspan="5">Surface Normal</td></tr><tr><td colspan="2">(Higher Better)</td><td colspan="2">(Lower Better)</td><td colspan="2">Angle Distance (Lower Better)</td><td colspan="3">Within  $t^{\circ}$ (Higher Better)</td></tr><tr><td>mIoU</td><td>Pix Acc</td><td>Abs Err</td><td>Rel Err</td><td>Mean</td><td>Median</td><td>11.25</td><td>22.5</td><td>30</td></tr><tr><td rowspan="2">Single Task</td><td>3</td><td>One Task</td><td>n.a.</td><td>15.10</td><td>51.54</td><td>0.7508</td><td>0.3266</td><td>31.76</td><td>25.51</td><td>22.12</td><td>45.33</td><td>57.13</td></tr><tr><td>4.56</td><td>STAN</td><td>n.a.</td><td>15.73</td><td>52.89</td><td>0.6935</td><td>0.2891</td><td>32.09</td><td>26.32</td><td>21.49</td><td>44.38</td><td>56.51</td></tr><tr><td rowspan="15">Multi Task</td><td rowspan="3">1.75</td><td rowspan="3">Split, Wide</td><td>Equal Weights</td><td>15.89</td><td>51.19</td><td>0.6494</td><td>0.2804</td><td>33.69</td><td>28.91</td><td>18.54</td><td>39.91</td><td>52.02</td></tr><tr><td>Uncert. Weights [14]</td><td>15.86</td><td>51.12</td><td>0.6040</td><td>0.2570</td><td>32.33</td><td>26.62</td><td>21.68</td><td>43.59</td><td>55.36</td></tr><tr><td>DWA, T = 2</td><td>16.92</td><td>53.72</td><td>0.6125</td><td>0.2546</td><td>32.34</td><td>27.10</td><td>20.69</td><td>42.73</td><td>54.74</td></tr><tr><td rowspan="3">2</td><td rowspan="3">Split, Deep</td><td>Equal Weights</td><td>13.03</td><td>41.47</td><td>0.7836</td><td>0.3326</td><td>38.28</td><td>36.55</td><td>9.50</td><td>27.11</td><td>39.63</td></tr><tr><td>Uncert. Weights [14]</td><td>14.53</td><td>43.69</td><td>0.7705</td><td>0.3340</td><td>35.14</td><td>32.13</td><td>14.69</td><td>34.52</td><td>46.94</td></tr><tr><td>DWA, T = 2</td><td>13.63</td><td>44.41</td><td>0.7581</td><td>0.3227</td><td>36.41</td><td>34.12</td><td>12.82</td><td>31.12</td><td>43.48</td></tr><tr><td rowspan="3">4.95</td><td rowspan="3">Dense</td><td>Equal Weights</td><td>16.06</td><td>52.73</td><td>0.6488</td><td>0.2871</td><td>33.58</td><td>28.01</td><td>20.07</td><td>41.50</td><td>53.35</td></tr><tr><td>Uncert. Weights [14]</td><td>16.48</td><td>54.40</td><td>0.6282</td><td>0.2761</td><td>31.68</td><td>25.68</td><td>21.73</td><td>44.58</td><td>56.65</td></tr><tr><td>DWA, T = 2</td><td>16.15</td><td>54.35</td><td>0.6059</td><td>0.2593</td><td>32.44</td><td>27.40</td><td>20.53</td><td>42.76</td><td>54.27</td></tr><tr><td rowspan="3"> $\approx 3$ </td><td rowspan="3">Cross-Stitch [20]</td><td>Equal Weights</td><td>14.71</td><td>50.23</td><td>0.6481</td><td>0.2871</td><td>33.56</td><td>28.58</td><td>20.08</td><td>40.54</td><td>51.97</td></tr><tr><td>Uncert. Weights [14]</td><td>15.69</td><td>52.60</td><td>0.6277</td><td>0.2702</td><td>32.69</td><td>27.26</td><td>21.63</td><td>42.84</td><td>54.45</td></tr><tr><td>DWA, T = 2</td><td>16.11</td><td>53.19</td><td>0.5922</td><td>0.2611</td><td>32.34</td><td>26.91</td><td>21.81</td><td>43.14</td><td>54.92</td></tr><tr><td rowspan="3">1.77</td><td rowspan="3">MTAN (Ours)</td><td>Equal Weights</td><td>17.72</td><td>55.32</td><td>0.5906</td><td>0.2577</td><td>31.44</td><td>25.37</td><td>23.17</td><td>45.65</td><td>57.48</td></tr><tr><td>Uncert. Weights [14]</td><td>17.67</td><td>55.61</td><td>0.5927</td><td>0.2592</td><td>31.25</td><td>25.57</td><td>22.99</td><td>45.83</td><td>57.67</td></tr><tr><td>DWA, T = 2</td><td>17.15</td><td>54.97</td><td>0.5956</td><td>0.2569</td><td>31.60</td><td>25.46</td><td>22.48</td><td>44.86</td><td>57.24</td></tr></table>

Table 3: 13-class semantic segmentation, depth estimation, and surface normal prediction results on the NYUv2 validation dataset. #P shows the number of network parameters, and the best performing combination of multi-task architecture and weighting is highlighted in bold. The top validation scores for each task are annotated with boxes.

![](images/4e1b37ce426635d7a1b5df82f8aa493a207869b8362a49886d425a7f67d46b76.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph LR
  subgraph Input Image
  A["Input Image"] --> B["Growth Truth (Semantic)"]
  B --> C["Vanilla Single-Task Learning"]
  C --> D["Multi-Task Attention Network"]
  end

  subgraph Growth Truth
    E["Growth Truth (Depth)"]
    F["Vanilla Single-Task Learning"]
    G["Multi-Task Attention Network"]
  end

  subgraph Visualization
    H["Heatmap Data Distribution"]
    I["Color Gradient Representation"]
    J["Visual Intensity Map Processing"]
  end

  A --> H
  H --> I
  I --> J
  J --> K["Final Output Result"]
```
</details>

Figure 4: CityScapes validation results on 7-class semantic labelling and depth estimation, trained with equal weighting. The original images are cropped to avoid invalid points for better visualisation. The red boxes are regions of interest, showing the effectiveness of the results provided from our method and single task method.

multi-task methods since it is able to fully utilise network parameters in a simple manner for the simple task. However, for greater task complexity, the multi-task methods encourage the sharing of features for a more efficient use of available network parameters, which then leads to better results. We also observe that, whilst the relative performance gain increases for all implementations as the task complexity increases, our MTAN method increases at a greater rate.

![](images/dee3871da89aac690b98372c0669a34116a91ba9eedcf776abc3d9d9043bcd54.jpg)

<details>
<summary>line</summary>

| Category | Single-Task, STAN | Multi-Task, Split | Multi-Task, Dense | Multi-Task, Cross-Stitch | Multi-Task, MTAN (ours) |
| --- | --- | --- | --- | --- | --- |
| 2-class | ~2 | ~0 | ~1.5 | ~0 | ~2 |
| 7-class | ~3.5 | ~2.5 | ~5.5 | ~1 | ~6.8 |
| 19-class | ~7.8 | ~7.2 | ~9 | ~5.8 | ~14 |
</details>

<table><tr><td>Method</td><td>#P.</td><td>ImNet.</td><td>Airc.</td><td>C100</td><td>DPed</td><td>DTD</td><td>GTSR</td><td>Flwr</td><td>Oglt</td><td>SVHN</td><td>UCF</td><td>Mean</td><td>Score</td></tr><tr><td>Scratch [23]</td><td>10</td><td>59.87</td><td>57.10</td><td>75.73</td><td>91.20</td><td>37.77</td><td>96.55</td><td>56.3</td><td>88.74</td><td>96.63</td><td>43.27</td><td>70.32</td><td>1625</td></tr><tr><td>Finetune [23]</td><td>10</td><td>59.87</td><td>60.34</td><td>82.12</td><td>92.82</td><td>55.53</td><td>97.53</td><td>81.41</td><td>87.69</td><td>96.55</td><td>51.20</td><td>76.51</td><td>2500</td></tr><tr><td>Feature [23]</td><td>1</td><td>59.67</td><td>23.31</td><td>63.11</td><td>80.33</td><td>45.37</td><td>68.16</td><td>73.69</td><td>58.79</td><td>43.54</td><td>26.8</td><td>54.28</td><td>544</td></tr><tr><td>Res. Adapt.[23]</td><td>2</td><td>59.67</td><td>56.68</td><td>81.20</td><td>93.88</td><td>50.85</td><td>97.05</td><td>66.24</td><td>89.62</td><td>96.13</td><td>47.45</td><td>73.88</td><td>2118</td></tr><tr><td>DAN [25]</td><td>2.17</td><td>57.74</td><td>64.12</td><td>80.07</td><td>91.30</td><td>56.54</td><td>98.46</td><td>86.05</td><td>89.67</td><td>96.77</td><td>49.38</td><td>77.01</td><td>2851</td></tr><tr><td>Piggyback [19]</td><td>1.28</td><td>57.69</td><td>65.29</td><td>79.87</td><td>96.99</td><td>57.45</td><td>97.27</td><td>79.09</td><td>87.63</td><td>97.24</td><td>47.48</td><td>76.60</td><td>2838</td></tr><tr><td>Parallel SVD [24]</td><td>1.5</td><td>60.32</td><td>66.04</td><td>81.86</td><td>94.23</td><td>57.82</td><td>99.24</td><td>85.74</td><td>89.25</td><td>96.62</td><td>52.50</td><td>78.36</td><td>3398</td></tr><tr><td>MTAN (Ours)</td><td>1.74</td><td>63.90</td><td>61.81</td><td>81.59</td><td>91.63</td><td>56.44</td><td>98.80</td><td>81.04</td><td>89.83</td><td>96.88</td><td>50.63</td><td>77.25</td><td>2941</td></tr></table>

Table 4: Left: CityScapes performance gain in percentage for all implementations compared with the vanilla single-task method. Right: Top-1 classification accuracy on the Visual Decathlon Challenge online test set. #P is the number of parameters as a factor of a single-task implementation. The upper part of table presents results from single task learning baselines; lower part of table presents results from multi-task learning baselines.

## 4.1.6 Attention Masks as Feature Selectors

To understand the role of the proposed attention modules, in Figure 5 we visualise the first layer attention masks learned with our network based on CityScapes dataset. We can see a clear difference in attention masks between the two tasks, with each mask working as a feature selector to mask out uninformative parts of the shared features, and focus on parts which are useful for each task. Notably, the depth masks have a much higher contrast than the semantic masks, suggesting that whilst all shared features are generally useful for the semantic task, the depth task benefits more from extraction of task-specific features.

![](images/579447bf9b3fbaccbd4e03a14e7479ede2646810e83dbf6d480948833b9b4e25.jpg)  
Figure 5: Visualisation of the first layer of 7-class semantic and depth attention features of our proposed network. The colours for each image are rescaled to fit the data.

## 4.2. Visual Decathlon Challenge (Many-to-Many)

Finally, we evaluate our approach on the recently introduced Visual Decathlon Challenge, consisting of 10 individual image classification tasks (many-to-many predictions). Evaluation on this challenge reports per-task accuracies, and assigns a cumulative score with a maximum value of 10,000 (1,000 per task) based on these accuracies. The complete details about the challenge settings, evaluation, and datasets used, can be found at http://www. robots.ox.ac.uk/ vgg/decathlon/.

Table 4 (right) shows results for the online test set of the challenge. As consistent with the prior works, we apply MTAN built on Wide Residual Network [31] with a depth of 28, widening factor of 4, and a stride of 2 in the first convolutional layer of each block. We train our model using a batch size of 100, learning rate of 0.1 with SGD, and weight decay of $5 \cdot 1 0 ^ { - 5 } $ for all 10 classification tasks. We halve the learning rate every 50 epochs for a total of 300 epochs. Then, we fine-tune 9 classification tasks (all except ImageNet) with a learning rate 0.01 until convergence. The results show that our approach surpasses most of the baselines and is competitive with the current state-of-theart, without the need for complicated regularisation strategies such as applying DropOut [28], regrouping datasets by size, or adaptive weight decay for each dataset, as required.

## 5. Conclusions

In this work, we have presented a new method for multitask learning, the Multi-Task Attention Network (MTAN). The network architecture consists of a global feature pool, together with task-specific attention modules for each task, which allows for automatic learning of both task-shared and task-specific features in an end-to-end manner. Experiments on the NYUv2 and CityScapes datasets with multiple dense-prediction tasks, and on the Visual Decathlon Challenge with multiple image classification tasks, show that our method outperforms or is competitive with other methods, whilst also showing robustness to the particular task weighting schemes used in the loss function. Due to our method’s ability to share weights through attention masks, our method achieves this state-of-the-art performance whilst also being highly parameter efficient.

## References

[1] Vijay Badrinarayanan, Alex Kendall, and Roberto Cipolla. Segnet: A deep convolutional encoder-decoder architecture for image segmentation. IEEE transactions on pattern analysis and machine intelligence, 39(12):2481–2495, 2017.  
[2] Rich Caruana. Multitask learning. In Learning to learn, pages 95–133. Springer, 1998.  
[3] Zhao Chen, Vijay Badrinarayanan, Chen-Yu Lee, and Andrew Rabinovich. Gradnorm: Gradient normalization for adaptive loss balancing in deep multitask networks. In International Conference on Machine Learning, pages 793–802, 2018.  
[4] Marius Cordts, Mohamed Omran, Sebastian Ramos, Timo Rehfeld, Markus Enzweiler, Rodrigo Benenson, Uwe Franke, Stefan Roth, and Bernt Schiele. The cityscapes dataset for semantic urban scene understanding. In Proceedings ofthe IEEE Conference on Computer Vision and Pattern Recognition, pages 3213–3223, 2016.  
[5] Camille Couprie, Clement Farabet, Laurent Najman, and´ Yann Lecun. Indoor semantic segmentation using depth information. In International Conference on Learning Representations (ICLR2013), April 2013, 2013.  
[6] Carl Doersch and Andrew Zisserman. Multi-task selfsupervised visual learning. In The IEEE International Conference on Computer Vision (ICCV), Oct 2017.  
[7] David Eigen and Rob Fergus. Predicting depth, surface normals and semantic labels with a common multi-scale convolutional architecture. In Proceedings of the IEEE International Conference on Computer Vision, pages 2650–2658, 2015.  
[8] Theodoros Evgeniou and Massimiliano Pontil. Regularized multi–task learning. In Proceedings of the tenth ACM SIGKDD international conference on Knowledge discovery and data mining, pages 109–117. ACM, 2004.  
[9] Georgia Gkioxari, Bharath Hariharan, Ross Girshick, and Jitendra Malik. R-cnns for pose estimation and action detection. arXiv preprint arXiv:1406.5212, 2014.  
[10] Michelle Guo, Albert Haque, De-An Huang, Serena Yeung, and Li Fei-Fei. Dynamic task prioritization for multitask learning. In European Conference on Computer Vision, pages 282–299. Springer, 2018.  
[11] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In Proceedings ofthe IEEE conference on computer vision and pattern recognition, pages 770–778, 2016.  
[12] Geoffrey Hinton, Oriol Vinyals, and Jeff Dean. Distilling the knowledge in a neural network. arXiv preprint arXiv:1503.02531, 2015.  
[13] Justin Johnson, Alexandre Alahi, and Li Fei-Fei. Perceptual losses for real-time style transfer and super-resolution. In European Conference on Computer Vision, pages 694–711. Springer, 2016.  
[14] Alex Kendall, Yarin Gal, and Roberto Cipolla. Multi-task learning using uncertainty to weigh losses for scene geometry and semantics. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 7482– 7491, 2018.  
[15] Diederik P Kingma and Jimmy Ba. Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980, 2014.  
[16] Iasonas Kokkinos. Ubernet: Training a universal convolutional neural network for low-, mid-, and high-level vision using diverse datasets and limited memory. In The IEEE Conference on Computer Vision and Pattern Recognition (CVPR), July 2017.  
[17] Abhishek Kumar and Hal Daume III. Learning task grouping´ and overlap in multi-task learning. In Proceedings of the 29th International Coference on International Conference on Machine Learning, pages 1723–1730. Omnipress, 2012.  
[18] Mingsheng Long, Jianmin Wang, Guiguang Ding, Jiaguang Sun, and Philip S Yu. Transfer feature learning with joint distribution adaptation. In Proceedings of the IEEE international conference on computer vision, pages 2200–2207, 2013.  
[19] Arun Mallya, Dillon Davis, and Svetlana Lazebnik. Piggyback: Adapting a single network to multiple tasks by learning to mask weights. In Proceedings of the European Conference on Computer Vision (ECCV), pages 67–82, 2018.  
[20] Ishan Misra, Abhinav Shrivastava, Abhinav Gupta, and Martial Hebert. Cross-stitch networks for multi-task learning. In Proceedings ofthe IEEE Conference on Computer Vision and Pattern Recognition, pages 3994–4003, 2016.  
[21] Pushmeet Kohli Nathan Silberman, Derek Hoiem and Rob Fergus. Indoor segmentation and support inference from rgbd images. In ECCV, 2012.  
[22] Sinno Jialin Pan and Qiang Yang. A survey on transfer learning. IEEE Transactions on knowledge and data engineering, 22(10):1345–1359, 2010.  
[23] Sylvestre-Alvise Rebuffi, Hakan Bilen, and Andrea Vedaldi. Learning multiple visual domains with residual adapters. In Advances in Neural Information Processing Systems, pages 506–516, 2017.  
[24] Sylvestre-Alvise Rebuffi, Hakan Bilen, and Andrea Vedaldi. Efficient parametrization of multi-domain deep neural networks. In Proceedings ofthe IEEE Conference on Computer Vision and Pattern Recognition, pages 8119–8127, 2018.  
[25] Amir Rosenfeld and John K Tsotsos. Incremental learning through deep adaptation. IEEE transactions on pattern analysis and machine intelligence, 2018.  
[26] Andrei A Rusu, Neil C Rabinowitz, Guillaume Desjardins, Hubert Soyer, James Kirkpatrick, Koray Kavukcuoglu, Razvan Pascanu, and Raia Hadsell. Progressive neural networks. arXiv preprint arXiv:1606.04671, 2016.  
[27] Karen Simonyan and Andrew Zisserman. Very deep convolutional networks for large-scale image recognition. arXiv preprint arXiv:1409.1556, 2014.  
[28] Nitish Srivastava, Geoffrey Hinton, Alex Krizhevsky, Ilya Sutskever, and Ruslan Salakhutdinov. Dropout: a simple way to prevent neural networks from overfitting. The Journal of Machine Learning Research, 15(1):1929–1958, 2014.  
[29] Sebastian Thrun and Lorien Pratt. Learning to learn. Springer Science & Business Media, 2012.  
[30] Fei Wang, Mengqing Jiang, Chen Qian, Shuo Yang, Cheng Li, Honggang Zhang, Xiaogang Wang, and Xiaoou Tang.  
Residual attention network for image classification. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 3156–3164, 2017.  
[31] Sergey Zagoruyko and Nikos Komodakis. Wide residual networks. In Edwin R. Hancock Richard C. Wilson and William A. P. Smith, editors, Proceedings of the British Machine Vision Conference (BMVC), pages 87.1–87.12. BMVA Press, September 2016.