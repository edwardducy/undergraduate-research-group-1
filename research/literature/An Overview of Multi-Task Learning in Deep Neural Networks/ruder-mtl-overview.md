# An Overview of Multi-Task Learning in Deep Neural Networks

Sebastian Ruder  
*arXiv preprint*, arXiv:1706.05098, 2017  

---

## Abstract

Multi-Task Learning (MTL) has led to successes across many applications of machine learning, from natural language processing and speech recognition to computer vision and drug discovery. This paper aims to provide a comprehensive introduction to MTL in deep neural networks, focusing on the two most common parameter sharing mechanisms: hard parameter sharing and soft parameter sharing. The survey explores the underlying statistical mechanisms that enable positive inductive transfer, reviews various multi-task architectures, and discusses practical considerations for selecting auxiliary tasks and balancing loss functions.

---

## Core Concepts

1. **Hard Parameter Sharing vs. Soft Parameter Sharing**
   - **Hard Parameter Sharing:** Intermediate hidden layers are shared across all tasks while task-specific output layers branch from the top shared representation. This reduces parameter storage and computational complexity, amortizing feature extraction into a single forward pass ($\mathcal{O}(1)$). It also acts as a strong regularizer, reducing the risk of overfitting on individual tasks.
   - **Soft Parameter Sharing:** Each task has its own model parameters, and the distance between parameters is encouraged to be small through regularization penalties (such as $L_2$ distance or trace norm).

2. **Five Mechanisms of Positive Inductive Transfer**
   - **Implicit Data Augmentation:** Multi-task learning increases the effective sample size because different tasks have different noise distributions, encouraging the shared representation to capture invariant domain features.
   - **Attention Focusing:** Tasks that provide clear signals about important features help other noisy tasks focus attention on relevant dimensions.
   - **Eavesdropping:** Difficult tasks can access intermediate representations learned directly by easier auxiliary tasks.
   - **Representation Bias:** Biases the network toward representations that are preferred by multiple tasks, improving out-of-distribution generalization.
   - **Regularization:** Acts as an inductive regularizer that shrinks the effective hypothesis space and reduces Rademacher complexity.
