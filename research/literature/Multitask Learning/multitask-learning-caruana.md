# Multitask Learning

Rich Caruana  
*Machine Learning*, Vol. 28, No. 1, pp. 41–75, 1997  
DOI: 10.1023/A:1007379606734

---

## Abstract

Multitask Learning (MTL) is an approach to inductive transfer that improves generalization by using the domain information contained in the training signals of related tasks as an inductive bias. It does this by learning tasks in parallel while using a shared representation; what is learned for each task can help other tasks be learned better. In this paper we describe multitask learning for artificial neural nets, vector-valued kernel regression, and k-nearest neighbor. We explain how multitask learning works and show that it improves generalization performance across multiple real-world and synthetic problem domains.

---

## Core Theoretical Concepts

1. **Inductive Transfer and Inductive Bias**
   - Traditional machine learning learns single tasks in isolation ($f: X \to Y_k$).
   - Multitask Learning learns multiple related tasks concurrently ($F: X \to (Y_1, Y_2, \dots, Y_K)$) using a shared intermediate representation.
   - The extra tasks provide an inductive bias that guides search in the hypothesis space toward representations that explain multiple phenomena simultaneously.

2. **Hard Parameter Sharing Architecture**
   - Intermediate hidden layers are shared across all tasks ($\boldsymbol{\theta}_{\text{sh}}$).
   - Task-specific output layers ($\boldsymbol{\theta}_k$) branch from the top shared hidden layer.
   - Forward pass feature extraction is amortized into a single shared execution ($\mathcal{O}(1)$ relative to $K$), rather than executing $K$ separate forward passes ($\mathcal{O}(K)$).

3. **Mechanisms of Positive Transfer**
   - **Statistical Data Amplification:** Tasks amplify effective sample size through shared weights.
   - **Feature Selection:** Shared representation learns to focus on feature dimensions relevant across multiple tasks.
   - **Representation Bias:** Biases network toward representations that other tasks also prefer, avoiding local minima.
   - **Eavesdropping:** Tasks that are difficult to learn can directly access features discovered by easier auxiliary tasks.
