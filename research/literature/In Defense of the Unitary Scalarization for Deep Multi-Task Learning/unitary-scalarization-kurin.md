# In Defense of the Unitary Scalarization for Deep Multi-Task Learning

Vitaly Kurin, Alessandro De Palma, Ilya Kostrikov, Shimon Whiteson, and M. Pawan Kumar  
*Advances in Neural Information Processing Systems* (NeurIPS 2022), Vol. 35, pp. 28657–28670, 2022  

---

## Abstract

Multi-Task Learning (MTL) is frequently formulated as a multi-objective optimization problem to handle trade-offs between tasks. A popular approach in recent years has been to replace simple linear scalarization with specialized Multi-Task Optimization (SMTO) algorithms designed to balance gradients or navigate the Pareto front. In this work, we conduct a comprehensive empirical evaluation of SMTO methods across standard benchmarks. We find that simple unitary scalarization (equal weighting with static weights $w_i = 1$) combined with standard single-task training practices (such as the Adam optimizer, learning rate scheduling, and weight decay) achieves competitive performance with—and often outperforms—complex SMTO algorithms, while avoiding significant runtime and memory overhead.

---

## Key Findings & Core Arguments

1. **The Unitary Scalarization Baseline**
   - Unitary scalarization defines the total loss simply as the unweighted sum of task losses: $\mathcal{L}_{\text{total}} = \sum_{k=1}^K \mathcal{L}_k$.
   - When combined with adaptive optimizers like Adam and standard hyperparameter tuning, unitary scalarization provides a remarkably strong baseline.

2. **Computational Overhead of SMTO Methods**
   - Specialized MTO algorithms (such as MGDA, GradNorm, PCGrad, and CAGrad) introduce substantial computational slowdowns, often training 2 to 35 times slower than unitary scalarization due to per-task gradient backpropagations and quadratic programming solvers.
   - They also increase peak memory allocation during training.

3. **Role of SMTO Methods as Implicit Regularizers**
   - The empirical analysis suggests that reported gains from specialized MTO methods often stem from under-tuned baseline models or the implicit regularizing effect of gradient manipulation rather than true Pareto optimization.
