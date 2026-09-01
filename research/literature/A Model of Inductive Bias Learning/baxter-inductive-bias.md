# A Model of Inductive Bias Learning

Jonathan Baxter  
*Journal of Artificial Intelligence Research* (JAIR), Vol. 12, pp. 149–198, 2000  
DOI: 10.1613/jair.731  

---

## Abstract

A key challenge in machine learning is how to automatically learn or refine an inductive bias (the hypothesis space and learning algorithm) to improve generalization on future tasks. In this paper, we introduce a formal model of bias learning where a learner is embedded within an environment of related learning tasks. By learning multiple tasks sampled from the environment, the learner can identify a shared representation (a restricted hypothesis space $\mathcal{H}$) that contains good solutions for all tasks in that environment. We prove rigorous theoretical generalization bounds showing that learning multiple tasks simultaneously reduces the sample complexity required per task, explaining the theoretical foundations of inductive transfer and multi-task representation learning.

---

## Core Theoretical Foundations

1. **The Formal Bias Learning Model**
   - The environment is modeled as a probability distribution $P$ over learning tasks.
   - Instead of searching an unconstrained hypothesis space for each task in isolation, the learner searches a family of hypothesis spaces $\mathbb{H} = \{\mathcal{H}\}$.
   - Multi-task learning acts as bias learning by finding a shared feature representation that minimizes empirical error across all sampled tasks.

2. **Sample Complexity Bounds and Capacity Reduction**
   - In single-task learning, the number of samples required to guarantee generalization error $\epsilon$ scales with the capacity (VC dimension or covering number) of the unconstrained hypothesis space: $m = \mathcal{O}\left(\frac{\text{capacity}(\mathcal{H})}{\epsilon^2}\right)$.
   - In multi-task learning across $K$ tasks, sharing an internal representation restricts the effective hypothesis space to the intersection $\bigcap_{k=1}^K \mathcal{H}_k$.
   - The sample complexity per task decreases because the shared representation capacity is amortized across all $K$ tasks: $m_{\text{per-task}} = \mathcal{O}\left(\frac{\text{capacity}(\mathbb{H})}{K \cdot \epsilon^2} + \frac{\text{capacity}(\mathcal{H})}{\epsilon^2}\right)$.

3. **Theoretical Justification for Multi-Task Low-Resource Learning**
   - For low-resource tasks where labeled data is scarce, single-task learning suffers from high generalization error due to capacity mismatch.
   - Multi-task representation learning constrains the search space to representations validated by auxiliary tasks, directly enabling generalization on sparse or noisy datasets.
