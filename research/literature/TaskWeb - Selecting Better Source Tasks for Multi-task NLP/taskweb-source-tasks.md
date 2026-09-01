# TASKWEB: Selecting Better Source Tasks for Multi-task NLP

Joongwon Kim† , Akari Asai† , Gabriel Ilharco† , Hannaneh Hajishirzi†♡ †University of Washington ♡Allen Institute for AI **{jwonkim,akari,gamaga,hannaneh}@cs.washington.edu**

## Abstract

Recent work in NLP has shown promising results in training models on large amounts of tasks to achieve better generalization. However, it is not well-understood how tasks are related, and how helpful training tasks can be chosen for a new task. In this work, we investigate whether knowing task relationships via pairwise task transfer improves choosing one or more source tasks that help to learn a new target task. We provide TASKWEB, a largescale benchmark of pairwise task transfers for 22 NLP tasks using three different model types, sizes, and adaptation methods, spanning about 25,000 experiments. Then, we design a new method TASKSHOP based on our analysis of TASKWEB. TASKSHOP uses TASKWEB to estimate the benefit of using a source task for learning a new target task, and to choose a subset of helpful training tasks for multi-task training. Our method improves overall rankings and top-k precision of source tasks by 10% and 38%, respectively. We also use TASKSHOP to build much smaller multi-task training sets that improve zero-shot performances across 11 different target tasks by at least 4.3%. [<sup>1</sup>](#page-0-0)

## 1 Introduction

Recent studies have revealed that large language models are able to generalize to unseen tasks when jointly trained on many different tasks, with their performance scaling to the size and diversity of the training data [\(Sanh et al.,](#page-11-0) [2022;](#page-11-0) [Wang et al.,](#page-12-0) [2022b;](#page-12-0) [Wei et al.,](#page-12-1) [2022a;](#page-12-1) [Chung et al.,](#page-9-0) [2022;](#page-9-0) [Long](#page-10-0)[pre et al.,](#page-10-0) [2023\)](#page-10-0). As more and more tasks are added to build general-purpose models, it has been noted that knowing inter-task relationships may be helpful but that it remains unclear how to select helpful tasks for multi-task learning [\(Ye et al.,](#page-12-2) [2021;](#page-12-2) [Min](#page-10-1) [et al.,](#page-10-1) [2022;](#page-10-1) [Asai et al.,](#page-9-1) [2022;](#page-9-1) [Chan et al.,](#page-9-2) [2022\)](#page-9-2).

In this work, we investigate whether quantifying the relationship between different NLP tasks via

<span id="page-0-1"></span>![](_page_0_Diagram_3.jpeg)

Figure 1: We use pairwise transfer scores in TASKWEB to score (source, target) pairs where the source task is in TASKWEB and the target task is unseen (i.e., access to only a few examples). Then, we select helpful tasks and perform multi-task learning for the target task.

pairwise task transfer helps *task selection*, which we define as choosing one or more source tasks that better initialize a model for an unseen target task as shown in Figure [1.](#page-0-1) We begin from a pairwise setup as it is often used to quantify task relationships [\(Zamir et al.,](#page-13-0) [2019;](#page-13-0) [Vu et al.,](#page-12-3) [2020\)](#page-12-3) and is more tractable than larger combinations of tasks.

First, we construct TASKWEB, a large-scale benchmark for pairwise task transfers across different model architectures (encoder-only, decoderonly, encoder-decoder), parameter count (60M to 770M) and adaptation methods including finetuning, Adapter-tuning [\(Houlsby et al.,](#page-10-2) [2019\)](#page-10-2) and Bit-Fit [\(Zaken et al.,](#page-13-1) [2022\)](#page-13-1), resulting in 25,000 transfers. From our results, we discover a *transitive* property where having strong, positive transfers <sup>A</sup> → <sup>B</sup> and <sup>B</sup> → <sup>C</sup> for tasks A, <sup>B</sup> and <sup>C</sup> makes it more likely that A → C is also a positive transfer.

Then, we introduce a new method TASKSHOP that predicts the transferability from a source task to a target task associated with only a few examples. TASKSHOP builds upon the transitive behavior to

<span id="page-0-0"></span><sup>1</sup>Our code is available at [https://github.com/](https://github.com/danieljkim0118/TaskWeb) [danieljkim0118/TaskWeb](https://github.com/danieljkim0118/TaskWeb).

construct different paths with "pivot" tasks between the source and target tasks. It combines TASKWEB scores between the source and pivot and textual similarity scores between the pivot and target to estimate (source→target) transfers.

We evaluate our methods in both single-task and multi-task settings. First, we show that TASKSHOP assigns better transferability scores both in terms of the overall ranking and identifying top helpful tasks. Then, we demonstrate that models trained on small multi-task sets built with TASKSHOP outperform models trained on larger sets of tasks. We perform additional analyses and discover that there is a tradeoff for building multitask sets of varying sizes with TASKSHOP, and that the proportion of helpful tasks in the training set affects performance.

To summarize, our contributions are as follows:

- 1. We build and analyze TASKWEB, a benchmark of pairwise transfer experiments across various tasks, models and adaptation methods.
- 2. We define task selection for single-task and multi-task setups and propose TASKSHOP which uses pairwise transfer scores to predict transfer to an unseen target task.
- 3. We use TASKSHOP and TASKWEB to choose helpful source tasks and build small multitask training sets that result in better zero-shot performance for unseen targets.

## 2 Background and Overview

We use pairwise task transfer to quantify task similarities, select better source tasks for unseen tasks and improve performance via multi-task finetuning.

### 2.1 Overview

Figure [2](#page-2-0) depicts how we use task relationships to select better source tasks. We first quantify task relations with pairwise task transfer, which is a process of sequentially learning one task—the *source task* and then another task—the *target task*. We use this to build TASKWEB, a collection of 22 diverse, high-resource tasks in NLP and their pairwise task transfer scores across seven different training setups (Sections [3.1,](#page-2-1) [3.2\)](#page-3-0). From our analysis, we find that pairwise task transfer indicates *transitive* behavior between positive transfers (Section [3.3\)](#page-3-1).

We then explore task selection, where for a target task t with n examples and a set of source tasks <sup>S</sup>, we select a helpful task <sup>s</sup> ∈ <sup>S</sup> for <sup>t</sup>. Here, we assume that the target task is *unseen*, that is, with

access only to a small number of examples from t (<sup>n</sup> ≤ <sup>32</sup>). We propose a new task selection method TASKSHOP that builds upon the transitive behavior to select the best source task to transfer to an unseen target task, even without pairwise transfer scores for the target (Section [4.1\)](#page-4-0). We evaluate the overall task rankings and the precision of top-k helpful tasks returned by TASKSHOP (Section [5.1\)](#page-4-1).

Moreover, we extend task selection to a multitask setup. By selecting tasks k > 1 times, we obtain a set of k source tasks as a multi-task training set (Section [4.2\)](#page-4-2). We train models on these multi-task sets and perform evaluations and analyses on 11 different target tasks (Sections [5.2,](#page-5-0) [5.3\)](#page-6-0).

## 2.2 Related Work

Pairwise Task Transfer. Pairwise task transfer, also known as intermediate task transfer, is used to quantify relationships between different tasks in computer vision [\(Zamir et al.,](#page-13-0) [2019;](#page-13-0) [Achille et al.,](#page-8-0) [2019\)](#page-8-0) and NLP [\(Vu et al.,](#page-12-3) [2020;](#page-12-3) [Poth et al.,](#page-11-1) [2021\)](#page-11-1). It is also used in NLP to study factors impacting task transfer [\(Pruksachatkun et al.,](#page-11-2) [2020;](#page-11-2) [Albalak](#page-9-3) [et al.,](#page-9-3) [2022\)](#page-9-3) and identify helpful source tasks for parameter-efficient methods [\(Vu et al.,](#page-12-4) [2022;](#page-12-4) [Su](#page-12-5) [et al.,](#page-12-5) [2022;](#page-12-5) [Asai et al.,](#page-9-1) [2022\)](#page-9-1). Building upon previous work, we address more diverse tasks, models, and adaptation methods.

Task Selection. Task selection is used in many studies to better initialize models for learning new tasks. Some methods assume access to the entire training set and model [\(Vu et al.,](#page-12-3) [2020;](#page-12-3) [Poth et al.,](#page-11-1) [2021;](#page-11-1) [Vu et al.,](#page-12-4) [2022;](#page-12-4) [Su et al.,](#page-12-5) [2022\)](#page-12-5), while other methods only access a small portion of the training data [\(Jang et al.,](#page-10-3) [2023;](#page-10-3) [Paranjape et al.,](#page-11-3) [2023\)](#page-11-3). We build upon the second case in this work.

Multi-task Fine-tuning. Multi-task fine-tuning is used to train models that generalize across many tasks [\(Khashabi et al.,](#page-10-4) [2020;](#page-10-4) [Mishra et al.,](#page-11-4) [2022;](#page-11-4) [Sanh et al.,](#page-11-0) [2022\)](#page-11-0). While studies report that adding more tasks generally improve performance, [\(Agha](#page-8-1)[janyan et al.,](#page-8-1) [2021;](#page-8-1) [Wei et al.,](#page-12-1) [2022a;](#page-12-1) [Wang et al.,](#page-12-0) [2022b\)](#page-12-0), others report that using a subset of tasks provide better performance [\(Padmakumar et al.,](#page-11-5) [2022;](#page-11-5) [Chan et al.,](#page-9-2) [2022\)](#page-9-2) but that it is not clear how to identify such subset [\(Aribandi et al.,](#page-9-4) [2022\)](#page-9-4). Previous work retrieves the top-k relevant source examples based on the target examples [\(Lin et al.,](#page-10-5) [2022;](#page-10-5) [Ivison et al.,](#page-10-6) [2022\)](#page-10-6). In this work, we take a simpler approach and select helpful *tasks* based on target examples to build multi-task training sets.

<span id="page-2-0"></span>![](_page_2_Diagram_0.jpeg)

Figure 2: Overview of single and multi-task selection using TASKSHOP and TASKWEB. Section [3](#page-2-2) describes the pairwise task transfer involved in TASKWEB as well as its analysis. Section [4](#page-4-3) details TASKSHOP and describes task selection in single task and multi-task setups. Section [5](#page-4-4) presents our experiments as well as additional analyses.

#### <span id="page-2-2"></span>3 TASKWEB: A Benchmark for Pairwise Task Transfer

Previous studies in pairwise task transfer tend to focus on specific models, adaptation methods or task domains [\(Vu et al.,](#page-12-3) [2020;](#page-12-3) [Poth et al.,](#page-11-1) [2021;](#page-11-1) [Albalak](#page-9-3) [et al.,](#page-9-3) [2022\)](#page-9-3). We introduce TASKWEB, which consists of pairwise task transfer experiments that span a wide variety of tasks, models, and adaptation methods. TASKWEB can be used as a benchmark to evaluate task transferability, and as a repository for selecting helpful source tasks (Section [4\)](#page-4-3).

#### <span id="page-2-1"></span>3.1 Focus and Experimental Setup

Tasks. To build TASKWEB, we choose a set of 22 representative tasks in NLP that span diverse categories and require various forms of knowledge, as shown in Table [1.](#page-2-3) We perform a total of about 25,000 transfers between all pairs of tasks.[<sup>2</sup>](#page-2-4)

Training Procedure. We finetune a pre-trained language model on the full dataset associated with a source task s, and further finetune the model on a set of 1,000 random examples of the target task t. [<sup>3</sup>](#page-2-5) Then, we compare the performance gain from initializing the model on s to finetuning the model on the same subset of t without starting from s. We repeat this process over eight random seeds to reduce variability [\(Dodge et al.,](#page-10-7) [2020\)](#page-10-7).

Models. We study the impacts of three different model architectures on task transfer—T5 (encoder-

<span id="page-2-3"></span>

Table 1: All tasks used in our pairwise transfer experiments, grouped by high-level task categories. Citations for all datasets are provided in Table [8](#page-15-0) in the appendix.

decoder; [Raffel et al.](#page-11-6) [2020\)](#page-11-6), GPT-2 (decoderonly; [Radford et al.](#page-11-7) [2019\)](#page-11-7) and RoBERTa (encoderonly; [Liu et al.](#page-10-8) [2019\)](#page-10-8). We use the LM-adapted versions[<sup>4</sup>](#page-2-6) [\(Lester et al.,](#page-10-9) [2021\)](#page-10-9) of T5-small/base/large, as well as GPT-2 medium and RoBERTa-base.

Adaptation Settings. We investigate pairwise task transfer with three widely-adopted adaptation methods—full fine-tuning, Adapter-tuning [\(Houlsby et al.,](#page-10-2) [2019\)](#page-10-2) and BitFit [\(Zaken et al.,](#page-13-1) [2022\)](#page-13-1)—while fixing T5-base as the base model.

Metrics for Task Transferability. We follow [Vu](#page-12-3) [et al.](#page-12-3) [\(2020\)](#page-12-3) and use the average percentage change to measure task transfer. Also, we measure the proportion of models with positive transfer across all random seeds. We combine both metrics to account for both the magnitude and consistency of transfers across all random seeds. The formal definition is provided in Section [A.1](#page-14-0) in the appendix.

<span id="page-2-4"></span><sup>2</sup>We use SQuAD2.0 as only a source task due to difficulties associated with running SQuAD evaluation for all transfers.

<span id="page-2-5"></span><sup>3</sup>This number was chosen for the model to not overfit to t, but also learn enough from t to provide a measure of how it would perform on the task, in line with previous studies.

<span id="page-2-6"></span><sup>4</sup>The original T5 checkpoints have been trained on datasets that overlap with ours. We aim to separate the effects of multitask supervised pretraining in our pairwise transfer analysis.

<span id="page-3-2"></span>![](_page_3_Figure_0.jpeg)

Figure 3: (Left) visualization of TASKWEB, our collection of pairwise transfer between 22 different NLP tasks, averaged over seven training setups. Positive transfers are blue and negative transfers are red. All transfers point from the source to the target. (Center) transfer scores between a subset of source tasks (three more helpful/three less helpful) and a subset of target tasks. The full set of scores is given in Figure [5](#page-15-1) in the appendix. (Top-right) similarities between pairwise transfer results in our experiment of 22 tasks obtained for seven different training setups. (Bottom-right) probability of identifying positive source → target transfers as the minimum threshold for (source → pivot, pivot → target) transfers is increased. Results with all setups are in Figure [<sup>14</sup>](#page-20-0) in the appendix. t5s/b/l: T5-small/base/large, ft: finetuning, ad: adapter-tuning, bf: BitFit, gpt2: GPT-2 medium, rob: RoBERTa-base.

#### <span id="page-3-0"></span>3.2 Observations from TASKWEB

Results. Figure [3](#page-3-2) visualizes TASKWEB—the left shows all transfers, and the center gives examples of pairwise transfer scores. All scores are averaged over seven training configurations. Refer to Figures [5](#page-15-1) to [12](#page-19-0) in the appendix for the full results.

We note that positive transfers (blue) occur between intuitively similar tasks such as CosmosQA to SocialIQA (+0.15), both of which are multiplechoice commonsense questions. In contrast, negative transfers (red) occur for tasks that seem to require unrelated skills, such as from QQP to CosmosQA (-0.12). Surprisingly, positive transfers exist between tasks that do not seem similar, such as a positive transfer from SocialIQA to RTE (+0.10).

Effects of Training Setup. We investigate how the training setup affects pairwise task transfer. To this end, we build matrices of pairwise transfer scores for each training setup as shown in Figure [5](#page-15-1) and compute their normalized dot products.

Refer to the top-right subfigure of Figure [3.](#page-3-2) We observe more similar pairwise transfers when 1) the same adaptation method is applied to models of the same class but different sizes, or 2) different adaptation methods are applied to the same model. For example, T5-base finetune exhibits more similar transfer with T5-small/large finetune or T5-base adapter/BitFit than GPT-2 or RoBERTa finetune.

#### <span id="page-3-1"></span>3.3 Analysis of Mathematical Properties

Computing pairwise transfer scores can become costly as more tasks are added. Would it be possible to predict transferability beforehand using existing scores? We formulate pairwise task transfer as a mathematical relationship and investigate two properties—*commutativity* and *transitivity*.

We define *commutativity* in our setup as whether <sup>A</sup> → <sup>B</sup> being a positive/negative transfer implies that <sup>B</sup> → <sup>A</sup> is also a positive/negative transfer. If <sup>A</sup> → <sup>B</sup> is known, the commutativity would help us predict B → A before performing the transfer.

Meanwhile, we define *transitivity* in our setup as whether knowing the transfer scores of <sup>A</sup> → <sup>B</sup> and <sup>B</sup> → <sup>C</sup> allows us to infer about <sup>A</sup> → C. This property would also provide us more flexibility to predict pairwise transfer in advance.

Commutativity often does not hold. Based on the pairwise transfer scores shown in Figure [3](#page-3-2) (center), we compute the proportion of transfer pairs that exhibit commutativity. Of the 210 unique transfer pairs in our setup, we find that 97 exhibit commutativity. The results are visualized in Figure [13](#page-19-1) in the appendix. We uniquely observe from our experiments that pairwise transfer does not display strong signs of commutativity. One possible reason is that while knowledge acquired from task A may be helpful for task B, the reverse may not be true.

Transitivity holds for positive transfers. We perform a small experiment where we predict transfer <sup>A</sup> → <sup>B</sup> as positive if both <sup>A</sup> → <sup>B</sup> and <sup>B</sup> → <sup>C</sup> score above a threshold. Here, we call A the source task, C the target task, and B the "pivot" task.

Refer to the bottom-right subfigure of Figure [3.](#page-3-2) We observe that as stricter criteria is imposed for source → pivot and pivot → target, the likelihood of observing positive transfers steadily increase across all training setups. For example, the probability of observing positive transfers increases from 88% to 97% when the intermediate thresholds increase from 0.01 to 0.04. These results indicate a transitive behavior between positive transfers.

## <span id="page-4-3"></span>4 Task Selection for Unseen Target Tasks

Pairwise transfer scores are not always available for a new target task. We introduce TASKSHOP to estimate transfer from a source task in TASKWEB to an unseen target task with only a small number of examples (Figure [2\)](#page-2-0). Then, we perform task selection in two settings: a single-task setup where we identify a helpful source task, and a multi-task setup where we locate a set of helpful source tasks.

## <span id="page-4-0"></span>4.1 TASKSHOP: Selecting Helpful Tasks

The objective of task selection in a single-task setup is to predict the benefit of initializing a model on a source task for learning a target task. We introduce a new method TASKSHOP which uses pairwise transfer scores to estimate the transfer from source tasks in TASKWEB to an unseen target task.

Setup. Given a source task <sup>s</sup> ∈ <sup>S</sup> and an unseen target task t, we seek to predict the transferability of s to t. We assume access to pairwise transfer scores between <sup>s</sup> and other source tasks <sup>S</sup>\{s}. Meanwhile, we have a small number of examples (<sup>n</sup> ≤ <sup>32</sup>) but no pairwise transfer scores for <sup>t</sup>.

Overview. Our method searches over paths from s to t via a set of pivot tasks in TASKWEB where each pivot <sup>p</sup> forms a path <sup>s</sup> → <sup>p</sup> → <sup>t</sup>, and averages their scores to estimate <sup>s</sup> → <sup>t</sup>. It builds upon our previous observation that the strengths of <sup>s</sup> → <sup>p</sup> and <sup>p</sup> → <sup>t</sup> help us estimate the strength of <sup>s</sup> → <sup>t</sup>.

Method. The TASKSHOP method is summarized in Equation [4.1.](#page-4-5) Given a pivot task <sup>p</sup> ∈ <sup>S</sup>\{s} for which transfer <sup>s</sup> → <sup>p</sup> is already known, we first use an off-the-shelf task selection method F to obtain <sup>F</sup>(<sup>p</sup> → <sup>t</sup>). <sup>F</sup> can be any method that only uses a small number of task examples. Then, we find the pairwise transfer score <sup>T</sup>(<sup>s</sup> → <sup>p</sup>) from <sup>T</sup>ASKWEB, and average the two scores. We repeat this process over all pivot tasks <sup>p</sup> ∈ <sup>S</sup>\{s} and average the resulting scores. Finally, we linearly interpolate our estimate with a direct estimate <sup>F</sup>(<sup>s</sup> → <sup>t</sup>) using a hyperparameter λ tuned on a held-out task.

TS(s, t) = <sup>λ</sup> · 1 ∥S\{s}∥ X <sup>p</sup>∈S\{s} <sup>T</sup>(<sup>s</sup> → <sup>p</sup>) + <sup>F</sup>(<sup>p</sup> → <sup>t</sup>) 2 +(1 − <sup>λ</sup>) · <sup>F</sup>(<sup>s</sup> → <sup>t</sup>) (1)

<span id="page-4-5"></span>TASKSHOP is directional. One interesting feature of TASKSHOP is its *directionality*—our predictions for <sup>A</sup> → <sup>B</sup> differs from <sup>B</sup> → <sup>A</sup>. Our method deviates from conventional techniques that use task embeddings and select tasks using cosine similarities, which results in symmetric predictions. Hence our method is more aligned with the noncommutative property observed in Section [3.3.](#page-3-1)

TASKSHOP is modular. Another feature of TASKSHOP is its *modularity* since any task selection method that only uses a small number of target examples can be used for F. Likewise, we utilize recent methods that only use a small number of target task examples, thereby excluding methods that require the fine-tuned model or the full training set. Specifically, we use Retrieval-of-Experts (RoE) from [Jang et al.](#page-10-3) [\(2023\)](#page-10-3) and the LLM similarity method from [Paranjape et al.](#page-11-3) [\(2023\)](#page-11-3) for F.

## <span id="page-4-2"></span>4.2 Extension to Multi-Task Selection

While choosing a single, appropriate source task is beneficial for learning a target task [\(Vu et al.,](#page-12-3) [2020,](#page-12-3) [2022\)](#page-12-4), it has also been observed that using multiple source tasks provides additional benefits [\(Asai et al.,](#page-9-1) [2022\)](#page-9-1). Hence we extend task selection from a single-task to a multi-task setup.

Given a target task t and a task selection method, we first select the top-k highest scoring source tasks <sup>S</sup><sup>k</sup> <sup>=</sup> {<sup>s</sup>1, ..., sk} for <sup>t</sup>. Here, the task selection method can be TASKSHOP or other methods. We then randomly sample n prompted examples from each task, resulting in a small training set of kn examples. Table [6](#page-14-1) in the appendix shows examples of top-5 tasks selected by TASKSHOP with F=RoE.

## <span id="page-4-4"></span>5 Experiments and Results

## <span id="page-4-1"></span>5.1 Single-Task Selection

Comparisons. We compare to Retrieval-of-Experts (RoE) from [Jang et al.](#page-10-3) [\(2023\)](#page-10-3) and LLMsimilarity in [Paranjape et al.](#page-11-3) [\(2023\)](#page-11-3). For Retrievalof-Experts, we take 100 examples of the source

<span id="page-5-1"></span>

Table 2: Results of task selection experiments. We use TASKWEB to evaluate TASKSHOP and two task selection methods that only use target examples : LLM similarity [\(Paranjape et al.,](#page-11-3) [2023\)](#page-11-3) and RoE [\(Jang et al.,](#page-10-3) [2023\)](#page-10-3). TASKSHOP LLM uses F = LLM-similarity and TASKSHOP <sup>R</sup>O<sup>E</sup> uses F = ROE in Equation [4.1.](#page-4-5) TASKSHOP <sup>R</sup>O<sup>E</sup> exhibits the best performance in task selection both in terms of the overall ranking (NDCG) and top-5 precision (Regret@5). Note that a higher score is better for NDCG (above) and a lower score is better for Regret@5 (below).

task and 32 examples of the target task and compute the similarity between text embeddings of the prompts. For LLM-similarity, we input a prompt to text-davinci-003 [\(Ouyang et al.,](#page-11-8) [2022\)](#page-11-8) to assign probability scores to whether the two tasks are similar or not. For TASKSHOP, we use RoE and LLM-similarity for F in Equation [4.1.](#page-4-5) More details are provided in Section [A.1](#page-14-0) in the appendix.

Metrics. To evaluate task selection, we use two metrics: normalized discounted cumulative gain (NDCG) and Regret@k, following [Poth et al.](#page-11-1) [\(2021\)](#page-11-1). We use NDCG to evaluate the overall ranking, and Regret@k to measure the performance drop of the predicted top-k source tasks from the actual top-k source tasks. We evaluate task selection for all tasks in our setup grouped by categories in Table [1,](#page-2-3) and use TASKWEB for the gold labels.

Experimental Setup. While we use target tasks from TASKWEB to use their transfer scores as labels, we wish to simulate a scenario in which there are only 32 examples for each target. Therefore we perform our experiments in a leave-one-out setup, where for each experiment we assume access to pairwise scores amongst our set of tasks except for the given target task. In this way, we maintain the assumption that only a small number of examples of the target task are available during evaluation.

Results. Table [2](#page-5-1) reports our results. Combining pairwise transfer scores with LLM and RoE improves both NDCG and Regret@5 compared to their base methods, with the best gains from RoE. We hypothesize that the improvement occurs because the pairwise transfer scores capture the transferability between each source task and the set of tasks textually similar to the target task. Due to

transitive behavior between positive task transfers, these transfer scores would provide additional information about the transferability from the helpful source tasks to the target. Moreover, our method considers the direction of the pairwise transfer unlike the other methods, thereby better accounting for the non-commutativity observed in Section [3.3.](#page-3-1)

#### <span id="page-5-0"></span>5.2 Multi-Task Selection

We now investigate whether TASKSHOP can also be used to select multiple source tasks that collectively improve target task performance.

Comparisons. We use the following baselines. T0-3B has the same architecture as T5-3B but trained on millions of examples spanning 35 different tasks [\(Sanh et al.,](#page-11-0) [2022\)](#page-11-0). T5-3B + most similar is the LM-adapted T5-3B [\(Lester et al.,](#page-10-9) [2021\)](#page-10-9) trained on a handpicked, similar source task from the same category as each target task. T5-3B + all tasks is the LM-adapted T5-3B trained with samples from all 22 tasks from TASKWEB except each target task in a leave-one-out setup.

We then train T5-3B models on small training sets sampled from the five highest-scoring source tasks based on the following task selection methods: Retrieval-of-Experts from [\(Jang et al.,](#page-10-3) [2023\)](#page-10-3), LLM-similarity from [\(Paranjape et al.,](#page-11-3) [2023\)](#page-11-3) and TASKSHOP <sup>R</sup>O<sup>E</sup> with F = ROE in Equation [4.1.](#page-4-5)

Finally, we consider the case where TASKWEB scores for the target task are available and select the five highest-scoring source tasks for each target. We train T5-3B on samples from these tasks.

Training Setup. Given a target task t and a task selection method, we first select the five highestscoring source tasks s1, ..., s<sup>5</sup> for t. We then randomly sample 2,000 prompted examples from each

<span id="page-6-1"></span>

Table 3: Results of multi-task learning experiments. We perform all evaluations in zero-shot settings, meaning that we do not fit the model parameters to the target task - however, we still assume access to a small number of labeled examples of the target. We average results over multiple prompts. The first group corresponds to our baselines, the second group corresponds to two existing task selection methods, as well as TASKSHOP without access to TASKWEB scores for the target task (but access to TASKWEB scores between other tasks), and the third group uses <sup>T</sup>ASKWEB scores for the target task to select source tasks. <sup>⋆</sup> is from [Jang et al.](#page-10-3) [\(2023\)](#page-10-3) and ⋄ is from [Paranjape et al.](#page-11-3) [\(2023\)](#page-11-3). † has access to <sup>T</sup>ASKWEB scores directly to the target task. All methods below the dotted line use the top-5 scoring source tasks to build multi-task training sets, while the three above utilize different numbers of source tasks.

task and randomly shuffle all examples to create a multitask training set. For the T5-3B most similar baseline, we sample 10,000 examples of the similar task in the same category in order to ensure that the size of the training set is the same as the size of the multitask training sets in our other experiments. Meanwhile, for the T5-3B + all tasks baseline, we select 21 tasks except the target and use 2,000 examples from each task. We provide more training details in the appendix.

As it is costly to compute pairwise transfer scores with bigger language models, we use TASKWEB scores from T5-large. This is based on our observation that models with similar architectures and adaptation methods share more similar transferabilities (Section [3.2\)](#page-3-0). We hypothesize that T5-large can learn the complexities of our source tasks and represent their transferabilities—this is supported by how both our T5-large transfers and T5-3B expert models in [Jang et al.](#page-10-3) [\(2023\)](#page-10-3) found CosmosQA and SocialIQA to be great source tasks.

Evaluation setup. We use the same set of evaluation tasks used by [Jang et al.](#page-10-3) [\(2023\)](#page-10-3). For ANLI-R1/R2 which are not included in TASKWEB, we apply the subset of tasks chosen for ANLI-R3 for the upper baseline. Meanwhile, for the Story Cloze task which is not included in TASKWEB due to its lack of training set, we use a subset of five tasks with the best transfer scores for the upper baseline. For each target task, we perform the evaluation in a leave-one-out setup by removing the target task from TASKWEB along with its scores. This is to maximize the number of available source tasks

while ensuring that the target task is unseen in our setup. By doing so, we simulate using TASKSHOP and TASKWEB across various categories of target tasks with access only to their examples (<sup>n</sup> ≤ <sup>32</sup>). We perform all evaluations in a zero-shot setting.

Results. Table [3](#page-6-1) summarizes the results of our experiments. The middle section details the performances of task selection methods that assume no access to pairwise transfer scores to the target. Two out of three methods improve target task performance compared to all baselines. Most notably, TASKSHOP outperforms both baselines as well as other task selection methods, improving by 14.7% over T0-3B and by 4.3% over our strongest baseline while using a small portion of the training set.

Finally, we observe that using the top-5 source tasks for each target according to TASKWEB consistently improves target performance. Our results support previous observations that using smaller multi-task training sets with a more careful task selection strategy can improve target performance [\(Pruksachatkun et al.,](#page-11-2) [2020;](#page-11-2) [Chan et al.,](#page-9-2) [2022\)](#page-9-2).

## <span id="page-6-0"></span>5.3 Discussion

The results of our experiments indicate that singletask transfer metrics can help improve multi-task transfers. We perform further experiments to support this hypothesis and address three questions.

How many source tasks do we need? We investigate whether different numbers of source tasks in the training set affect target task performance. To this end, we train T5-3B on training sets with top-1, 3, 10 and 21 source tasks in addition to five tasks.

<span id="page-7-0"></span>

Table 4: Results of choosing different numbers of source tasks for multi-task learning with TASKSHOP <sup>R</sup>O<sup>E</sup>. For each target task, the highest scoring setup is bolded. Results for top-5 are taken from TASKSHOPRO<sup>E</sup> in Table [3.](#page-6-1)

<span id="page-7-2"></span>![](_page_7_Figure_2.jpeg)

Figure 4: Variations in the zero-shot target performance as the top-5 source tasks for each target are incrementally replaced by the bottom-5 source tasks according to TASKWEB while maintaining the size of the training set.

<span id="page-7-1"></span>

Table 5: Results of choosing random and worst sets of tasks according to TASKSHOP and TASKWEB for three example target tasks, as well as the mean over all target tasks. Table [9](#page-15-2) in the appendix provides the full results.

Table [4](#page-7-0) shows the results. We observe that most target tasks achieve performance improvements from training on 3 to 5 source tasks. Using five source tasks results in the best overall performance and ranks first or second across most targets. Meanwhile, using ten source tasks results in a worse overall performance. The performance drops considerably when 21 tasks are used. According to our results, most targets only require a careful selection of three to five source tasks except several tasks such as Winogrande. Our findings differ from previous work which finds performance to scale with the number of tasks [\(Sanh et al.,](#page-11-0) [2022;](#page-11-0) [Wei et al.,](#page-12-1) [2022a;](#page-12-1) [Wang et al.,](#page-12-0) [2022b\)](#page-12-0) because while they add tasks in a target-agnostic manner, we add helpful source tasks based on the target task.

Do our methods identify both helpful and unhelpful source tasks? We demonstrate that our methods can also identify *unhelpful* tasks in multitask settings. To this end, we pick the bottom-5 source tasks for each target with TASKSHOP and TASKWEB, as well as five random source tasks.

Table [5](#page-7-1) summarizes the results. A random set of source tasks underperforms the T0-3B baseline, and the bottom-5 tasks from TASKSHOP further observes decreases in 3.4 accuracy points on average. Finally, the bottom-5 tasks based on TASKWEB results in similarly low performances. These results indicate that negative pairwise transfers between source and target tasks impact multi-task learning.

What happens if we mix helpful and unhelpful source tasks? While grouping helpful sources improves target performance and vice versa, it is unclear what happens in between. To address this, we experiment with different proportions of helpful tasks and measure the target task performance. We repeat this process over four target tasks in our evaluation setup—ANLI (R3), COPA, HellaSwag and RTE. For each task, we start with the top-5 tasks according to TASKWEB and replace a task with a bottom-5 task until all top-5 tasks are replaced. We perform the same evaluations as Tables [3,](#page-6-1) [4](#page-7-0) and [5.](#page-7-1)

Figure [4](#page-7-2) visualizes the results. As each helpful source task is replaced with an unhelpful source task, the target performance decreases across all four tasks. However, there are several instances where such replacement *increases* performance, as can be seen from 0→1 in HellaSwag and 4→5 in ANLI. These results indicate that while pairwise transferability between the source and target heavily impacts target performance during multi-task learning, other factors such as negative interference between the source tasks may also be involved, which is an interesting direction for future work.

## 6 Conclusion

In this work, we investigate how using prior knowledge of task relationships quantified via pairwise task transfer aids selecting helpful source tasks for multi-task NLP. We build TASKWEB, a benchmark and repository of pairwise task transfers across different tasks, models and adaptation methods in NLP. Based on our analysis of TASKWEB, we propose TASKSHOP, our method for selecting helpful source tasks for a new target task. We show that TASKSHOP outperforms existing methods in choosing helpful source tasks for different target tasks. Moreover, we use TASKSHOP and TASKWEB to build small multi-task training sets and outperform other methods that use much larger training sets.

## 7 Limitations

Our work contains several limitations. First, our set of tasks does not constitute the entirety of NLP tasks. While we use 22 NLP tasks that are representative enough to cover various types of reasoning, we do not include long-form tasks (e.g., summarization, LFQA) or domain-specific tasks (e.g., law, medicine) to facilitate experiments across various model architectures such as encoder-only models. In order to add entirely new forms of task to TASKWEB, one would have to compute pairwise transfer scores between the new task and other tasks in TASKWEB. If the model is known beforehand, this would require ∥T∥ iterations of fine-tuning with 1,000 examples where T is the set of tasks in TASKWEB. On the other hand, if the model is not known beforehand, this would require ∥M∥ × ∥T∥ iterations where M is the set of models used in TASKWEB.

Moreover, our datasets are in English and we do not incorporate multilinguality in our experiments. Second, our work focuses on models with at most three billion parameters. Our finding may not be directly applicable to models with orders of magnitude more parameters considering factors such as emergence [\(Wei et al.,](#page-12-6) [2022b\)](#page-12-6), which can be explored in future work. Third, we perform our multi-task finetuning experiments by uniformly sampling 2,000 examples from each source task following the style of [Wang et al.](#page-12-0) [\(2022b\)](#page-12-0). Therefore, different behavior may arise when other sampling strategies are used. Finally, recent work shows the effectiveness of using diverse instruction-output pairs which do not necessarily have clear boundaries as our tasks do [\(Ouyang et al.,](#page-11-8) [2022;](#page-11-8) [Wang](#page-12-7)

[et al.,](#page-12-7) [2022a,](#page-12-7) [2023\)](#page-12-8). Recently, [Wang et al.](#page-12-8) [2023](#page-12-8) report that large language models finetuned on specific instruction datasets perform better on related target tasks, which is closely related to our findings. Future work could extend our approach to setups without clear boundaries between tasks and explore ways to perform target-specific instruction tuning. Considering these limitations, we encourage the NLP community to contribute to quantifying the transferabilities between different language tasks.

## Ethics Statement

TASKWEB is based on a set of representative NLP tasks that have widely been used in the NLP community. While this work explores pairwise task transfer and multi-task finetuning using nonharmful datasets, an adversary could potentially misuse our approach to build another version of TASKWEB containing harmful tasks and quickly train models specifically for malicious target tasks. Hence we emphasize the importance of monitoring the content of tasks newly added to TASKWEB.

## Acknowledgements

We thank members of the H2Lab and UW NLP for their discussion and constructive feedback. This work was funded in part by the DARPA MCS program through NIWC Pacific (N66001-19-2-4031), NSF IIS-2044660, and gifts from AI2. Joongwon Kim is supported by the National Science Foundation Graduate Research Fellowship under Grant No. DGE-2140004. Akari Asai is funded by the IBM PhD Fellowship.

## References

<span id="page-8-1"></span><span id="page-8-0"></span>Alessandro Achille, Michael Lam, Rahul Tewari, Avinash Ravichandran, Subhransu Maji, Charless C. Fowlkes, Stefano Soatto, and Pietro Perona. 2019. [Task2vec: Task embedding for meta-learning.](https://doi.org/10.1109/ICCV.2019.00653) In *2019 IEEE/CVF International Conference on Computer Vision, ICCV 2019, Seoul, Korea (South), October 27 - November 2, 2019*, pages 6429–6438. IEEE. Armen Aghajanyan, Anchit Gupta, Akshat Shrivastava, Xilun Chen, Luke Zettlemoyer, and Sonal Gupta. 2021. [Muppet: Massive multi-task representations](https://doi.org/10.18653/v1/2021.emnlp-main.468) [with pre-finetuning.](https://doi.org/10.18653/v1/2021.emnlp-main.468) In *Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, EMNLP 2021, Virtual Event / Punta Cana, Dominican Republic, 7-11 November, 2021*, pages 5799–5811. Association for Computational Linguistics.

<span id="page-13-1"></span>

<span id="page-13-0"></span>

<span id="page-13-2"></span>

## A Appendix

#### <span id="page-14-0"></span>A.1 More Experimental Details

Full list of the datasets. Table [5](#page-15-1) presents the complete list of the 22 tasks studied in TASKWEB, along with references to the original papers.

Pairwise Task Transfer Metric For a source s and target t, evaluation function p, model m<sup>t</sup> tuned on <sup>t</sup> and a model <sup>m</sup>s→<sup>t</sup> tuned from s to t,

PC(s, t) ∝ <sup>m</sup>∈<sup>M</sup> <sup>p</sup>(ms→t) − <sup>p</sup>(mt) p(mt) PM(s, t) ∝ <sup>m</sup>∈<sup>M</sup> <sup>1</sup> (p(ms→t) > p(mt))

PC refers to the average percentage change of the model performance across all random seeds, and PM refers to the proportion of models that resulted in a positive transfer across all random seeds.

Implementation Details of Task Selection. For Retrieval-of-Experts, we use a similar implementation by taking 100 examples of the source task and 32 examples of the target task and computing the similarity between text embeddings of the prompts. We use PromptSource [\(Bach et al.,](#page-9-5) [2022\)](#page-9-5) to extract prompts and Sentence Transformers [\(Reimers and](#page-11-9) [Gurevych,](#page-11-9) [2019\)](#page-11-9) to obtain text embeddings.

For LLM-similarity, we write a prompt that contains several pairs of tasks not used in our setup, where each pair has 1) an example of each task, and 2) an answer noting whether the two tasks are similar or not. Then, for each source-target pair, we pass the prompt prepended to source and target examples to text-davinci-003 [\(Ouyang et al.,](#page-11-8) [2022\)](#page-11-8). We use the ratio of the log probabilities of the answers "yes" and "no" to assign a score between the source and target tasks.

Multi-Task Finetuning Details. We construct our multi-task training set by randomly selecting 2,000 examples with prompts from each task. For our T5-3B + all tasks baseline we choose all 21 tasks in TASKWEB apart from the target task, resulting in 42,000 examples. For all other methods (Retrieval-of-Experts, LLM-similarity, TASKSHOPROE, TASKWEB), we choose the five highest-scoring tasks according to each method, resulting in 10,000 examples. Then, we fully finetune LM-adapted T5-3B on our training set for five epochs, with an Adam Optimizer using a learning rate of 1e-4 and batch sizes ranging from 4 to 16 depending on the maximum length of each dataset.

<span id="page-14-1"></span>

Table 6: Top-5 source tasks selected using TASKSHOP.

<span id="page-14-2"></span>

Table 7: Top-5 source tasks selected using TASKWEB.

#### A.2 More Pair-wise Transfer Results

Full results. Figure [5](#page-15-1) displays pairwise transfer scores for all tasks in TASKWEB averaged over training setups. Scores for individual setups are shown in Figure [6](#page-16-0) (T5-large finetune), Figure [7](#page-16-1) (T5 base finetune), Figure [8](#page-17-0) (RoBERTa-base finetune), Figure [9](#page-17-1) (GPT2-medium finetune), Figure [10](#page-18-0) (T5 base Adapters), Figure [11](#page-18-1) (T5-base BitFit) and Figure [12](#page-19-0) (T5-small finetune).

Commutativity results. Figure [13](#page-19-1) shows the commutativity experiment results.

Transitivity results. Figure [14](#page-20-0) shows the experimental results of the transitivity analysis for all setups in our experiments.

### A.3 More Multi-Task Selection Results

Tasks chosen for the multi-task setup. Tables [6](#page-14-1) and [7](#page-14-2) list the top-5 (left to right) source tasks chosen for our multi-task setup using TASKSHOP and TASKWEB, respectively.

Bottom-5 and random-5 full results. Table [9](#page-15-2) presents the evaluation results for the bottom-5 source tasks selected with TASKSHOP and TASKWEB as summarized in Table [5,](#page-7-1) as well as five random source tasks.

<span id="page-15-0"></span>

<span id="page-15-1"></span>![](_page_15_Figure_3.jpeg)

<span id="page-15-2"></span>

<span id="page-16-0"></span>![](_page_16_Figure_0.jpeg)

<span id="page-16-1"></span>

![](_page_16_Figure_2.jpeg)

<span id="page-17-0"></span>![](_page_17_Figure_0.jpeg)

<span id="page-17-1"></span>

![](_page_17_Figure_2.jpeg)

<span id="page-18-0"></span>![](_page_18_Figure_0.jpeg)

<span id="page-18-1"></span>

![](_page_18_Figure_2.jpeg)

<span id="page-19-0"></span>![](_page_19_Figure_0.jpeg)

<span id="page-19-1"></span>

![](_page_19_Figure_2.jpeg)

<span id="page-20-0"></span>![](_page_20_Figure_0.jpeg)

Figure 14: Results for Figure [3](#page-3-2) (right) but for all setups in TASKWEB, with the probability of identifying positive source → target transfers as the minimum threshold for (source → pivot, pivot → target) transfers is increased.