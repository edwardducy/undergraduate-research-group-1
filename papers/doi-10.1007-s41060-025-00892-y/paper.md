International Journal of Data Science and Analytics (2026) 21:77 https://doi.org/10.1007/s41060-025-00892-y 

**~~REVIEW~~** 



# **Deep multi-task learning: a review of concepts, methods, and cross-domain applications** 

**Mahmoud Mohamed Abdelsamie**<sup>**1**</sup> **· Shahira Shaaban Azab**<sup>**1**</sup> **· Hesham A. Hefny**<sup>**1**</sup> 

Received: 14 December 2024 / Accepted: 2 November 2025 / Published online: 18 December 2025 © The Author(s) 2025 

#### **Abstract** 

Multi-task learning (MTL) is a machine learning method that has witnessed exponential traction due to its ability to solve multiple tasks simultaneously. Joining training-related tasks in MTL improves both predictive performance and data efficiency. The shared knowledge across tasks enables MTL models to outperform traditional single-task models. The effectiveness of MTL depends on several factors, including how tasks are chosen, how relationships between them are structured, and the choice of shared and task-specific model components. In literature, several architectures and strategies for multi-task learning have been successfully applied across different fields such as natural language processing, computer vision, healthcare, and others. However, MTL has some challenges, such as task interference, data availability, optimal task selection, and hyperparameter tuning. Recent research has introduced new techniques to address these issues, such as adaptive task balancing and parameter sharing mechanisms, although each approach has its limitations. In this review, we provide a comprehensive examination of the multi-task learning concept, and the strategies used in several different domains. It also explores the key factors influencing the success of MTL and provides a detailed classification of research efforts applying MTL in a wide range of fields, highlighting existing achievements and challenges. 

**Keywords** Multi-task learning · Single-task learning · Transfer learning · Hard parameter sharing · Soft parameter sharing · Deep learning 

## **1 Introduction** 

In today’s data-rich world, machine learning stands out as a leading branch of artificial intelligence methods, which are applied in many fields [1]. There are many different learning algorithms and methods, each method comes with its own strengths and limitations, often evaluated in terms of factors like performance and scalability. Among these, multitask learning has emerged as particularly dominant. These methods have shaped the landscape of machine learning by offering powerful tools for tackling complex tasks and large datasets. 

- B Shahira Shaaban Azab Shahiraazazy@cu.edu.eg Mahmoud Mohamed Abdelsamie 12422021452995@pg.cu.edu.eg 

Hesham A. Hefny hehefny@cu.edu.eg 

> 1 Department of Computer Science, Cairo University, Cairo, Egypt 

Humans have a natural ability to learn multiple tasks at once, often using skills gained in one area to enhance learning in another [2]. From this human learning ability, multi-task learning [3–5] is a learning model, aiming to learn multiple related tasks jointly so that the knowledge in one task can be used by other tasks, hoping to improve generalization performance for all tasks at hand [6]. 

The authors in [7] have made a superior comparison of multi-task learning and single-task learning approaches. 

Single-task learning is a machine learning approach where a model is trained to perform only one specific task [8]. It also focuses exclusively on optimizing performance for a single objective [7]. This approach simplifies the learning process and typically involves task-specific data and model parameters, which can lead to higher accuracy for that specific task, especially when large amounts of task-specific data are available. However, STL does not use any benefits of shared knowledge from related tasks, as seen in multi-task learning. 

Unlike multi-task learning, which focuses on training a single model to handle multiple tasks at once, for example, it 

123 

**77** Page 2 of 26 

International Journal of Data Science and Analytics (2026) 21 :77 

can involve learning both hate speech and sentiment analysis together or developing two topic classifiers simultaneously. 

The goal of MTL is to improve the generalization performance of a model on multiple related tasks by using the information shared across tasks [9]. By sharing certain network parameters, the model can develop a more efficient and compact data representation, which is especially advantageous when tasks are related or share common features. 

There are many multi-task learning architectures which focus on developing neural network frameworks that facilitate the efficient sharing of knowledge across different tasks to enhance overall performance [10]. Successful multi-task learning models strategically share parameters to mitigate the risk of negative transfer [11], which occurs when excessivesharingnegativelyimpactstheperformanceofindividual tasks. At the same time, they use shared information to enhance generalization capabilities [51, 52]. These architectures can be classified into several categories, including single-domain, multimodal, learned, and conditional types, each designed to meet distinct needs related to domain characteristics, input methods, or data-driven computations [74]. 

MTL is valuable in domains such as natural language processing (NLP), computer vision, and healthcare. In NLP, tasks like part-of-speech tagging, named entity recognition, hate speech, sarcasm detection, and sentiment analysis often benefit from shared syntactic and semantic features [12, 56]. The authors [13] emphasized that multi-task learning is increasingly becoming a major future trend in NLP tasks, especially in areas such as abusive language and hate speech detection. Similarly, in computer vision, object detection and segmentation can share features such as edge detection and texture recognition, improving both performance and data efficiency [14]. MTL is also particularly helpful in situations with limited data, as the shared knowledge across tasks can improve the model’s generalization by preventing overfitting andenhancingrobustness,especiallyinspecializedfieldslike healthcare where labeled data are often rare [10]. 

While numerous reviews have explored MTL from various perspectives, gaps exist in terms of scope, depth, and integration across domains. The authors [2] provided a seminal survey focusing on algorithmic models, theoretical analyses, and task relationships, offering a taxonomy of MTL approaches. Other work, such as recent surveys [15, 22, 45] on transfer learning, intelligent transportation, and cybersecurity, has examined MTL in the context of specific domains or convergent paradigms. For example, applications of MTL in intelligent transportation (e.g., traffic prediction and autonomous driving) and cybersecurity (e.g., intrusion detection and malware analysis) have been discussed in domain-specific reviews. However, these studies often lack a unified view across domains or fail to capture recent methodological developments in deep MTL. In contrast, our review aims to provide a comprehensive and up-to-date overview of 

deep MTL, covering not only the algorithmic foundations but also a wide range of applications across domains including healthcare, finance, language technologies, and more. 

It explores several factors that influence the success of MTL, including the choice of related tasks, the architecture for sharing knowledge across tasks, the methods for balancing task-specific and shared parameters, and the optimization techniques used. Additionally, the paper discusses the benefits and challenges associated with each MTL strategy. By linking theory, methods, and application perspectives, this survey fills a critical gap and supports both researchers and practitioners seeking to apply MTL across diverse fields. 

To this end, the contributions of this paper are outlined as follows:First,weofferquantitativeinsightsintotheeffectiveness of MTL across different domains. Second, we introduce the core concepts and general framework of MTL, the key approaches for sharing knowledge among tasks, and the factors that impact the performance of MTL models. Furthermore, we analyze the structure of various MTL architectures, their advantages, limitations, and how they are classified. Lastly,weprovideacomprehensivesurveyofresearchefforts that have successfully applied MTL in a wide range of fields. 

The remainder of this paper is organized as follows: Section 2 presents a comprehensive overview of the foundational concepts of MTL, including the factors that influence the effectiveness of any MTL approach. Section 3 provides quantitative insights into the research on MultiTask Learning focusing on studies indexed in "Scopus". Section 4 explores various MTL architectures with a particular focus on shared knowledge and task-specific components. The categories of multi-task learning architectures are presented in section 5. Section 6 shows the differences between MTL and transfer learning paradigms. Section 7 reviews the applications of MTL across diverse domains such as natural language processing, computer vision, and healthcare. We also present the pros and cons of using MTL in Section 8. Finally, Section 9 concludes the paper and suggests potential future directions for MTL research. 

## **2 Foundational concepts of MTL** 

In this section, we lay the groundwork for understanding multi-task learning by introducing its key concepts and principles. MTL is a machine learning approach where multiple related tasks are learned simultaneously, with the goal of leveraging shared knowledge to improve model performance across tasks. By sharing representations between tasks, MTL aims to enhance the generalization ability of the model, especially in cases where data for individual tasks are scarce or limited [10]. This highlights a more efficient learning paradigm that can minimize memory usage, reduce data 

123 

International Journal of Data Science and Analytics (2026) 21 :77 

Page 3 of 26 **77** 

consumption, and enhance both training speed and testing performance [15]. The primary objective of MTL is to improve model generalization by utilizing shared representations that can be applied across different tasks [16]. Instead of training separate models for each task, MTL enables a model to learn common features that can benefit multiple tasks. 

The effectiveness of MTL stems from several fundamental principles that govern its performance. Key among these is the use of shared representations, where hidden layers or parameters are shared across tasks to capture underlying commonalities [84]. This mechanism enables the model to generalize better, especially in scenarios with limited taskspecific data. Additionally, MTL introduces a regularization effect through shared parameters, imposing an inductive bias that discourages overfitting and promotes robust learning across related tasks [85]. The success of MTL also depends on task interdependence joint learning is most beneficial when tasks share underlying semantics or distinct structures. However, combining unrelated tasks can lead to negative transfer [86], where performance deteriorates due to conflicting learning signals. Finally, MTL involves multi-objective optimization, where the model must balance competing taskspecific losses [53]. This is typically managed by minimizing the weighted sum of individual task loss functions, requiring careful calibration to ensure that no single task dominates the learning process. 

MTL can be implemented through various architectural strategies, primarily classified into hard parameter sharing and soft parameter sharing [87]. In hard parameter sharing, the hidden layers are shared across all tasks, while the output layers remain task-specific. Soft parameter sharing, on the other hand, maintains separate model parameters for each task but applies regularization to maintain their similarity, allowing greater flexibility in learning task-specific nuances while leveraging shared knowledge. These techniques form the basis of many multi-task learning systems and will be discussed in more detail in the following sections. 

In conclusion, MTL offers several key advantages. It improves generalization by leveraging shared knowledge across related tasks, enabling models to learn more robust and transferable representations. MTL also enhances data efficiency, making it particularly useful in situations where labeled data are limited. Additionally, by combining multiple tasks into a single model, MTL reduces the complexity of the overall model, simplifying deployment and reducing computational requirements. Finally, MTL contributes to enhanced training stability, as joint learning helps mitigate the impact of noise or anomalies in individual task datasets. 

## **3 Quantitative analysis of multi-task learning research** 

Multi-task learning has gained significant attention in the field of machine learning due to its ability to improve model performance by sharing knowledge across multiple related tasks. As MTL becomes increasingly popular, understanding its growth trend and application across various domains is essential. This section presents a quantitative analysis of MTL research using “Scopus,” one of the most authoritative databases for scientific publications. The goal is to examine how research on MTL has evolved over the years, from 2020 to 2025, and to explore the key areas of its application. 

We used inclusion and exclusion criteria to identify relevant studies on multi-task learning. We conducted our analysis using the Scopus database. The search query included the terms “multi-task learning” OR “multitask learning,” applied to the title, abstract, and keywords fields. We limited the results to publications from 2020 to 2025 and further refined the search by restricting the subject areas to relevant domains, including but not limited to computer science, engineering, mathematics, physics, medicine, decision sciences, neuroscience, business, and social sciences. 

To ensure the quality and relevance of the retrieved documents, we included only articles, conference papers, and review papers. Additionally, we applied filters to include records with the exact keywords “multi-task learning” or “multitask learning.” A total of 8,245 publications on multitask learning were retrieved from the Scopus database between 2020 and mid-2025. After filtering by document type (articles, conference papers, reviews) and relevant subject areas and refining the results by removing duplicates based on DOIs and exact title matches, 6,872 unique records were retained for analysis. Figure 1 shows the yearly growth and document type distribution of MTL publications (2020–2025). 

The majority of research on multi-task learning between 2020 and mid-2025 was published in the field of computer science, accounting for 4,698 papers. This was followed by engineering (1,135 papers), mathematics (498 papers), medicine and health sciences (241 papers), and decision sciences (113 papers). These numbers highlight the strong dominance of technical and computational domains in driving advancements in multi-task learning research. We present the distribution of MTL usage across various research fields in Fig. 2, and also Fig. 3 reveals the top contributing fields. Other fields, such as physics, astronomy, decision sciences, pharmacology, and immunology, also contribute notably. 

This distribution highlights the relative focus and volume of research across different scientific disciplines, showcasing areas of intense research activity and those that may be emerging or less active. 

123 

**77** Page 4 of 26 

International Journal of Data Science and Analytics (2026) 21 :77 



**Fig. 1** Trends in multi-task learning publications (2020–2025): yearly growth and document type distribution 



**Fig. 2** The distribution of MTL usage across various research fields 

Based on the statistical analysis, it is evident that MTL research is growing rapidly, with more publications and increased interest from the scientific community each year. In particular, the year 2024 shows the highest number of MTL-related papers, indicating a surge in research efforts. 

## **4 MTL common approaches** 

MTL can be implemented through various approaches, including weight sharing [6, 17, 18] and architectural adaptations [54]. In deep learning, it is commonly achieved via hard or soft parameter sharing in hidden layers [10]. 

123 

International Journal of Data Science and Analytics (2026) 21 :77 

Page 5 of 26 

**77** 

**Fig. 3** The top contributing fields using MTL 



**Fig. 4** The architecture of hard parameter sharing 



### **4.1 Hard parameter sharing** 

#### **4.1.1 Mathematical definition** 

Hard parameter sharing is one of the most widely used strategiesinMTL.Itinvolvessharinghiddenlayersamongalltasks while employing task-specific output layers for predictions [18]. The model weights or parameters are shared between multiple tasks, and each weight is modified to minimize multiple loss functions [19]. This approach promotes knowledge transfer and reduces the risk of overfitting by forcing the model to learn representations that are useful across tasks. They also find applications in diverse fields, from natural language processing to computer vision, etc. Figure 4 illustrates the architecture of hard parameter sharing. 

For the task-specific representation, let **_W s_** represent the shared parameters of the network. The input **_x_** is passed through these shared layers to produce a shared representation **_hs_** : 



where **_f s_** is the function representing the shared layers. 

123 

**77** Page 6 of 26 

International Journal of Data Science and Analytics (2026) 21 :77 

For the task-specific heads, for each task **t** ∈{1, 2, _. . ._ , **T** }, there is a task-specific set of parameters **wt** . The taskspecific output **_yt_** is computed as: 



where **_f t_** represents the task-specific layers. 

For the loss function, the total loss **_L_** for all tasks is a weighted sum of individual task losses: 



where **_λt_** is the weight for task **t** , **_L t_** is the loss function for task **t** , and **_y_**<sup>**_true_**</sup> **_t_** is the ground truth for task **t** . 

The shared parameters **_W_** _s_ are updated using gradients computed from all tasks, while task-specific parameters **_W t_** are updated based only on their respective task losses. 

### **4.2 Soft parameter sharing** 

In contrast, soft parameter sharing assigns each task its own model with separate parameters and distance between the weights of models for every task, which are regularized to encourage similarity across tasks [20, 21]. In soft parameter sharing, there is no explicit sharing of parameters; instead, the models for different tasks are encouraged to have similar parameters through regularization. This approach is particularly useful when negative transfer occurs between tasks, necessitating reduced sharing [19]. Various optimization techniques are employed to achieve effective soft parameter sharing. This allows for greater flexibility in accommodating task-specific nuances while maintaining knowledge sharing. Figure 5 illustrates the architecture of soft parameter sharing. 

In soft parameter sharing it differs, the overall loss function for soft parameter sharing is given by: 



where: 

�{ **_Tt_** =1}<sup>**_Lt_**:aggregatesthetask-specificlosses,</sup><sup>**_L_**</sup><sup>_t_bethe</sup> loss function for task **_t_** , **_α R_** _(_ **_W_** 1, **_W_** 2, _. . ._ , **_W T_** _)_ : A regularization term is added to the loss function to encourage the task-specific parameters **_W_** _T_ to remain similar across tasks (similarity does not mean that all task parameters are identical, they are only allowed to deviate when needed to account for task-specific nuances.), and **_α_** is a hyperparameter controlling the strength of regularization. 

To conclude this section, we provide a comparison between hard parameters and soft parameters sharing. Table 1 presents a comparison between them. 

## **5 Categories of MTL architectures** 

The nature of tasks is always responsible for the design of multi-task neural network architecture. When creating a shared architecture, it is crucial to balance the shared parameters well. Oversharing in multi-task models can lead to negative transfer and affect performance. Conversely, too little sharing prevents the model from effectively taking advantage of shared information, limiting its ability to exploit relationships between tasks [74]. Successful architecture achieves a balance that enhances the overall learning of all tasks involved. In the survey [74], the authors have categorized MTL architectures into four main groups: Single-domain architectures tailored for specific task domains such as computer vision and NLP, multimodal architectures that handle inputs from multiple modalities such as visual and linguistic data in visual picture storybooks, learned architectures with structures determined during architecture search, and conditional architectures that adapt dynamically based on input data. 

### **5.1 Single-domain architectures** 

#### **5.1.1 Computer vision multi-task architectures** 

In the field of computer vision, many multi-task learning architectures aim to partition networks into common taskspecific components. 

For instance, shared trunk where a common backbone (trunk) is used to extract common features, while taskspecific heads process these features for individual tasks. The authors in [75, 76] proposed architecture based on the shared trunk. For instance, [75] proposed a multi-task attention network (MTAN) that includes a shared network with a global feature set and a soft attention unit dedicated to each task. These attention units enable the extraction of task-specific features from the global set while allowing feature sharing across tasks. This design ensures a balance between using shared information and maintaining task-specific adaptability, enhancing overall performance in multi-task scenarios. A simple architecture of a shared trunk example is shown in Figure 6 where the feature extractor is made of a series of convolutional layers which are shared between all tasks, and the extracted features are used as input to task-specific output heads. 

Another architecture for multi-task learning, different from the shared trunk commonly used with task-specific modules, is the cross-talk network [74]. Unlike shared trunk models, these architectures assign a separate network to each task while facilitating the flow of information between parallel layers of task-specific networks. 

123 

International Journal of Data Science and Analytics (2026) 21 :77 

Page 7 of 26 

**77** 

**Fig. 5** The architecture of soft parameter sharing 



**Table 1** Hard parameter sharing and soft parameter sharing comparison 

|Aspect|Hard parameter sharing|Soft parameter sharing|
|---|---|---|
|Definition|All tasks share a common set of parameters|Tasks have their own parameters but share knowledge<br>indirectly|
|Task relatedness|Requires tasks to be highly related|Suitable for tasks with partial or low relatedness|
|Flexibility|Less flexible; tasks cannot have task-specific features|More flexible; tasks can learn their specific features|
|Computational cost|Low, as few parameters are used|Higher, due to maintaining separate parameters for<br>tasks|
|Risk of overfitting|Lower, due to reduced model complexity|Higher, as each task has more freedom to learn<br>independently|
|Adaptability|Limited adaptability to diverse task requirements|Better adaptability to unique task requirements|
|Implementation|Simpler and easier to implement|More complex, requiring mechanisms for task-specific<br>learning|



An example of this approach is the cross-stitch network proposed by [14]. In a cross-stitch network, each task maintains an individual network, but the inputs to each layer are derived from a weighted linear combination of the outputs from the corresponding layers of all task networks. These weights are learned during training and are task-specific, allowing each layer to selectively use information from other tasks. Figure 7 shows the cross-stitch network architecture by [14]. 

Prediction distillation is another approach, inspired by knowledge distillation, involves one task guiding another by providing its predictions as supervision. For example, the authors in [77] take advantage of this prediction distillation for MTL of computer vision tasks by making initial predictions for each task and then combining these predictions to produce improved final outputs. The earliest example of this approach was made by [78] which is the PAD-Net architecture for prediction distillation [74] as shown in Figure 8. 

#### **5.1.2 NLP multi-task architectures** 

Implementing MTL architectures in NLP parallels the implementation of neural networks, moving from feedforward architectures to recurrent models and, more recently, attention-based architectures like transformers [74]. The main MTL architectural approaches used in NLP are as follows: 

Traditional Feedforward is the earlier MTL models which shared a global feature extractor (like word embeddings) with task-specific output heads. Furthermore, the authors in [55] categorized MTL architecture in NLP into four architectures (parallel, hierarchical, modular, generative). Parallel architecture: Shares the model across tasks with task-specific output layers for each task. In hierarchical architecture, the modelshierarchicaltaskrelationshipsbycombiningfeatures, using the output of one task as input to another, or explicitly modeling task interactions. However, modular architecture 

123 

**77** Page 8 of 26 

International Journal of Data Science and Analytics (2026) 21 :77 



**Fig. 6** A simple architecture of a shared trunk 



Fig. 7 Cross-stitch network<br>architecture [14], where a new<br>cross-stitch unit combines two<br>networks into a multi-task<br>network in a way that the tasks<br>supervise how much sharing is<br>needed<br>
decomposes the model into shared, task-specific components to learn task-invariant and task-specific features. Unlike, generative adversarial architecture uses generative adversarial networks to enhance the model’s capabilities for multi-task learning. 

[79, 80], decompose networks into reusable submodules, allowing for task-independent and transferable representations. Additionally, auxiliary tasks can guide reinforcement of learning models by enhancing representation learning. 

### **5.2 Multimodal architectures** 

#### **5.1.3 Reinforcement learning architectures** 

Reinforcementlearningarchitecturesrangefromsimplefully connected, convolutional, and recurrent networks to more complex modular designs for multi-task learning [74]. Joint task training architectures use shared or discrete feature extractors to improve task performance through shared learning signals. Modular policies, such as those developed by 

Multimodal architectures extend MTL by integrating data from diverse domains, such as vision and language, to share representations across tasks and media (like images), enhancing generalization. Approaches such as OmniNet [81] use hierarchical or joint-attention mechanisms to efficiently process multiple media. This approach shows improved performance across diverse tasks, especially where datasets 

123 

International Journal of Data Science and Analytics (2026) 21 :77 

Page 9 of 26 **77** 



**Fig. 8** PAD-Net architecture for prediction distillation [78] 



**Fig. 9** OmniNet architecture [81] 

are limited, or tasks involve complex interactions between media. Figure 9 shows the OmniNet architecture. 

### **5.3 Learned architectures** 

Instead of manually designing shared structures, approaches are being developed that allow models to learn both the structure and the weights. This approach allows for the sharing of dynamic parameters tailored to the relationships between 

tasks, mitigating negative transfer and maximizing positive transfer [74]. These approaches include structure search, branched sharing, modular sharing, and fine-grained sharing. Structure search algorithms, such as reinforcement learning or evolutionary strategies, help design models that can automatically adjust their architecture based on task relationships. Branched sharing gradually partitions shared layers into task-specific layers. Modular sharing enables flexible 

123 

**77** Page 10 of 26 

International Journal of Data Science and Analytics (2026) 21 :77 



**Fig. 10** MTL and transfer learning paradigms 

reuse of recombined components for different tasks. Finegrained sharing applies techniques such as binary masks or pruning techniques to customize parameter sharing, improving task performance. These approaches aim to maximize the benefits of multi-task learning by adapting the sharing strategy to the specific needs of the task, minimizing negative transfer, and enhancing positive transfer. 

### **5.4 Conditional architectures** 

Conditional architectures in multi-task learning dynamically adjust their architecture based on the input or task, enhancing computational efficiency and task generalization. [74]. In multi-task learning, this approach dynamically adapts between inputs and tasks, with components shared across tasks to enhance generalization ability. This flexibility enables the network to efficiently adapt to diverse requirements without sacrificing performance across different tasks. 

## **6 MTL VS transfer learning** 

MTL and transfer learning are both approaches to leveraging shared knowledge across tasks but differ fundamentally in their paradigms. The core principle of multi-task learning is that knowledge gained from learning one task can enhance the learning of other tasks [19]. In MTL, all tasks are trained simultaneously by using shared representations. The model requires training on all datasets for all tasks at the same time. On adding a new task multimodal inputs must retrain the network, as it may require performance of architectural changes. 

In contrast, transfer learning sequentially applies knowledge from a source task to a target task, often through pretrained models, to enhance performance in scenarios where the target task has limited data [22]. Transfer learning refers to exploiting what the model has already been learned 

in one setting task to improve the learning in another setting [19]. Knowledge transfer happens from the source domain to the target domain [69]. It only focuses on enhancing the performance of the target task. For example, a model previously trained on public text data can be adapted to identify hate speech or offensive language in social media posts or classify sentiment in customer reviews. The knowledge gained from the large-scale language understanding task is transferred to thetargettask(e.g.,hatespeechorsentimentanalysis),allowing the model to perform better, especially when labeled data for the target task is scarce. Figure 10 shows the paradigms of MTL and transfer learning. While MTL requires datasets for all tasks at the same time, transfer learning typically involves a large dataset for the source task and adapts its knowledge to the target task later. 

Transfer learning and multi-task learning can involve heterogeneous tasks, such as classification, segmentation, or regression, and both can share features and parameters across tasks [19]. While multi-task learning is often considered a type of transfer learning [82] due to its inductive transfer approach, the two models differ significantly. Finally, the key architectural differences between MTL and transfer learning are presented in Table 2. 

## **7 Applications across diverse domains** 

Multi-task learning has proved its power and effectiveness in a variety of domains and applications [83] by using shared representations to improve performance across related tasks and domains. However, learning such multi-tasks becomes even more challenging when the tasks are associated with different domains, as there is less correlation information available both among the tasks and across the domains [68]. Below, we review key applications of multi-task learning in 

123 

International Journal of Data Science and Analytics (2026) 21 :77 

Page 11 of 26 **77** 

**Table 2** Key architectural differences between MTL and transfer learning 

|Aspect|Transfer learning|Multi-Task learning|
|---|---|---|
|Training mode|Sequential: source task first, then target|Simultaneous: all tasks at once|
|Knowledge flow|From source to target only|Bidirectional, among all tasks|
|Goal|Boost target task (using source knowledge)|Improve all tasks via shared representations|
|Model structure|Often reuses/fine-tunes model for new task|Shared backbone with task-specific heads/outputs|
|Typical usage|When target data is scarce; domain adaptation|Related tasks with shared features, multi-label|



many different areas such as computer science, engineering, and medicine. 

In NLP, MTL is applied to classification and prediction tasks such as hate speech detection, sentiment analysis, sarcasm detection [34, 35], and machine translation. By learning shared representations across tasks, MTL enables models to generalize better, capturing linguistic nuances such as dialects and context. For instance, the authors [84] addressed the challenge of Arabic dialectal variation ambiguity and its effect on hate speech detection performance. They implemented an MTL approach consisting of a shared encoder based on transformer architecture and five task-specific heads for five Arabic dialects. Their approach achieved superior results across all tasks compared to the baseline models. Another study [88] implemented a framework that improves implicit sentiment analysis by combining multi-task learning, multi-task machine learning, and adaptive weight adjustment. It addresses uncertainties at the data and task levels but is limited by potential hallucinations in the generated auxiliary data. [34] proposes a multi-task approach for detecting hate speech in Spanish tweets, combining polarity and emotion classification to improve the accuracy. Also, in [35] a multi-task learning-based framework is proposed to improve sentiment analysis by modeling the association with sarcasm detection, outperforming existing methods with an F1 score of 94%. Another MTL framework for text classification is proposed by [36] using unsupervised subword and phrase recognition as an auxiliary task, which reduces the need to generate supervised labels and improves classification performance across multiple datasets. The work [42] proposed a multi-task learning framework for multilingual neural machine translation, which jointly trains translation and noise removal tasks, improving translation quality for resource-rich and resource-poor languages and enhancing non-redundancy capabilities. 

In computer vision, MTL is currently used in the field of computer vision due to its success in achieving advanced scene understanding [56]. The shared features of MTL models allow them to understand spatial hierarchy, improving the model’s accuracy and performance. For medical image analysis, MTL has proved essential for tasks such as disease classification, lesion segmentation [57], and anomaly 

detection, often simultaneously processing different modalities such as MRI and CT to enhance diagnostic accuracy. For instance [89], they proposes a multi-task, deep learning framework for cross-platform malware image classification (Windows, Android, Linux, Mac) by converting binary malware files into RGB images. The model, which uses a CNN with PReLU activation, achieves over 99.9% accuracy, although it currently lacks integration of code-level features, such as API calls. MTL is also used in medical imaging in [90], where the authors introduce AMTLUS framework, a novel multi-task learning approach for simultaneous skin lesion segmentation and classification by integrating attention mechanisms and uncertainty estimation. This enhances both accuracy and interpretability, addressing key challenges in dermatological image analysis. 

Furthermore, in medical imaging, the study [23] introduces the cross-task attention network (CTAN), an attentionbased MTL framework, demonstrating improved performance over standard single-task learning and baseline MTL methods like hard parameter sharing and multi-task attention networks in diverse medical imaging tasks. [37] explores multi-task learning for facial landmark detection using related tasks such as head pose estimation and facial feature inference and improves robustness to occlusions and pose changes through task-constrained learning and early stopping. YOLOPv3, an efficient anchor-based multi-task visual perception network, is proposed by [38], which demonstratesstate-of-the-artperformanceintrafficobjectdetection, driven area segmentation, and path discovery while maintaining competitive inference speed. [39] proposed a multi-task learning framework for object detection and semantic segmentation using partially labeled data is proposed, leveraging knowledge distillation to improve performance compared to single-task learning and fully supervised scenarios. [43] proposed a deep multi-task learning framework with a multidepth fusion module for brain tumor segmentation in MRI, achieving high accuracy and robust segmentation contours. 

In healthcare, MTL is used for tasks like predicting patient illness and discovering disease biomarkers [58, 59]. These insights enable personalized medicine and early diagnosis. In [28], they proposed a novel multimodal multi-task learning 

123 

**77** Page 12 of 26 

International Journal of Data Science and Analytics (2026) 21 :77 

approach for healthcare (M3H), integrating data from language, vision, tabular, and time-series sources. It supports classification, regression, and clustering across diverse medical tasks, achieving superior accuracy and explainability. Also, GenHPF [91] is a multi-purpose framework designed for multi-task, multi-source healthcare prediction by transforming heterogeneous electronic health record (EHR) data into a unified text format. It utilizes seamless parameter sharing through shared representations and self-learning, enabling scalable learning across diverse datasets and tasks. Another study [24] shows that MTL improves rare phenotype detection using EHR, enhancing robustness and performance with more auxiliary tasks, but underperforms for common phenotypes. The study [73] proposes a drug recommendation model using multi-task learning with soft parameter sharing, integrating structured data and unstructured texts to predict drugsatisfactionratings,andovercomethecoldstartproblem in healthcare recommendations. 

More fields in one study such as medicine, healthcare, and NLP, the authors [25] present MedRec, a graph-based model for medicine recommendation that alleviates data sparsity by leveraging knowledge and attribute graphs. 

It demonstrates superior performance over state-of-theart methods and shows that learning graph representations and medicine recommendation tasks mutually enhance each other. 

Another domain, biological sciences, MTL is used for protein structure prediction [60], drug interaction analysis, and gene expression modeling, enabling researchers to make connections between related activities to gain better understanding. Recent studies in biological data analysis use multi-task learning to integrate multi-omics data, improving prediction and interpretability. The work [30] builds on this by proposing an interpretable MTL framework that jointly addresses multimodal classification and prediction, offering better performance and insight compared to traditional single-task or modal-specific approaches. Another work [92] introduced the HBNMM model, a heterogeneous biological network model that uses multi-task learning and graph attention networks, has been proposed to predict the relationships between non-homologous RNA (ncRNA) and disease, ncRNA and drug, or drug and disease. This model improves prediction accuracy and generalization but still faces challenges such as data scarcity and task imbalance. 

In power systems, MTL enhances tasks such as demand forecasting in power systems [61] and fault detection. For instance, seasonal cycles and energy usage anomalies are better captured via common representations. MTL improves system reliability and operational efficiency in engineering by learning from auxiliary jobs in predictive maintenance, problem categorization, and resource allocation. The paper [93] proposes a multi-task learning approach using a spatiotemporal recurrent imputation network (SRIN) to jointly 

handle missing performance unit data imputation and shortterm voltage stability assessment. This method effectively combines history-based and feature-based imputation and adapts well to diverse spatiotemporal imputation scenarios. The work [31] proposes a multi-task deep learning framework for fault classification and localization in power systems, achieving high performance with minimal error in fault localization. 

In Energy, [32] designs a multi-task learning system for four building energy tasks, achieving superior prediction performance for energy forecasting and anomaly detection. 

In autonomous systems, MTL is utilized in tasks such self-driving cars, for tasks including path prediction, lane segmentation, and object identification [62]. For safer and more effective navigation, the joint learning framework combines temporal and spatial features. 

In finance, by utilizing common patterns across financial instruments and markets, MTL improves activities like fraud detection [63], credit risk assessment, and stock price prediction in the finance industry. Additionally, it enhances the ability to analyze consumer behavior and offers tailored financial advice. 

In climate science, MTL helps with climate science tasks like environmental monitoring, climate change modeling, and weather prediction [64]. Reducing computational complexity and improving predictions are two benefits of sharing representations across various activities. 

In educational technology, MTL is used for adaptive learning systems, combining tasks such as predicting student performance [65], recommending resources, and detecting at-risk students [67]. Shared representations enable comprehensive strategies for personalization and intervention. 

In psychology, [33] investigates multi-task learning to predict mental health factors, improving predictions of depression and stress but not anxiety, while enhancing temporal considerations. 

In robotics, MTL incorporates motion planning, obstacle avoidance, and object manipulation into robotics. For example, [66] presented a multi-task transformer for robotic manipulation. As a result, robots can acquire logical techniques for effectively navigating and interacting with their surroundings. 

Furthermore, MTL in physics allows for shared learning across related problems, like predicting quantum characteristics or solving differential equations for fluid dynamics and heat transfer. To address challenging physics problems, researchers have recently investigated physics-informed neural networks (PINNs) utilizing MTL frameworks. For example [70], M-PINN, a new network structure based on PINNs, improves the representation of multi-physics coupling phenomena by applying soft parameter sharing to fluid–structure interaction (FSI) problems in porous media. This technique showed the potential of MTL to further physics research 

123 

International Journal of Data Science and Analytics (2026) 21 :77 

Page 13 of 26 **77** 

by outperforming conventional methods in tackling constant pressure production challenges. [40] introduced an MTObased PINN training framework that uses auxiliary tasks to enhance performance via knowledge transfer, achieving improved results in traffic density prediction compared to traditional methods. 

Inchemistry,[26]usedhardparametersharingandsharing parameters of the LSTM and character embeddings between the named entity recognition (NER) model and the chemical compound paraphrase model. This allows both models to benefit from shared knowledge during training, leading to improved performance in chemical NER. Also, [27] introduces a multi-task deep learning model (MDNN) to predict four flammability-related properties of chemicals. The model combines Tree-LSTM and feedforward networks to automatically extract molecular features, outperforming traditional methods and eliminating the need for manual feature extraction. 

In biochemistry, [41] presented a new SGNN-EBM method that takes advantage of the important relationship graphs for structured modeling of tasks in molecular property prediction, enhancing MTL by modeling the latent space and predicting the structured output. 

In bioinformatics/drug discovery, researchers in [70] have explored physics-informed neural networks (PINNs) with MTL frameworks to address complex physics problems. For example, M-PINN, a novel network architecture based on PINNs, applies soft parameter sharing to fluid–structure interaction (FSI) problems in porous media, enhancing the representation of Multiphysics coupling phenomena. This method outperforms conventional approaches in solving static pressure generation problems, demonstrating the potential of MTL in advancing physics research. 

Also, in the field of drug discovery, MTL addresses challenges such as sparse data and task diversity by combining tasks such as molecular property prediction, toxicity estimation, and drug-target interaction. For instance, the study [71] introduces a multi-task learning framework with soft parameter sharing, aimed at improving drug-target affinity (DTA) prediction, particularly for unknown drugs. 

Also, in agriculture, researchers used multi-task learning to solve complex challenges that are addressed by using common patterns across tasks, such as crop yield prediction, pest detection, and precision fertilization. By using common representations, MTL models can integrate diverse agricultural data, including climate, soil nutrients, and crop characteristics, to improve forecast accuracy and resource management. For example [72], MTL frameworks have been used to enhance precision fertilization predictions, and their approach aids intelligent, precise, and personalized farm management. In [29], they proposed a hybrid deep learning model combining classification and regression for rice crop yield prediction, outperforming other models. 

Across all these fields and more, MTL has the potential to drive innovation and efficiency in solving real-world problems. Table 3 summarizes the work of applications of different fields including domain, approach, architecture, task settings, and limitations. 

By analyzing the reviewed studies, we found that multitask learning has been applied in various fields such as computer science, medicine, energy, engineering, physics, biochemistry, agriculture, and health, demonstrating its versatility and effectiveness in solving complex tasks. The decision between hard and soft parameter sharing in MTL systems relies on the characteristics of the issue. For tasks that are highly similar, hard parameter sharing is used where parameters are utilized across tasks with unique output layers. However, in situations where tasks significantly differ or need increased flexibility, soft parameter sharing, which allows tasks to share essential features while keeping distinct parameters for layers, could lead to improved performance. 

## **8 Advantages and limitations** 

Although MTL has shown significant advantages, it also comes with its own set of challenges and limitations. In this section, we provide a summary of its advantages and disadvantages as follows: 

### **8.1 Advantages of MTL** 

MTL offers several important benefits. One major benefit is improved generalization, as sharing knowledge across related tasks helps reduce the risk of overfitting and enables the model to generalize better to unseen data [44, 45]. By learning from multiple tasks simultaneously, MTL enhances the model’s ability to perform well in a variety of situations of unseen data. 

Another important benefit is the efficient use of data. Multi-task learning is particularly useful when data for individual tasks are scarce [46], as the joint learning process allows the model to leverage information from all tasks, improving overall performance. This efficiency is particularly important in domains where labeled data are limited. 

Multi-task learning also promotes better feature learning, as the model is encouraged to learn common features across tasks [2]. This results in more robust and richer representations, which are particularly useful in domains such as natural language processing (for example, hate speech detection) and computer vision (for example, object detection, and image segmentation), where tasks often share similar features and structures. 

Additionally, reduced training time and required resources are a notable benefit of multi-task learning [47, 48]. Because multiple tasks are learned together, a model can often be 

123 

**77** Page 14 of 26 

International Journal of Data Science and Analytics (2026) 21 :77 

|**Limitations**|Dialect Imbalance and<br>out-of-vocab problem<br>- Data-level uncertainty due to<br>LLM hallucinations<br>- Task-level uncertainty from<br>different model capacities<br>- Performance depends on<br>accurate weight estimation<br>Sensitive to implicit hate;<br>struggles with nuanced<br>polarity/emotion cues;<br>mislabeled training data<br>impacts learning; not<br>evaluated beyond Spanish<br>datasets<br>Limited to image-based<br>features; does not yet include<br>code/API analysis<br> <br>Limited generalization to other<br>datasets, no domain<br>adaptation, future work<br>needed for task expansion|
|---|---|
|Tasks|Hate speech detection<br>Sentiment analysis<br>Hate Speech Detection,<br>Sentiment Analysis<br>Malware classification<br>across PE, APK, ELF,<br>Mach-O formats<br>Skin lesion segmentation<br>and classification|
|Architecture|Transformer shared encoder<br>with task-specific output<br>heads (e.g., dialect classifiers<br>or binary hate speech<br>classifiers)<br>LLM-based MTL with auxiliary<br>task construction and weight<br>learning<br>Transformer-based<br>CNN with 11 layers, PReLU<br>activation<br>CNN with MTL+Attention+<br>Uncertainty modules|
|Parameter<br>Sharing<br>Hard<br>Soft|✓<br>–<br>–<br>✓<br>✓<br>–<br>✓<br>–<br>✓<br>✓|
|Field/Domain<br>Approach|NLP<br>Multi-task learning<br>across<br>dialect-specific<br>hate speech tasks<br>NLP<br>Automatic Weight<br>Learning with<br>LLMs<br>NLP<br>MTL Approach to<br>Hate Speech<br>Detection using<br>Sentiment<br>Analysis<br>Computer vision<br>Deep Multi-Task<br>Learning for<br>Malware Image<br>Classification<br>Computer vision /Medical Imaging<br>Attention-guided<br>Multi-Task<br>Learning|
|Ref<br>Year|[84]<br>2025<br>[88]<br>2024<br>[34]<br>2021<br>[89]<br>2022<br>[90]<br>2024|



123 

International Journal of Data Science and Analytics (2026) 21 :77 

Page 15 of 26 **77** 

|**Limitations**|Limited task/pattern coverage<br>(e.g., lack of image<br>segmentation, MS in<br>Computer Science);<br>robustness to data<br>disturbance; lacks integration<br>between omics/wearables;<br>task set design suffers from<br>the curse of dimensionality;<br>numerical instability; future<br>potential in prediction then<br>optimization<br>Limited to tabular data and<br>same-language EHRs;<br>excludes some EHR event<br>types due to memory<br>constraints<br>Requires fine-tuning of the<br>architecture/hyperparameters;<br>randomness in training; and<br>scalability for more tasks that<br>are still being explored<br>Sparse associations, imbalance<br>of drug-disease links, limited<br>labeled data, potential for<br>enhancement<br>Needs further testing across<br>more complex/multiple grids;<br>depends on RNN<br>interpretability|
|---|---|
|Tasks|40 diagnoses, 3<br>operation forecasts, 1<br>phenotyping<br>(classification,<br>regression, clustering)<br>12 clinical prediction<br>tasks across 3 EHR<br>datasets<br>Joint group dentification,<br>Cross-modal<br>prediction<br>ncRNA-disease,<br>ncRNA-drug,<br>drug-disease<br>association prediction<br>Missing data imputation,<br>Voltage stability<br>assessment|
|Architecture|Attention-based model<br>Hierarchical Text Encoding<br>Encoder-Decoder-<br>Discriminator<br>Graph Attention+MTL<br>SRIN (RNN)|
|eter<br>ng<br>Soft|✓<br>✓<br>✓<br>✓<br>✓|
|Param<br>Shari<br>Hard|✓<br>–<br>–<br>–<br>–|
|Approach|M3H Multimodal<br>MTL<br>GenHPF,<br>Multi-task<br>Multi-source<br>Learning<br>UnitedNet, an<br>explainable<br>multi-task deep<br>neural network<br>ed<br>heterogeneous<br>biological<br>network<br>multi-task<br>learning model<br>(HBNMM)<br>multi-task learning<br>approach based<br>on<br>spatial–temporal<br>recurrent<br>imputation<br>network (SRIN)|
|Field/Domain|Healthcare<br>Healthcare<br>Bioinformatics<br>Bioinformatics/Healthcare/Biom<br>Electrical Engineering/Power<br>Systems|
|Year|2024<br>2023<br>2023<br>2024<br>2025|
|Ref|[28]<br>[91]<br>[30]<br>[93]<br>[94]|



123 

**77** Page 16 of 26 

International Journal of Data Science and Analytics (2026) 21 :77 

|**Limitations**|Limited improvement in<br>segmentation tasks;<br>performance drop in complex<br>tasks like COVID severity<br>classification<br>Random auxiliary task<br>selection may not optimize<br>performance; rule-based<br>ground truth may introduce<br>artifacts; lack of temporal<br>modeling in feature<br>representation; limited<br>phenotype scope<br>Effectiveness depends on graph<br>quality; lacks temporal<br>modeling; limited external<br>validation<br>Dependent on paraphrase<br>quality; limited to chemical<br>domain; requires large<br>paraphrase data<br>Requires large, curated<br>datasets; may underperform<br>on small or simple molecules<br>like methane; interpretability<br>limited<br>Limited data availability;<br>reliance on feature<br>engineering; no temporal<br>dynamics modeled deeply<br>Data collection challenges;<br>complex spatiotemporal<br>modelingassumptions|
|---|---|
|Tasks|Segmentation,<br>Diagnosis, Severity<br>Prediction, Dose<br>Planning across 4<br>datasets (Prostate,<br>OpenKBP,<br>HAM10000, STOIC)<br>Phenotyping+<br>Auxiliary Tasks<br>Graph Embedding,<br>Medicine<br>Recommendation<br>NER, Paraphrase<br>Generation<br>Predict Lower<br>Flammability Limit<br>(LFL), Upper<br>Flammability Limit<br>(UFL), Flash Point,<br>and Auto-Ignition<br>Temperature (AIT)<br>Regression (yield),<br>Classification<br>(high/low yield)<br>Predict fertilizer amount<br>and timing across<br>farms and months|
|Architecture|Attention-based MTL<br>MTL Neural Networks<br>GCN+Attention+Attribute<br>Fusion<br>Shared Bi-LSTM+ANMT<br>Tree-LSTM encoder+multiple<br>FNN decoders (MDNN)<br>MKCNN+Bi-LSTM with<br>Shared Layers<br>Spatiotemporal Tensor+CP<br>Decomposition|
|meter<br>ring<br>d<br>Soft|✓<br>✓<br>✓<br>✓<br>✓<br>✓<br>✓|
|Para<br>Sha<br>Har|–<br>–<br>–<br>–<br>–<br>–<br>–|
|Approach|Cross-Task<br>Attention<br>Network (CTAN)<br>Multi-task Neural<br>Nets for<br>Phenotyping<br>(EHR Data)<br>MedRec (Medicine<br>Recommendation<br>System)<br>Multi-task<br>Learning (NER<br>+Paraphrase)<br>Multi-task Deep<br>Neural Network<br>(MDNN) for<br>Predicting<br>Flammability<br>Properties<br>Hybrid Deep<br>Learning Model<br>Tensor-based<br>Multi-Task<br>Learning|
|inued)<br>Field/Domain|Medical Imaging<br>Medicine, Healthcare<br>Healthcare (NLP/RecSys)<br>Biomedical NLP, Chemistry<br>Chemistry<br>Agriculture<br>Smart Agriculture|
|**3** (cont<br>Year|2023<br>2018<br>2022<br>2019<br>2021<br>2024<br>2023|
|**Table**<br>Ref|[23]<br>[24]<br>[25]<br>[26]<br>[27]<br>[29]<br>[72]|



123 

International Journal of Data Science and Analytics (2026) 21 :77 

Page 17 of 26 **77** 

|**Limitations**|Slight degradation on non-HS<br>class, error propagation from<br>auxiliary tasks, label noise in<br>datasets<br>Misclassification of neutral<br>class, sensitivity to sarcastic<br>context, data dependency,<br>preprocessing issues<br>Dependent on unsupervised<br>subword-phrase extraction<br>quality; no task-weighting<br>scheme implemented<br>Tasks have different<br>convergence rates; balancing<br>learning across tasks is<br>challenging<br>Underutilized high-resolution<br>features and contextual<br>dependencies in previous<br>works; real-time constraints;<br>current model lacks instance<br>segmentation and<br>compression for edge devices<br>Limited by partial annotations;<br>cannot optimize both tasks<br>simultaneously without<br>distillation<br>Complexity in balancing loss<br>components; scalability to<br>more tasks|
|---|---|
|Tasks|Hate Speech Detection,<br>Polarity Classification,<br>Emotion Classification<br>Sentiment Analysis,<br>Sarcasm Detection<br>Text classification,<br>Subword-phrase<br>recognition<br>Facial landmark<br>detection (main), head<br>pose estimation, facial<br>attribute inference,<br>expression<br>recognition, gender,<br>and age estimation<br>Object detection,<br>drivable area<br>segmentation, lane<br>detection<br>Object detection,<br>Semantic segmentation<br>Traffic density<br>prediction (main),<br>traffic speed prediction<br>(auxiliary)|
|Architecture|Transformer-based BETO<br>Bi-LSTM+Task-Specific<br>Perceptrons<br>Pretrained NLM+<br>Task-specific FC layers<br>Task-Constrained Deep CNN<br>(TCDCN)<br>YOLOPv3 (based on YOLOv7,<br>encoder–decoder with<br>task-specific heads)<br>Shared encoder+task-specific<br>heads<br>Multiple PINNs with shared<br>architecture and adaptive<br>parameter transfer|
|eter<br>ng<br>Soft|✓<br>–<br>–<br>–<br>✓<br>✓<br>✓|
|Param<br>Shari<br>Hard|✓<br>✓<br>✓<br>✓<br> <br>✓<br>✓<br>✓|
|Approach|Multi-Task<br>Learning using<br>BETO<br>Deep Multi-Task<br>Learning<br>Multi-task learning<br>with<br>subword-phrase<br>extraction<br>Task-Constrained<br>Learning for<br>Facial Landmark<br>Detection<br>YOLOPv3,<br>Intelligent<br>Driving (Traffic<br>Object Detection,<br>Drivable Area<br>Segmentation,<br>Lane Detection)<br>MTL with<br>knowledge<br>distillation<br>Multi-Task<br>Optimization<br>(MTO) for<br>training Physics-<br>Informed Neural<br>Networks<br>(PINNs)|
|Field/Domain|NLP<br>NLP<br>NLP/Text Classification<br>Computer Vision<br>Intelligent Driving/Computer<br>Vision<br>Computer Vision<br>Traffic Prediction/physics|
|Year|2021<br>2023<br>2022<br>2014<br>2024<br>2023<br>2023|
|Ref|[34]<br>[35]<br>[36]<br>[37]<br>[38]<br>[39]<br>[40]|



123 

**77** Page 18 of 26 

International Journal of Data Science and Analytics (2026) 21 :77 

|**Limitations**|Focused on constant pressure<br>production;<br>Computational complexity;<br>performance sensitive to<br>noise distribution in NCE<br>training<br>Requires careful scheduling<br>Data<br>imbalance<br>across<br>lan-<br>guages<br>Performance drop in<br>high-resource languages<br>- Weight coefficients set<br>empirically<br>- Not fully optimal<br>- Module suitability (fusion)<br>not validated against other<br>attention types<br>- Overfitting on small labeled<br>data<br>- Limited impact from<br>swapping pretrained models<br>- No overall rating supervision<br>- Lacks personalized utility<br>modeling<br>- Limited patient data (no<br>symptoms/medical history)<br>Meant for clinician use only|
|---|---|
|Tasks|Coupled prediction of<br>fluid and structure<br>fields<br>Molecular property<br>prediction (382 tasks)<br>- Translation (bitext)<br>- MLM (masked<br>language modeling)<br>- DAE (denoising<br>autoencoding)<br>- Tumor mask<br>segmentation<br>- Distance transform<br>estimation<br>- DTA (Drug-Target<br>Affinity) prediction<br>- Pretraining<br>(protein/drug)<br>- Predict satisfaction<br>scores (multicriteria)<br>- Medicine<br>recommendation|
|Architecture|MTL-PINN Soft-shared MTL<br>architecture<br>Soft-sharing via Structured<br>Graph Neural Network<br>(SGNN) and Energy-Based<br>Model (EBM)<br>Transformer-based MNMT<br>Modified V-Net with two<br>decoders+multi-depth<br>fusion module<br>Protein encoder (Transformer),<br>Drug encoder Graph<br>Convolutional Network<br>(GCN), DTA predictor, MTL<br>framework with dual<br>adaptation<br>Multivariate Bayesian model<br>with shared priors; Gibbs<br>sampling; Latent variable<br>ordinal probit model|
|eter<br>g<br>Soft|✓<br>✓<br>–<br>✓<br>✓<br>✓|
|Param<br>Sharin<br>Hard|–<br>–<br>✓<br>✓<br>✓<br>✓|
|Approach|Physics-Informed<br>Neural Networks<br>(PINNs) utilizing<br>MTL<br>frameworks to<br>fluid–structure<br>interaction<br>problems in<br>porous media<br>SGNN-EBM<br>(Structured GNN<br>+Energy-Based<br>Model)<br>Multi-task<br>Learning with<br>task and data<br>scheduling<br>Deep Multi-Task<br>Learning with<br>weighted losses<br>Multi-task learning<br>+self-supervised<br>pretraining<br>(protein+drug)<br>with dual<br>adaptation<br>mechanism<br>Bayesian<br>Multi-task<br>Learning with<br>LASSO for<br>ordinal<br>regression|
|Field/Domain|Physics<br>Drug Discovery/biochemistry<br>NLP/Multilingual Neural Machin<br>Translation<br>Medical Imaging/MRI<br>Segmentation<br>Bioinformatics/Drug Discovery<br>Healthcare/Recommender Systems|
|Year|2024<br>2022<br>2020<br>2021<br>2022<br>2023|
|Ref|[70]<br>[41]<br>[42]<br>[43]<br>[71]<br>[73]|



123 

International Journal of Data Science and Analytics (2026) 21 :77 

Page 19 of 26 **77** 

|**Limitations**|- First MTL for power grid<br>faults<br>- Outperforms STL &<br>traditional methods<br>- Noise resilient up to 40dB<br>- Fast inference (< 1 cycle),<br>fewer trainable parameters<br>- First model to combine<br>anomaly detection+<br>prediction and forecasting in<br>one framework<br>- New public dataset created<br>for heterogeneous building<br>tasks<br>- Load forecasting error<br>reduced by ~60% via MTL<br>- MTL benefits STL via<br>correlated learning<br>- Training difficulty &<br>resource consumption varies<br>across tasks<br>- Larger models̸ =better<br>accuracy<br>- Transformer-based model<br>uses 5x more memory than<br>CNN<br>Limited impact of<br>wearable-only features;<br>generalizability beyond<br>college students not explored<br>Depth Estimation shows low<br>inter-task correlation (ITC),<br>reducing joint performance;<br>high complexity in<br>optimizing MTL<br>configuration|
|---|---|
|Tasks|1. Fault Classification<br>(multi-class)<br>2. Fault Localization<br>(regression)<br>1. Electricity Load<br>Forecasting<br>(regression)<br>2. Air Temperature<br>Forecasting<br>(regression)<br>3. Energy Anomaly<br>Detection<br>(classification)<br>4. Energy Anomaly<br>Prediction<br>(classification)<br>Predict Depression,<br>Anxiety, Stress<br>Object Detection, Lane<br>Detection, Drivable<br>Area Segmentation,<br>Depth Estimation|
|Architecture|- CNN blocks (Conv1D+BN<br>+tanh)<br>- Shared encoder<br>- Two heads: softmax<br>(classification) and linear<br>(regression)<br>- Multi-gate mixture-of-experts<br>with sparse-constrained<br>transformer<br>- Separate heads for each task<br>- Shared encoder with<br>attention-based fusion layers<br>- Inter-task weighting strategy<br>Shared base encoder+<br>task-specific heads<br>Shared Backbone+<br>Task-Specific Subnets (UNet,<br>FPN, Transformer)|
|Parameter<br>Sharing<br>Hard<br>Soft|✓<br>–<br>✓<br>✓<br>–<br>✓<br>✓<br>✓|
|Approach|CNN-based<br>Multi-task<br>Learning with<br>shared encoder,<br>separate heads<br>Multi-task<br>Learning with<br>Mixture-of-<br>Experts+<br>Self-attention<br>(Sparse<br>Transformer)+<br>Multi-gate<br>Feature Fusion<br>MTL for Mental<br>Health<br>(Depression,<br>Stress, Anxiety)<br>MTL with MDO<br>(Decision and<br>Optimization)|
|Ref<br>Year<br>Field/Domain|[31]<br>2024<br>Power Systems/Smart Grids<br>[32]<br>2024<br>Smart Buildings/Energy<br>[33]<br>2024<br>Mental Health/psychology<br>[62]<br>2023<br>Autonomous Driving|



123 

**77** Page 20 of 26 

International Journal of Data Science and Analytics (2026) 21 :77 

|**Limitations**|High computational cost; long<br>training and tuning time; risk<br>of overfitting with large<br>synthetic sample size; quality<br>and diversity of generated<br>data require improvement;<br>Limited interpretability; no<br>attention mechanism;<br>requires future work on<br>identifying spatially<br>informative regions;<br>computational complexity<br>due to temporal modeling<br>Focuses only on<br>assignment-related features;<br>does not consider content<br>understanding or external<br>factors; model complexity<br>may increase with task<br>expansion<br>Ignores psychological/family<br>factors; assumes prior course<br>info is sufficient; limited<br>dataset availability;<br>generalization across<br>institutions untested<br>Limited to discrete actions;<br>relies on sampling-based<br>motion planner; hard to<br>generalize to dexterous<br>continuous control (e.g.,<br>multi-fingered hands)|
|---|---|
|Tasks|1. Credit scoring with<br>imbalanced data<br>2. Synthetic data<br>generation<br>1. Temperature<br>prediction<br>2. Humidity estimation<br>3. Visibility estimation<br>4. Wind speed<br>estimation<br>1. Predict student<br>performance (main<br>task) 2. Predict<br>knowledge point<br>mastery (auxiliary<br>task)<br>1. Predict student<br>performance in<br>upcoming courses<br>2. Academic early<br>warning<br>1. Multi-task 6-DoF<br>robotic manipulation<br>2. Language-<br>conditioned behavior<br>cloning on 18<br>RLBench & 7 real<br>tasks|
|Architecture|CWGAN-GP generator+MTL<br>with shared MLP layers<br>CNN for feature extraction+<br>2D-RNN for temporal<br>modeling<br>Multi-layer LSTM with shared<br>parameters+Attention<br>MIML with iterative soft<br>feature fusion<br>Perceiver Transformer<br>(PERACT)|
|eter<br>ng<br>Soft|-<br>-<br>✓<br>✓<br>✓|
|Param<br>Shari<br>Hard|✓<br>✓<br>-<br>✓<br>✓|
|Approach|CWGAN-GP+<br>Multi-Task<br>Learning<br>Multi-Task<br>Learning+<br>2D-RNN<br>Multi-task learning<br>with multi-layer<br>LSTM and<br>attention<br>mechanism<br>Multi-task MIML<br>(MIML-Circle)<br>Multi-task<br>Transformer|
|Field/Domain|Finance<br>Climate science<br>Educational Data Mining<br>Educational Data Mining<br>Robotics/Manipulation|
|Ref<br>Year|[63]<br>2022<br>[64]<br>2021<br>[65]<br>2019<br>[67]<br>2020<br>[66]<br>2022|



123 

International Journal of Data Science and Analytics (2026) 21 :77 

Page 21 of 26 **77** 

trained faster than training separate models for each individual task. This can greatly improve efficiency, in large-scale applications. 

Finally, MTL enables positive knowledge transfer [49]. When tasks are interdependent, knowledge gained from one task can directly enhance the performance of other tasks [50], making MTL particularly valuable in scenarios where some tasks may have limited data. 

hundreds of tasks requires sophisticated strategies for coordinating memory usage, model architecture, and optimization processes, highlighting a major limitation when applying MTL to large-scale real-world scenarios [98]. 

Finally, and to the best of our knowledge, shared parameters may not always benefit all tasks equally, as tasks with different objectives may perform better with separate models than with a shared model. 

### **8.3 Future trends and approaches** 

### **8.2 Disadvantages and limitations of MTL** 

Despite its potential benefits, MTL presents several drawbacks and limitations. One major challenge is negative transfer, which occurs when tasks are too different or conflicting, causing the shared learning process to affect the performance when trained jointly [51, 52, 56]. 

Task imbalance is another problem, where some tasks may dominate the learning process due to larger datasets, stronger gradients, or simpler objectives, leading to poor performance on low-resource or complex tasks. [53]. 

Furthermore, Architectural Complexity, designing suitable architectures that allow for both shared and task-specific components, is non-trivial, especially when tasks differ in modality, structure, or scale [10]. 

Another one of the most significant limitations of MTL is the sensitivity of hyperparameters. Unlike single-task models, multi-task learning systems often require fine-tuning of various hyperparameters, such as task-specific weights, learning rates, and optimization strategies [94, 95]. This additional layer of complexity can significantly increase the effort required during the model development phase, as finding the optimal balance between tasks is critical to avoid poor performance or negative transfer. 

Another notable challenge in MTL is the task selection dilemma. Choosing which tasks to jointly train on is a significant and complex problem [96]. When tasks are improperly grouped, either because they are too different or because their objectives conflict, the benefits of joint learning may be diminished or even reversed. This can degrade the overall performanceof themodel, as interferencebetweenincompatible tasks undermines the effectiveness of knowledge transfer and generalization. 

Finally, as the number of tasks in MTL framework increases, the complexity of training increases significantly [97]. Managing multiple objectives simultaneously requires more computational resources and memory, especially when each task has distinct requirements or large datasets. This increased demand can slow down the training process and increase the difficulty of optimization. Furthermore, scheduling updates effectively across tasks and ensuring balanced learning becomes critical to avoid task overlapping and reduced efficiency. Scaling MTL to include dozens or even 

There are several promising directions emerging in the field of MTL research: 

Recent advances in multi-task learning have introduced dynamicapproaches[90, 100, 101]thatadjusttheimportance of tasks during training. Techniques such as attention mechanisms and reinforcement learning are increasingly used to adaptively evaluate task contributions based on their importance and learning progress. These methods enable the model to prioritize the most useful or least useful tasks, while reducing the importance of tasks that may cause interference, mitigating the risk of negative transfer. By dynamically steering information gradients or allocating resources according to task-specific signals, these methods enhance learning efficiency and improve overall task performance. 

Meta-learning, often referred to as “learning to learn,” is gaining increasing traction as an effective tool for enhancing multi-task learning [102]. In the context of multi-task learning, meta-learning techniques are used to automatically discover optimal strategies for sharing representations across tasks or for rapidly adapting to new task combinations. Instead of relying on handcrafted sharing schemes, metalearning algorithms can learn how to balance shared and task-specific components based on experience gained across tasks [103]. This enables more flexible and scalable multitask models that can better generalize to unseen or evolving tasks, making meta-learning a promising direction for future multi-task learning research. 

Large language models (LLMs), such as GPT and T5, inherently demonstrate strong multi-task learning capabilities thanks to their pretraining on diverse and large-scale datasets [104]. These models can perform a wide range of tasks with minimal or no fine-tuning, making them ideal foundations for MTL frameworks. By leveraging the rich contextual representations and transfer learning capabilities of LLMs, researchers can build more general, versatile, and efficient architectures. Combining MTL techniques with LLMs also opens the door to multi-stage (zero-dose) and multi-stage (low-dose) learning, enabling models to adapt to new tasks with limited data while maintaining high performance across multiple domains. 

Finally, with the increasing use of MTL systems in highstakes domains, such as healthcare, finance, and autonomous 

123 

**77** Page 22 of 26 

International Journal of Data Science and Analytics (2026) 21 :77 

systems, the demand for interpretable and reliable models such as [105, 106] is growing. In these environments, decision transparency, robustness, and accountability are essential, prompting researchers to develop MTL approaches that provide insights into task interactions, representation sharing,andpredictionfoundations.Enhancinginterpretability not only enhances user confidence but also aids in debugging and regulatory compliance. Consequently, interpretable MTL is becoming a critical focus in building reliable AI systems for practical applications. 

To conclude, future research in multi-task learning should aimtoaddresscurrentlimitationswhileexploringnewopportunities for advancement. Continued innovation will lead to more robust, adaptable, and efficient learning frameworks across a variety of domains. 

**Author contributions** The first author, Mahmoud Abdelsamie, led the conceptualization and organization of the review, conducted the literature analysis, and drafted the initial manuscript. All co-authors contributed to refining the scope of the review, participated in the interpretation and critical discussion of relevant studies, and provided substantive revisions throughout the writing process. All authors reviewed and approved the final version of the manuscript and agree to be accountable for all aspects of the work, ensuring its accuracy and integrity. 

**Funding** Open access funding provided by The Science, Technology & Innovation Funding Authority (STDF) in cooperation with The Egyptian Knowledge Bank (EKB). 

**Data availability** No datasets were generated or analysed during the current study. 

### **Declarations** 

**Competing interests** The authors declare no competing interests. 

## **9 Conclusion** 

In this review, we examine the most recent applications and developments of MTL across various domains, with a particular focus on its effectiveness and challenges. Multitask learning has emerged as a powerful paradigm in machine learning, enabling models to learn multiple related tasks simultaneously, resulting in improved generalization, reduced training time, and the ability to exploit relationships between tasks. The reviewed studies highlight the broad application of MTL in domains such as healthcare, agriculture, natural language processing, computer vision, and others. Despite these promising results, several challenges remain. Many multi-task learning frameworks struggle to deal with the task overlap, where optimization of one task can negatively impact learning of another. Additionally, the computational resources required to train multi-task models can be significant, especially when dealing with large and complex datasets. Furthermore, while some multi-task learning approaches leverage shared knowledge effectively, others struggle to manage task dependencies, which can limit their overall performance. Future research directions should focus on improving interconnections between tasks, enhancing model interpretability, and reducing reliance on large, annotated datasets. In conclusion, multi-task learning holds great promise as a fundamental approach to solving complex and multifaceted problems across diverse domains. As the field continues to evolve, overcoming existing challenges will pave the way for more efficient, effective, and scalable solutions. 

**Acknowledgements** We would like to express our sincere gratitude to our supervisors for their valuable guidance, insightful comments, and continuous support. Their expertise and encouragement have played an instrumental role in shaping this work. We also extend our appreciation to the Science, Technology & Innovation Funding Authority (STDF) for their financial support, which has been pivotal in enabling this research. 

**Open Access** This article is licensed under a Creative Commons Attribution 4.0 International License, which permits use, sharing, adaptation, distribution and reproduction in any medium or format, as long as you give appropriate credit to the original author(s) and the source, provide a link to the Creative Commons licence, and indicate if changes were made. The images or other third party material in this article are included in the article’s Creative Commons licence, unless indicated otherwise in a credit line to the material. If material is not included in the article’s Creative Commons licence and your intended use is not permitted by statutory regulation or exceeds the permitted use, you will need to obtain permission directly from the copyright holder. To view a copy of this licence, visit http://creativecommons.org/licenses/by/4.0/. 

## **References** 

1. Mohammed, A., Kora, R.: A comprehensive review on ensemble deep learning: opportunities and challenges. J. King Saud Univ. - Comput. Inf. Sci. **35** (2), 757–774 (2023). https://doi.org/10.1016/ j.jksuci.2023.01.014 

2. Zhang, Y., Yang, Q.: A survey on multi-task learning. IEEE Trans. Knowl. Data Eng. **34** (12), 5586–5609 (2021). https://doi.org/10. 1109/tkde.2021.3070203 

3. Zhang, Y., and Yeung, D.: Learning high-order task relationships in multi-task learning. International Joint Conference on Artificial Intelligence, 1917–1923 (2013). https://repository.ust.hk/ir/bitstr eam/1783.1-60877/1/283.pdf 

4. Sener, O., Koltun, V.: Multi-task learning as multi-objective optimization. Neural Information Processing Systems **31** , 527–538 (2018) 

5. Kollias, D., Sharmanska, V., and Zafeiriou, S.: Distribution Matching for Heterogeneous Multi-Task Learning: a Large-scale Face Study. arXiv (Cornell University) (2021). https://doi.org/10. 48550/arxiv.2105.03790 

6. Caruana, R.: Multitask learning. Mach. Learn. **28** (1), 41–75 (1997) 

7. Marquet, T., and Oswald, E.: A comparison of multi-task learning and single-task learning approaches. in lecture notes in computer science pp. 121–138 (2023) https://doi.org/10.1007/978-3-03141181-6_7 

8. Nunes, M., Gerding, E., McGroarty, F., Niranjan, M.: A comparison of multitask and single task learning with artificial neural 

123 

International Journal of Data Science and Analytics (2026) 21 :77 

Page 23 of 26 **77** 

   - networks for yield curve forecasting. Expert Syst. Appl. **119** , 362–375 (2018). https://doi.org/10.1016/j.eswa.2018.11.012 

9. Banayeeanzade, M., Soltanolkotabi, M., and Rostami, M.: Theoretical insights into overparameterized models in multi-task and replay-based continual learning. arXiv (Cornell University) (2024). https://doi.org/10.48550/arxiv.2408.16939 

10. Ruder, S.: An Overview of Multi-Task Learning in Deep Neural Networks. arXiv preprint arXiv:1706.05098 (2017) 

11. Standley, T., Zamir, A., Chen, D., Guibas, L., Malik, J., and Savarese, S.: Which tasks should be learned together in multi-task learning? In Proceedings of the 37th International Conference on Machine Learning 119: 9120–9132 (2020) 

12. Collobert, R., and Weston, J.: A unified architecture for natural language processing: Deep neural networks with multitask learning. In Proceedings of the 25th International Conference on Machine Learning pp. 160–167 (2008) 

13. Abdelsamie, M.M., Azab, S.S., Hefny, H.A.: A comprehensive review on Arabic offensive language and hate speech detection on social media: methods, challenges and solutions. Soc. Netw. Anal. Min. (2024). https://doi.org/10.1007/s13278-024-01258-1 

14. Misra, I., Shrivastava, A., Gupta, A., and Hebert, M.: Crossstitch networks for multi-task learning. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition pp. 3994–4003 (2016) 

15. Yu, J., Dai, Y., Liu, X., Huang, J., Shen, Y., Zhang, K., Zhou, R., Adhikarla, E., Ye, W., Liu, Y., Kong, Z., Zhang, K., Yin, Y., Namboodiri, V., Davison, B. D., Moore, J. H., and Chen, Y.: Unleashingthepowerofmulti-tasklearning:acomprehensivesurvey spanning traditional, deep, and pretrained foundation model eras. arXiv (Cornell University) (2024). https://doi.org/10.48550/ arxiv.2404.18961 

16. Guo, H., Pasunuru, R., and Bansal, M.: AutoSeM: automatic task selection and mixing in multi-task learning. arXiv (Cornell University) (2019). https://doi.org/10.48550/arxiv.1904.04153 

17. Mormont, R., Geurts, P., Maree, R.: Multi-task pre-training of deep neural networks for digital pathology. IEEE J. Biomed. Health Inform. **25** (2), 412–421 (2020). https://doi.org/10.1109/ jbhi.2020.2992878 

18. Graham, S., Vu, Q.D., Jahanifar, M., Raza, S.E.A., Minhas, F., Snead, D., Rajpoot, N.: One model is all you need: multi-task learning enables simultaneous histology image segmentation and classification. Med. Image Anal. **83** , 102685 (2022). https://doi. org/10.1016/j.media.2022.102685 

19. Upadhyay, R., Phlypo, R., Saini, R., and Liwicki, M.: Sharing to learn and learning to share -- fitting together Meta-learning, multi-task learning, and transfer learning: a meta review. arXiv (Cornell University) (2021). https://doi.org/10.48550/arxiv.2111. 12146 

20. Duong, L., Cohn, T., Bird, S., Cook, P.: Low resource dependencyparsing:cross-lingualparametersharinginaneuralnetwork parser (2015). https://doi.org/10.3115/v1/p15-2139 

21. Yang, Y., Hospedales, T.M.,: Trace norm regularized deep multitask learning. arXiv preprint arXiv:1606.04038 (2016) 

22. Zhuang, F., Qi, Z., Duan, K., Xi, D., Zhu, Y., Zhu, H., Xiong, H., and He, Q.: A comprehensive survey on transfer learning. arXiv (Cornell University) (2019). https://doi.org/10.48550/arxiv.1911. 02685 

23. Kim, S., Purdie, T.G., and McIntosh, C.: Cross-task attention network: improving multi-task learning for medical imaging applications. In Lecture notes in computer science pp. 119–128 (2023). https://doi.org/10.1007/978-3-031-47401-9_12 

24. Ding, D.Y., Simpson, C., Pfohl, S., Kale, D.C., Jung, K., Shah, N.H.: The effectiveness of multitask learning for phenotyping with electronic health records data. Biocomputing **24** , 18–29 (2018). https://doi.org/10.1142/9789813279827_0003 

25. Zhang, Y., Wu, X., Fang, Q., Qian, S., Xu, C.: Knowledgeenhanced attributed multi-task learning for medicine recommendation. ACM Transactions on Office Information Systems **41** (1), 1–24 (2022). https://doi.org/10.1145/3527662 

26. Watanabe, T., Tamura, A., Ninomiya, T., Makino, T., and Iwakura, T.: Multi-Task learning for chemical named entity recognition with chemical compound paraphrasing (2019). https://doi.org/10. 18653/v1/d19-1648 

27. Yang, A., Su, Y., Wang, Z., Jin, S., Ren, J., Zhang, X., Shen, W., Clark, J.H.: A multi-task deep learning neural network for predicting flammability-related properties from molecular structures. Green Chem. **23** (12), 4451–4465 (2021). https://doi.org/10. 1039/d1gc00331c 

28. Bertsimas, D., and Ma, Y.: M3H: multimodal multitask machine learning for healthcare. arXiv (Cornell University) (2024). https:// doi.org/10.48550/arxiv.2404.18975 

29. Chang,C.,Lin,J.,Chang,J.,Huang,Y.,Lai,M.,Chang,Y.:Hybrid deep neural networks with multi-tasking for rice yield prediction using remote sensing data. Agriculture **14** (4), 513 (2024). https:// doi.org/10.3390/agriculture14040513 

30. Tang, X., Zhang, J., He, Y., Zhang, X., Lin, Z., Partarrieu, S., Hanna, E.B., Ren, Z., Shen, H., Yang, Y., Wang, X., Li, N., Ding, J., Liu, J.: Explainable multi-task learning for multi-modality biological data analysis. Nat. Commun. (2023). https://doi.org/10. 1038/s41467-023-37477-x 

31. Bhardwaj, D., Londhe, N.D., Raj, R.: Fault-MTL: a multi-task deep learning approach for simultaneous fault classification and localization in power systems. J. Control Autom. Electr. Syst. **35** (5), 884–898 (2024). https://doi.org/10.1007/s40313-024-01 119-4 

32. Wang, R., Rayhana, R., Gholami, M., Herrera, O.E., Liu, Z., Mérida, W.: Multi-task deep learning for large-scale buildings energy management. Energy Build. **307** , 113964 (2024). https:// doi.org/10.1016/j.enbuild.2024.113964 

33. Saylam, B.,<sup>˙</sup> Incel, Ö.D.: Multitask learning for mental health: depression, anxiety, stress (DAS) using wearables. Diagnostics **14** (5), 501 (2024). https://doi.org/10.3390/diagnostics14050501 

34. Plaza-Del-Arco, F.M., Molina-Gonzalez, M.D., Urena-Lopez, L.A., Martin-Valdivia, M.T.: A multi-task learning approach to hate speech detection leveraging sentiment analysis. IEEE Access **9** , 112478–112489 (2021). https://doi.org/10.1109/access.2021. 3103697 

35. Tan, Y.Y., Chow, C., Kanesan, J., Chuah, J.H., Lim, Y.: Sentiment analysis and sarcasm detection using deep multi-task learning. Wirel. Pers. Commun. **129** (3), 2213–2237 (2023). https://doi.org/ 10.1007/s11277-023-10235-4 

36. Kimura, Y., Komamizu, T., Hatano, K.: Multi-task learning-based text classification with subword-phrase extraction. Proceedings of the 11th International Symposium on Information and Communication Technology **2007** , 23–30 (2022). https://doi.org/10.1145/ 3568562.3568635 

37. Zhang, Z., Luo, P., Loy, C. C., and Tang, X.: Facial landmark detection by deep multi-task learning. In Lecture notes in computer science pp. 94–108 (2014). https://doi.org/10.1007/978-3319-10599-4_7 

38. Zhan, J., Liu, J., Wu, Y., Guo, C.: Multi-task visual perception for object detection and semantic segmentation in intelligent driving. Remote Sens. **16** (10), 1774 (2024). https://doi.org/10.3390/rs16 101774 

39. Lê, H., and Pham, M.: Data exploitation: multi-task learning of object detection and semantic segmentation on partially annotated data. arXiv (Cornell University) (2023). https://doi.org/10.48550/ arxiv.2311.04040 

40. Wang, B., Qin, A. K., Shafiei, S., Dia, H., Mihaita, A., and Grzybowska, H.: Training physics-informed neural networks via multi-task optimization for traffic density prediction. arXiv 

123 

**77** Page 24 of 26 

International Journal of Data Science and Analytics (2026) 21 :77 

(Cornell University) (2023). https://doi.org/10.48550/arxiv.2307. 03920 

41. Liu, S., Qu, M., Zhang, Z., Cai, H., and Tang, J.: Structured multitask learning for molecular property prediction. arXiv (Cornell University) (2022). https://doi.org/10.48550/arxiv.2203.04695 

42. Wang, Y., Zhai, C., and Awadalla, H.H.: Multi-task learning for multilingual neural machine translation. arXiv (Cornell University) (2020). https://doi.org/10.48550/arxiv.2010.02523 

43. Huang, H., Yang, G., Zhang, W., Xu, X., Yang, W., Jiang, W., Lai, X.: A Deep Multi-Task Learning Framework for Brain Tumor Segmentation. Front. Oncol. **11** , 690244 (2021). https://doi.org/ 10.3389/fonc.2021.690244 

44. Wang, X., Yu, H., Meng, X., Cao, H., Zhang, H., Sun, H., Liu, X., Hu, C.: MTL-TRANSFER: leveraging multi-task learning and transferred knowledge for improving fault localization and program repair. ACM Trans. Softw. Eng. Methodol. **33** (6), 1–31 (2024). https://doi.org/10.1145/3654441 

45. Alzahrani, M., Wang, Q., Liao, W., Chen, X., and Yu, W.: Survey on multi-task learning in smart transportation. IEEE Access, 1 (2024). https://doi.org/10.1109/access.2024.3355034 

46. Aljoufi, R. and Lasebae, A.: Multi-task learning for intrusion detection and analysis of computer network traffic, E3S Web of Conferences,229:01057(2021). https://doi.org/10.1051/e3sconf/ 202122901057 

47. Ibrahim, S., Catal, C., Kacem, T.: The use of multi-task learning in cybersecurity applications: a systematic literature review. Neural Comput. Appl. (2024). https://doi.org/10.1007/s00521-024-10 436-3 

48. Kung, P., Yin, S., Chen, Y., Yang, T., and Chen, Y.: Efficient multi-task auxiliary learning: selecting auxiliary data by feature similarity. Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, 416–428 (2021). https://doi.org/10.18653/v1/2021.emnlp-main.34 

49. Graffeuille, O., Koh, Y.S., Wicker, J., and Lehmann, M.: Enabling asymmetric knowledge transfer in multi-task learning with selfauxiliaries. arXiv (Cornell University) (2024). https://doi.org/10. 48550/arxiv.2410.15875 

50. Gao, F., Yoon, H., Wu, T., Chu, X.: A feature transfer enabled multi-task deep learning model on medical imaging. Expert Syst. Appl. **143** , 112957 (2019). https://doi.org/10.1016/j.eswa.2019. 112957 

51. Lakkapragada, A., Sleiman, E., Surabhi, S., and Wall, D.P.: Mitigating Negative transfer in multi-task learning with exponential moving average loss weighting strategies. arXiv (Cornell University) (2022). https://doi.org/10.48550/arxiv.2211.12999 

52. Meng, Z., Yao, X., and Sun, L.: Multi-task distillation: towards mitigating the negative transfer in multi-task learning. 2022 IEEE International Conference on Image Processing (ICIP). (2021). https://doi.org/10.1109/icip42928.2021.9506618 

53. Mao, Y., Wang, Z., Liu, W., Lin, X., and Xie, P.: Metaweighting: learning to weight tasks in multi-task learning. findings of the association for computational linguistics: ACL 2022, 3436–3448 (2022). https://doi.org/10.18653/v1/2022.findings-acl.271 

54. Strezoski, G., Nanne, V.N., and Worring, M.: Many task learning with task routing. arXiv (Cornell University) (2019). https://doi. org/10.48550/arxiv.1903.12117 

55. Chen, S., Zhang, Y., and Yang, Q.: Multi-task learning in natural language processing: an overview. arXiv (Cornell University) (2021). https://doi.org/10.48550/arxiv.2109.09138 

56. Fontana, M., Spratling, M., and Shi, M.: When multi-task learning meets partial supervision: a computer vision review. arXiv (Cornell University) (2023).https://doi.org/10.48550/arxiv.2307. 14382 

57. Kasukurthi, A., Davuluri, R.L.: AMTLUS: attention-guided multi-task learning with uncertainty estimation in skin lesion 

segmentation and classification. Multimed. Tools Appl. (2024). https://doi.org/10.1007/s11042-024-19360-z 

58. Wang, Z., He, Z., Shah, M., Zhang, T., Fan, D., Zhang, W.: Network-based multi-task learning models for biomarker selection and cancer outcome prediction. Bioinformatics **36** (6),1814–1822(2019). https://doi.org/10.1093/bioinformatics/ btz809 

59. Nguyen, T.N.Q., García-Rudolph, A., Saurí, J., Kelleher, J.D.: Multi-task learning for predicting quality-of-life and independence in activities of daily living after stroke: a proof-of-concept study. Front. Neurol. (2024). https://doi.org/10.3389/fneur.2024. 1449234 

60. Mesrabadi, H.A., Faez, K., Pirgazi, J.: Multi source deep learning method for drug-protein interaction prediction using k-mers and chaos game representation. Chemometr. Intell. Lab. Syst. **246** , 105065 (2024). https://doi.org/10.1016/j.chemolab.2024.105065 

61. Liu,J.,Zhang,Z.,Fan,X.,Zhang,Y.,Wang,J.,Zhou,K.,Liang,S., Yu, X., Zhang, W.: Power system load forecasting using mobility optimization and multi-task learning in COVID-19. Appl. Energy **310** , 118303 (2022). https://doi.org/10.1016/j.apenergy.2021.11 8303 

62. Jun, W., Son, M., Yoo, J., Lee, S.: Optimal configuration of multi-task learning for autonomous driving. Sensors **23** (24), 9729 (2023). https://doi.org/10.3390/s23249729 

63. Kang, Y., Chen, L., Jia, N., Wei, W., Deng, J., Qian, H.: A CWGAN-GP-based multi-task learning model for consumer credit scoring. Expert Syst. Appl. **206** , 117650 (2022). https:// doi.org/10.1016/j.eswa.2022.117650 

64. Chu, W., Liang, Y., Ho, K.: Visual weather property prediction by multi-task learning and two-dimensional RNNs. Atmosphere **12** (5), 584 (2021). https://doi.org/10.3390/atmos12050584 

65. Qu, S., Li, K., Wu, B., Zhang, X., Zhu, K.: Predicting student performance and deficiency in mastering knowledge points in MOOCs using multi-task learning. Entropy **21** (12), 1216 (2019). https://doi.org/10.3390/e21121216 

66. Shridhar, M., Manuelli, L., and Fox, D.:Perceiver-actor: a multitask transformer for robotic manipulation. arXiv (Cornell University) (2022). https://doi.org/10.48550/arxiv.2209.05451 

67. Ma, Y., Cui, C., Yu, J., Guo, J., Yang, G., Yin, Y.: Multi-task MIML learning for pre-course student performance prediction. Front. Comput. Sci. (2020). https://doi.org/10.1007/s11704-0199062-8 

68. Feng, Q., Chen, S.: Learning multi-tasks with inconsistent labels by using auxiliary big task. Front. Comput. Sci. (2023). https:// doi.org/10.1007/s11704-022-2251-x 

69. Guo, W., Zhuang, F., Zhang, X., Tong, Y., Dong, J.: A comprehensive survey of federated transfer learning: challenges, methods and applications. Front. Comput. Sci. (2024). https://doi.org/10. 1007/s11704-024-40065-x 

70. Zhao, Y., Yao, J., Wang, J., Xie, X., Ablameyko, S.V.: Multi-task learning enhanced physics-informed neural network for solving fluid-structure interaction equations. 2020 ieee 4th information technology, networking. Electronic and Automation Control Conference (ITNEC) **7** , 591–595 (2024). https://doi.org/10.1109/itne c60942.2024.10732993 

71. Lin, S., Shi, C., Chen, J.: GeneralizedDTA: combining pretraining and multi-task learning to predict drug-target binding affinity for unknown drug discovery. BMC Bioinformatics (2022). https://doi.org/10.1186/s12859-022-04905-6 

72. Zhang, Y., Wang, X., Liu, T., Wang, R., Li, Y., Xue, Q., Yang, P.: Sustainable fertilisation management via tensor multi-task learning using multi-dimensional agricultural data. J. Ind. Inf. Integr. **34** , 100461 (2023). https://doi.org/10.1016/j.jii.2023.100461 

73. Cheng, Y., Xia, Y., Wang, X.: Bayesian multitask learning for medicine recommendation based on online patient reviews. Bioinformatics (2023). https://doi.org/10.1093/bioinformatics/btad491 

123 

International Journal of Data Science and Analytics (2026) 21 :77 

Page 25 of 26 **77** 

74. Crawshaw, M.: Multi-task learning with deep neural networks: a survey. arXiv (Cornell University) (2020). https://doi.org/10.48 550/arxiv.2009.09796 

75. Liu, S., Johns, E., Davison, A.J.: End-To-End Multi-Task Learning With Attention. IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) **2022** , 1871–1880 (2019). https://doi.org/10.1109/cvpr.2019.00197 

76. Zheng, F., Deng, C., Sun, X., Jiang, X., Guo, X., Yu, Z., Huang, F., and Ji, R.: Pyramidal person re-identification via multi-loss dynamic training. arXiv (Cornell University) (2018). https://doi. org/10.48550/arxiv.1810.12193 

77. Vandenhende, S., Georgoulis, S., and Luc, V.G.: MTI-Net: multiscale task interaction networks for multi-task learning. arXiv (Cornell University) (2020). https://doi.org/10.48550/arxiv.2001. 06902 

78. Xu, D., Ouyang, W., Wang, X., and Sebe, N.: PAD-Net: multitasks guided prediction-and-distillation network for simultaneous depth estimation and scene parsing. arXiv (Cornell University) (2018). https://doi.org/10.48550/arxiv.1805.04409 

79. Heess, N., Wayne, G., Tassa, Y., Lillicrap, T., Riedmiller, M., and Silver, D.: Learning and transfer of modulated locomotor controllers. arXiv (Cornell University) (2016). https://doi.org/10. 48550/arxiv.1610.05182 

80. Devin, C., Gupta, A., Darrell, T., Abbeel, P., and Levine, S.: Learning modular neural network policies for multi-task and multi-robot transfer. arXiv (Cornell University) (2016). https:// doi.org/10.48550/arxiv.1609.07088 

81. Subhojeet, P., Priyanka, A.„ and Aman, H.,: Omninet: A unified architecture for multi-modal multi-task learning. arXiv preprint arXiv:1907.07804, (2019) 

82. Luo, S., Chu, V.W., Li, Z., et al.: Multi-task learning by hierarchical Dirichlet mixture model for sparse failure prediction. Int. J. Data Sci. Anal. **12** , 15–29 (2021). https://doi.org/10.1007/s4 1060-020-00219-z 

83. Zhao, J., Lv, W., Du, B., et al.: Deep multi-task learning with flexible and compact architecture search. Int. J. Data Sci. Anal. **15** , 187–199 (2023). https://doi.org/10.1007/s41060-021-00274-0 

84. Abdelsamie, M.M., Azab, S.S., Hefny, H.A.: The dialects gap: a multi-task learning approach for enhancing hate speech detection in Arabic dialects. Expert Syst. Appl. **295** , 128584 (2025). https:// doi.org/10.1016/j.eswa.2025.128584 

85. Lindsey, J. W., and Lippl, S.: Implicit regularization of multi-task learning and finetuning in overparameterized neural networks. arXiv (Cornell University) (2023). https://doi.org/10.48550/arxiv. 2310.02396 

86. Roy, A., Koutlis, C., Papadopoulos, S., Ntoutsi, E.: FairBranch: Mitigating bias transfer in fair Multi-task learning. International Joint Conference on Neural Networks (IJCNN) **33** , 1–8 (2024). https://doi.org/10.1109/ijcnn60899.2024.10651221 

87. Wang, S., Wang, Q., Gong, M.: Multi-task learning based network embedding. Front. Neurosci. **13** , 1387 (2020). https://doi.org/10. 3389/fnins.2019.01387 

88. Lai,W.,Xie,H.,Xu,G.,andLi,Q.:Multi-tasklearningwithLLMs for implicit sentiment analysis: data-level and Task-level automatic weight learning. arXiv (Cornell University) (2024). https:// doi.org/10.48550/arxiv.2412.09046 

89. Bensaoud, A., Kalita, J.: Deep multi-task learning for malware image classification. J. Inf. Secur. Appl. **64** , 103057 (2021). https://doi.org/10.1016/j.jisa.2021.103057 

90. Kasukurthi, A., Davuluri, R.L.: AMTLUS: attention-guided multi-task learning with uncertainty estimation in skin lesion segmentation and classification. Multimedia Tools Appl. **83** (37), 84885–84909 (2024). https://doi.org/10.1007/s11042024-19360-z 

framework for multi-task multi-source learning. IEEE J. Biomed. Health Inform. **28** (1), 502–513 (2023). https://doi.org/10.1109/ jbhi.2023.3327951 

92. Yuan, Y., Liu, J., Pan, X., Zhang, R., Su, W.: Heterogenous biological network multi-task learning model for ncRNA-disease-drug association prediction. Knowl. Based Syst. **300** , 112222 (2024). https://doi.org/10.1016/j.knosys.2024.112222 

93. Li, Q., Ren, C., Zhang, R., and Xu, Y. (2025). A multi-task learning-based approach for power system short-term voltage stability assessment with missing PMU data. IEEE Transactions on Automation Science and Engineering, 1.(2025). https://doi.org/ 10.1109/tase.2025.3551593 

94. Xin, D., Ghorbani, B., Garg, A., Firat, O., and Gilmer, J.: Do current multi-task optimization methods in deep learning even help?arXiv(CornellUniversity)(2022). https://doi.org/10.48550/ arxiv.2209.11379 

95. Peng, C., Yuan, M., and Wang, A.: An exploration of multi-task learning over minBERT (By Stanford CS224N DefaultProject). https://web.stanford.edu/class/cs224n/final-repo rts/256906365.pdf 

96. Fifty, C., Amid, E., Zhao, Z., Yu, T., Anil, R., and Finn, C.: Efficiently identifying task groupings for multi-task learning. arXiv (Cornell University) (2021). https://doi.org/10.48550/arxiv.2109. 04617 

97. He, X., Alesiani, F., and Shaker, A.: Efficient and scalable multitask regression on massive number of tasks. arXiv (Cornell University) (2018). https://doi.org/10.48550/arxiv.1811.05695 

98. Sherif, A., Abid, A., Elattar, M., ElHelw, M.: STG-MTL: scalable task grouping for multi-task learning using data maps. Mach. Learn. Sci. Technol. **5** (2), 025068 (2024). https://doi.org/10.1088/ 2632-2153/ad4e04 

99. Wang, X., Zhang, M., Chen, B., Wei, D., Shao, Y.: Dynamic weighted multitask learning and contrastive learning for multimodal sentiment analysis. Electronics **12** (13), 2986 (2023). https://doi.org/10.3390/electronics12132986 

100. Bohn, C., Freeman, I., Tercan, H., and Meisen, T.: Task weighting through gradient projection for multitask learning. arXiv (Cornell University) (2024). https://doi.org/10.48550/arxiv.2409.01793 

101. Mao, L., Ma, Z., Li, X.: A multi-task dynamic weight optimization framework based on deep reinforcement learning. Appl. Sci. **15** (5), 2473 (2025). https://doi.org/10.3390/app15052473 

102. Tarunesh, I., Khyalia, S., Kumar, V., Ramakrishnan, G., and Jyothi, P.: Meta-Learning for effective multi-task and multilingual modelling. arXiv (Cornell University) (2021). https://doi.org/10. 48550/arxiv.2101.10368 

103. Goldie, A.D., Wang, Z., Foerster, J.N., and Whiteson, S.: How should we meta-learn reinforcement learning algorithms? arXiv.org (2025). https://arxiv.org/abs/2507.17668v1 

104. Fu, Z., Wu, X., Wang, Y., Wang, W., Ye, S., Yin, H., Chang, Y., Zheng, Y., and Zhao, X.: Training-free LLM merging for multitask learning. arXiv.org (2025). https://arxiv.org/abs/2506.12379 

105. Guo, L., Tahir, A.M., Hore, M., Collins, A., Rideout, A., Wang, Z.J.: A multi-task learning model for clinically interpretable sesamoiditis grading. Comput. Biol. Med. **182** , 109179 (2024). https://doi.org/10.1016/j.compbiomed.2024.109179 

106. Zelaszczyk,<sup>˙</sup> M., Ma´ndziuk, J.: Interpretable multi-task learning with shared variable embeddings. arXiv (Cornell University) (2024). https://doi.org/10.48550/arxiv.2405.06330 

**Publisher’s Note** Springer Nature remains neutral with regard to jurisdictional claims in published maps and institutional affiliations. 

91. Hur, K., Oh, J., Kim, J., Kim, J., Lee, M.J., Cho, E., Moon, S., Kim, Y., Atallah, L., Choi, E.: GenHPF: general healthcare predictive 

123 

