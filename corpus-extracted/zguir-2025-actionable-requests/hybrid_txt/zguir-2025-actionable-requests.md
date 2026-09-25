# Detecting Actionable Requests and Ofers on Social Media During Crises Using LLMs

Ahmed El Fekih Zguir

Qatar Computing Research Institute, Hamad Bin Khalifa University, Doha, Qatar azguir@hbku.edu.qa

Ferda Ofli

Qatar Computing Research Institute, Hamad Bin Khalifa University, Doha, Qatar fofli@hbku.edu.qa

Muhammad Imran

Qatar Computing Research Institute, Hamad Bin Khalifa University, Doha, Qatar mimran@hbku.edu.qa

## ABSTRACT

Natural disasters often result in a surge of social media activity, including requests for assistance, ofers of help, sentiments, and general updates. To enable humanitarian organizations to respond more eficiently, we propose a fine-grained hierarchical taxonomy to systematically organize crisis-related information about requests and ofers into three critical dimensions: supplies, emergency personnel, and actions. Leveraging the capabilities of Large Language Models (LLMs), we introduce Query-Specific Few-shot Learning (QSF Learning) that retrieves class-specific labeled examples from an embedding database to enhance the model’s performance in detecting and classifying posts. Beyond classification, we assess the actionability of messages to prioritize posts requiring immediate attention. Extensive experiments demonstrate that our approach outperforms baseline prompting strategies, efectively identifying and prioritizing actionable requests and ofers.

## Keywords

Large language models, Disaster management, Taxonomy, Social media, Query-Specific Few-shot Learning

## INTRODUCTION

Social media platforms have become vital during crises, as they enable real-time communication and coordination during emergencies. During past events like Hurricane Dorian, Hurricane Harvey, and COVID-19, platforms like Twitter (now X) and Facebook played a crucial role, allowing people to request help and organize assistance quickly ([23, 19]). For emergency responders and humanitarian organizations, this stream of information presents an opportunity to enhance situational awareness, allocate resources more eficiently, and provide aid where it is most needed. However, much of the data, including emotional expressions and commentary from those outside the afected area, is irrelevant to immediate response needs. To fully harness the value of social media during disasters, it is essential to develop efective methods for identifying the most relevant and actionable information amidst the noise [11].

This study focuses on identifying actionable social media posts categorized as “requests” (messages seeking assistance) and “ofers” (messages providing assistance). Actionability, as defined by Zade et al. [33], refers to posts containing suficient contextual information to assess urgency, time, and the specific location for the required assistance or available resources. Previous eforts to identify such posts have primarily focused on high-level categorization into requests and ofers, with limited attention to fine-grained distinctions based on specific types of assistance. Moreover, while supervised machine learning models trained on human-labeled data are ideal for such tasks, they are often impractical in disaster scenarios due to the time, resources, and efort required for data annotation. Furthermore, adapting these models to new events or categories demands additional training data, which is challenging to obtain during emergencies.

To address these limitations, this study focuses on fine-grained identification of requests and ofers without relying on extensive human-labeled data. We propose a fine-grained hierarchical taxonomy that organizes crisis-related information along three critical dimensions: supplies, emergency personnel, and actions. This taxonomy was developed using a “top-down” approach informed by documents, guidelines, and expertise from humanitarian organizations, resulting in a structured framework comprising 1,093 elements across six levels of granularity. Unlike previous “bottom-up” approaches that rely on text mining and topic modeling techniques [8], our taxonomy ensures consistency, practicality, and alignment with real-world disaster management needs.

A significant challenge in prior research has been the oversimplification of social media classification tasks as single-label multi-class problems, which fail to account for the multifaceted nature of disaster-related posts. For instance, a tweet like, “We’re cooking mealsfor displacedfamilies tonight. DM me ifyou want to help or donate. #CaliforniaFires,” contains both an ofer (meals for displaced families) and a request (volunteers to help). To address this, we frame the problem as a multi-label multi-class classification task, where a post can be labeled as both a request and an ofer, and each post can belong to multiple other fine-grained categories.

Additionally, existing approaches often lack the granularity required for efective disaster response. For example, the category “medical supplies” can encompass diverse items such as bandages, oxygen tanks, or mobility aids (e.g., wheelchairs), which are critical to diferentiate during a crisis. Traditional research has also predominantly focused on tangible supplies, overlooking requests for actions (e.g., search and rescue) or specific personnel (e.g., military support during riots). Our taxonomy addresses this gap by preserving granular and actionable information, enabling a more comprehensive analysis of social media posts.

To implement this taxonomy, we leverage Large Language Models (LLMs) with various prompting strategies. Additionally, we propose a Query-Specific Few-shot Learning (QSF learning) approach supported by Retrieval-Augmented Generation [18]. This method enables the classification of posts into fine-grained categories without extensive labeled data and outperforms several baselines on both real-world and synthetic data. By framing the problem as three distinct multi-label, multi-class classification tasks corresponding to supplies, actions, and emergency personnel, our framework provides a robust and scalable solution for detecting actionable requests and ofers on social media data during disasters. We provide the dataset, taxonomy, and other related resources at the following URL: https://crisisnlp.qcri.org/requests\_offers/.

## RELATED WORK

The growing reliance on social media during natural disasters has spurred significant research into identifying and categorizing actionable information, particularly requests and ofers, to aid humanitarian eforts. Early research primarily focused on traditional machine learning techniques for classifying disaster-related posts. For instance, Purohit et al. [25] developed and released labeled datasets and regular expressions to identify requests and ofers on Twitter. Their work used cascading SVM classifiers, prioritizing precision over recall, to classify tweets into categories like money, shelter, and medical supplies. However, their approach was limited to posts containing either requests or ofers, without accounting for posts that included both. Building on this, Nazer et al. [20] incorporated additional features, such as URLs and hashtags, and used decision tree classifiers to enhance classification performance. Similarly, Devaraj et al. [6] employed GloVe word vectors [24] to distinguish urgent posts from non-urgent ones, demonstrating the evolving sophistication of feature engineering in this domain. Taking a step further, Basu et al. [3] presented a utility-driven model for optimized resource allocation in a post-disaster scenario, based on information extracted from microblogs in real time.

The introduction of transformer architectures [31] marked a paradigm shift in natural language processing (NLP), enabling eficient processing of long text sequences through attention mechanisms. This breakthrough paved the way for models like BERT [7] and GPT-3 [5], which established new benchmarks in text classification. Fine-tuning these pre-trained models on disaster-related tasks became a popular approach [12]. For instance, Seeberger and Riedhammer [26] fine-tuned BERT to classify disaster-related tweets into actionable categories. Prompt engineering and few-shot prompting, introduced with GPT-3, further reduced dependence on large labeled datasets, showcasing the potential for efective performance with minimal examples [5].

More recent work has leveraged multiple pre-trained transformer models to address the complexity of disaster-related tasks. For example, Zhou et al. [35] employed BERT, RoBERTa, and XLNet to classify tweets across various disaster-related categories, outperforming traditional machine learning methods. Ziaullah et al. [37] highlighted the zero-shot capabilities of large language models (LLMs) for monitoring critical infrastructure during emergent disasters. Furthermore, Lamsal et al. [17] introduced crisis-specific fine-tuned transformers (CrisisTransformers) to classify tweets into requests and ofers, demonstrating significant improvements over earlier approaches.

Several studies have emphasized the importance of structured taxonomies for organizing disaster-related information. RweetMiner [28] introduced a formal framework for identifying and categorizing “rweets” (request tweets) into sub-types such as medical, food, and shelter, using machine learning classifiers with high precision. Similarly, Basu et al. [1] analyzed WhatsApp messages during the 2015 Nepal earthquake to curate resource requirements and delays, demonstrating the value of taxonomy-driven approaches for disaster preparedness. More recently, Durham et al. [8] employed text mining and topic modeling techniques, such as latent Dirichlet allocation, to develop a bottom-up taxonomy from tweets. In contrast, our work adopts a top-down approach, leveraging humanitarian guidelines and expertise to define a fine-grained hierarchical taxonomy. This taxonomy captures three critical dimensions—supplies, emergency personnel, and actions—providing a robust framework for classifying posts into actionable categories.

Fine-grained classification has proven essential for improving resource allocation during crises. For instance, Basu et al. [2] experimented with supervised and unsupervised models for identifying resource needs and availabilities, emphasizing the importance of granular classifications when high-quality training data is available. Ullah et al. [28] categorized tweets into sub-types such as medical, food, and shelter using machine learning classifiers with high precision. Similarly, Zhang et al. [34] employed a topic model-based framework to identify the spatial distribution of demand for relief supplies, while Dutt et al. [9] proposed a methodology to match resource needs and availabilities, considering resource type, quantity, and geographical proximity. However, earlier works often simplified the problem to single-label classification, overlooking the complexity of posts containing both requests and ofers. For example, a post ofering food while simultaneously requesting volunteers exemplifies the need for multi-label classification.

Highlighting the need for multilingual support, Vitiugin and Purohit [32] introduced MulTMR, a multilingual serviceability model leveraging knowledge distillation with task-related and behavior-guided teacher models to detect and rank help requests on social media. Their approach, validated across multiple languages and disaster events, demonstrated substantial performance improvements in multilingual scenarios. Similarly, Lamsal et al. [17] proposed CReMa, a systematic framework integrating textual, temporal, and spatial features for cross-lingual identification and matching of requests and ofers. Their multilingual embedding space and crisis-specific pretrained model significantly advanced performance benchmarks and highlighted the importance of cross-lingual analysis in disaster response.

In addition to other challenges, earlier research has also emphasized the need for actionable intelligence tailored to responders’ roles. Zade et al. [33] highlighted issues like information overload and misinformation in integrating social media data into disaster response. They proposed shifting from general situational awareness to actionable intelligence, aligning with our redefinition of actionability. Our approach prioritizes posts with suficient context to drive direct actions, addressing gaps in traditional urgency-based classifications and supporting humanitarian organizations in efective decision-making during crises.

## METHODOLOGY

Our goal is to develop a robust approach capable of identifying any predefined categories of requests or ofers while enabling rapid deployment without requiring large amounts of labeled data or supervised model training. To this end, we design a comprehensive taxonomy comprising three key dimensions (supplies, actions, emergency personnel). We then propose a Query-Specific Few-shot Learning (QSF learning) approach leveraging Retrieval-Augmented Generation (RAG) [18] to construct few-shot prompts for message classification. Our goal is not only to improve classification performance but also to assess how these improvements generalize across diferent LLMs. To ensure broad applicability, we evaluate our approach using multiple instruction-tuned LLMs of varying sizes and architectures, including Llama 3 8B, Llama 3.1 8B [10], Gemma 2 9B [27], Mistral 7B v0.2 [14], and GPT-4o mini. While all models are tested on the full range of baseline prompts and our proposed solution, GPT-4o mini—being a paid API—was evaluated exclusively on our solution (QSF Learning) to benchmark its performance relative to the other models. This diverse set of models allows us to systematically examine whether our findings generalize across LLMs with diferent capacities and training backgrounds. Next, we provide details of our methodology.

## Taxonomy Generation

Most prior works, including the work by Lamsal et al. [17], build on the taxonomy by Purohit et al. [25], which categorizes resources as tangible supplies or services requested or ofered during disasters, such as monetar donations, volunteer work, shelter, clothing, and medical supplies. While valuable, these taxonomies face two key limitations: (i) They group distinct resource types—tangible supplies (e.g., medical supplies, clothing) and intangible services (e.g., volunteer work)—failing to capture their unique characteristics and complicating accurate categorization. (ii) They often omit critical resources frequently highlighted during disasters. For example, social media posts during Hurricane Harvey in 2017 frequently requested bottled water and baby formula, while the 2020 Beirut explosion saw significant demand for dust masks and personal protective equipment—resources absent in existing taxonomies. These gaps hinder comprehensive disaster response and underscore the need for more fine-grained categorization.

Furthermore, previous research overlooks two critical types of requests and ofers: “actions” and “emergency personnel.” Many social media posts during disasters highlight urgent actions, such as search and rescue operations, debris clearance, medical aid, food distribution, and crowd control. For example, posts during the 2015 Nepal earthquake frequently requested search and rescue teams for locating survivors [4], while the 2023 T¨urkiye-Syria earthquake emphasized coordinated debris removal and emergency medical care [22]. Beyond actions, messages contain requests for trained personnel, e.g., firefighters, medical professionals, and law enforcement. For instance, the California wildfires saw appeals for firefighting reinforcements, and the Ebola outbreak required infectious disease specialists and emergency nurses.

To address these gaps, we propose a taxonomy with three main branches: supplies, actions, and emergency personnel.

• Supplies: tangible resources such as medical supplies, shelter, food, water, hygiene products, and more.  
• Actions: represent tasks like search and rescue, medical care, debris clearance, and food distribution.  
• Emergency personnel: refers to trained responders, including paramedics, firefighters, structural engineers, and specialized volunteers.

Next, we utilized oficial online resources, including situation reports, guidelines, and articles from organizations such as UN OCHA, UNDP, FEMA, Red Cross, and UNHCR [30, 29, 21]. The selection of these three dimensions—supplies, actions, and emergency personnel—is based on a thorough manual evaluation of the collected documents. Through careful analysis, we observed that the majority of disaster response activities naturally cluster around these three core areas: tangible resources (which we categorize as supplies), human responders and teams (personnel), and required operational activities (actions). This observation reflects the real-world practices and language used by humanitarian agencies, ensuring that our taxonomy is aligned with operational workflows and suficiently comprehensive to capture the essential elements of disaster response.

Information from 20 such sources was processed using GPT-4o, a state-of-the-art language model. to generate and augment categories within the three main branches. Table 1 outlines the prompts used to create the diferent taxonomy levels. The model suggested categories and sub-categories at diferent depths of the taxonomy underwent thorough human review, with adjustments made as needed. In the prompts, we ensure that each level provides increasingly fine-grained information about its parent category, incorporating synonyms, regional variations, and linguistic nuances. For example, the category “bandages” was expanded to include terms such as “Band-Aids,” “adhesive strips,” and “plasters,” reflecting diverse social media expressions. The final taxonomy comprises 1,093 elements distributed as follows at diferent depth levels: Level-1: 3, Level-2: 33, Level-3: 129, Level-4: 635, Level-5: 271, and Level-6: 22. Figure 1 shows a partial view of our taxonomy, highlighting the root branches, all categories at depth two that are directly under the root, and an expansion of some selected branches.

## Task Definition and Classifiers

## Task Definition

The goal of our task is to process an input message (e.g., a tweet) and extract key structured information relevant to crisis response. Specifically, for each input message, we aim to generate an 8-element tuple capturing its essential attributes. The tuple consists of:

• Type: A list indicating the type of message, such as "request", "offer", or "other".  
• Actions (r): A list of requested actions (e.g., "Search and Rescue").  
• Supplies (r): A list of requested supplies (e.g., "Medical", "Clothing and Warmth").  
• Personnel (r): A list of requested personnel (e.g., "Medical and Health Teams").  
• Actions (o): A list of ofered actions (e.g., "Infrastructure Repair and Debris Clearance").  
• Supplies (o): A list of ofered supplies (e.g., "Money").  
• Personnel (o): A list of ofered personnel (e.g., "Medical and Health Teams").  
• Actionability: A boolean value indicating whether the message contains actionable information.

Table 1. Prompts used to generate and expand the taxonomy

<table><tr><td>Extraction Prompt: Extracting Relevant Terms</td><td>Level 2 Prompt: Creating the Hierarchy (Per Category)</td></tr><tr><td></td><td></td></tr><tr><td>Shared Context (given below)</td><td>Shared Context</td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td>You are provided with multiple documents from humanitarian organizations that contain terms related to disaster relief. These terms can broadly fall under three categories: actions, emergency personnel, and supplies. Extract all relevant terms and group them based on which category they fall under.</td><td>You are provided with a list of terms that fall under the category (actions, supplies, or personnel). Your task is to group the terms into distinct, non-overlapping groupings within this category. Ensure that the groupings cover all terms and are logical, clear, and comprehensive.</td></tr><tr><td></td><td></td></tr><tr><td>Level 3 &amp; 4 Prompt: Refining the Combined Taxonomy</td><td>Levels 5 &amp; 6 Prompt: Expansion</td></tr><tr><td></td><td></td></tr><tr><td>Shared Context</td><td>Shared Context</td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td>You are provided with a taxonomy that organizes terms under actions, emergency personnel, and supplies. Your task is to refine and improve this hierarchy by reorganizing or expanding branches as needed. Different branches may vary in depth, but ensure the taxonomy remains logical, comprehensive, and well-organized.</td><td>You are provided with a detailed taxonomy. Your task is to review and expand leaf terms by:1. Adding synonyms for terms that are commonly referred to by different names. Make sure to include terms used both in social media and by humanitarian organizations.2. Adding subcategories or breaking down elements where justified.Only make expansions where they are necessary to improve clarity, usability, or completeness. Keep the taxonomy concise and avoid unnecessary additions.</td></tr><tr><td></td><td></td></tr><tr><td colspan="2">Shared context: In this task, you are assisting in the development of a taxonomy to classify and organize requests and offers made during disaster scenarios. The goal is to create a structured, top-down taxonomy based on a corpora of documents sourced from humanitarian organizations. These documents include situation reports, needs assessments, articles, websites, and guidelines which describe various forms of needs and offers.</td></tr></table>

![](images/0489880490cb33179896f42627d1483d7c9de9d1984d165a0dc79c6c9238f8ad.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
  A["Supplies {11}"] --> B["Medical {8}"]
  A --> C["Wound Care and First Aid Supplies {4}"]
  A --> D["&quot;Antiseptics and Sanitizers [9"]"]
  
  B --> E["Wound Care and First Aid Supplies {4}"]
  B --> F["&quot;Medical and Health Teams [8"]"]
  B --> G["&quot;Community and Social Support Teams [4"]"]
  B --> H["Transportation Services {3}"]
  
  C --> I["Wound Care and First Aid Supplies {4}"]
  C --> J["&quot;Antiseptics and Sanitizers [9"]"]
  
  K["Emergency Personnel {11}"] --> L["&quot;Medical and Health Teams [8"]"]
  K --> M["&quot;Community and Social Support Teams [4"]"]
  K --> N["Actions {11}"]
  
  O["&quot;Search and Rescue Teams [5"]"] -.-> P["&quot;Medical and Health Teams [8"]"]
  O -.-> Q["&quot;Community and Social Support Teams [4"]"]
  
  R["Search and Rescue {3}"] -.-> S["Medical Assistance {3}"]
  R -.-> T["Transportation Services {3}"]
  
  U["Water, Sanitation, and Hygiene (WASH) {9}"] -.-> V["Medical & Medical Assistant {8}"]
  U -.-> W["Transportation Services {3}"]
  
  X["Shelter and Housing {3}"] -.-> Y["Medical & Medical Assistant {8}"]
  X -.-> Z["Transportation Services {3}"]
  
  AA["Equipment and Tools {5}"] -.-> AB["Medical & Medical Assistant {8}"]
  AA -.-> AC["Transportation Services {3}"]
  
  AD["Food and Nutrition {7}"] -.-> AE["Medical & Medical Assistant {8}"]
  AD -.-> AF["Transportation Services {3}"]
  
  AG["Clothing and Warmth {3}"] -.-> AH["Medical & Medical Assistant {8}"]
  AG -.-> AI["Transportation Services {3}"]
  
  AJ["Communication and Power {3}"] -.-> AK["Medical & Medical Assistant {8}"]
  AJ -.-> AL["Transportation Services {3}"]
  
  AM["Education and Recreation {2}"] -.-> AN["Medical & Medical Assistant {8}"]
  AM -.-> AO["Transportation Services {3}"]
  
  AP["&quot;Money [5"]"] -.-> AQ["Medical & Medical Assistant {8}"]
  AP -.-> AR["Transportation Services {3}"]
  
  AS["&quot;Infant Care [4"]"] -.-> AT["Medical & Medical Assistant {8}"]
  AS -.-> AU["Transportation Services {3}"]
  
  AV["Water, Sanitation, and Hygiene (WASH) {9}"] -.-> AW["Medical & Medical Assistant {8}"]
  AV -.-> AX["Transportation Services {3}"]
  
  BA["Shelter and Housing {3}"] -.-> B["Medical & Medical Assistant {8}"]
  BA -.-> CA["Transportation Services {3}"]
  
  BA -.-> CD["Transportation Services {3}"]
  
  DA["Equipment and Tools {5}"] -.-> AE["Medical & Medical Assistant {8}"]
  DA -.-> AF["Transportation Services {3}"]
  
  DA -.-> AG["Transportation Services {3}"]
  
  AH["Food and Nutrition {7}"] -.-> AI["Medical & Medical Assistant {8}"]
  AH -.-> AJ["Transportation Services {3}"]
  
  AK["Clothing and Warmth {3}"] -.-> AL["Medical & Medical Assistant {8}"]
  AK -.-> AM["Transportation Services {3}"]
  
  AN["Communication and Power {3}"] -.-> AO["Medical & Medical Assistant {8}"]
  AN -.-> AP
  
  AO["Education and Recreation {2}"] -.-> AP
  
  AQ["&quot;Money [5"]"] -.-> AR
  AQ -.-> AS
  
  AR --> AF
  AR --> AG
  AR --> AH
  
  AI --> AJ
  AI --> AK
  
  AJ --> AL
  AJ --> AM
  
  AK --> AN
  AK --> AO
  
  AL --> AP
  AL --> AQ
  
  AM --> AR
  AM --> AS
  
  AN --> AR
  AN --> AO
  
  AO --> AP
  
  AP --> AQ
  
  AR --> AR
  AR --> AS
  
  AS --> AR
  AS --> AS
  
  AR --> AR
  AS --> AS
  
  AS --> AR
```
</details>

Figure 1. Request and Ofer Taxonomy: A partial representation

Here, (r) denotes request and (o) denotes ofer. All elements of the tuple are treated as multi-label classification tasks, except for Actionability, which is framed as a binary classification problem.

Building upon prior work [33], we define actionability as: Information related to a crisis that either requires a response or constitutes an ofer, where details such as time, location, urgency, and specific needs (e.g., quantities or actions) determine its usefulness. For any category without relevant information, the output is either an empty list (for list-based fields) or False (for actionability).

Formally, let 𝑇 be the set of input messages, and $t \in T$ an individual message. We define a mapping function 𝑓 as:

$$
f (t) = (\text {Type}, A _ {r}, S _ {r}, P _ {r}, A _ {o}, S _ {o}, P _ {o}, \text {Actionability})
$$

where:

• Type: A subset of {"request", "offer", "other"}.  
$A _ { r } \mathrm { : }$ Set of requested actions $( A _ { r } \subseteq A )$  
$S _ { r } { : }$ Set of requested supplies $( S _ { r } \subseteq S )$  
• $P _ { r } \mathbf { : }$ : Set of requested personnel $( P _ { r } \subseteq P )$  
$A _ { o } \colon$ Set of ofered actions $( A _ { o } \subseteq A )$  
$S _ { o } \colon$ Set of ofered supplies $( S _ { o } \subseteq S )$  
$P _ { o } \colon$ Set of ofered personnel $( P _ { o } \subseteq P ) .$  
• Actionability: A boolean value (True or False).

Here, 𝐴, 𝑆, and 𝑃 represent predefined sets of target labels for actions, supplies, and personnel, respectively. Each set contains 11 distinct labels, derived from our taxonomy at depth 2.

## Baseline classifiers

We approach the classification of crisis-related messages using LLMs, as illustrated in Figure 2. Our method relies on prompt engineering to guide the model’s output, starting with a baseline prompt and progressively refining it by incorporating additional contextual information. Each refinement results in a new classifier, designed to evaluate the impact of various prompting strategies. These classifiers difer in the level of detail they provide, specificall in terms of taxonomy depth and the inclusion of labeled examples (few-shots). The baseline prompt follows this structure:

• Instruction: In this section, we explain the LLM’s role as an advanced AI trained to classify social media posts related to crises, emphasizing the use of taxonomy and its knowledge of natural disasters.  
• Taxonomy: Provides the taxonomy, defining the labels the model should output.  
• Context: Discusses social media’s critical role during disasters, categorizing posts into three types: Request, Ofer, and Other. It highlights the importance of identifying urgent, actionable messages, using in-context learning principles [36].  
• Output Format: Specifies the output as a JSON object with structured labels.

![](images/21d8fe572128b03e3faba72f8c419ba0a0a4c5f3f249bcda4965e1e575831cf3.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph LR
  A["Query (Social Media post)"] --> B["Prompt"]
  B --> C["LLM inference"]
  C --> D["Post processing"]
  D --> E["Classification"]
```
</details>

Figure 2. Diagram of baseline (BL) classifiers. Each BL has a diferent prompt in the diagram.

The baseline prompt, used with BL 1 (baseline classifier 1), can be seen below, which served as the foundation for subsequent variants detailed as follows:

• BL 1 (Baseline Classifier 1): Uses the baseline prompt with taxonomy limited to depth 2 and no few-shot examples.  
• BL 2: Extends the taxonomy to depth 3, providing a more detailed hierarchical structure to improve label comprehension.  
• BL 3: Incorporates few-shot prompting by adding a small set of labeled examples to the prompt, improving task understanding as shown in prior work [5].

• BL 4: Combines BL 2 and BL 3, integrating both a detailed taxonomy and few-shot examples.  
• BL 5: Builds on BL 4 by adding chain-of-thought (CoT) prompting [16]. This classifier addresses ambiguous cases with detailed explanations and requires step-by-step reasoning for each classification decision, including actionability.

Baseline Prompt for Classification  
```txt
<Instruction>
You are an advanced AI trained to label social media posts related to crisis situations, specifically natural disasters. Your goal is to label the given posts.
You make use of the vast knowledge that you have about natural disasters, your knowledge of social media posts of people during those disasters, the
provided taxonomy and information mentioned below to label the data.
</Instruction>
<Context>
Natural disasters, such as hurricanes, wildfires, earthquakes, floods, tornadoes, landslides, etc., have significant impacts on communities. During
these events, social media platforms (such as Twitter, Facebook, and Instagram) become primary channels of sharing and receiving information. This
information can be broadly categorized into Request, Offer, or Other. Note that some posts are both Request and Offer at the same time ... (context was
omitted to fit on a page)
</Context>
<Taxonomy>
{Taxonomy till depth 2 here}
</Taxonomy>
<Output formats>
This is the output format when the type is either Request or Offer or both:
```

```json
{
    "text": "The social media post text here",
    "type": ["Request" | "Offer" | "Offer", "Request"],
    "action_request": [...],
    "personnel_request": [...],
    "supplies_request": [...],
    "action_offer": [...],
    "personnel_offer": [...],
    "supplies_offer": [...],
    "actionability": true | false
}
```  
This is the output format when the type is Other:

```txt
{
    "text": "The social media post text here",
    "type": ["Other"]
}
```

```txt
</Output formats>
<Task>
Your task is to label the following social media post based on the taxonomy and rules mentioned above. Only output the JSON dictionary and nothing else.
</Task>
```

## Query-Specific Few-Shot Learning Approach

Prior studies show that including labeled examples (few-shots) in prompts improves LLM performance on classification tasks. However, in a multi-class setting, adding an equal number of shots to all classes often leads to a decline in performance, as demonstrated by Imran et al. [13]. Building on this insight, we introduce a query-specific few-shot learning strategy (QSF learning) using Retrieval-Augmented Generation (RAG) [18] to retrieve relevant, query-specific examples dynamically for each input message.

For each message, we compute its embedding using OpenAI’s text-embedding-3-small model and retrieve the 𝑘/2 most similar labeled examples from a pre-built embedding database. This database contains examples from previous crisis events. To maintain variability and prevent bias toward specific classes, we also include 𝑘/2 randomly selected examples. These examples are then appended to the prompt. The detailed steps of the QSF Learning algorithm can be seen in Figure 3 and are as follows:

1. Top-k retrieval: Embeddings for each input are computed using OpenAI’s text-embedding-3-small model, and the 𝑘/2 most relevant examples are retrieved using cosine similarity.  
2. Random sampling: 𝑘/2 random examples are added for variability and to avoid over-fitting.  
3. Mapping and appending: Retrieved embeddings are mapped to their labeled examples and appended to the prompt.  
4. Prompt creation: A tailored prompt combines structured sections with retrieved examples.  
5. Inference, cleanup, and evaluation: The same steps as BL 1–5 are applied to process and evaluate outputs.

![](images/dcb1a16d774a175195bc3694c81bc38d77cf60a09afec7d95053341ce3c1181d.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph LR
  A["Query (Social Media post)"] --> B["Embedding Model"]
  B --> C["Vector db"]
  C --> D["Relevant Few-shot examples"]
  D --> E["Prompt"]
  E --> F["QSF prompt"]
  F --> G["LLM inference"]
  G --> H["Post processing"]
  H --> I["Classification"]
```
</details>

Figure 3. A high-level overview of the Query-Specific Few-shot Learning approach

## DATA AND EVALUATION

## Synthetic Data Generation

To evaluate the proposed methodology, we sought ground-truth data aligned with our detailed taxonomy. However, to the best of our knowledge, no such fine-grained dataset currently exists. As a result, we opted to generate synthetic data (tweets) to mimic social media posts during disasters. To achieve this, we tested various data generation methods, ultimately combining them to produce a diverse, high-quality dataset with unique elements and accurate labels. We used GPT-4o (with a May 2023 knowledge cutof) as our data generation model.

Our initial data generation approach involved a simple prompt containing the classes from depth 2 of our taxonomy. In this approach, we asked the model to generate 100 tweets and their labels simultaneously. However, the generated data lacked naturalness and appeared robotic, as shown below:

“Urgently need water, sanitation, and hygiene (WASH) at Westside Park. Please help!”

“Ofering water, sanitation, and hygiene (WASH) at St. Andrews Church. Available anytime.”

“Urgently need medical supplies at Westside Park.”

We observed that the model overly relied on the provided taxonomy labels, resulting in repetitive and unnatural outputs. Additionally, asking the model to generate 100 labeled examples in a single prompt led to repetitive text, as the model got “lazy” and started stitching examples by merely substituting labels from the taxonomy in the same text. We observed significant improvements when we seeded the prompt with real disaster event names. This way, we ask the model to use its knowledge of the events mentioned. For instance, when focusing on the T¨urkiye-Syria earthquake of 2023, the generated data appeared more realistic:

“This is so heartbreaking. Entire neighborhoodsflattened in Kahramanmaras¸. Stay strong. #TurkeySyriaEarthquake”

Table 2. Data generation prompt evaluation results

<table><tr><td>Prompt</td><td>Avg. Examples</td><td>Mean Similarity</td><td>Max Similarity</td></tr><tr><td>P1</td><td>38</td><td>0.4169</td><td>0.8154</td></tr><tr><td>P2</td><td>40</td><td>0.4022</td><td>0.7617</td></tr><tr><td>P3</td><td>68</td><td>0.3660</td><td>0.8189</td></tr><tr><td>P4</td><td>70</td><td>0.3727</td><td>0.8302</td></tr></table>

We further redesigned the prompt structure to address other issues, e.g., related to the length of the tweets, realisticness, etc., through iterative rounds of prompt engineering. The final structure included detailed instructions, context, output format, and enhanced instructions, formatted as follows:

## Synthetic Data Generation Prompt

## <Instruction>

You are an advanced AI trained to generate realistic and diverse synthetic social media posts related to crisis situations, specifically natural disasters. Your goal is to create posts that closely mimic real-life data while ensuring creativity, uniqueness, and variability. You make use of the vast knowledge that you have about natural disasters and social media posts of people during those disasters to generate the data.

## </Instruction>

## <Context>

{same context as baseline prompt for classification}

## </Context>

## <EnhancedInstructions>

The data that you generate should adhere to the following guidelines:

• All generated data must be unique. Avoid creating similar posts with minor variations (e.g., changing only the place or disaster name).  
• Data should look as realistic as possible, simulating posts created by humans during a natural disaster.  
• Posts should be diverse, including noisy data, partial data, and complete data.  
• Incorporate realistic elements such as links, phone numbers, hashtags, grammatical mistakes, abbreviations, etc., to enhance authenticity.  
• Integrate references to actual past events to ground the data in reality. Use the following disasters as reference points: {list of chosen disasters}.  
• Ensure a balance of content types (Request, Ofer, and Other) in the generated posts.

## </EnhancedInstructions>

## <OutputFormat>

The output should be a JSON object containing 100 generated posts in the following format:

```json
{
    "posts": [
        "generated text 1 here",
        "generated text 2 here",
        ...
        "generated text n"
    ]
}
```

## </OutputFormat>

## <Task>

Your task is to generate 𝑁 realistic social media posts based on the context and instructions provided above. Use the same context as the baseline prompt for classification. Each post should adhere to the guidelines, reflect a variety of disaster-related content, and maintain uniqueness.

## </Task>

To test the best data generation approach through prompting, in total, we created four prompts based on the above-mentioned structure, as follows:

• Prompt 1: Taxonomy + labeling (data and labels generated together).  
• Prompt 2: No taxonomy + labeling.  
• Prompt 3: Taxonomy (data generation only).  
• Prompt 4: No taxonomy (data generation only).

We instructed the LLM to generate 75 examples per prompt, using a temperature setting of 0.8 to enhance variability. Each prompt was executed 10 times to evaluate which prompting strategy produced better results, particularly in terms of generating more unique tweets. We calculated the mean and maximum similarity of the generated texts and recorded the number of messages produced. Specifically, we calculated the cosine similarity between all pairs of embeddings of generated examples. OpenAI’s text-embedding-3-small model was used to create the embeddings. While the model was instructed to generate 75 examples, the output varied across runs. Table 2 summarizes the averaged metrics across the 10 iterations.

The results indicate that generating and labeling simultaneously (Prompts 1 and 2) produces fewer examples due to token limitations and results in higher similarity between outputs. This aligns with prior research showing that breaking complex tasks into smaller chunks improves the performance of LLMs [15].

Based on these findings, we selected Prompt 4 (no taxonomy, data generation only) for 80% of the data and Prompt 3 (taxonomy, data generation only) for 20% to balance realism and coverage. We focused on disasters such as the Pakistan floods (2014), T¨urkiye-Syria earthquakes (2023), Australian bushfires (2019), Hurricane Maria (2017), and Haiti earthquake (2010). From an initial pool of 2,000 generated examples, duplicates (pairs with cosine similarity > 0.925) and unnatural outputs were removed, resulting in 1,346 high-quality examples.

![](images/9b5bc4485523a47849d83a52badff40f9bc7c3fe00cd424c8031e6b49e561e07.jpg)  
Figure 4. Distribution of the synthetic generated data across (a) Type and (b) Supply categories

The generated data was labeled using few-shot learning, guided by human-labeled examples. Human annotators reviewed 40% of the data (538 examples) and found fewer than 10% mislabeled, which were manually corrected. Specifically, for the message “type” task, only 10 examples were mislabeled, and for other categories (actions, supplies, personnel), 50 examples had minor issues with extra or missing labels. Posts labeled as type “Other” (neither a request nor an ofer) were excluded from further labeling for actions, supplies, personnel, and actionability categories. Figures 4 & 5 show the distribution of the generated data for the message types, supplies, actions, and emergency personnel for both requests and ofers. As for actionability, from the 899 posts that were either requests or ofers, 748 were actionable while 151 were not actionable.

## Real-World Data

While synthetic data enabled us to create a controlled and balanced dataset aligned with our detailed taxonomy, it was essential to evaluate the robustness of our approach on real-world data, where posts are inherently noisier, less structured, and often incomplete. To this end, we utilized an existing dataset of disaster-related tweets collected and originally annotated by Purohit et al. [25] and later improved by Lamsal et al. [17]. Their dataset contains tweets labeled as either requests or ofers during the Hurricane Sandy disaster, providing a solid foundation for real-world evaluation.

From this dataset, we randomly sampled 300 tweets and manually annotated them according to our taxonomy. Each tweet was reviewed to assign the relevant labels across all applicable categories, including type, supplies, actions, personnel, and actionability. Importantly, this subset of real-world data allowed us to test the model’s performance in scenarios where noise and incomplete information are prevalent. For example, consider the following tweet:

#LiveWire Game Donated \$10,000 to Hurricane Sandy Voters: The rapper wanted to help storm victi... http://t.co/EWkLK4ph #LiveWireRecords

Here, part of the message is truncated, and the text lacks explicit mentions of key elements such as location or clear action verbs—common challenges encountered in authentic social media posts. By incorporating such examples, we ensured that the model’s ability to generalize extends beyond synthetic, well-formed data and can handle the ambiguity and noise characteristic of real-world platforms like Twitter.

Due to the labor-intensive nature of manual labeling, especially given the multi-label setup and the breadth of categories, we limited the annotation to 300 examples. Nevertheless, this subset proved suficient to validate tha our methodology remains efective even when applied to naturally occurring, less structured data.

![](images/73fe9349cd2a9bddf2b03eafddd5f4bb912ccd41bdf0dd9d0a0609537ceb80e4.jpg)

<details>
<summary>bar</summary>

| Category | Request | Offer |
| --- | --- | --- |
| Security and Crowd Control | ~10 | ~5 |
| Food and Water Distribution | ~20 | ~45 |
| Transportation Services | ~27 | ~25 |
| Shelter and Housing Support | ~28 | ~36 |
| Medical Assistance | ~31 | ~38 |
| Search and Rescue | ~32 | ~2 |
| Community and Social Support | ~35 | ~16 |
| Recovery and Rehabilitation | ~36 | ~23 |
| Infrastructure Repair and Debris Clearance | ~53 | ~14 |
| Resource Coordination and Supply Management | ~61 | ~34 |
| Information and Communication | ~128 | ~49 |
</details>

(a) Actions

![](images/c4a9fa750928e1156e28c96343458d6998295d514335887855c4a6350a6cf2d1.jpg)

<details>
<summary>bar</summary>

| Category | Request | Offer |
| --- | --- | --- |
| Communication and IT Teams | ~7 | ~2 |
| Law Enforcement and Security | ~9 | ~4 |
| Shelter and Housing Teams | ~14 | 0 |
| Fire and Hazard Control | ~16 | ~2 |
| Energy and Infrastructure Repair Teams | ~20 | 0 |
| Legal and Advocacy Teams | ~20 | ~5 |
| Recovery and Rehabilitation Teams | ~21 | ~1 |
| Medical and Health Teams | ~24 | ~13 |
| Community and Social Support Teams | ~24 | ~9 |
| Logistics and Coordination Teams | ~25 | ~3 |
| Search and Rescue Teams | ~31 | ~1 |
</details>

(b) Emergency Personne  
Figure 5. Distribution of the synthetic generated data across (a) Action and (b) Emergency Personnel categories

## Evaluation metrics

We split the synthetic dataset into 50% training (673 examples) and 50% evaluation (673 examples). Examples used for few-shot prompting in BL 3, 4, and 5 are strictly from the training set. For the QSF Learning method, relevant examples are retrieved exclusively from the training set to ensure no data leakage into the evaluation set. For the real-world dataset, we use a training set of 107 examples and a test set of 200 examples. We evaluate the classification performance using micro F1-scores across all multi-label tasks: type, supplies, actions, and personnel. Micro averaging aggregates contributions of all classes globally, treating each instance-label pair equally, making it particularly well-suited for multi-label classification tasks. For the binary classification task of actionability, we report macro F1-scores to account for both classes equally, regardless of class distribution.

## RESULTS AND DISCUSSION

Our task includes one binary classification task (i.e., actionability) and seven multi-label classification tasks. Despit careful prompt design, the model occasionally produces problematic outputs, such as assigning labels outsid the taxonomy. Figure 6 shows the distribution of the errors from models (Llama 3) at the classification/inference time. BL 1 and 2 exhibit a substantial number of errors (359 and 410, respectively), whereas BL 3 through 6 show a significant reduction. This sharp decline indicates that adding few-shot examples to the prompts significantly enhances the LLM’s ability to generate taxonomy-compliant outputs. BL 3 and 4, which include few-shot examples, produce similar counts, as do BL 5 and our QSF learning technique. The diference between BL 4 and 5 can be attributed to the inclusion of chain-of-thought prompting in BL 5, which encourages step-by-step reasoning and results in better alignment with the taxonomy.

In addition to errors, the model sometimes provides broken responses that cannot be processed and evaluated even after post processing. Only BL 3 and 4 result in such responses, with BL 3 resulting in 38 broken responses, and BL 4 resulting in 8 broken responses. Both prompts lack chain-of-thought prompting, indicating that while few-shot examples reduce mislabels, incorporating structured reasoning further stabilizes the output quality.

Table 3 and 4 present evaluation results for our multi-task, multi-label classification tasks on both synthetic and real-world data. We run our evaluation on multiple models, namely: Llama 3 8B, Llama 3.1 8B, Gemma 2 9B, Mistral 7B v0.2, and GPT-4o mini. For the Type task, baseline prompts (BL1–BL5) exhibit gradual improvements, with BL5 consistently outperforming earlier versions across all models. This is likely due to the relative simplicity of the Type task and its limited label space (request, ofer, other). However, incorporating QSF learning yields the highest F1-scores across most models. For instance, on synthetic data, Llama 3.1 improves from 0.86 (BL5) to 0.89, while GPT-4o mini achieves a peak F1-score of 0.92. On real-world data, although absolute F1-scores are lower due to increased complexity, the same pattern persists. Models such as Gemma 2 and Llama 3 benefit from a 2-3% improvement over their best baselines, further confirming the advantage of QSF Learning for this task.

![](images/82220bb0e63a1ee8410240c3747042c0ff003a49eb929dc98aa227aa24e0c5ac.jpg)

<details>
<summary>heatmap</summary>

| Category | BL 1 | BL 2 | BL 3 | BL 4 | BL 5 | QSF |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| action_offer | 58 | 33 | 18 | 13 | 9 | 8 |
| action_request | 55 | 43 | 26 | 33 | 21 | 17 |
| personnel_offer | 54 | 61 | 11 | 11 | 4 | 9 |
| personnel_request | 22 | 54 | 17 | 20 | 13 | 11 |
| supplies_offer | 78 | 85 | 22 | 7 | 13 | 7 |
| supplies_request | 90 | 135 | 13 | 15 | 7 | 9 |
| type | 0 | 0 | 0 | 0 | 0 | 0 |
</details>

Figure 6. Error distribution during inference on synthetic data using Llama 3

For the Supplies task, baseline prompts improve steadily from BL1 to BL5, particularly when few-shot prompting is introduced. Given the task’s complexity, involving 11 labels and 2048 label combinations $( 2 ^ { 1 1 } )$ , static prompts alone achieve moderate performance. The QSF Learning consistently leads to the highest scores across both synthetic and real-world data. For example, in the synthetic setting, Llama 3.1 improves from 0.82 (BL5) to 0.84, while GPT-4o mini achieves an F1 score of 0.88. A similar trend is observed in real data, where the QSF classifier (“QSF”) consistently boosts model performance to around 0.83–0.85, suggesting that context-aware prompting is particularly efective in providing the necessary contextual variety to handle high-variability multi-label tasks.

The Actions task follows a similar trajectory. Baseline prompts struggle, especially in real data, with BL1 and BL2 achieving F1 scores as low as 0.20–0.30 across models. However, each subsequent enhancement results in gradual gains. The most notable improvements are observed with QSF learning, which increases F1 scores by up to 15% compared to the strongest static baseline. For instance, on synthetic data, Mistral improves from 0.62 (BL5) to 0.77, while on real-world data, Llama 3’s performance increases from 0.42 (BL5) to 0.57. This indicates that tailoring examples dynamically allows the model to better capture complex action patterns.

The Personnel task presents significant challenges across both datasets, reflected by generally lower F1 scores. In real-world data, baseline prompts achieve particularly poor performance, often under 0.30. Despite this, our QSF learning approach consistently yields the highest improvements. For example, Llama 3.1 improves from 0.27 (BL5) to 0.47, and Gemma 2 increases from 0.25 to 0.38. On synthetic data, the trend is similar, with performance gains of approximately 10–15% across models. These results suggest that dynamically selecting query-relevant few-shot examples is crucial for addressing class imbalance and label sparsity in the Personnel task.

Examining the detailed taxonomy reveals that supplies are the most granular dimension, followed by actions and personnel. This reflects the humanitarian organization corpora used to build the taxonomy, which primarily emphasize supplies. The taxonomy itself contains 946 supply-related elements, compared to 90 for actions and 57 for personnel, contributing to richer coverage in that dimension.

Furthermore, supplies are typically mentioned explicitly in social media posts, making them easier to classify. In contrast, references to actions or personnel are often implicit, requiring the model to infer intent, which increases prediction dificulty. This challenge is compounded by class prevalence: supplies dominate both our synthetic and real datasets, while actions and personnel are less frequently labeled.

In our real-world dataset (307 labeled posts: 107 for retrieval, 200 for testing), posts labeled with supplies significantly outnumber those mentioning actions or personnel. Consequently, the embedding database used in our QSF classifier is denser and more representative for supplies, providing the model with more relevant examples. These factors—the taxonomy’s granularity, explicit mentions, and higher class frequency—collectively explain the stronger classification performance observed for supplies.

<table><tr><td>Task</td><td>Model</td><td>BL1</td><td>BL2</td><td>BL3</td><td>BL4</td><td>BL5</td><td>QSF</td></tr><tr><td rowspan="5">Type</td><td>Mistral</td><td>0.73</td><td>0.74</td><td>0.80</td><td>0.81</td><td>0.87</td><td>0.87</td></tr><tr><td>Llama 3.1</td><td>0.71</td><td>0.72</td><td>0.82</td><td>0.82</td><td>0.86</td><td>0.89</td></tr><tr><td>Gemma 2</td><td>0.73</td><td>0.74</td><td>0.85</td><td>0.83</td><td>0.88</td><td>0.86</td></tr><tr><td>Llama 3</td><td>0.75</td><td>0.77</td><td>0.84</td><td>0.82</td><td>0.85</td><td>0.89</td></tr><tr><td>GPT-4o mini</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>0.92</td></tr><tr><td rowspan="5">Supplies</td><td>Mistral</td><td>0.62</td><td>0.57</td><td>0.79</td><td>0.82</td><td>0.82</td><td>0.81</td></tr><tr><td>Llama 3.1</td><td>0.57</td><td>0.58</td><td>0.75</td><td>0.77</td><td>0.82</td><td>0.84</td></tr><tr><td>Gemma 2</td><td>0.69</td><td>0.63</td><td>0.79</td><td>0.75</td><td>0.75</td><td>0.79</td></tr><tr><td>Llama 3</td><td>0.38</td><td>0.41</td><td>0.76</td><td>0.78</td><td>0.79</td><td>0.79</td></tr><tr><td>GPT-4o mini</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>0.88</td></tr><tr><td rowspan="5">Actions</td><td>Mistral</td><td>0.49</td><td>0.51</td><td>0.55</td><td>0.57</td><td>0.62</td><td>0.77</td></tr><tr><td>Llama 3.1</td><td>0.47</td><td>0.48</td><td>0.55</td><td>0.59</td><td>0.71</td><td>0.77</td></tr><tr><td>Gemma 2</td><td>0.49</td><td>0.53</td><td>0.61</td><td>0.61</td><td>0.69</td><td>0.73</td></tr><tr><td>Llama 3</td><td>0.40</td><td>0.49</td><td>0.54</td><td>0.57</td><td>0.64</td><td>0.75</td></tr><tr><td>GPT-4o mini</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>0.77</td></tr><tr><td rowspan="5">Personnel</td><td>Mistral</td><td>0.44</td><td>0.40</td><td>0.46</td><td>0.49</td><td>0.54</td><td>0.64</td></tr><tr><td>Llama 3.1</td><td>0.45</td><td>0.41</td><td>0.51</td><td>0.52</td><td>0.62</td><td>0.70</td></tr><tr><td>Gemma 2</td><td>0.51</td><td>0.53</td><td>0.57</td><td>0.56</td><td>0.59</td><td>0.66</td></tr><tr><td>Llama 3</td><td>0.31</td><td>0.35</td><td>0.46</td><td>0.47</td><td>0.54</td><td>0.69</td></tr><tr><td>GPT-4o mini</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>0.72</td></tr><tr><td rowspan="5">Actionability</td><td>Mistral</td><td>0.64</td><td>0.63</td><td>0.70</td><td>0.67</td><td>0.72</td><td>0.67</td></tr><tr><td>Llama 3.1</td><td>0.54</td><td>0.56</td><td>0.50</td><td>0.44</td><td>0.54</td><td>0.59</td></tr><tr><td>Gemma 2</td><td>0.62</td><td>0.64</td><td>0.68</td><td>0.67</td><td>0.78</td><td>0.49</td></tr><tr><td>Llama 3</td><td>0.68</td><td>0.60</td><td>0.60</td><td>0.64</td><td>0.58</td><td>0.81</td></tr><tr><td>GPT-4o mini</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>0.60</td></tr></table>

Table 3. Results (F1-scores) for all models and baselines on synthetic data.

The Actionability task shows mixed results. On synthetic data, the improvements due to our QSF learning approach vary by model. For instance, Llama 3.1 sees a moderate increase from 0.54 (BL5) to 0.59, while Llama 3 shows a significant gain from around 0.58 to 0.81. However, Gemma 2’s QSF score (0.49) is lower than its best baseline (0.78), indicating that the benefits of QSF learning might depend on how well the model’s underlying representation aligns with the task. In real-world data, though, our method generally yields better F1-scores i.e., 0.66 for Llama 3.1 and 0.70 for Gemma 2, which points to its potential to enhance recall for less frequent classes.

These findings reveal key insights into both model behavior and prompt design in multi-task, multi-label classification. Across both synthetic and real-world datasets, QSF learning consistently delivers the most substantial performance gains, particularly for tasks with complex label structures such as Actions and Personnel. By tailoring contextual examples to each input, this technique enables models to better handle label imbalance and variability, leading to superior generalization. While static prompt enhancements (BL1–BL5) result in incremental improvements, they plateau quickly, especially in granular tasks. Interestingly, the results show that QSF learning not only improves performance across tasks but also narrows the gap between models of diferent sizes and capabilities. High-performing models like GPT-4o mini benefit from this technique, but smaller models such as Llama 3.1 and Gemma 2 also achieve competitive results when QSF learning is applied. This indicates that prompt engineering, particularly dynamic few-shot approaches, plays a pivotal role in bridging the performance disparity between models, allowing even smaller models to generalize efectively in complex, real-world scenarios.

## LIMITATIONS AND FUTURE WORK

In this section, we outline the key limitations and biases of our study. One primary source of bias stems from the dataset used. While a portion of our evaluation is conducted on a real-world dataset comprising 300 manually labeled tweets, the relatively small size of this dataset limits its ability to fully capture the variability, noise, and evolving nature of social media posts during crises. To address the scarcity of large-scale curated datasets, we also utilized synthetically generated data produced using GPT-4o. Although GPT-4o provides diverse and partiall grounded examples, it may not entirely replicate the informal language, emerging trends, or unpredictability present in real-world streams.

<table><tr><td>Task</td><td>Model</td><td>BL1</td><td>BL2</td><td>BL3</td><td>BL4</td><td>BL5</td><td>QSF</td></tr><tr><td rowspan="5">Type</td><td>Mistral</td><td>0.51</td><td>0.56</td><td>0.63</td><td>0.65</td><td>0.69</td><td>0.74</td></tr><tr><td>Llama 3.1</td><td>0.64</td><td>0.66</td><td>0.74</td><td>0.73</td><td>0.72</td><td>0.75</td></tr><tr><td>Gemma 2</td><td>0.55</td><td>0.58</td><td>0.69</td><td>0.67</td><td>0.74</td><td>0.77</td></tr><tr><td>Llama 3</td><td>0.54</td><td>0.54</td><td>0.75</td><td>0.73</td><td>0.74</td><td>0.77</td></tr><tr><td>GPT-4o mini</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>0.72</td></tr><tr><td rowspan="5">Supplies</td><td>Mistral</td><td>0.58</td><td>0.53</td><td>0.83</td><td>0.84</td><td>0.79</td><td>0.83</td></tr><tr><td>Llama 3.1</td><td>0.41</td><td>0.45</td><td>0.63</td><td>0.69</td><td>0.80</td><td>0.82</td></tr><tr><td>Gemma 2</td><td>0.76</td><td>0.57</td><td>0.80</td><td>0.79</td><td>0.71</td><td>0.85</td></tr><tr><td>Llama 3</td><td>0.41</td><td>0.32</td><td>0.79</td><td>0.81</td><td>0.81</td><td>0.83</td></tr><tr><td>GPT-4o mini</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>0.85</td></tr><tr><td rowspan="5">Actions</td><td>Mistral</td><td>0.21</td><td>0.22</td><td>0.37</td><td>0.36</td><td>0.43</td><td>0.54</td></tr><tr><td>Llama 3.1</td><td>0.23</td><td>0.21</td><td>0.36</td><td>0.39</td><td>0.38</td><td>0.55</td></tr><tr><td>Gemma 2</td><td>0.34</td><td>0.31</td><td>0.48</td><td>0.49</td><td>0.50</td><td>0.50</td></tr><tr><td>Llama 3</td><td>0.29</td><td>0.26</td><td>0.41</td><td>0.43</td><td>0.42</td><td>0.57</td></tr><tr><td>GPT-4o mini</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>0.61</td></tr><tr><td rowspan="5">Personnel</td><td>Mistral</td><td>0.07</td><td>0.08</td><td>0.00</td><td>0.19</td><td>0.21</td><td>0.21</td></tr><tr><td>Llama 3.1</td><td>0.06</td><td>0.07</td><td>0.19</td><td>0.17</td><td>0.27</td><td>0.47</td></tr><tr><td>Gemma 2</td><td>0.04</td><td>0.14</td><td>0.22</td><td>0.19</td><td>0.25</td><td>0.38</td></tr><tr><td>Llama 3</td><td>0.06</td><td>0.00</td><td>0.19</td><td>0.22</td><td>0.22</td><td>0.31</td></tr><tr><td>GPT-4o mini</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>0.35</td></tr><tr><td rowspan="5">Actionability</td><td>Mistral</td><td>0.60</td><td>0.53</td><td>0.49</td><td>0.45</td><td>0.47</td><td>0.58</td></tr><tr><td>Llama 3.1</td><td>0.43</td><td>0.47</td><td>0.47</td><td>0.40</td><td>0.60</td><td>0.66</td></tr><tr><td>Gemma 2</td><td>0.50</td><td>0.51</td><td>0.69</td><td>0.64</td><td>0.63</td><td>0.70</td></tr><tr><td>Llama 3</td><td>0.45</td><td>0.48</td><td>0.52</td><td>0.54</td><td>0.45</td><td>0.47</td></tr><tr><td>GPT-4o mini</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>0.72</td></tr></table>

Table 4. Results (F1-scores) for all models and baselines on real-world data.

Another limitation is the focus solely on English-language posts, which excludes the multilingual nature of crisis communication in many global contexts. Expanding to multilingual datasets is an important direction for enhancing the generalizability of our approach.

Additionally, while the taxonomy introduces valuable structure and granularity to the classification process, its evaluation has primarily been qualitative. A more systematic, quantitative assessment of the taxonomy’s depth, completeness, and impact on classification performance would provide deeper insights and potentially inform refinements to better support crisis response applications.

For future work, we recommend expanding the real-world dataset, incorporating multilingual posts, and exploring alternative data generation strategies to improve robustness. Further, developing quantitative metrics to assess the taxonomy’s design and conducting controlled experiments to evaluate its efect on classification outcomes will be key to enhancing its utility. Incorporating optimization techniques, such as fine-tuning or multi-agent frameworks, may also ofer additional performance gains.

## CONCLUSION

In this work, we introduced a fine-grained hierarchical taxonomy and a dynamic few-shot prompting technique to improve the detection of actionable requests and ofers in social media posts during natural disasters. Our taxonomy organizes crisis-related information related to requests and ofers into three core dimensions: supplies, emergency personnel, and actions. By leveraging the capabilities of Large Language Models, we demonstrated through extensive experiments that our approach significantly outperforms baseline prompting methods in accuratel identifying and prioritizing actionable content. These contributions provide a valuable framework for enhancing the eficiency of humanitarian organizations in crisis management and rapid response. Future work will focus on expanding the approach to diverse disaster scenarios, integrating real-world data, and incorporating advancements in next-generation LLMs to further refine performance and adaptability.

## REFERENCES

[1] Moumita Basu, Saptarshi Ghosh, Arnab Jana, Somprakash Bandyopadhyay, and Ravikant Singh. Resource mapping during a natural disaster: a case study on the 2015 nepal earthquake. Internationaljournal ofdisaster risk reduction, 24:24–31, 2017.  
[2] Moumita Basu, Anurag Shandilya, Prannay Khosla, Kripabandhu Ghosh, and Saptarshi Ghosh. Extracting resource needs and availabilities from microblogs for aiding post-disaster relief operations. IEEE Transactions on Computational Social Systems, 6(3):604–618, 2019.  
[3] Moumita Basu, Sipra Das Bit, and Saptarshi Ghosh. Utilizing microblogs for optimized real-time resource allocation in post-disaster scenarios. Social Network Analysis and Mining, 12:1–20, 2022.  
[4] Carla Bleiker. Social media after nepal earthquake, April 27 2015. URL https://www.dw.com/en/ nepal-searching-for-missing-loved-ones-with-google/a-18411530.  
[5] T. Brown, B. Mann, N. Ryder, M. Subbiah, J. Kaplan, P. Dhariwal, A. Neelakantan, P. Shyam, G. Sastry, and A. Askell. Language Models are Few-Shot Learners. In Advances in Neural Information Processing Systems, volume 33, 2020.  
[6] A. Devaraj, D. Murthy, and A. Dontula. Machine-learning methods for identifying social media-based requests for urgent help during hurricanes. International Journal ofDisaster Risk Reduction, 51:Art. no. 101757, 2020. doi: 10.1016/j.ijdrr.2020.101757.  
[7] J. Devlin, M.-W. Chang, K. Lee, and K. Toutanova. BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. In Proceedings ofthe 2019 Conference ofthe North American Chapter ofthe Association for Computational Linguistics: Human Language Technologies, pages 4171–4186. Association for Computational Linguistics, 2019. doi: 10.18653/v1/N19-1423. URL https://doi.org/10.18653/ v1/N19-1423.  
[8] J. Durham, S. Chowdhury, and A. Alzarrad. Unveiling key themes and establishing a hierarchical taxonomy of disaster-related tweets: A text mining approach for enhanced emergency management planning. Information, 14(7):385, 2023. doi: 10.3390/info14070385. URL https://doi.org/10.3390/info14070385.  
[9] Ritam Dutt, Moumita Basu, Kripabandhu Ghosh, and Saptarshi Ghosh. Utilizing microblogs for assisting post-disaster relief operations via matching resource needs and availabilities. Information Processing & Management, 56(5):1680–1697, 2019.  
[10] A. Grattafiori, A. Dubey, A. Jauhri, A. Pandey, A. Kadian, A. Al-Dahle, A. Letman, A. Mathur, A. Schelten, A. Vaughan, A. Yang, A. Fan, A. Goyal, A. Hartshorn, A. Yang, A. Mitra, A. Sravankumar, A. Korenev, A. Hinsvark, and Z. Ma. The llama 3 herd of models, July 2024. URL https://arxiv.org/abs/2407. 21783. arXiv preprint.  
[11] Xingsheng He. Di Lu. Drew Margolin. Mengdi Wang, Salma El Idrissi, and Yu-Ru Lin. The signals and noise: actionable information in improvised social media channels during a disaster. In Proceedings of the 2017 ACM on web science conference, pages 33–42, 2017.  
[12] J. Howard and S. Ruder. Universal language model fine-tuning for text classification. In Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 328–339. Association for Computational Linguistics, 2018. doi: 10.18653/v1/P18-1031. URL https://doi.org/10.18653/v1/P18-1031.  
[13] Muhammad Imran, Abdul Wahab Ziaullah, Kai Chen, and Ferda Ofli. Evaluating robustness of llms on crisisrelated microblogs across events, information types, and linguistic features. arXiv preprint arXiv:2412.10413, 2025.  
[14] Albert Q. Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Florian Bressand, Gianna Lengyel, Guillaume Lample, Lucile Saulnier, Lelio Renard Lavaud,´ Marie-Anne Lachaux, Pierre Stock, Teven Le Scao, Thibaut Lavril, Thomas Wang, Timothee Lacroix, and´ William El Sayed. Mistral 7b, 2023. URL https://arxiv.org/abs/2310.06825.  
[15] Tushar Khot, Harsh Trivedi, Ashish Sabharwal, and Peter Clark. Decomposed prompting: A modular approach for solving complex tasks. In Proceedings of the International Conference on Learning Representations (ICLR), 2023.  
[16] Takeshi Kojima, Shixiang Shane Gu, Machel Reid, Yutaka Matsuo, and Yusuke Iwasawa. Large language models are zero-shot reasoners. arXiv preprint arXiv:2205.11916, 2022. URL https://arxiv.org/abs/ 2205.11916.  
[17] Rabindra Lamsal, MariaRodriguez Read, Shanika Karunasekera, and Muhammad Imran. Crema: Crisi response through computational identification and matching of cross-lingual requests and ofers shared on social media. IEEE Transactions on Computational Social Systems, 2024.  
[18] Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich K¨uttler, Mike Lewis, Wen-tau Yih, Tim Rocktaschel, Sebastian Riedel, and Douwe Kiela. Retrieval-augmented¨ generation for knowledge-intensive nlp tasks. In Advances in Neural Information Processing Systems (NeurIPS), 2020.  
[19] V. V. Mihunov, N. S. N. Lam, L. Zou, Z. Wang, and K. Wang. Use of twitter in disaster rescue: lessons learned from hurricane harvey. International Journal ofDigital Earth, 2020.  
[20] T. H. Nazer, F. Morstatter, H. Dani, and H. Liu. Finding requests in social media for disaster relief. In 2016 IEEE/ACM International Conference on Advances in Social Networks Analysis and Mining (ASONAM), pages 1410–1413. IEEE, 2016. doi: 10.1109/ASONAM.2016.7752424.  
[21] OCHA. Response planning and coordination, 2023. URL https://www.unocha.org. Retrieved from OCHA Oficial Website.  
[22] OCHA. Turkey/Syria: Earthquakes - feb 2023, 2023. URL https://reliefweb.int/disaster/ eq-2023-000015-tur. asdfdfsdfas.  
[23] S. Olawale. Social media and crisis management: A review and analysis of existing studies. LAU Sosyal<sup>¨</sup> Bilimler Dergisi, 9(2):199–215, 2018.  
[24] J. Pennington, R. Socher, and C. D. Manning. GloVe: Global vectors for word representation. In Proceedings ofthe Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 1532–1543, 2014.  
[25] Hemant Purohit, Carlos Castillo, Fernando Diaz, Amit Sheth, and Patrick Meier. Emergency-relief coordination on social media: Automatically matching resource requests and ofers. First Monday, 19(1), 2014. doi: 10.5210/fm.v19i1.4848. URL https://doi.org/10.5210/fm.v19i1.4848.  
[26] P. Seeberger and K. Riedhammer. Enhancing crisis-related tweet classification with entity-masked language modeling and multi-task learning. In Proceedings of the Workshop on Natural Language Processing for Positive Impact (NLP4PI), pages 70–78, 2022. URL https://aclanthology.org/2022.nlp4pi-1.9. pdf. Available online: https://aclanthology.org/2022.nlp4pi-1.9.pdf.  
[27] Gemma Team, Morgane Riviere, Shreya Pathak, Pier Giuseppe Sessa, Cassidy Hardin, Surya Bhupatiraju, Leonard Hussenot, Thomas Mesnard, Bobak Shahriari, Alexandre Ram´ e, Johan Ferret, Peter Liu, Pouya´ Tafti, Abe Friesen, Michelle Casbon, Sabela Ramos, Ravin Kumar, Charline Le Lan, Sammy Jerome, Anton Tsitsulin, Nino Vieillard, Piotr Stanczyk, Sertan Girgin, Nikola Momchev, Matt Hofman, Shantanu Thakoor, Jean-Bastien Grill, Behnam Neyshabur, Olivier Bachem, Alanna Walton, Aliaksei Severyn, Alicia Parrish, Aliya Ahmad, Allen Hutchison, Alvin Abdagic, Amanda Carl, Amy Shen, Andy Brock, Andy Coenen, Anthony Laforge, Antonia Paterson, Ben Bastian, Bilal Piot, Bo Wu, Brandon Royal, Charlie Chen, Chintu Kumar, Chris Perry, Chris Welty, Christopher A. Choquette-Choo, Danila Sinopalnikov, David Weinberger, Dimple Vijaykumar, Dominika Rogozinska, Dustin Herbison, Elisa Bandy, Emma Wang, Eric Noland, Erica´ Moreira, Evan Senter, Evgenii Eltyshev, Francesco Visin, Gabriel Rasskin, Gary Wei, Glenn Cameron, Gus Martins, Hadi Hashemi, Hanna Klimczak-Plucinska, Harleen Batra, Harsh Dhand, Ivan Nardini, Jacinda´ Mein, Jack Zhou, James Svensson, Jef Stanway, Jetha Chan, Jin Peng Zhou, Joana Carrasqueira, Joana Iljazi, Jocelyn Becker, Joe Fernandez, Joost van Amersfoort, Josh Gordon, Josh Lipschultz, Josh Newlan, Ju yeong Ji Kareem Mohamed, Kartikeya Badola, Kat Black, Katie Millican, Keelin McDonell, Kelvin Nguyen, Kiranbir Sodhia, Kish Greene, Lars Lowe Sjoesund, Lauren Usui, Laurent Sifre, Lena Heuermann, Leticia Lago, Lilly McNealus, Livio Baldini Soares, Logan Kilpatrick, Lucas Dixon, Luciano Martins, Machel Reid, Manvinder Singh, Mark Iverson, Martin Gorner, Mat Velloso, Mateo Wirth, Matt Davidow, Matt Miller, Matthew Rahtz,¨ Matthew Watson, Meg Risdal, Mehran Kazemi, Michael Moynihan, Ming Zhang, Minsuk Kahng, Minwoo Park, Mofi Rahman, Mohit Khatwani, Natalie Dao, Nenshad Bardoliwalla, Nesh Devanathan, Neta Dumai, Nilay Chauhan, Oscar Wahltinez, Pankil Botarda, Parker Barnes, Paul Barham, Paul Michel, Pengchong Jin, Petko Georgiev, Phil Culliton, Pradeep Kuppala, Ramona Comanescu, Ramona Merhej, Reena Jana,  
Reza Ardeshir Rokni, Rishabh Agarwal, Ryan Mullins, Samaneh Saadat, Sara Mc Carthy, Sarah Cogan, Sarah Perrin, Sebastien M. R. Arnold, Sebastian Krause, Shengyang Dai, Shruti Garg, Shruti Sheth, Sue´ Ronstrom, Susan Chan, Timothy Jordan, Ting Yu, Tom Eccles, Tom Hennigan, Tomas Kocisky, Tulsee Doshi, Vihan Jain, Vikas Yadav, Vilobh Meshram, Vishal Dharmadhikari, Warren Barkley, Wei Wei, Wenming Ye, Woohyun Han, Woosuk Kwon, Xiang Xu, Zhe Shen, Zhitao Gong, Zichuan Wei, Victor Cotruta, Phoebe Kirk, Anand Rao, Minh Giang, Ludovic Peran, Tris Warkentin, Eli Collins, Joelle Barral, Zoubin Ghahramani, Raia Hadsell, D. Sculley, Jeanine Banks, Anca Dragan, Slav Petrov, Oriol Vinyals, Jef Dean, Demis Hassabis, Koray Kavukcuoglu, Clement Farabet, Elena Buchatskaya, Sebastian Borgeaud, Noah Fiedel, Armand Joulin, Kathleen Kenealy, Robert Dadashi, and Alek Andreev. Gemma 2: Improving open language models at a practical size, 2024. URL https://arxiv.org/abs/2408.00118.  
[28] Irfan Ullah, Sharifullah Khan, Muhammad Imran, and Young-Koo Lee. Rweetminer: Automatic identification and categorization of help requests on twitter during disasters. Expert Systems with Applications, 176:114787, 2021.  
[29] UNHCR. Emergency response and guidelines, 2023. URL https://www.unhcr.org. Retrieved from UNHCR Oficial Website.  
[30] UNICEF. Humanitarian action reports and guidelines, 2023. URL https://www.unicef.org. Retrieved from UNICEF Oficial Website.  
[31] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, L. Kaiser, and I. Polosukhin. Attention is all you need. In Advances in Neural Information Processing Systems, volume 30, 2017.  
[32] Fedor Vitiugin and Hemant Purohit. Multilingual serviceability model for detecting and ranking help requests on social media during disasters. In Proceedings ofthe International AAAI Conference on Web and Social Media, volume 18, pages 1571–1584, 2024.  
[33] H. Zade, A. Shah, M. Imran, and F. O. Ostermann. From situational awareness to actionability: Towards improving the utility of social media data for crisis response. Proceedings of the ACM on Human-Computer Interaction, 2(CSCW):1–18, 2018.  
[34] Ting Zhang, Shi Shen, Changxiu Cheng, Kai Su, and Xiangxue Zhang. A topic model based framework for identifying the distribution of demand for relief supplies using social media data. International Journal of Geographical Information Science, 35(11):2216–2237, 2021.  
[35] L. Zhou, X. Wu, Z. Xu, and H. Fujita. VictimFinder: Harvesting rescue requests in disaster response from social media with BERT. Computers, Environment and Urban Systems, 95:101824, 2022. doi: 10.1016/j.compenvurbsys.2022.101824.  
[36] Yang Zhou, Jing Li, Yifan Xiang, Hao Yan, Lin Gui, and Yulan He. The mystery of in-context learning: A comprehensive survey on interpretation and analysis. In Proceedings ofthe 2024 Conference on Empirical Methods in Natural Language Processing, pages 14365–14378. Association for Computational Linguistics, 2024.  
[37] Abdul Wahab Ziaullah, Ferda Ofli, and Muhammad Imran. Monitoring critical infrastructure facilities during disasters using large language models. In Proceedings ofthe International ISCRAM Conference, 2024. doi: https://doi.org/10.59297/755e8b64.