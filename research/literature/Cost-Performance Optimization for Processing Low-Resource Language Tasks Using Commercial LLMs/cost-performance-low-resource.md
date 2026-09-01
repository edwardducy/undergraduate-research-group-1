# **Cost-Performance Optimization for Processing Low-Resource Language Tasks Using Commercial LLMs**

**Arijit Nag** IIT Kharagpur

arijitnag@iitkgp.ac.in

**Animesh Mukherjee**

IIT Kharagpur animeshm@cse.iitkgp.ac.in

**Niloy Ganguly** IIT Kharagpur

niloy@cse.iitkgp.ac.in

**Soumen Chakrabarti** IIT Bombay

soumen@cse.iitb.ac.in

# **Abstract**

Large Language Models (LLMs) exhibit impressive zero/few-shot inference and generation quality for high-resource languages (HRLs). A few of them have been trained on low-resource languages (LRLs) and give decent performance. Owing to the prohibitive costs of training LLMs, they are usually used as a network service, with the client charged by the count of input and output tokens. The number of tokens strongly depends on the script and language, as well as the LLM's subword vocabulary. We show that LRLs are at a pricing disadvantage, because the wellknown LLMs produce more tokens for LRLs than HRLs. This is because most currently popular LLMs are optimized for HRL vocabularies. Our objective is to level the playing field: reduce the cost of processing LRLs in contemporary LLMs while ensuring that predictive and generative qualities are not compromised. As means to reduce the number of tokens processed by the LLM, we consider code-mixing, translation, and transliteration of LRLs to HRLs. We perform an extensive study using the IndicXTREME classification and six generative tasks dataset, covering 15 Indic and 3 other languages, while using GPT-4 (one of the costliest LLM services released so far[<sup>1</sup>](#page-0-0) ) as a commercial LLM. We observe and analyze interesting patterns involving token count, cost, and quality across a multitude of languages and tasks. We show that choosing the best policy to interact with the LLM can reduce cost by ∼90% while giving better or comparable performance, compared to communicating with the LLM in the original LRL.

# **1 Introduction**

Large Language Models (LLMs) like GPT-4 [\(Ope](#page-10-0)[nAI et al.,](#page-10-0) [2023\)](#page-10-0), [ChatGPT](https://openai.com/blog/chatgpt), Llama-2 [\(Touvron](#page-10-1) [et al.,](#page-10-1) [2023\)](#page-10-1), PaLM [\(Chowdhery et al.,](#page-9-0) [2022\)](#page-9-0), *inter alia*, are greatly contributing to the advancement

of NLP with their exceptional zero/few-shot inference and generation abilities.

LLMs can also reduce dependence on expensive human-generated gold data for finetuning models for various downstream tasks, particularly for LRLs where gold data is scarce. However, our experience suggests that this benefit is offset by one problem. Commercial LLM services like GPT-4 charge by the number of tokens exchanged with the client. The typical message from the client to the LLM consists of a *task description* or *instruction*, followed by zero or more *few-shot* "in-context examples", and the payload *instance* to be solved. The output tokens from the LLM express the solution to the payload instance.

The number of tokens exchanged between the client and LLM service depends on the (subword) vocabulary of the LLM, and the language(s) that the client uses. If the client's language is a LRL, chances are, the LLM will heavily segment the LRL tokens into subwords, because LRL subwords have minority status in most popular LLMs today [\(Hong et al.,](#page-10-2) [2021\)](#page-10-2). Consequently, use of the LLM will be expensive compared to HRL clients.

Despite giant leaps in reasoning and instructionfollowing capabilities, GPT-4 effectively discriminates against LRL clients in the same manner, and for the same reason, as its predecessor multilingual models [mBERT](https://github.com/google-research/bert/blob/master/multilingual.md) and XLM-R [\(Conneau et al.,](#page-9-1) [2020\)](#page-9-1). This results in LRL clients being disadvantaged in terms of pricing, if not also quality. Our goal is to reduce this inequity.

Our choice of GPT-4 is driven by the fact that it is one of the costliest (yet most popular) commercial blackbox LLM-as-a-service. The cost of using the GPT-4 8K context model API is \$0.03 per 1,000 input tokens and \$0.06 per 1,000 output tokens. For the 32K context model, the cost is as high as \$0.06 per 1,000 input tokens and \$0.12 per 1,000 output tokens. Thus, a task of summarizing the Wikipedia (∼6 billion tokens) to half its size

<span id="page-0-0"></span><sup>1</sup> <http://tinyurl.com/llm-costing>

would cost \$720,000 and \$360,000 with GPT-4 having context length of 32K and 8K respectively.

Our initial study shows that GPT-4 generates diverse numbers of tokens for translations of a source sentence in different Indian languages. We also get a profile of default LRL task performances in these languages, if directly communicated to the LLM. The tool at our disposal is to preprocess the LRL messages from client to LLM service, in the face of a black-box service that we cannot influence. Specifically, we try to decrease the API cost by reducing the number of tokens exchanged between the client and LLM, while trying to maintain task quality. This can be done in various ways, as shown in Figure [1](#page-1-0) and detailed later.

To compare these approaches, we define (in equation [1\)](#page-3-0) a metric called the *RelaTive Performance to Cost Ratio* (RTPCR), which specifies the task performance we can get, given the tokendriven cost we pay using various preprocessing methods M, relative to using the original LRL text. We desist from directly comparing the cost of LLM service access against operating costs of an in-house LLM, because the latter depends on too many hard-to-model factors like power and cooling costs (and comparing against the LLM service).

<span id="page-1-0"></span>![](_page_1_Diagram_5.jpeg)

**Cricket ভারতে ক্রিকেট ভারতে** We summarize our contributions. (1) We identify a key cost-performance disadvantage faced by LLM clients that wish to solve LRL problems. (2) We demonstrate that this disadvantage is present in GPT-4 with respect to 15 Indian languages for diverse classification and generation tasks, using a new measure we introduce, called RTPCR. (3) We experiment with implicit and explicit client message translation and compare their RTPCR scores, to show that a free translator can reduce cost *and* improve performance. (4) We experiment with wordmixing and transliteration and show that both help reduce cost, but transliteration may result in modest performance impacts. (5) Apart from aggregative performance, we also present many case studies with interesting findings: The number of token counts varies greatly across Indian languages; translation to English may both increase and decrease the number of English words, depending on the source language; and LLM performance is rather sensitive to syntax.

Figure 1: Different numbers of tokens generated by GPT-4 before and after preprocessing a LRL sentence using various techniques.

Explicit instance translation using LLM before inference will accrue extra cost, compared to freeof-charge translation using an open-source machine translation tool. An alternative is *implicit translation*, in which we instruct the LLM to translate the payload instance to English in the background before solving it, thus saving explicit translation cost. (We do not ask the LLM for the translated version.) We observe that the improvement using implicit translation trails behind ex-

plicit translation. Therefore, using a dedicated LRL-to-HRL translator outside the LLM service can save cost *and* improve LRL task quality. A problem may arise if the LRL does not have a high-quality, open-sourced translator available and falling back on the LLM for translation will increase costs and decrease RTPCR. We next strike out in a *third* direction: *can mixing LRL and HRL words help*? To reduce cost, we can use a LRL-HRL or LRL-LRL dictionary, because such translations are often context-independent. Lastly, as a natural companion to selective wordmixing, we experiment with a *transliteration* technique, in which we transliterate the whole LRL message into Latin script. After all, the giant corpora used to train LLMs are more likely to contain LRL words heavily transliterated into Latin script, wordmixed with native LRL and HRL words, than 'pure' LRLs.

# **2 Related work**

Recently LLMs like [ChatGPT](https://openai.com/blog/chatgpt), GPT-4 [\(OpenAI](#page-10-0) [et al.,](#page-10-0) [2023\)](#page-10-0), Llama-2 [\(Touvron et al.,](#page-10-1) [2023\)](#page-10-1), PaLM [\(Chowdhery et al.,](#page-9-0) [2022\)](#page-9-0), BLOOM [\(Scao](#page-10-3) [et al.,](#page-10-3) [2023\)](#page-10-3), etc, have shown incredible zeroshot and few-shot capabilities across language and tasks. Although their performance in highresource languages (HRLs) is impressive, studies [\(Hendy et al.,](#page-9-2) [2023;](#page-9-2) [Jiao et al.,](#page-10-4) [2023;](#page-10-4) [Bang](#page-9-3) [et al.,](#page-9-3) [2023\)](#page-9-3) find that the same level of performance

is not seen for low-resource languages (LRLs). Various benchmark datasets like XTREME [\(Hu](#page-10-5) [et al.,](#page-10-5) [2020\)](#page-10-5), XTREME-R [\(Ruder et al.,](#page-10-6) [2021\)](#page-10-6), and XGLUE [\(Liang et al.,](#page-10-7) [2020\)](#page-10-7) have been designed to gauge the cross-lingual capabilities of a multilingual LLM. Benchmarks are also available in LRLs, e.g., IndicXTREME [\(Doddapaneni et al.,](#page-9-4) [2023\)](#page-9-4) for Indian languages. Benchmarks are also available in other language families like African [\(Ade](#page-9-5)[lani et al.,](#page-9-5) [2022\)](#page-9-5) and Indonesian [\(Wilie et al.,](#page-10-8) [2020\)](#page-10-8). In this work, we work with IndicXTREME. MEGA [\(Ahuja et al.,](#page-9-6) [2023\)](#page-9-6) witnesses improved task performance when translating the test instance to English. Having said that, they experiment from the performance perspective using the Bing translator (not an open-sourced tool). Unlike them, our focus is not only on performance but also on the cost. As inference using GPT-4 is costly for LRLs, we experiment with different techniques to reduce cost without hurting the performance. Similarly, a recent study [\(Huang et al.,](#page-10-9) [2023\)](#page-10-9) finds that a prompt written in English performs better than one written in other low-resource languages. There are works like [\(Ahia et al.,](#page-9-7) [2023\)](#page-9-7) and [\(Petroni et al.,](#page-10-10) [2019\)](#page-10-10) which explore the over-fragmentation of tokenizer in different languages, and later one proposes a metric called Parity premium to quantify the tokenization disparity between two languages. We compared the Parity premium metric with our RTPCR and found latter more robust. Frugal-GPT [\(Chen et al.,](#page-9-8) [2023\)](#page-9-8) tries to address the cost aspect of GPT-4 on a different level, like selecting the fewer but more effective in-context examples for few-shot inference, caching previous queries and responses for future use, or LLM-cascading where cheap LLMs are tried first to check if the response is reasonable and only go for expensive ones if previous responses are not satisfactory. They have experimented with English, where the words are not fragmented much. LRLs present a steeper challenge, where first we need to reduce the cost by controlling over-fragmentation of LRL words. FrugelGPT can be applied thereafter to further reduce the cost. As per our knowledge, we are first to study the effect on performance when prepossessing the LRL input instance in a variety of different ways and relating it to the LLM API cost.

# **3 Methodology**

In this work, we primarily preprocess the original LRL input instances using the techniques below to reduce the number of tokens generated.

#### **3.1 Native**

Here, we pass the query instance as it is in the native language script. We compare this scheme with all those stated below.

#### **3.2 Translation**

Here, we translate the LRL input instances before passing through the GPT-4. To translate, we consider with three possibilities.

**Using open-source MT:** We rely on off-the-shelf machine translation tools to translate LRL to English. We use IndicTrans2 [\(Gala et al.,](#page-9-9) [2023\)](#page-9-9) for this purpose.

**Using GPT-4 explicitly:** We first use GPT-4 as a translator to translate the LRL sentences to English; we then pass those translated English sentences to GPT-4 for inference.

**Using GPT-4 implicitly:** Here, our prompt (see Appendix Figure [7\)](#page-16-0) instructs GPT-4 to translate LRL instances to English "within its premises" if it faces difficulty in understanding the LRL instances. The hope is that some cost will be saved because we do not ask for the English translation. The main problem with implicit translation is that we cannot control which LRL instances it chooses to translate, or the quality or effects of translation. Between the two explicit translation approaches, from the point of view of cost, obviously, the opensource MT tool is better, but it might be possible that for some LRLs, such MT may not be readily available. In such cases, explicit translation using GPT-4 can help, but it will come with the extra cost burden of API calling to do the translation beforehand.

#### **3.3 Wordmix**

In this approach, if a LRL word is excessively fragmented wrt the LLM subword vocabulary (see Algorithm [1](#page-15-0) in Appendix [B](#page-15-1) for details), then we will replace that word with its corresponding English translation. We call the new input **Wordmix(En)** as it is now a 'synthetic' text that contains a mix of LRL and English (or a more advantaged LRL, say Hindi — **Wordmix(Hi)**) words thus giving it the name. Note that, in the wordmix approach, we do not need any translation tool; an LRL-HRL or LRL-LRL dictionary suffices. The extreme case is where all LRL words are individually translated to English — **W2W**. For results of Wordmix(Hi) and W2W see Appendix [A.1.](#page-12-0)

<span id="page-3-2"></span>![](_page_3_Figure_0.jpeg)

Figure 2: Average token generated per word by GPT-4. Here, we show the average number of tokens generated per word by GPT-4 when a sentence is processed in its own script (Native), translated via GPT-4 (labeled: Translation(GPT-4)), translated via open-sourced machine translation tool (labeled: Translation(IndicTrans)), and transliterated in English (labeled: Transliteration) for different Indian languages across various tasks.

#### **3.4 Transliteration**

We transliterate[<sup>2</sup>](#page-3-1) the LRL script into English (Latin script) in this approach. As we will now process the original LRL input instance in Latin script, the hypothesis is that fewer tokens shall be generated (as shown in Figure [1\)](#page-1-0) than the original LRL script. Consequently, the cost should decrease. Another motivation behind using transliteration is that the training corpus of GPT-4 (and other popular LLMs, although mostly undisclosed) is likely to include social media or marketplace data, which includes a great deal of LRL text in Latin script.

#### **3.5 RTPCR**

To rank these techniques relative to using native script wrt both cost and performance, we devised a metric called RTPCR (RelaTive Performance to Cost Ratio):

<span id="page-3-0"></span>RTPCR(*M*) = Perf(M) Perf(*Native*) Cost(M) Cost(*Native*) (1)

Here Perf(M) and Perf(*Native*) denote the taskspecific performance of the method M and native LRL script, respectively. Similarly, Cost(M) and Cost(*Native*) represent the cost incurred by the method M and native LRL script, respectively. RTPCR will be high when the performance of M is better than native, and the cost is lower for M than native. Conversely, a low RTPCR value indicates that M lags in terms of performance but accrues more cost than using native text. Table [5](#page-5-0) shows RTPCR values of different techniques.

## **4 Experiments and results**

#### **4.1 Datasets and evaluation metric**

We experiment with all the classification tasks available in the IndicXTREME [\(Doddapaneni](#page-9-4) [et al.,](#page-9-4) [2023\)](#page-9-4) benchmark dataset and six generative datasets which include summarization (XLSUM [\(Hasan et al.,](#page-9-10) [2021\)](#page-9-10)), Question-Answering(Chaii [\(Addison Howard,](#page-9-11) [2021\)](#page-9-11), TyDi QA [\(Clark et al.\)](#page-9-12)), Machine Translation(Samanantar [\(Ramesh et al.,](#page-10-11) [2022\)](#page-10-11)) and mathematical reasoning tasks (MGSM [\(Shi](#page-10-12) [et al.,](#page-10-12) [2022\)](#page-10-12), Conic10K [\(Wu et al.,](#page-10-13) [2023\)](#page-10-13) and ArMath [\(Alghamdi et al.,](#page-9-13) [2022\)](#page-9-13)) total covering 15 Indian languages along with Thai, Swahili and Chinese. (details in Table [10\)](#page-11-0). We report the accuracy score as a performance metric for classification tasks, ROUGE for summarization, BLEU for machine translation and Exact Match for question-answering and mathematical reasoning tasks. To make the GPT-4's output consistent and deterministic as much as possible, we set the seed and temperature parameters constant throughout the experiments. The Appendix discusses all the details related to different prompt designs (Figure [7\)](#page-16-0) and GPT-4 API hyperparameters (Table [13\)](#page-15-2).

<span id="page-3-1"></span><sup>2</sup> <https://github.com/libindic/indic-trans>

<span id="page-4-0"></span>

<span id="page-4-1"></span>Table 1: Classification tasks: the change in accuracy (in terms of %) wrt the native script. The best results are in **boldface** and underlined. The last column shows the average performance across languages.

Table 2: Generation tasks: the change in BLEU(MT)/ROUGE(Summarization)/Exact match(QA,Mathematical reasoning) (in terms of %) wrt the native script. The best results are in **boldface** and underlined. The last column shows the average performance across languages.

<span id="page-4-2"></span>

Table 3: Classification tasks: the change in token generation per instance (in terms of %) wrt the native script is shown. The best results are in **boldface** and underlined. The last column shows the average cost across languages.

#### **4.2 Latin script reduces token generation**

In Figure [2,](#page-3-2) we show the average number of tokens generated per word if a sentence passes through the 15685

<span id="page-5-1"></span>

Table 4: Generation tasks: the change in token generation per instance (in terms of %) wrt the native script is shown. The best results are in **boldface** and underlined. The last column shows the average cost across languages.

<span id="page-5-0"></span>

Table 5: RTPCR values for all techniques across all classification datasets and languages. The best results are in **boldface** and underlined. The last column shows the average RTPCR across languages.

GPT-4 in its own script, is translated to English using open-sourced MT and GPT-4, and is transliterated to English script. The general observation is that token generation is greatly reduced when processing the instance in Latin script (translation or transliteration). The reduction can be anywhere between 2× and 7×, depending on the languages. Another interesting observation is that the reduction is higher for Dravidian languages like Malayalam, Tamil, Telugu, Kannada, etc., while the reduction is comparatively less in North-Indian languages like Hindi and Bengali. This might imply that GPT -4 is better trained in North-Indian languages than South-Indian ones. Another thing to notice here is that although transliteration does not necessarily produce a valid English word, it still helps to reduce the number of token generation.

#### **4.3 Translation performs better**

Table [1](#page-4-0) and [2](#page-4-1) shows the change in performance (%) compared to the native script for classification and generation tasks, respectively (complete result in Appendix Table [14](#page-17-0) & [15\)](#page-18-0). For classification, it is clear that, on the whole, translation improves the GPT-4 performance, be it translation by GPT-4, open-source MT like IndicTrans or implicit translation by GPT-4. However, translation by IndicTrans has the upper hand most of the time. From this observation, it is clear that GPT-4 better understands text in English than LRLs. Next, Wordmix(En) also shows promise by giving a comparable performance to native script, given that our wordmix technique is a bit crude, and further improvement can be possible by improving the word replacement heuristics. Wordmix(Hi) also

<span id="page-6-0"></span>

Table 6: RTPCR values for all techniques across all generation datasets and languages. The best results are in **boldface** and underlined. The last column shows the average RTPCR across languages.

performs comparable to Wordmix(En), but the latter is more fruitful when considering the cost reduction (details in Appendix [A.1\)](#page-12-0). On the other hand, in transliteration, performance drops, and it is expected as we are not translating the word to English but changing the script. For generation tasks, except the reasoning tasks, direct translation does not seem to be beneficial from the performance point of view, but here, the implicit translation improves the performance of Summarization, MT and QA tasks.

#### **4.4 Open-source MT reduces cost**

In Table [3](#page-4-2) and [4,](#page-5-1) we compare the cost incurred by different techniques for GPT-4 inference. As the cost is proportional to the tokens generated from the inputs and outputs, we use the average token generated per query instance as a proxy for cost. We show the percentage change in token generation compared to the native script (While reporting the average token generation per instance for different techniques in Appendix Table [16](#page-19-0) & [17\)](#page-20-0). Here, the observation is clear: all the techniques reduce cost except the translation via GPT-4. It is expected since in this method, we first explicitly translate the instance using GPT-4 and then use the translated instance for classification/generation again using GPT-4. So here, we need to take the extra burden of translation costs using GPT-4. On the other hand, translation via IndicTrans (open-source MT) reduces the cost by as much as 90%. But in case of unavailability of such free MT,

one can go for wordmix or transliteration, which also reduces the cost significantly. Having said that, transliteration can impact performance (see Table [1](#page-4-0) & [2\)](#page-4-1).

#### **4.5 RTPCR helps find best value for money**

Table [5](#page-5-0) and [6](#page-6-0) shows RTPCR for different methods compared to the Native script. Here, the translation via IndicTrans is the clear winner for obvious reasons. The translation in this case is free, and GPT-4 performance as we see in Table [1](#page-4-0) improved when using English-translated query instances. The lowest RTPCR score is for translation using GPT-4 because although the performance improves after translation, the cost we pay for translation outweighs the performance gain. RTPCR for transliteration is also high — though the performance drops a bit, it reduces the cost significantly compared to using native script. RTPCRs for wordmix and implicit translation via GPT-4 are also better than native (RTPCR is 1), as wordmix performs comparably as native but reduces cost. On the other hand, the cost of implicit translation by GPT-4 is the same as the native script, but the performance gain is more for implicit translation.

#### **4.6 Comparing RTPCR to Parity premium**

We compare RTPCR with the recent Parity premium score [\(Petrov et al.,](#page-10-14) [2023\)](#page-10-14), which measures |t(sa)| |t(sb)| . Here |t(si)| represents the tokenization length of the sentence s in language i using tokenizer t. For our case, we take a and b as a partic-

<span id="page-7-0"></span>

<span id="page-7-1"></span>Table 7: Parity and RTPCR agree. Here, we show the RTPCR and Parity values of the sentence classification dataset for different Indian languages. Detailed results are in Appendix Table [19.](#page-20-1) The highest and lowest values are marked with underline and asterisk (∗), respectively.

Table 8: Parity and RTPCR disagree. Here, we show the RTPCR and Parity values of the mutable Wikidata object prediction dataset for different Indian languages. **Acc**↑ **(%)** represents the performance improvement (%) from native scripts to English translation. Detailed results are in Appendix Table [18.](#page-20-2) The highest RTPCR and Parity values are marked with underline and asterisk (∗), respectively.

ular native language and English, respectively. In Table [7,](#page-7-0) we show the RTPCR (for both translation and transliteration method) and Parity premium scores for the Indian sentiment analysis dataset. From the table, it is evident that both scores are highly correlated. The scores are highest for Odia and lowest in case of Urdu, which suggests English scripts can help Odia much higher compared to Urdu.

Next, inspired by Mulan [\(Fierro et al.,](#page-9-14) [2024\)](#page-9-14), we create a Wikidata object prediction dataset (dataset details in Appendix Section [A.1\)](#page-11-1) given the subject and relation for various Indian languages. Here the relation can be of three types, namely time-dependent (*mutable*) like *head of the country*, time-independent one-to-one (*immutable*-1) like *father* and one-to-many (*immutable*-*n*) like *border shared with*, making the task more challenging than normal sentiment classification tasks. In Table [8,](#page-7-1) we show the RTPCR and Parity scores for all the relation and language types. We also report the performance improvement percentage going from native script to English in the **Acc**↑ **(%)** column for all relation types. Here we see disagreement between the RTPCR and Parity scores. As an example, for the immutable-1 relation type, Gujarati and Odia produce the highest RTPCR and Parity scores, respectively. However, while the performance

improvement from Gujarati to English scripts is around 200%, for Odia, it is just around 15%. This observation is consistent with immutable-n and mutable relation types, where Malayalam and Odia have the highest RTPCR and Parity premium, respectively, but the improvement of Malayalam is higher compared to Odia. So our understanding from the whole exercise is as follows: tasks can be classified into two broad categories, one where the model finds it difficult as it does not understand the task in that particular language, and the second type is the task is difficult to the model irrespective of the language. RTPCR as a metric is more appropriate for the latter case, and for the former, both RTPCR and parity will portray a similar picture.

### <span id="page-7-2"></span>**4.7 Analysis of the results**

In Table [9,](#page-8-0) we show examples covering different cases in our methodology. The first two examples are related to the translation method. As shown earlier, translation improves the inference performance; however, this solely depends on the quality of the translation. In the first example, the original sentence in Bengali expresses a negative sentiment sarcastically; GPT-4 finds it challenging to comprehend it in Bengali and, as a result, produces a wrong classification. However, when we translate it to English (translation quality is good), GPT-4

<span id="page-8-0"></span>

Table 9: Case analysis. The first two examples show the translation technique's good and bad sides, followed by the next two examples related to the Wordmix(En) method. The last example shows how implicit translation can help with inference. (Note: To reduce cost we do not explicitly output the translation of the sentence; this is just for illustration purposes.) Detailed discussion is in section [4.7.](#page-7-2) Key sentiment-related words are underlined.

can classify it correctly. In the second example, although the translation looks acceptable on the surface, it misses the negative emotion latent in the Hindi sentence, resulting in a wrong prediction. The subsequent two cases are related to the wordmix technique. Here, the first one talks about the terrible (Bn: ভয়¿র) storytelling of a movie and also uses the phase "েসরা উদাহরণ" (great example); GPT-4 focuses on the later to make a wrong positive prediction. Nevertheless, when we replace ভয়¿র with 'terrible', GPT-4 produces the correct result. One problem with this approach is that we replace any word in the original sentence with its translation without looking at the context, and it can sometimes produce non-meaningful sentences. The following example shows such a case where wordmixing produces an incoherent Odia-English wordmix sentence, and GPT-4 fails to predict anything. The last example corresponds to the implicit translation using the GPT-4 while inference. Here, the original sentence in Bengali produces the wrong prediction, but when we advise GPT-4 to translate the Bn sentence to En "in the background" if needed, it produces the correct result. To clarify, we do not output the translation explicitly to reduce the output token generation; this

is just to verify and illustrate the technique's effectiveness.

# **5 Conclusion**

We study the cost of solving five classification and six generation task datasets across various Indian languages using GPT-4, the leading commercial LLM. While GPT-4 does well on these tasks, LRL words get highly fragmented, leading to high cost of API calls. We wish to retain the performance advantage of GPT-4 while reducing API cost. To that end, we try different pre-processing techniques involving translation, wordmixing, and transliteration. We find that translating the whole LRL sentence to English reduces the cost and improves performance (given a free LRL→HRL MT system). In the absence of such an MT system, implicit translation using GPT-4 or even an LRL-HRL dictionary can be used to replace highly fragmented words and reduce costs. We introduce the RTPCR metric that considers both the performance and cost aspects and gives a holistic picture of the gain achieved among the two transformation methods. In future, we wish to explore wordmix more deeply while using other HRLs related to the source LRL.

### **6 Limitations**

- <span id="page-9-8"></span>Despite high costs, GPT-4 is widely regarded as providing better quality than open-source counterparts like BLOOM [\(Scao et al.,](#page-10-3) [2023\)](#page-10-3) and GPTneo. GPT-4 has been compared and shown to be superior to BARD in multiple application scenarios including text processing and understanding[<sup>3</sup>](#page-9-15) . For this reason, and to contain costs, we only experiment with one commercial LLM, GPT-4. Although it will increase the experiment cost significantly, bringing other paid LLMs into the experimental setup will make our observations more robust. Also, our wordmix technique, where we replace an LRL word with its translation, without taking sentence context into account, is crude it sometimes impacts sentence structure. Replacing a word or phrase while considering sentence context will be more appropriate. In this work, we only focus on Indian languages. Although we cover as many as 15 languages, an immediate extension can be to check the hypothesis for other LRLs worldwide. **References** divy thakkar Julia Elliott Partha Talukdar Phil Culliton Addison Howard, Deepak Nathani. 2021. [chaii](https://kaggle.com/competitions/chaii-hindi-and-tamil-question-answering)  [hindi and tamil question answering.](https://kaggle.com/competitions/chaii-hindi-and-tamil-question-answering) David Ifeoluwa Adelani et al. 2022. [MasakhaNER 2.0:](http://arxiv.org/abs/2210.12391) [Africa-centric transfer learning for named entity](http://arxiv.org/abs/2210.12391) [recognition](http://arxiv.org/abs/2210.12391). Orevaoghene Ahia, Sachin Kumar, Hila Gonen, Jungo Kasai, David R. Mortensen, Noah A. Smith, and Yulia Tsvetkov. 2023. [Do all languages cost the same?](http://arxiv.org/abs/2305.13707) [tokenization in the era of commercial language mod](http://arxiv.org/abs/2305.13707)[els.](http://arxiv.org/abs/2305.13707) Kabir Ahuja, Harshita Diddee, Rishav Hada, Millicent Ochieng, Krithika Ramesh, Prachi Jain, Akshay Nambi, Tanuja Ganu, Sameer Segal, Maxamed Axmed, Kalika Bali, and Sunayana Sitaram. 2023. [Mega: Multilingual evaluation of generative ai.](http://arxiv.org/abs/2303.12528) Reem Alghamdi, Zhenwen Liang, and Xiangliang Zhang. 2022. [ArMATH: a dataset for solving Ara](https://aclanthology.org/2022.lrec-1.37)[bic math word problems](https://aclanthology.org/2022.lrec-1.37). In *Proceedings of the Thirteenth Language Resources and Evaluation Conference*, pages 351–362, Marseille, France. European Language Resources Association. Yejin Bang, Samuel Cahyawijaya, Nayeon Lee, Wenliang Dai, Dan Su, Bryan Wilie, Holy Lovenia, Ziwei Ji, Tiezheng Yu, Willy Chung, Quyet V. Do, Yan 3 [https://themeisle.com/blog/](https://themeisle.com/blog/chatgpt-vs-google-bard/#gref) [chatgpt-vs-google-bard/#gref](https://themeisle.com/blog/chatgpt-vs-google-bard/#gref) Xu, and Pascale Fung. 2023. [A multitask, multilin](http://arxiv.org/abs/2302.04023)[gual, multimodal evaluation of chatgpt on reasoning,](http://arxiv.org/abs/2302.04023) [hallucination, and interactivity](http://arxiv.org/abs/2302.04023). Lingjiao Chen, Matei Zaharia, and James Zou. 2023. [Frugalgpt: How to use large language models while](http://arxiv.org/abs/2305.05176) [reducing cost and improving performance](http://arxiv.org/abs/2305.05176). Aakanksha Chowdhery et al. 2022. [Palm: Scaling lan](http://arxiv.org/abs/2204.02311)[guage modeling with pathways](http://arxiv.org/abs/2204.02311). Jonathan H. Clark, Eunsol Choi, Michael Collins, Dan Garrette, Tom Kwiatkowski, Vitaly Nikolaev, and Jennimaria Palomaki. Tydi qa: A benchmark for information-seeking question answering in typologically diverse languages. Alexis Conneau, Kartikay Khandelwal, Naman Goyal, Vishrav Chaudhary, Guillaume Wenzek, Francisco Guzmán, Edouard Grave, Myle Ott, Luke Zettlemoyer, and Veselin Stoyanov. 2020. [Unsupervised](https://doi.org/10.18653/v1/2020.acl-main.747) [cross-lingual representation learning at scale](https://doi.org/10.18653/v1/2020.acl-main.747). In *Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics*, pages 8440– 8451, Online. Association for Computational Linguistics. Sumanth Doddapaneni, Rahul Aralikatte, Gowtham Ramesh, Shreya Goyal, Mitesh M. Khapra, Anoop Kunchukuttan, and Pratyush Kumar. 2023. [Towards](http://arxiv.org/abs/2212.05409) [leaving no indic language behind: Building monolin](http://arxiv.org/abs/2212.05409)[gual corpora, benchmark and models for indic lan](http://arxiv.org/abs/2212.05409)[guages.](http://arxiv.org/abs/2212.05409) Constanza Fierro, Nicolas Garneau, Emanuele Bugliarello, Yova Kementchedjhieva, and Anders Søgaard. 2024. [Mulan: A study of fact mutability](http://arxiv.org/abs/2404.03036) [in language models.](http://arxiv.org/abs/2404.03036) Jay Gala, Pranjal A Chitale, A K Raghavan, Varun Gumma, Sumanth Doddapaneni, Aswanth Kumar M, Janki Atul Nawale, Anupama Sujatha, Ratish Puduppully, Vivek Raghavan, Pratyush Kumar, Mitesh M Khapra, Raj Dabre, and Anoop Kunchukuttan. 2023. [Indictrans2: Towards high](https://openreview.net/forum?id=vfT4YuzAYA)[quality and accessible machine translation models](https://openreview.net/forum?id=vfT4YuzAYA) [for all 22 scheduled indian languages](https://openreview.net/forum?id=vfT4YuzAYA). *Transactions on Machine Learning Research*. Tahmid Hasan, Abhik Bhattacharjee, Md. Saiful Islam, Kazi Mubasshir, Yuan-Fang Li, Yong-Bin Kang,
  - M. Sohel Rahman, and Rifat Shahriyar. 2021. [XL](https://doi.org/10.18653/v1/2021.findings-acl.413)[sum: Large-scale multilingual abstractive summa](https://doi.org/10.18653/v1/2021.findings-acl.413)[rization for 44 languages.](https://doi.org/10.18653/v1/2021.findings-acl.413) In *Findings of the Association for Computational Linguistics: ACL-IJCNLP 2021*, pages 4693–4703, Online. Association for Computational Linguistics. Amr Hendy, Mohamed Abdelrehim, Amr Sharaf, Vikas Raunak, Mohamed Gabr, Hitokazu Matsushita, Young Jin Kim, Mohamed Afify, and Hany Hassan Awadalla. 2023. [How good are gpt models at ma](http://arxiv.org/abs/2302.09210)[chine translation? a comprehensive evaluation](http://arxiv.org/abs/2302.09210).

<span id="page-9-15"></span><span id="page-9-14"></span><span id="page-9-13"></span><span id="page-9-12"></span><span id="page-9-11"></span><span id="page-9-10"></span><span id="page-9-9"></span><span id="page-9-7"></span><span id="page-9-6"></span><span id="page-9-5"></span><span id="page-9-4"></span><span id="page-9-3"></span><span id="page-9-2"></span><span id="page-9-1"></span><span id="page-9-0"></span>

<span id="page-10-14"></span><span id="page-10-13"></span><span id="page-10-12"></span><span id="page-10-11"></span><span id="page-10-10"></span><span id="page-10-9"></span><span id="page-10-8"></span><span id="page-10-7"></span><span id="page-10-6"></span><span id="page-10-5"></span><span id="page-10-4"></span><span id="page-10-3"></span><span id="page-10-2"></span><span id="page-10-1"></span><span id="page-10-0"></span>Jimin Hong, TaeHee Kim, Hyesu Lim, and Jaegul Choo. 2021. [AVocaDo: Strategy for adapting vocabu](https://doi.org/10.18653/v1/2021.emnlp-main.385)[lary to downstream domain](https://doi.org/10.18653/v1/2021.emnlp-main.385). In *Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing*, pages 4692–4700, Online and Punta Cana, Dominican Republic. Association for Computational Linguistics. Junjie Hu, Sebastian Ruder, Aditya Siddhant, Graham Neubig, Orhan Firat, and Melvin Johnson. 2020. [XTREME: A massively multilingual multi](http://arxiv.org/abs/2003.11080)[task benchmark for evaluating cross-lingual general](http://arxiv.org/abs/2003.11080)[ization.](http://arxiv.org/abs/2003.11080) Haoyang Huang, Tianyi Tang, Dongdong Zhang, Xin Zhao, Ting Song, Yan Xia, and Furu Wei. 2023. [Not](https://doi.org/10.18653/v1/2023.findings-emnlp.826) [all languages are created equal in LLMs: Improv](https://doi.org/10.18653/v1/2023.findings-emnlp.826)[ing multilingual capability by cross-lingual-thought](https://doi.org/10.18653/v1/2023.findings-emnlp.826) [prompting](https://doi.org/10.18653/v1/2023.findings-emnlp.826). In *Findings of the Association for Computational Linguistics: EMNLP 2023*, pages 12365– 12394, Singapore. Association for Computational Linguistics. Wenxiang Jiao, Wenxuan Wang, Jen tse Huang, Xing Wang, Shuming Shi, and Zhaopeng Tu. 2023. [Is](http://arxiv.org/abs/2301.08745) [chatgpt a good translator? yes with gpt-4 as the en](http://arxiv.org/abs/2301.08745)[gine.](http://arxiv.org/abs/2301.08745) Yaobo Liang et al. 2020. [Xglue: A new benchmark](http://arxiv.org/abs/2004.01401) [dataset for cross-lingual pre-training, understanding](http://arxiv.org/abs/2004.01401) [and generation.](http://arxiv.org/abs/2004.01401) OpenAI et al. 2023. [Gpt-4 technical report](http://arxiv.org/abs/2303.08774). Fabio Petroni, Tim Rocktäschel, Sebastian Riedel, Patrick Lewis, Anton Bakhtin, Yuxiang Wu, and Alexander Miller. 2019. [Language models as knowl](https://doi.org/10.18653/v1/D19-1250)[edge bases?](https://doi.org/10.18653/v1/D19-1250) In *Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP)*, pages 2463–2473, Hong Kong, China. Association for Computational Linguistics. Aleksandar Petrov, Emanuele La Malfa, Philip H. S. Torr, and Adel Bibi. 2023. [Language model tokeniz](http://arxiv.org/abs/2305.15425)[ers introduce unfairness between languages.](http://arxiv.org/abs/2305.15425) Gowtham Ramesh, Sumanth Doddapaneni, Aravinth Bheemaraj, Mayank Jobanputra, Raghavan AK, Ajitesh Sharma, Sujit Sahoo, Harshita Diddee, Mahalakshmi J, Divyanshu Kakwani, Navneet Kumar, Aswin Pradeep, Srihari Nagaraj, Kumar Deepak, Vivek Raghavan, Anoop Kunchukuttan, Pratyush Kumar, and Mitesh Shantadevi Khapra. 2022. [Samanantar: The largest publicly available parallel](https://doi.org/10.1162/tacl_a_00452) [corpora collection for 11 Indic languages.](https://doi.org/10.1162/tacl_a_00452) *Transactions of the Association for Computational Linguistics*, 10:145–162. Sebastian Ruder, Noah Constant, Jan Botha, Aditya Siddhant, Orhan Firat, Jinlan Fu, Pengfei Liu, Junjie Hu, Dan Garrette, Graham Neubig, and Melvin Johnson. 2021. [Xtreme-r: Towards more challenging and nu](http://arxiv.org/abs/2104.07412)[anced multilingual evaluation](http://arxiv.org/abs/2104.07412). Teven Le Scao et al. 2023. [Bloom: A 176b-parameter](http://arxiv.org/abs/2211.05100) [open-access multilingual language model](http://arxiv.org/abs/2211.05100). Freda Shi, Mirac Suzgun, Markus Freitag, Xuezhi Wang, Suraj Srivats, Soroush Vosoughi, Hyung Won Chung, Yi Tay, Sebastian Ruder, Denny Zhou, Dipanjan Das, and Jason Wei. 2022. [Language models](http://arxiv.org/abs/2210.03057) [are multilingual chain-of-thought reasoners.](http://arxiv.org/abs/2210.03057) Hugo Touvron et al. 2023. [LLaMA 2: Open foundation](http://arxiv.org/abs/2307.09288) [and fine-tuned chat models](http://arxiv.org/abs/2307.09288). Bryan Wilie et al. 2020. [IndoNLU: Benchmark and re](https://aclanthology.org/2020.aacl-main.85)[sources for evaluating Indonesian natural language](https://aclanthology.org/2020.aacl-main.85) [understanding.](https://aclanthology.org/2020.aacl-main.85) In *Proceedings of the 1st Conference of the Asia-Pacific Chapter of the Association for Computational Linguistics and the 10th International Joint Conference on Natural Language Processing*, pages 843–857, Suzhou, China. Association for Computational Linguistics. Haoyi Wu, Wenyang Hui, Yezeng Chen, Weiqi Wu, Kewei Tu, and Yi Zhou. 2023. [Conic10K: A chal](https://doi.org/10.18653/v1/2023.findings-emnlp.427)[lenging math problem understanding and reasoning](https://doi.org/10.18653/v1/2023.findings-emnlp.427) [dataset.](https://doi.org/10.18653/v1/2023.findings-emnlp.427) In *Findings of the Association for Computational Linguistics: EMNLP 2023*, pages 6444–6458, Singapore. Association for Computational Linguistics.

# **Cost-Performance Optimization for Processing Low-Resource Language Tasks Using Commercial LLMs (Appendix)**

# **A Supplementary results**

<span id="page-11-0"></span>

Table 10: Dataset details. Examples (in English) are for illustration; actual datasets are in Indian languages.

#### **Datasets**

# <span id="page-11-1"></span>**A.1 IndicMulan**

The details of the Indic-Mulan dataset are in Table [11](#page-12-1) and Table [12.](#page-12-2) Table [11](#page-12-1) shows the number of <subject, relation, object> triplets across 11 Indian languages. And Table [12](#page-12-2) shows the relation-wise query count across languages. In Figure [7,](#page-16-0) we show the prompt used for object prediction.

**Distinct words reduces after translation** In Figure [4,](#page-13-0) we plot the change in distinct words in an LRL sentence before and after the translation using both the GPT-4 and open-source MT tools. Surprisingly, we

<span id="page-12-1"></span>

Table 11: Triplets across relation types for 11 Indian languages in Indic-MULAN.

<span id="page-12-2"></span>

Table 12: Number of triplets for each relation across 11 Indian languages in Indic-MULAN.

observe that for most of the languages, the number of distinct words drops significantly after translation for both methods. The number of distinct words increased after translation only for languages – Hindi, Urdu, and Punjabi. One thing that needs to be noted here is that the reduction of distinct words is more for the South-Indian languages than other Indic languages. This can potentially lead to a loss of information, and as an effect, the model's performance after translation can suffer.

**Wordmix(Hi)** In place of English it is also possible to use a more advantaged LRL. We did our experiments with Hindi as such a choice since among the Indian LRLs, Hindi words are comparatively less fragmented (as shown in Figure [2\)](#page-3-2). We call this variant **Wordmix(Hi)**.

<span id="page-12-0"></span>**Effect of wordmixing** In Figure [5,](#page-14-0) we check the effect of wordmixing in more detail. We compare two different wordmixing techniques involving English and Hindi as target languages to translate an LRL

<span id="page-13-0"></span>![](_page_13_Figure_4.jpeg)

Figure 4: Change in distinct words after translation. Here, we compare change in distinct words after translation using (IndicTrans and GPT-4) with sentence in native scripts for different Indian languages across tasks.

word. While the choice of English is understandable, we choose Hindi because it is fragmented less compared to other Indian languages (shown in Figure [2\)](#page-3-2) and also with the assumption that a language from the same language family can cooperate better in a sentence. As per observation, although Wordmix(Hi) cannot reduce the *cost* as compared to Wordmix(En), Wordmix(Hi) often *performs* better than Wordmix(En). To some extent, this validates a hypothesis of effective transfer between LRL members of the same language family. Having said that, these observations demand a deeper exploration involving different target languages to get a deeper understanding.

**W2W (Dictionary)** Here, we take a LRL-English dictionary and replace each LRL word with its corresponding English translation. It is quite obvious that W2W will hamper the sentence syntax and semantics, but having said that, we use it as we are getting a free approximate translation of the LRL sentence and also to check if word order really matters in GPT-4 inference.

**Sentence structure matters** In Figure [6,](#page-14-1) we compare the performance of GPT-4 between techniques that import English words in the original native script sentence. These techniques change the structure or syntax of the sentence differently, unlike the full-sentence translation using MT, which keeps the sentence syntax intact. For instance, wordmix can change sentence structure as it heuristically replaces any word with its corresponding English translation, totally oblivious of the context. Similarly, the extreme case can be W2W translation, where we use a dictionary to replace each word in the original sentence with

<span id="page-14-0"></span>![](_page_14_Figure_0.jpeg)

Figure 5: GPT-4 wordmixing performance vs cost reduction comparison while mixing native script with an HRL (English) and LRL (Hindi). Here, we plot the accuracy using native script, Wordmix(Native-En), and Wordmix(Native-Hi) sentence inference along with the change of API cost compared to native script inferencing for different Indian languages across tasks.

<span id="page-14-1"></span>![](_page_14_Figure_2.jpeg)

Figure 6: Impact on different ways of incorporating English words while GPT-4 inferencing. Here, we compare performance using native script, full sentence translation using MT (IndicTrans), partial native word replacement to English (Wordmix(En)), and full word-to-word translation (W2W) using LRL-HRL dictionary for different Indian languages across tasks.

its corresponding English word. This can produce a completely incoherent sentence without proper syntax. Our experiment shows out of these three techniques, total translation using MT always does better, followed by Wordmix(En). W2W gives the worst performance. This establishes the fact that although English script is much easier for GPT-4 to understand, word structure matters.

## <span id="page-15-1"></span>**B Word selection algorithm**

<span id="page-15-0"></span>**Algorithm 1** wordmix: Word selection. **inputs** LRL corpus <sup>D</sup>, LLM tokenizer T , LRL-HRL dictionary DicLtH **outputs** wordmix LRL corpus DCM 1: <sup>D</sup>CM ← empty list 2: <sup>L</sup><sup>W</sup> ← empty list <sup>▷</sup> *Word length list* 3: SR<sup>W</sup> ← empty list <sup>▷</sup> *Token to word length split ratio list* 4: W ← distinct words from corpus <sup>D</sup> 5: **for** <sup>w</sup> ∈ W **do** 6: <sup>L</sup><sup>W</sup> ← len(w) 7: SR<sup>W</sup> ← len(<sup>T</sup> (w)) len(w) 8: **for** <sup>s</sup> ∈ <sup>D</sup> **do** 9: **for** <sup>w</sup> ∈ <sup>s</sup> **do** 10: **if** len(w) ≥ mean(L<sup>W</sup> ) & len(T (w)) len(w) ≥ mean(SR<sup>W</sup> ) **then** 11: Replace w with DicLtH(w) in s 12: <sup>D</sup>CM ← <sup>s</sup> **return** DCM

## **C Experimental settings**

<span id="page-15-2"></span>

Table 13: Details of GPT-4 hyperparameters.

We use GPT-4-0613 model, which costs \$0.03 / 1K tokens for input and \$0.06 / 1K tokens for output. We run all the experiments in 16GB RAM CPU based system, without any GPU usage. To preserve cost, we do all the experiments one time, and to make them reproducible we fix the seed value to 42 and set temperature close to zero for classification tasks of the GPT-4 API.

#### **Native Script Prompt description: Classify the sentiment(positive/negative) of the Hindi sentence.**

**Sentence:**

# <span id="page-16-0"></span>**वीडियो कॉलिं<sup>ग</sup> के लिए, वे आपको अपना मैसेंजर इनस्टॉल करने के लि<sup>ए</sup> कहेंगे। तो मू<sup>ल</sup> रू<sup>प</sup> से एक सोशल मीडिया के लिए, आपको 2 अलग-अलग ऐप्स की आवश्यकता है, क्या यह बकवास नहीं है? Codemix instance**

**Label:**

**Translated/W2W instance**

**Prompt description: Classify the sentiment(positive/negative) of the sentence.**

**Sentence: For video calling, they will ask you to install your messenger. So basically for a social media, you need 2**

**different apps, isn't that rubbish?**

**Label:**

**Transliterated instance**

**Prompt description: Classify the sentiment(positive/negative) of the Hindi to English transliterated sentence. Sentence: video colling ke liye, ve aapako apana massenger inastol karne ke liye kahenge. to mul rup se ek soshal**

**media ke liye, aapako 2 alag-alag eps kii aavashyaktaa he, kya yah bakawaas nahin he?**

**Label:**

**Prompt description: Classify the sentiment(positive/negative) of the Hindi-English codemixed sentence.**

**Sentence: Video calling** 

**के लिए, वे आपको अपना Messenger Install करने के लिए will say. तो मू<sup>ल</sup> रू<sup>प</sup> से एक सोशल मीडिया के लिए, आपको 2 different ऐप्स की Necessity है, क्या यह बकवास नहीं है? वीडियो कॉलिं<sup>ग</sup> के लिए, वे आपको अपना मैसेंजर इनस्टॉल करने के लि<sup>ए</sup> कहेंगे। तो मू<sup>ल</sup> रू<sup>प</sup> से एक सोशल मीडिया के लिए, आपको 2 अलग-अलग ऐप्स की आवश्यकता है, क्या यह बकवास नहीं है?** Figure 7: Prompts used for different techniques. Here, all these prompts are related to the IndicSentiment dataset.

**Label:**

#### **Implicit translation instance**

**Prompt description: Classify the sentiment(positive/negative) of the Sentence. For your better understanding you**

**can translate the Hindi Sentence to English in background when needed.**

**Sentence:**

**Label:**

<span id="page-17-0"></span>

Table 14: Accuracy of all techniques for all classification datasets and languages. Below each accuracy value, the change in performance (in terms of %) wrt the native script is shown. The best results are in **boldface** and underlined. The last column shows the average performance across languages.

<span id="page-18-0"></span>

Table 15: Performance(BLEU for MT/ROUGE for Summarization/EM for QA and Mathematical Reasoning) of all techniques for all generation datasets and languages. Below each performance value, the change in performance (in terms of %) wrt the native script is shown. The best results are in **boldface** and underlined. The last column shows the average performance across languages.

<span id="page-19-0"></span>

Table 16: Average tokens generated per instance of all techniques for all classification datasets and languages. Below each average token value, the change in token generation (in terms of %) wrt the native script is shown. The best results are in **boldface** and underlined. The last column shows the average cost across languages.

<span id="page-20-0"></span>

Table 17: Average tokens generated per instance of all techniques for all generation datasets and languages. Below each average token value, the change in token generation (in terms of %) wrt the native script is shown. The best results are in **boldface** and underlined. The last column shows the average cost across languages.

<span id="page-20-2"></span>

Table 18: Parity and RTPCR disagree. Here, we show the RTPCR and Parity values of the Mutable Wikidata object prediction dataset for different Indian languages. Native(acc) and Trans(acc) represent the performance of native scripts and English translation, respectively.

<span id="page-20-1"></span>

Table 19: Parity and RTPCR agree. Here, we show the RTPCR and Parity values of various classification datasets for different Indian languages.