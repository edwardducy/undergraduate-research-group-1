# **Language Model Tokenizers Introduce Unfairness Between Languages** 

**Aleksandar Petrov, Emanuele La Malfa, Philip H.S. Torr, Adel Bibi** University of Oxford `aleks@robots.ox.ac.uk` 

## **Abstract** 

Recent language models have shown impressive multilingual performance, even when not explicitly trained for it. Despite this, there are concerns about the quality of their outputs across different languages. In this paper, we show how disparity in the treatment of different languages arises at the tokenization stage, well before a model is even invoked. The same text translated into different languages can have drastically different tokenization lengths, with differences up to 15 times in some cases. These disparities persist even for tokenizers that are intentionally trained for multilingual support. Character-level and byte-level models also exhibit over 4 times the difference in the encoding length for some language pairs. This induces unfair treatment for some language communities in regard to the cost of accessing commercial language services, the processing time and latency, as well as the amount of content that can be provided as context to the models. Therefore, we make the case that we should train future language models using multilingually fair subword tokenizers. 

## **1 Introduction** 

Language models are increasingly important in natural language processing tasks, as they can understand and generate human-like language. They have been deployed in applications such as virtual assistants (Chen et al., 2021; Ouyang et al., 2022), chatbots (Kuhail et al., 2023; Lee et al., 2023), machine translation (Stahlberg, 2020; Ranathunga et al., 2023), and text summarization (Kryściński et al., 2019; Xu et al., 2020). As general-purpose technologies, it is also projected that Large Language Models (LLMs) will have a significant impact on the economy and the labour market (Teubner et al., 2023; Eloundou et al., 2023). 

Such LLMs are often trained using large swaths of internet content regardless of language. Hence, these models often end up being multilingual, even if not by design. ChatGPT (OpenAI, 2022) is a prominent recent example (Bang et al., 2023; Jiao et al., 2023; Johnson, 2023). Given the economic benefits of LLMs and LLM-derived technology, it’s beneficial that they support multiple languages. Equal access is crucial, and multilingual support is a key component of this. 

However, this multilingualism is currently treated as a curious emergent phenomenon rather than a carefully designed, controlled and managed process. The performance of LLMs has been shown to be generally lower in non-target languages, a problem especially pronounced for low-resource languages (Virtanen et al., 2019; Ahuja et al., 2023). Providing access to the same technology in different languages but moderation and safety tools only for some has resulted in dire societal consequences before (Stecklow, 2018; Facebook, 2021; Leung, 2022). Differing cost of access could also reinforce inequality in opportunities for economic mobility and social participation (Lythreatis et al., 2022). Therefore, as LLM multilingualism emerges, we should pay attention to ensuring comparable performance and accessibility across the supported languages, regardless of whether by design or by chance. 

This work demonstrates how the unequal treatment of languages arises at the tokenization stage,<sup>1</sup> well before the language model sees any data at all. For instance, the tokenizer employed by ChatGPT (OpenAI, 2022) and GPT-4 (OpenAI, 2023) uses about 1.6 times more tokens to encode the same text in Italian as it does in English, 2.6 times for Bulgarian and 3 times for Arabic. For Shan —the native language of people from the Shan State in Myanmar— that difference can be as high as 15 times. Unicode character and byte-level tokenization also result in drastically different encoding lengths across languages: byte-level representation of the same text is over 4 times longer for Burmese or Tibetan than Chinese. 

We discuss three fairness implications of these differences in tokenization: 

1. **Cost:** Commercial services charge users per token or Unicode character. In either case, these discrepancies lead to users of some languages paying at least 2.5 times more for the same task as users of English. 

2. **Latency:** The number of tokens has a direct effect on the processing time for a task. Some languages can require twice the time to process the same content as English. This may be critical for real-time applications like emergency services. 

3. **Long context processing:** Many models have a fixed-size context. Users of languages that are more token-efficient can use these systems to process or generate texts that may be more than an order of magnitude longer than users of other languages. This may lead to significant discrepancies in the quality of service. 

Therefore, we make the case for _multilingual tokenization parity_ : tokenizers should produce similar encoded lengths for the same content across languages. Hence, we advocate for multilingually fair tokenizers for the next generation of language models. 

## **2 Intriguing Properties of Tokenization Across Languages** 

Subword tokenization is currently the preferred approach for state of the art language models (Kudo and Richardson, 2018). In this section, we show how artefacts from data collection might result in technical terms or rare words having dedicated tokens, while more commonly used words and non-Latin characters end up requiring multiple tokens. 

Using large corpora scraped from the internet results in _peculiar_ choices for tokens. For instance, GPT-2 contains _glitch tokens_ which can be usernames or concepts from games (Rumbelow and Watkins, 2023b; Miles and Riley, 2023). As an example, `BuyableInstoreAndOnline` , likely coming from an online store backend, has a dedicated token. Another such token is `rawdownloadcloneembedreportprint` . 

While such obscure terms get their own tokens, the frequently used Arabic word “لماذا ” (meaning “why”) is broken into letters with each letter having its own token. The same word in Bulgarian (“защо”) is not only broken down to letters, but some of the letters require two tokens to be represented, resulting in 6 tokens for this 4 letter word. 



One may argue that this is because Arabic and Bulgarian are not target languages for GPT-2. However, glitch tokens also exist for Japanese: there are dedicated tokens for “ゼウス”, the name of the ancient Greek god Zeus and “サ–ティワン”, the name of an ice cream chain (Rumbelow and Watkins, 2023a). At the same time, GPT-2 requires 3 tokens to represent the much more commonly used kanji character for “to say”: 



In fact, more than half of the Japanese kanji characters require three tokens. 

> 1We offer a summary of the relevant tokenization approaches in Appendix A. 

2 

The existence of glitch tokens like “ゼウス” and “サ–ティワン” despite the lack of a dedicated token for “言” shows that tokenizers are heavily influenced by the biases of the corpus source. If one uses non-natural inputs, log files, or specialist forums, the tokenizer vocabulary would reflect this. While `cl100k_base` , the tokenizer used for the newer ChatGPT and GPT-4, may not have glitch tokens it still requires two tokens to represent some Cyrillic letters and three tokens for more than 65% of kanji characters. Therefore, to place all languages on an equal footing, it is important to have the tokens balanced across languages. 

## **3 Measuring Tokenizer Parity** 

To demonstrate that the above examples are not anecdotal evidence, we introduce the notion of _tokenizer parity_ to systematically assess how fairly tokenizers treat equivalent sentences in different languages. Parity occurs when a tokenizer exhibits similar tokenized lengths for the same sentence in different languages. Take a sentence _sA_ in language _A_ and its translation _sB_ to language _B_ . Then, a tokenizer _t_ achieves parity for _A_ with respect to _B_ at _sA_ and _sB_ if<sup>_|t_(</sup><sup>_sA_)</sup><sup>_|_</sup> / _|t_ ( _sB_ ) _| ≈_ 1, where _t_ ( _sA_ ) is the tokenization of the sentence _sA_ and _|t_ ( _sA_ ) _|_ represents its length. The ratio<sup>_|t_(</sup><sup>_sA_)</sup><sup>_|_</sup> / _|t_ ( _sB_ ) _|_ is the _premium_ for _A_ relative to _B_ .<sup>2</sup> 

## **4 Tokenization Length Differences Across Languages** 

Languages vary significantly in the number of tokens required to encode the same content, as demonstrated in the examples in Section 2. Hence, following Section 3, we measure the tokenization premium of different tokenizers. To this end, we use the FLORES-200 parallel corpus, comprising of the same 2000 sentences taken from Wikipedia and human-translated to 200 different languages (Guzmán et al., 2019; Goyal et al., 2021; Costa-jussà et al., 2022). We look at subword tokenization models which target English, languages other than English, language varieties, multi-lingual tokenizers, as well as tokenizer-free (byte-level) modelling. 

### **4.1 Parity for English-centric Models** 

As most models target English, we report in Table 1 the tokenization parity for a subset of languages in FLORES-200. The parities for all 200 languages are in Appendix C.<sup>3</sup> GPT2 (Radford et al., 2019), RoBERTa (Liu et al., 2019), and the `r50k_base` , `p50k_base` and `p50k_edit` tokenizers (OpenAI, 2022) have close<sup>4</sup> tokenization lengths so we report them together. ChatGPT and GPT-4 share the same `cl100k_base` tokenizer and are also reported together. Some models, such as FlanT5 (Chung et al., 2022), use a special `UNK` token to model unknown symbols not encountered during training. Hence, to ensure a fair comparison, we report only languages where no more than 10% of the input characters are mapped to `UNK` tokens (marked with —). 

Table 1 shows large variations in the tokenizer parity for all tokenizers. For GPT-2 and RoBERTa, Pangasinan, the language with shortest tokenization, is already 66% more expensive to process than English. ChatGPT and GPT-4 are slightly closer to parity, likely 

Table 1: Premiums with respect to English on FLORES-200 for several **Englishcentric** models. The languages in the top or bottom three for any tokenizer, as well as the ones discussed in the text, are shown. 

||GPT-2<br>RoBERTa|ChatGPT<br>GPT-4|FlanT5|
|---|---|---|---|
|Bulgarian|5.51|2.64|—|
|Burmese<br>i|16.89|11.70|—|
|Chinese (Simplified)|3.21|1.91|—|
|Dzongkha|16.36|12.33|—|
|English|1.00|1.00|1.00|
|French|2.00|1.60|1.60|
|German|2.14|1.58|1.37|
|Italian|2.01|1.64|2.18|
|Japanese|3.00|2.30|—|
|Jingpho|2.65|2.35|3.41|
|Maori|2.45|2.35|3.28|
|Norwegian Bokmål|1.86|1.56|2.24|
|Odia|13.38|12.48|—|
|Pangasinan|1.66|1.57|2.18|
|Portuguese|1.94|1.48|2.21|
|Romanian|2.48|1.88|1.50|
|Santali|12.86|12.80|—|
|Shan|18.76|15.05|—|
|Spanish|1.99|1.55|2.23|
|Standard Arabic|4.40|3.04|—|
|Tumbuka|2.78|2.57|3.29|
|Vietnamese|4.54|2.45|—|



2The concurrent work by Ahia et al. (2023) also evaluates the tokenization premiums for different languages and reaches similar conclusions. 

> 3An interactive table of all the languages and tokenizers is also available on the project website. 

> 4The largest tokenizer parity difference between them is less than 0.005. 

3 

Table 2: Tokenizer premiums on the FLORES-200 dataset for **non-English centric models** . The premium is computed with respect to the target language (Modern Standard Arabic was used for Arabic BERT and Simplified Chinese for RoCBert). The languages that are in the top or bottom two for any tokenizer as well as the ones discussed are shown. 

||Arabic<br>BERT<br>|RoCBert<br>(Chinese)<br>|CamemBERT<br>(French)<br>|GottBERT<br>(German)|BERT<br>Japanese<br>|PhoBERT<br>(Vietnamese)|
|---|---|---|---|---|---|---|
|Belarusian<br>|4.74<br>|—|—|5.62<br>|—|3.46<br>|
|Bulgarian|4.30|—|—|4.73|—|3.09|
|Catalan<br>|2.36|2.86|1.59|1.89|1.95|1.57|
|Chinese (Simp.)|—|1.00|—|3.95|0.82|—|
|Chinese (Trad.)<br>|—|0.94|—|3.82|0.84|—|
|Dutch|2.52|2.92|1.68|1.73|1.98|1.58|
|Dzongkha|—|—|—|16.12|—|—|
|English<br>|1.83<br>|2.60<br>|1.20<br>|1.35<br>|1.49<br>|1.20<br>|
|French|2.42|3.10|1.00|1.99|2.03|1.66|
|Friulian|2.33|2.79|1.66|1.98|1.92|1.59|
|German<br>|2.63<br>|3.12<br>|1.85|1.00<br>|2.04|1.67<br>|
|Greek|4.93|3.00|—|6.73|—|3.73|
|Italian|2.58|3.10|1.63|1.93|2.04|1.60|
|Japanese<br>|1.85|1.34|—|4.35|1.00|—|
|Jingpho|3.12|3.12|2.13|2.55|2.47|1.84|
|Luxembourgish<br>|2.56|2.97|1.82|1.75|1.96|1.72|
|N. Lev. Arabic<br>|1.00|—|—|6.52<br>|—|—|
|Shan|—|—|—|16.88|—|—|
|Standard Arabic|1.00|—|—|7.03|—|—|
|Tagalog|2.84|3.28|2.00|2.20|2.39|1.74|
|Tosk Albanian|2.66|2.90|2.17|2.39|—|2.02|
|Tsonga|3.01|3.09|2.03|2.29|2.46|1.76|
|Tumbuka|3.27|3.49|2.21|2.61|—|2.00|
|Vietnamese<br>|2.52|2.55|—|4.12<br>|—|1.00|
|Yue Chinese|—|0.92|—|3.75|—|—|



Table 3: Tokenizer premiums on the FLORES-200 dataset for the MuRIL model focusing on **16 Indian languages and English** . The premium is computed with respect to English. 

||MuRIL|
|---|---|
|English|1.00|
|Nepali<br>|1.01|
|Bengali|1.01|
|Tamil|1.06|
|Marathi|1.06|
|Kannada|1.06|
|Hindi<br>|1.16|
|Malayalam|1.18|
|Gujarati|1.19|
|Sanskrit|1.21|
|Telugu|1.21|
|Odia|1.21|
|Sindhi<br>|1.22|
|Assamese|1.24|
|Urdu|1.26|
|Eastern Panjabi|1.35|
|Kashmiri (Arabic)|1.75|
|Kashmiri (Devanagari)|1.75|



due to their larger vocabulary size. However, the cheapest languages, Portuguese, Pangasinan and German, still see a premium of 50% when compared to English. Shan has the worst tokenizer parity for all four models. Take as an example “မႂ်း”, one of the Shan words for “you”. It is tokenized by ChatGPT and GPT-4 as: 



This word is constructed from one consonant and three diacritics. As the diacritics are encoded separately, there are four Unicode codepoints for this Shan character, resulting in 9 tokens. The English “you” has three characters but a single token. 

FlanT5 has more than 10% `UNK` tokens for 42% of languages (— in Table 1). It has a higher premium than the other tokenizers for all other languages except German and Romanian. 

**Summary.** All four English-centric tokenizers we consider are far from tokenization parity. Portuguese is closest to parity with English for the ChatGPT and GPT-4 tokenizer but still requires about 50% more tokens for the same content. Shan is furthest from parity for this tokenizer with 15 times longer encodings compared to English. FlanT5 is closer to parity with its premium range 1.37–3.41 but it encodes only 54% of the languages, so we cannot say that it is more multilingually fair than the other tokenizers. 

### **4.2 Parity for Models with Other Target Languages** 

There are models targeting languages other than English as well. Table 2 shows six such models based on the BERT architecture (Devlin et al., 2019): ArabicBERT (Safaya et al., 2020), RoCBert for Chinese (Su et al., 2022), CamemBERT for French (Martin et al., 2020), GottBERT for German (Scheible et al., 2020), BERT Japanese (Tohoku NLP Group, 2019) and PhoBERT for Vietnamese (Nguyen and Nguyen, 2020). 

4 

Table 4: Tokenizer premiums with respect to English on FLORES-200 for **multilingual models** . The languages that are in the top or bottom two for any tokenizer, as well as the ones discussed in the text, are shown. 

Table 5: Tokenizer premiums with respect to English on FLORES-200 for **byte-level models** . The languages that are in the top or bottom two for any tokenizer, as well as the ones discussed in the text, are shown. 

|X|LM-R N|LLB|mT5 M|2M100 B|LOOM||CANINE|ByT5|
|---|---|---|---|---|---|---|---|---|
||||||||UTF-32bytes<br>UT|F-8bytes|
|Bulgarian<br>|1.16|1.31|1.28|1.23|2.49||<br>||
|Central Kanuri<br>|2.60|2.54|2.43|2.49|2.10|Bulgarian|1.04|1.89|
|Chinese (Simp.)<br>|0.97|1.11|0.92|1.05|0.95|Burmese|1.24|3.51|
|Dzongkha<br>|—|1.48|4.24|—|7.36|Chinese (Simplified)|0.34|0.93|
|English<br>|1.00<br>|1.00|1.00|1.00|1.00|Chinese (Traditional)|0.32|0.89|
|Indonesian<br>|0.94<br>|0.93 <br>|1,08<br>|0.98<br>|0.96<br>|Dzongkha|1.25|3.64|
|Italian<br>|1.19<br>|1.25 <br>|1.34<br>|1.25<br>|1.62<br>|English|1.00|1.00|
|Japanese<br>|1.11<br>|1.01 <br>|0.90<br>|1.20<br>|1.81<br>|Italian|1.18|1.19|
|Kabiyè<br>|2.98|1.56 <br>|2.83|2.71|3.34<br>|Japanese|0.44|1.27|
|Santali<br>|—<br>|2.49<br>|—<br>|—<br>|12.71<br>|Shan|1.42|3.94|
|Shan<br>|4.43<br>|1.94 <br>|3.28<br>|4.63<br>|12.06<br>|Standard Arabic|0.88|1.60|
|Std. Arabic<br>|1.18|1.40 <br>|1.35<br>|1.29|1.14<br>|Standard Tibetan|1.13|3.31|
|Std. Tibetan<br>|—|1.44|3.68|—|6.66|TokPisin|1.28|1.28|
|Uyghur<br>|1.41|1.40|2.57|3.00|3.67|<br>Tumbuka|130|132|
|Yue Chinese|0.93|1.05|0.95|1.03|0.93|Yue Chinese|.<br>0.31|.<br>0.87|



The English premium for GottBERT (1.35) is lower than those for Dutch (1.73) and Luxembourgish (1.75), which are more linguistically similar to German. CamemBERT is similar: English has the lowest premium (1.20), while Catalan (1.59) and Friulian (1.66) have higher premiums. PhoBERT also has English with the lowest tokenizer premium (1.20). Thus, even models targeting other languages exhibit a preference for English tokenization. 

RoCBert and BERT Japanese differ by having the other target language as the one closest to parity, possibly due to the partially shared script. ArabicBERT demonstrates a similar behaviour, with Central Kanuri (1.27) and Acehnese (1.73), both written in Arabic script, and with English at 1.82. Sharing writing systems seems to improve tokenization parity. 

Across all tokenizers, the premium for English relative to the respective target language is significantly lower than the premium of RoBERTa for that target language. This asymmetry between English and all other languages likely stems from the extensive incorporation of English in documents written in other languages (Zhang et al., 2022). 

We also consider MuRIL, a BERT-based model trained on 16 Indian languages and English (Khanuja et al., 2021). Despite the model’s focus on Indian languages, it remains most token-efficient for English (see Table 3). 

Unequal treatment of dialects or linguistic varieties can lead to social and economic disadvantages making it important to also study the tokenization differences between the “standard” language and its varieties. For Swiss German and the Mauritian and Haitian Creoles, there are large differences in tokenization lengths compared respectively to High German (on GottBERT) and French (on CamemBERT). English is much closer to parity for both models than these language varieties. Therefore subword tokenizers might not be able to generalize to language varieties, such as dialects and creoles. The tokenizers of ArabicBERT and BERT Japanese, however, are close to parity across various dialects of both languages and have lower premiums for the dialects than for English. This is likely due to the good representation of the dialects in the dataset as well as the dialects being linguistically closer to the respective standard languages. The detailed analysis is deferred to Appendix B. 

**Summary.** We observed that the tokenizers targeting French, German and Vietnamese have English as the language closest to parity, rather than more linguistically close languages. On the other hand, tokenizers for Arabic, Chinese and Japanese have lower premiums for languages they share a script with. Notably, despite targeting Indian languages, MuRIL still has the shortest tokenizations for English. Finally, across all tokenizers, the premium for English is lower than the premium for the same language for the English-centric RoBERTa. Hence, we conclude that tokenizers for other languages give English preferential treatment. 

5 



Shan<br>Tamil<br>24<br>Burmese<br>Asturian<br>Central Kanuri (Arabic sc.)Central KurdishBulgarianBurmeseCatalan 22 Dzongkha<br>Chinese (Simplified)<br>Chinese (Traditional)Danish<br>EnglishFon<br>Haitian CreoleGeorgianGalician 20 Odia<br>Indonesian<br>JapaneseItalian Santali<br>JavaneseKannadaKabiyè Sango<br>MalayalamKikuyuKhmerLao 18 Hebrew Script family:<br>Norwegian NynorskMeitei (Bengali sc.)Norwegian BokmålPangasinanStd. ArabicShanNuerOdia Portuguese16 ItalianFrenchGerman StandardArabic Bulgarian JapaneseKorean ArabicArmenianNorthern BrahmiSouthern BrahmiCJK<br>Yue ChineseStd. MalaySwedishTurkishYorubaTeluguTamilTajikThai RoBERTaXLM-RoBERTa 14 EnglishMalaySpanish ZuluChinese (Simp.)Chinese (Trad.)Yue Chinese VietnameseBhojpuri GreekGeGeorgianHebrewBerberez<br>1 5 10 15 0.0 0.2 0.4 0.6 0.8 1.0<br>Tokenization premium Tokenisation length for the FLORES-200 parallel corpus 1e6<br>RoBERTa execution time [s]<br>
Figure 1: Comparison of the tokenization premiums for XLM-R and RoBERTa for the subset of languages that XLM-R encodes with less than 10% to the `UNK` token. 

Figure 2: Average processing time and length of the tokenized inputs of RoBERTa. Each FLORES-200 sentence is processed for 20 independent runs. The script family designation is only for illustration purposes. 

### **4.3 Parity for Multilingual Models** 

There has been a growing interest in multilingual language models, particularly for translation (Dabre et al., 2020). As these models are intended to support a variety of languages, one would expect them to be close to tokenizer parity. We compare several such multilingual models: XML-R (Conneau et al., 2020), NLLB (Costa-jussà et al., 2022), M2M100 (Fan et al., 2021) and mT5 (Xue et al., 2020). All of these models use the SentencePiece tokenizer with upsampling for rare languages. The final model, BLOOM (Scao et al., 2022), uses byte-level BPE instead of SentencePiece and is designed to maintain similar ratios of tokens per word for each language as reference monolingual tokenizers. 

BLOOM and NLLB encode all languages with less than 10% `UNK` tokens, respectively thanks to byte-level BPE tokenization and being trained on the same 200 languages as FLORES200 (see Table 4). The other three models fail to encode at least one language. All five models have languages with premiums of more than 2.5. Still, all models are better than the English-centric models in Table 1. Figure 1 shows how XLM-R is much closer to parity than RoBERTa (on which it is based), over all languages it can encode. However, none of the models uniformly reaches parity across all languages. Therefore even models which are intentionally designed to be multilingual suffer from a lack of tokenization parity. 

**Summary:** Multilingual models can improve the tokenization parity for different languages but challenges remain in achieving tokenization parity across all languages. 

### **4.4 Parity for Byte-level Tokenization Models** 

Byte-level representation is crucial for multilingual support, as it encodes any Unicode codepoint, even if unseen during training. One can also bypass vocabulary construction and directly employ the 256 byte values, enabling end-to-end training ( _byte-level tokenization_ ). CANINE (Clark et al., 2022) is a large model that operates at the Unicode codepoint level rather than the byte level. The CANINE tokenizer is thus equivalent to the UTF-32 encoding, resulting in an implicit tokenizer with a vocabulary of 1,114,112. ByT5 (Xue et al., 2022), on the other hand, uses the UTF-8 encoding: an implicit vocabulary of 256 tokens.<sup>5</sup> 

> 5To be consistent, we will refer to the characters and bytes in the encoding of the CANINE and ByT5 tokenizers as _tokens_ as they fulfil a similar role. 

6 

These byte-level models can represent any Unicode codepoint without an explicit tokenization step but there are still significant tokenization disparities. For CANINE, Shan has a premium of 4.58 relative to Yue Chinese. This can be attributed to the fact that CANINE provides a single token for each Unicode codepoint, which results in Chinese being more token-efficient (with a premium range 0.31–0.34 relative to English for the three Chinese languages) as each character is treated as a single token. This encoding also puts Shan at a disadvantage, as its encoding relies on diacritics represented as separate Unicode codepoints. Other languages, such as Tok Pisin and Tumbuka, which use the Latin script but require more characters than English for the same text, also face similar challenges. 

Tokenization disparity is also present in the ByT5 model. The tokenization premium for ByT5 ranges from 0.87 (for Yue Chinese) to 3.94 (for Shan). The introduction of the variablewidth UTF-8 encoding of Unicode characters in ByT5 creates another issue of unequal treatment. ASCII characters, which are sufficient for English, require only one byte. Other Latin script characters, as well as Greek, Cyrillic, Coptic, Armenian, Hebrew, Arabic and Syriac, require two bytes, while Chinese, Japanese and Korean characters require three bytes. Therefore, the tokenization of Chinese and Japanese is about three times as long for ByT5 as it is for CANINE (Table 5). Shan’s premium of 3.94 is due to the fact that all its consonants and diacritics require three bytes. For example, the word “မႂ်း” is encoded by ByT5 as 12 tokens, whereas the corresponding “you” requires 3 tokens. The situation is similar for other languages like Dzongkha, Tibetan and Burmese. 

**Summary.** Byte-level models also fail to achieve parity among the languages from FLORES-200 exhibiting a premium of over 4 times for some language pairs. There are two sources of multilingual tokenizer disparities. First, there are natural differences in the number of characters used in different languages to communicate the same content. Second, the UTF-8 standard uses different number of bytes to encode codepoints of different scripts. 

## **5 Fairness Implications of Tokenization Length Differences** 

We showed that no matter whether one uses subword, multilingual, or byte-level tokenization, none of the tokenizers gets close to parity for all languages in FLORES-200. This lack of tokenization parity is not merely a curiosity: it leads to unfairness in the cost to access language models, the latency of the service and the amount of data that can be processed. 

### **5.1 Cost** 

It is increasingly common to access LLMs as paid API services. One pricing approach, employed by OpenAI at the time of writing,<sup>6</sup> is to charge per token. Therefore, the tokenization premiums discussed in Section 4 directly map to cost premiums. For ChatGPT and GPT-4, the cost to process a text in German or Italian is about 50% higher than to process the same text in English (Table 1). Using them in Dzongkha, Odia, Santali or Shan, the most expensive languages for these services, costs more than 12 times more than in English. 

Another pricing strategy is per Unicode character: the approach currently taken by the Google Cloud Natural Language service.<sup>7</sup> However, as we showed in Section 4.4, the same content can have very different lengths when measured in Unicode characters. Burmese, Dzongkha, Shan, Tok Pisin or Tumbuka require more than 4 times more characters than Yue Chinese for the same text, resulting in a proportional cost difference. Therefore, both the per-token and the per-character approaches result in large disparities in the cost for users of different languages to use the exact same service. 

### **5.2 Latency** 

High latency of real-time interactions for users of certain languages can result in a suboptimal experience and communication breakdowns. For customer support or emergency services, delays in response time can lead to miscommunication or delayed assistance. 

> 6 `https://openai.com/pricing` 

> 7 `https://cloud.google.com/natural-language/pricing` 

7 

As some languages have significantly longer tokenized inputs, they would also experience longer processing times. The transformer attention mechanism has a quadratic complexity in the number of input tokens (Keles et al., 2023). However, the full model architecture contains other submodules and therefore the overall complexity might be different. 

To assess the effect of the tokenization length on the latency, in Figure 2 we plot the computation time of RoBERTa against the tokenization lengths. It appears that the processing time is linear in the tokenization length rather than quadratic, showing a strong correlation between sequence length and execution time. Therefore, tokenization disparities across languages also affect the latency and processing time for text in these languages. 

As expected, English is on the left lower corner, having the shortest tokenization and one of the fastest processing times. Shan is on the other extreme with the longest tokenization length and execution time (almost twice that of English). We can also observe clear trends dependent on the script used. Latin script and other Greek-derived scripts show the shortest tokenization lengths and processing times followed by the Chinese-Japanese-Korean (CJK) and Arabic languages. Other predominantly Asian and African scripts have longer tokenization lengths and processing times. 

The latency implications of tokenization disparity are not limited to text models. Speech recognition models often produce a series of tokens as their output sequentially. Similarly, speech synthesis takes as an input tokenized text (Latif et al., 2023). Therefore, differences in tokenization affect speech models too. 

### **5.3 Long context processing** 

Transformers models have difficulty processing long inputs (Liu et al., 2023). Given that the size of the input is contingent upon the tokenization process, inputs of greater length may impose a challenge for language models to adequately reason over. Such a predicament may result in reduced abilities or limited applicability for languages with high tokenization premiums. For example, RoBERTa has a fixed block size of 512, GPT-2 has 768, 1024, 1280, or 1600 Radford et al. (2019), GPT-4 comes in 8,000 and 16,000 context variants.<sup>8</sup> These models cannot process inputs longer than that. Therefore, one can process less than a tenth of the content in languages like Burmese and Dzongkha than they can in English. 

Alongside inconveniencing the users of these languages, this can also result in diminished performance on automated systems, such as content moderation. Reliable content moderation is crucial for tackling hate speech and diminished performance has already been shown to fail to prevent its spread (Stecklow, 2018; Facebook, 2021). Therefore, reduced long context capabilities for some languages could have severe real-world impacts. 

## **6 Towards Multilingual Tokenization Fairness** 

Section 5 showed that high values of tokenization parity for a language lead to increased cost and latency and decreased capacity for long context processing. In this section, we argue that training language models from scratch with a multilingually fair subword tokenizer is the only approach that can effectively address all these aspects of tokenization unfairness. 

**Subword tokenization is necessary to achieve parity.** In Section 4.4, we showed that neither character-level nor byte-level input representation can achieve tokenization parity. Therefore, a variation of subword tokenization is necessary. For example, Chinese characters could be individual tokens, Latin characters might be represented as tokens with an average length of about 3 characters while pairs of Burmese characters and their diacritics being assigned single tokens. Such an approach would account for Chinese requiring one-third the characters English does (as shown in Table 5). 

**A separate tokenizer for determining the processing cost is not sufficient.** An easy patch for existing models is to use a separate tokenizer for calculating how much a user should be charged. Using one tokenizer for computing the cost and another to process 

> 8 `https://openai.com/pricing` 

8 

Figure 3: How much longer will English language tokenization be if we dedicate a fraction of the `cl100k_base` vocabulary to other languages? This plot shows how many tokens will be necessary to encode the English language corpus of FLORES-200 for different subsets of the `cl100k_base` vocabulary. 



80000<br>78000<br>76000<br>74000<br>72000 A 10-fold reduction in the vocabulary<br>70000 30% longer sequences for English. would result in only<br>68000<br>66000<br>64000 With one-third of the vocabulary,<br>62000 just 10% longer for ChatGPT/GPT-4English sequences will become<br>60000<br>58000<br>56000<br>54000<br>52000<br>0 10000 20000 30000 40000 50000 60000 70000 80000 90000100000<br>Vocabulary size<br>Tokens necessary to encode FLORES-200<br>
the input can easily be applied to existing systems without the need to retrain the LLM itself. However, as the tokenizer for the language model is unchanged, this approach would still suffer from latency and inability to process long contexts. Therefore, to ensure similar processing times and long context capabilities across languages, the language model has to be trained with a multilingually fair tokenizer. 

**The tokenization needs to support all Unicode codepoints.** Amongst all tokenizers we examine in this paper, the ones which encode all FLORES-200 languages all have one thing in common: they build their tokenization on top of Unicode representation, allowing them them to represent all characters. Therefore, a multilingually fair tokenizer should also start from a Unicode (or equivalent) encoding. Considering that subword tokenization is necessary, building the vocabulary from UTF-8 would likely result in a smaller dictionary than building it on top of UTF-32. Hence, UTF-8 is likely the more appropriate choice. 

**Building a multilingually fair parallel corpus.** Building and evaluating multilingually fair tokenizers requires attention to the parallel corpus used. One must ensure a balanced representation of topics, otherwise, the resulting tokenizer might end up being multilingually fair only for a subset of topics. The presence of named entities must also be balanced. For example, in FLORES-200, there are many English-centric names and institutions, which might skew the results in favour of English. Additionally, the same sentence can have different translations with varying tokenization lengths. To account for this, a diversity of translations could ensure tokenization fairness across languages. These limitations also hold for the results in this paper. Hence, developing a well-curated and diverse parallel corpus is crucial for the development and evaluation of a multilingually fair tokenizer. 

**Building a multilingually fair tokenizer from monolinugal tokenizers.** As discussed in Section 4, byte-level, character-level and word-level tokenizers cannot achieve tokenization parity and subword tokenization is needed. However, simply training a subword tokenizer on a balanced dataset is also not sufficient as languages can share tokens. For example, “hotel” is written the same way in English, Spanish, Italian, Portuguese, Dutch, Danish, Hungarian, Polish, etc. Hence, languages from more numerous language families will also witness shorter tokenization lengths while more isolated languages and scripts, e.g. Korean, would see larger language premiums: “hotel” in Korean is “호텔” and no other language has the same spelling as no other language uses the Korean script. 

To address this issue, we suggest a two-stage process towards building a multilingually fair tokenizer. First, train individual monolingual tokenizers for all target languages. Then, merge them while maintaining parity. The merging can be done by starting with the 256 tokens corresponding to each value a byte can take and then repeatedly adding the most frequently used token for the language with the highest premium. 

While a multilingually fair tokenizer would lead to more tokens being needed for the dominant language, this additional cost would likely be much smaller than the benefit for the rest of the languages. The vocabulary size has diminishing returns: the additional tokens correspond to increasingly rare (parts of) words. For example, with only a third of the vocab- 

9 

ulary, English sequences will become just 10% longer for ChatGPT/GPT-4 (see Figure 3). Therefore, by removing rarely used tokens of the dominant language and replacing them with frequently used tokens in other languages, we would likely see an overall net benefit. 

## **7 Related Works** 

**Fairness and bias in language models.** The rapid increase in the size of language models has raised concerns regarding their biases and unfairness (Bender et al., 2021). For example, Bolukbasi et al. (2016), May et al. (2019) and Nadeem et al. (2021) showed that stereotypes and biases exist in language models, while Magee et al. (2021) identified the presence of intersectional biases which may be resistant to debiasing techniques. Language models were also shown to rely on social biases in question answering (Parrish et al., 2022). Another challenge is the generation of toxic content which can occur even without prompting (Gehman et al., 2020). Interestingly, Gururangan et al. (2022) point out that datasets consider one type of English as a higher quality depending on the location of the writer rather than on factuality or literary acclaim. Moreover, Ramesh et al. (2023) and Levy et al. (2023) highlighted the need to consider fairness issues of languages other than English, as they may have distinct sources of bias and solutions for English may not be applicable. 

**Multilingual performance.** One approach towards similar multilingual performance is to frame languages as entities as recently proposed by Choudhury and Deshpande (2021). Another method is to separately train vocabularies for different language clusters to balance cross-lingual and language-specific tokens (Chung et al., 2020). Still, multilingual models struggle to deliver on the promises of deep transfer learning for lower-resourced languages (Virtanen et al., 2019) and perform differently depending on the script and resource level of the language (Bang et al., 2023). Ahuja et al. (2023) found that generative models perform better on higher-resource languages and languages that use the Latin script, possibly due to the context length restrictions for some languages. Zhang et al. (2022) show that a balanced tokenizer corpus results in better translation performance. Separately, Hofmann et al. (2021, 2022) show that the BPE results in suboptimal token choices even for English and demonstrate that addressing this issue boosts performance. Similarly, Rajab (2022) and Oladipo et al. (2022) discuss how tokenization affects performance for African languages. 

**Measuring tokenization lengths.** Zhang et al. (2022) suggested using the ratio of the average sentence length in tokens to the length in characters as a measure of closeness to the character level. However, this method may not be suitable for comparing languages due to differences in sentence length across languages. On the other hand, Ács (2019) and Scao et al. (2022) measure the number of tokens created per word, but this method may not be effective for comparing languages due to differences in semantic content per word and the lack of word delineation in some languages. Rust et al. (2021) show that mBERT (Devlin et al., 2019) breaks down English words the least, in line with our findings of English receiving special treatment. However, to the best of our knowledge, we are the first to leverage a parallel corpus to compare tokenization lengths across languages. 

## **8 Conclusion** 

This paper highlights the significant disparities in tokenization across different languages which can lead to unequal treatment and disadvantages for certain language communities. The findings reveal that even tokenizers explicitly trained for multilingual support exhibit tokenization lengths that vary by up to a factor of 13. Furthermore, character-level and byte-level models also demonstrate encoding length discrepancies that are more than 4 times longer. These disparities have important real-world implications including increased costs for accessing commercial language services, longer processing times and limitations on the amount of contextual information provided to language models. To address these issues, we propose the development of multilingually fair tokenizers for future language models emphasizing the importance of ensuring comparable performance and accessibility across supported languages. By achieving tokenization parity, we can mitigate inequalities and promote fair access to language technologies across diverse linguistic communities. 

10 

## **Acknowledgements** 

We would like to thank Puyu Wang, Francisco Eiras, Ambre Bertrand and Carmen Scheidemann for their linguistic advice. Janet Pierrehumbert introduced us to many relevant prior works. We also extend special gratitude to Shinnosuke Takamichi and Hiroshi Saruwatari for open-sourcing the CPJD corpus for this project. Finally, we thank the reviewers; their feedback greatly improved this manuscript. 

AB has received funding from the Amazon Research Awards. This work is supported by a UKRI grant Turing AI Fellowship (EP/W002981/1) and the EPSRC Centre for Doctoral Training in Autonomous Intelligent Machines and Systems (EP/S024050/1). We also thank the Royal Academy of Engineering and FiveAI. 

## **References** 

- Ahmed Abdelali, Francisco Guzman, Hassan Sajjad, and Stephan Vogel. 2014. The AMARA corpus: Building parallel language resources for the educational domain. In _Proceedings of the Ninth International Conference on Language Resources and Evaluation_ ( _LREC_ ’ _14_ ). European Language Resources Association (ELRA). 

- Orevaoghene Ahia, Sachin Kumar, Hila Gonen, Jungo Kasai, David R. Mortensen, Noah A. Smith, and Yulia Tsvetkov. 2023. Do all languages cost the same? Tokenization in the era of commercial language models. _arXiv preprint arXiv:2305.13707_ . 

- Kabir Ahuja, Rishav Hada, Millicent Ochieng, Prachi Jain, Harshita Diddee, Samuel Maina, Tanuja Ganu, Sameer Segal, Maxamed Axmed, Kalika Bali, and Sunayana Sitaram. 2023. MEGA: Multilingual evaluation of generative AI. _arXiv preprint arXiv:2303.12528_ . 

- Jinze Bai, Shuai Bai, Shusheng Yang, Shijie Wang, Sinan Tan, Peng Wang, Junyang Lin, Chang Zhou, and Jingren Zhou. 2023. Qwen-VL: A versatile vision-language model for understanding, localization, text reading, and beyond. _arXiv preprint arXiv:2308.12966_ . 

- Yejin Bang, Samuel Cahyawijaya, Nayeon Lee, Wenliang Dai, Dan Su, Bryan Wilie, Holy Lovenia, Ziwei Ji, Tiezheng Yu, Willy Chung, Quyet V. Do, Yan Xu, and Pascale Fung. 2023. A multitask, multilingual, multimodal evaluation of ChatGPT on reasoning, hallucination, and interactivity. _arXiv preprint arXiv:2302.04023_ . 

- Loïc Barrault, Yu-An Chung, Mariano Cora Meglioli, David Dale, Ning Dong, PaulAmbroise Duquenne, Hady Elsahar, Hongyu Gong, Kevin Heffernan, John Hoffman, et al. 2023. SeamlessM4T – massively multilingual & multimodal machine translation. _arXiv preprint arXiv:2308.11596_ . 

- Reem Bassiouney. 2009. _Arabic Sociolinguistics_ . Edinburgh University Press. 

- Emily M. Bender, Timnit Gebru, Angelina McMillan-Major, and Shmargaret Shmitchell. 2021. On the dangers of stochastic parrots: Can language models be too big? In _Proceedings of the 2021 ACM Conference on Fairness, Accountability, and Transparency_ . 

- Yoshua Bengio, Réjean Ducharme, and Pascal Vincent. 2000. A neural probabilistic language model. _Advances in Neural Information Processing Systems_ . 

- Tolga Bolukbasi, Kai-Wei Chang, James Y Zou, Venkatesh Saligrama, and Adam T Kalai. 2016. Man is to computer programmer as woman is to homemaker? Debiasing word embeddings. In _Advances in Neural Information Processing Systems_ . 

- Houda Bouamor, Nizar Habash, Mohammad Salameh, Wajdi Zaghouani, Owen Rambow, Dana Abdulrahim, Ossama Obeid, Salam Khalifa, Fadhl Eryani, Alexander Erdmann, et al. 2018. The MADAR Arabic dialect corpus and lexicon. In _Proceedings of the Eleventh International Conference on Language Resources and Evaluation_ ( _LREC 2018_ ). 

- Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, Henrique Ponde de Oliveira Pinto, Jared Kaplan, Harri Edwards, Yuri Burda, Nicholas Joseph, Greg Brockman, et al. 2021. Evaluating large language models trained on code. _arXiv preprint arXiv:2107.03374_ . 

11 

- Monojit Choudhury and Amit Deshpande. 2021. How linguistically fair are multilingual pre-trained language models? In _Proceedings of the AAAI Conference on Artificial Intelligence_ . 

- Hyung Won Chung, Dan Garrette, Kiat Chuan Tan, and Jason Riesa. 2020. Improving multilingual models with language-clustered vocabularies. In _Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing_ ( _EMNLP_ ). 

- Hyung Won Chung, Le Hou, Shayne Longpre, Barret Zoph, Yi Tay, William Fedus, Eric Li, Xuezhi Wang, Mostafa Dehghani, Siddhartha Brahma, et al. 2022. Scaling instructionfinetuned language models. _arXiv preprint arXiv:2210.11416_ . 

- Junyoung Chung, Kyunghyun Cho, and Yoshua Bengio. 2016. A character-level decoder without explicit segmentation for neural machine translation. In _Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics_ ( _Volume 1: Long Papers_ ). 

- Jonathan H. Clark, Dan Garrette, Iulia Turc, and John Wieting. 2022. Canine: Pre-training an Efficient Tokenization-Free Encoder for Language Representation. _Transactions of the Association for Computational Linguistics_ . 

- Alexis Conneau, Kartikay Khandelwal, Naman Goyal, Vishrav Chaudhary, Guillaume Wenzek, Francisco Guzmán, Édouard Grave, Myle Ott, Luke Zettlemoyer, and Veselin Stoyanov. 2020. Unsupervised cross-lingual representation learning at scale. In _Annual Meeting of the Association for Computational Linguistics_ . 

- Marta R Costa-jussà, James Cross, Onur Çelebi, Maha Elbayad, Kenneth Heafield, Kevin Heffernan, Elahe Kalbassi, Janice Lam, Daniel Licht, Jean Maillard, et al. 2022. No language left behind: Scaling human-centered machine translation. _arXiv preprint arXiv:2207.04672_ . 

- Raj Dabre, Chenhui Chu, and Anoop Kunchukuttan. 2020. A survey of multilingual neural machine translation. _ACM Computing Surveys_ . 

- Raj Dabre and Aneerav Sukhoo. 2022. MorisienMT: A dataset for Mauritian Creole machine translation. _arXiv preprint arXiv:2206.02421_ . 

- Michel DeGraff. 2007. Kreyòl Ayisyen, or Haitian Creole (Creole French). 

- Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2019. BERT: Pretraining of deep bidirectional transformers for language understanding. In _Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies_ . 

- Pelin Dogan-Schönberger, Julian Mäder, and Thomas Hofmann. 2021. SwissDial: Parallel multidialectal corpus of spoken Swiss German. _arXiv preprint arXiv:2103.11401_ . 

- Tyna Eloundou, Sam Manning, Pamela Mishkin, and Daniel Rock. 2023. GPTs are GPTs: An early look at the labor market impact potential of large language models. _arXiv preprint arXiv:2303.10130_ . 

- Facebook. 2021. Sri Lanka human rights impact assessment. Accessed on April 11, 2023. 

- Angela Fan, Shruti Bhosale, Holger Schwenk, Zhiyi Ma, Ahmed El-Kishky, Siddharth Goyal, Mandeep Baines, Onur Celebi, Guillaume Wenzek, Vishrav Chaudhary, et al. 2021. Beyond English-centric multilingual machine translation. _The Journal of Machine Learning Research_ . 

- Philip Gage. 1994. A new algorithm for data compression. _C Users Journal_ . 

- Yingqiang Gao, Nikola I. Nikolov, Yuhuang Hu, and Richard H.R. Hahnloser. 2020. Character-level translation with self-attention. In _Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics_ . 

12 

- Samuel Gehman, Suchin Gururangan, Maarten Sap, Yejin Choi, and Noah A. Smith. 2020. RealToxicityPrompts: Evaluating neural toxic degeneration in language models. In _Findings of the Association for Computational Linguistics: EMNLP_ . Association for Computational Linguistics. 

- Naman Goyal, Cynthia Gao, Vishrav Chaudhary, Peng-Jen Chen, Guillaume Wenzek, Da Ju, Sanjana Krishnan, Marc’Aurelio Ranzato, Francisco Guzmán, and Angela Fan. 2021. The FLORES-101 evaluation benchmark for low-resource and multilingual machine translation. _Transactions of the Association for Computational Linguistics_ . 

- Suchin Gururangan, Dallas Card, Sarah K. Dreier, Emily K. Gade, Leroy Z. Wang, Zeyu Wang, Luke Zettlemoyer, and Noah A. Smith. 2022. Whose language counts as high quality? Measuring language ideologies in text data selection. _arXiv preprint arXiv:2201.10474_ . 

- Francisco Guzmán, Peng-Jen Chen, Myle Ott, Juan Pino, Guillaume Lample, Philipp Koehn, Vishrav Chaudhary, and Marc’Aurelio Ranzato. 2019. Two new evaluation datasets for low-resource machine translation: Nepali-English and Sinhala-English. In _Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing_ ( _EMNLP-IJCNLP_ ). 

- Shiro Hattori. 1973. Japanese dialects. In _Diachronic, areal, and typological linguistics_ . 

- Valentin Hofmann, Janet Pierrehumbert, and Hinrich Schütze. 2021. Superbizarre is not superb: Derivational morphology improves BERT’s interpretation of complex words. In _Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing_ ( _Volume 1: Long Papers_ ). 

- Valentin Hofmann, Hinrich Schuetze, and Janet Pierrehumbert. 2022. An embarrassingly simple method to mitigate undesirable properties of pretrained language model tokenizers. In _Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics_ ( _Volume 2: Short Papers_ ). 

- Michael A. Hogg, Nicholas Joyce, and Dominic Abrams. 1984. Diglossia in Switzerland? A social identity analysis of speaker evaluations. _Journal of Language and Social Psychology_ . 

- Wenxiang Jiao, Wenxuan Wang, Jen-tse Huang, Xing Wang, and Zhaopeng Tu. 2023. Is ChatGPT a good translator? Yes with GPT-4 as the engine. _arXiv preprint arXiv:2301.08745_ . 

- Johnson. 2023. ChatGPT is a marvel of multilingualism. _The Economist_ . 

- Alan S. Kaye. 2001. Diglossia: The state of the art. _International Journal of the Sociology of Language_ . 

- Feyza Duman Keles, Pruthuvi Mahesakya Wijewardena, and Chinmay Hegde. 2023. On the computational complexity of self-attention. In _International Conference on Algorithmic Learning Theory_ . 

- Simran Khanuja, Diksha Bansal, Sarvesh Mehtani, Savya Khosla, Atreyee Dey, Balaji Gopalan, Dilip Kumar Margam, Pooja Aggarwal, Rajiv Teja Nagipogu, Shachi Dave, Shruti Gupta, Subhash Chandra Bose Gali, Vish Subramanian, and Partha Talukdar. 2021. MuRIL: Multilingual representations for Indian languages. _arXiv preprint arXiv:2103.10730_ . 

- Wojciech Kryściński, Nitish Shirish Keskar, Bryan McCann, Caiming Xiong, and Richard Socher. 2019. Neural text summarization: A critical evaluation. In _Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing_ ( _EMNLP-IJCNLP_ ), Hong Kong, China. 

13 

- Taku Kudo. 2018. Subword regularization: Improving neural network translation models with multiple subword candidates. In _Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics_ ( _Volume 1: Long Papers_ ). 

- Taku Kudo and John Richardson. 2018. SentencePiece: A simple and language independent subword tokenizer and detokenizer for neural text processing. In _Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing: System Demonstrations_ . 

- Mohammad Amin Kuhail, Nazik Alturki, Salwa Alramlawi, and Kholood Alhejori. 2023. Interacting with educational chatbots: A systematic review. _Education and Information Technologies_ . 

- Siddique Latif, Aun Zaidi, Heriberto Cuayahuitl, Fahad Shamshad, Moazzam Shoukat, and Junaid Qadir. 2023. Transformers in speech processing: A survey. _arXiv preprint arXiv:2303.11607_ . 

- Jason Lee, Kyunghyun Cho, and Thomas Hofmann. 2017. Fully character-level neural machine translation without explicit segmentation. _Transactions of the Association for Computational Linguistics_ . 

- Peter Lee, Sebastien Bubeck, and Joseph Petro. 2023. Benefits, limits, and risks of GPT-4 as an AI chatbot for medicine. _New England Journal of Medicine_ . 

- Heather Lent, Emanuele Bugliarello, Miryam de Lhoneux, Chen Qiu, and Anders Søgaard. 2021. On language models for creoles. In _Proceedings of the 25th Conference on Computational Natural Language Learning_ . Association for Computational Linguistics. 

- Heather Lent, Kelechi Ogueji, Miryam de Lhoneux, Orevaoghene Ahia, and Anders Søgaard. 2022. What a creole wants, what a creole needs. In _Proceedings of the Thirteenth Language Resources and Evaluation Conference_ . 

- Janny Leung. 2022. Shortcuts and shortfalls in Meta’ s content moderation practices: A glimpse from its oversight board’ s first year of operation. _Comparative Law and Language_ . 

- Sharon Levy, Neha Anna John, Ling Liu, Yogarshi Vyas, Jie Ma, Yoshinari Fujinuma, Miguel Ballesteros, Vittorio Castelli, and Dan Roth. 2023. Comparing biases and the impact of multilingual training across multiple languages. _arXiv preprint arXiv:2305.11242_ . 

- Nelson F Liu, Kevin Lin, John Hewitt, Ashwin Paranjape, Michele Bevilacqua, Fabio Petroni, and Percy Liang. 2023. Lost in the middle: How language models use long contexts. _arXiv preprint arXiv:2307.03172_ . 

- Yinhan Liu, Jiatao Gu, Naman Goyal, Xian Li, Sergey Edunov, Marjan Ghazvininejad, Mike Lewis, and Luke Zettlemoyer. 2020. Multilingual denoising pre-training for neural machine translation. _Transactions of the Association for Computational Linguistics_ . 

- Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen, Omer Levy, Mike Lewis, Luke Zettlemoyer, and Veselin Stoyanov. 2019. RoBERTa: A robustly optimized bert pretraining approach. _arXiv preprint arXiv:1907.11692_ . 

- Georges Lüdi. 2007. The Swiss model of plurilingual communication. _Receptive multilingualism: Linguistic analyses, language policies and didactic concepts_ . 

- Sophie Lythreatis, Sanjay Kumar Singh, and Abdul-Nasser El-Kassar. 2022. The digital divide: A review and future research agenda. _Technological Forecasting and Social Change_ . 

- Liam Magee, Lida Ghahremanlou, Karen Soldatic, and Shanthi Robertson. 2021. Intersectional bias in causal language models. _arXiv preprint arXiv:2107.07691_ . 

- Louis Martin, Benjamin Muller, Pedro Javier Ortiz Suárez, Yoann Dupont, Laurent Romary, Éric Villemonte de La Clergerie, Djamé Seddah, and Benoît Sagot. 2020. CamemBERT: A tasty French language model. In _Annual Meeting of the Association for Computational Linguistics_ . 

14 

- Chandler May, Alex Wang, Shikha Bordia, Samuel R. Bowman, and Rachel Rudinger. 2019. On measuring social biases in sentence encoders. In _Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1_ ( _Long and Short Papers_ ). 

- Sabrina J Mielke, Zaid Alyafeai, Elizabeth Salesky, Colin Raffel, Manan Dey, Matthias Gallé, Arun Raja, Chenglei Si, Wilson Y Lee, Benoît Sagot, et al. 2021. Between words and characters: A brief history of open-vocabulary modeling and tokenization in NLP. _arXiv preprint arXiv:2112.10508_ . 

- Rob Miles and Sean Riley. 2023. Glitch tokens – Computerphile. Accessed on April 11, 2023. 

- Robert Munro. 2010. Crowdsourced translation for emergency response in Haiti: the global collaboration of local knowledge. In _Proceedings of the Workshop on Collaborative Translation: technology, crowdsourcing, and the translator perspective_ . Association for Machine Translation in the Americas. 

- Pieter Muysken and Norval Smith. 1994. The study of pidgin and creole languages. In _Pidgins and creoles: An introduction_ . 

- Moin Nadeem, Anna Bethke, and Siva Reddy. 2021. StereoSet: Measuring stereotypical bias in pretrained language models. In _Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing_ ( _Volume 1: Long Papers_ ). 

- Dat Quoc Nguyen and Anh-Tuan Nguyen. 2020. PhoBERT: Pre-trained language models for Vietnamese. In _Findings of the Association for Computational Linguistics: EMNLP_ . 

- Akintunde Oladipo, Odunayo Ogundepo, Kelechi Ogueji, and Jimmy Lin. 2022. An exploration of vocabulary size and transfer effects in multilingual language models for African languages. In _3rd Workshop on African Natural Language Processing_ . 

- OpenAI. 2022. Introducing ChatGPT. Accessed on April 11, 2023. 

OpenAI. 2022. tiktoken. Git commit: `82facf9` . 

- OpenAI. 2023. GPT-4 technical report. _arXiv preprint arXiv:2303.08774_ . 

- Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al. 2022. Training language models to follow instructions with human feedback. _Advances in Neural Information Processing Systems_ . 

- Alicia Parrish, Angelica Chen, Nikita Nangia, Vishakh Padmakumar, Jason Phang, Jana Thompson, Phu Mon Htut, and Samuel Bowman. 2022. BBQ: A hand-built bias benchmark for question answering. In _Findings of the Association for Computational Linguistics: ACL 2022_ . 

- Jonas Pfeiffer, Ivan Vulić, Iryna Gurevych, and Sebastian Ruder. 2021. UNKs everywhere: Adapting multilingual language models to new scripts. In _Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing_ . 

- Alec Radford, Jeff Wu, Rewon Child, David Luan, Dario Amodei, and Ilya Sutskever. 2019. Language models are unsupervised multitask learners. 

- Jenalea Rajab. 2022. Effect of tokenisation strategies for low-resourced Southern African languages. In _3rd Workshop on African Natural Language Processing_ . 

- Krithika Ramesh, Sunayana Sitaram, and Monojit Choudhury. 2023. Fairness in language models beyond English: Gaps and challenges. In _Findings of the Association for Computational Linguistics: EACL 2023_ . Association for Computational Linguistics. 

15 

- Surangika Ranathunga, En-Shiun Annie Lee, Marjana Prifti Skenduli, Ravi Shekhar, Mehreen Alam, and Rishemjit Kaur. 2023. Neural machine translation for low-resource languages: A survey. _ACM Computing Surveys_ . 

- Jessica Rumbelow and Matthew Watkins. 2023a. SolidGoldMagikarp III: Glitch token archaelogy. Accessed on April 11, 2023. 

- Jessica Rumbelow and Matthew Watkins. 2023b. SolidGoldMagikarp (plus, prompt generation). Accessed on April 11, 2023. 

- Charles Russ. 1990. _The Dialects of Modern German: A Linguistic Survey_ . 

- Phillip Rust, Jonas Pfeiffer, Ivan Vulić, Sebastian Ruder, and Iryna Gurevych. 2021. How good is your tokenizer? On the monolingual performance of multilingual language models. In _Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing_ ( _Volume 1: Long Papers_ ). 

- Ali Safaya, Moutasem Abdullatif, and Deniz Yuret. 2020. KUISAIL at SemEval-2020 task 12: BERT-CNN for offensive speech identification in social media. In _Proceedings of the Fourteenth Workshop on Semantic Evaluation_ . 

- Teven Le Scao, Angela Fan, Christopher Akiki, Ellie Pavlick, Suzana Ilić, Daniel Hesslow, Roman Castagné, Alexandra Sasha Luccioni, François Yvon, Matthias Gallé, et al. 2022. BLOOM: A 176B-parameter open-access multilingual language model. _arXiv preprint arXiv:2211.05100_ . 

- Raphael Scheible, Fabian Thomczyk, Patric Tippmann, Victor Jaravine, and Martin Boeker. 2020. GottBERT: A pure German language model. _arXiv preprint arXiv:2012.02110_ . 

- Mike Schuster and Kaisuke Nakajima. 2012. Japanese and Korean voice search. In _IEEE International Conference on Acoustics, Speech and Signal Processing_ ( _ICASSP_ ). 

- Rico Sennrich, Barry Haddow, and Alexandra Birch. 2016. Neural machine translation of rare words with subword units. In _Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics_ ( _Volume 1: Long Papers_ ). 

- Pieter A. M. Seuren. 1995. Notes on the history and the syntax of Mauritian Creole. _Linguistics_ . 

- Yan Shao, Christian Hardmeier, and Joakim Nivre. 2018. Universal word segmentation: Implementation and interpretation. _Transactions of the Association for Computational Linguistics_ . 

- Peter Sieber and Horst Sitta. 1987. Deutsch in der Schweiz. _Zeitschrift für Germanistik_ . 

- Felix Stahlberg. 2020. Neural machine translation: A review. _Journal of Artificial Intelligence Research_ . 

- Steve Stecklow. 2018. Hatebook. _Reuters_ . Accessed on April 11, 2023. 

- Hui Su, Weiwei Shi, Xiaoyu Shen, Zhou Xiao, Tuo Ji, Jiarui Fang, and Jie Zhou. 2022. RoCbert: Robust Chinese BERT with multimodal contrastive pretraining. In _Annual Meeting of the Association for Computational Linguistics_ . 

- Lichao Sun, Kazuma Hashimoto, Wenpeng Yin, Akari Asai, Jia Li, Philip Yu, and Caiming Xiong. 2020. Adv-BERT: BERT is not robust on misspellings! Generating nature adversarial samples on BERT. _arXiv preprint arXiv:2003.04985_ . 

- Shinnosuke Takamichi and Hiroshi Saruwatari. 2018. CPJD corpus: Crowdsourced parallel speech corpus of Japanese dialects. In _Proceedings of the Eleventh International Conference on Language Resources and Evaluation_ ( _LREC 2018_ ). 

16 

- Yuqing Tang, Chau Tran, Xian Li, Peng-Jen Chen, Naman Goyal, Vishrav Chaudhary, Jiatao Gu, and Angela Fan. 2020. Multilingual translation with extensible multilingual pretraining and finetuning. _arXiv preprint arXiv:2008.00401_ . 

- Timm Teubner, Christoph M Flath, Christof Weinhardt, Wil van der Aalst, and Oliver Hinz. 2023. Welcome to the era of ChatGPT et al: The prospects of large language models. _Business & Information Systems Engineering_ . 

- The Unicode Consortium. 2022. The Unicode standard, Version 15.0.0. 

- Jörg Tiedemann. 2012. Parallel data, tools and interfaces in OPUS. In _Proceedings of the Eight International Conference on Language Resources and Evaluation_ ( _LREC_ ’ _12_ ). 

- Tohoku NLP Group. 2019. BERT models for Japanese NLP. 

- Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, et al. 2023. LLaMA: Open and efficient foundation language models. _arXiv preprint arXiv:2302.13971_ . 

- Emma Trentman and Sonia Shiri. 2020. The mutual intelligibility of Arabic dialects: Implications for the language classroom. _Critical Multilingualism Studies_ . 

- Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. _Advances in Neural Information Processing Systems_ . 

- Antti Virtanen, Jenna Kanerva, Rami Ilo, Jouni Luoma, Juhani Luotolahti, Tapio Salakoski, Filip Ginter, and Sampo Pyysalo. 2019. Multilingual is not enough: BERT for Finnish. _arXiv preprint arXiv:1912.07076_ . 

- Jonathan J. Webster and Chunyu Kit. 1992. Tokenization as the initial phase in NLP. In _The International Conference on Computational Linguistics_ . 

- Jiacheng Xu, Zhe Gan, Yu Cheng, and Jingjing Liu. 2020. Discourse-aware neural extractive text summarization. In _Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics_ . Association for Computational Linguistics. 

- Linting Xue, Aditya Barua, Noah Constant, Rami Al-Rfou, Sharan Narang, Mihir Kale, Adam Roberts, and Colin Raffel. 2022. ByT5: Towards a token-free future with pretrained byte-to-byte models. _Transactions of the Association for Computational Linguistics_ . 

- Linting Xue, Noah Constant, Adam Roberts, Mihir Kale, Rami Al-Rfou, Aditya Siddhant, Aditya Barua, and Colin Raffel. 2020. mT5: A massively multilingual pre-trained textto-text transformer. In _Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies_ . 

- Joseph K. Yamagiwa. 1967. On dialect intelligibility in Japan. _Anthropological Linguistics_ . 

- Shiyue Zhang, Vishrav Chaudhary, Naman Goyal, James Cross, Guillaume Wenzek, Mohit Bansal, and Francisco Guzman. 2022. How robust is neural machine translation to language imbalance in multilingual tokenizer training? In _Proceedings of the 15th Biennial Conference of the Association for Machine Translation in the Americas_ ( _Volume 1: Research Track_ ). 

- Judit Ács. 2019. Exploring BERT’s vocabulary. Accessed on April 11, 2023. 

- Slavomír Čéplö, Ján Bátora, Adam Benkato, Jiří Milička, Christophe Pereira, and Petr Zemánek. 2016. Mutual intelligibility of spoken Maltese, Libyan Arabic, and Tunisian Arabic functionally tested: A pilot study. _Folia Linguistica_ . 

17 

## **A Background on Tokenization** 

To enable automatic processing of language, it must first be represented in a suitable form. The current practice is to use _tokenization_ which is the process of turning natural language into sequences of _tokens_ coming from a finite and pre-determined set called _vocabulary_ (Webster and Kit, 1992). Each token is typically associated with an integer value. Language models process such sequences of integers, rather than sequences of characters or words. In this section, we offer a brief overview of the contemporary tokenization methods. For further details, we recommend the comprehensive survey by Mielke et al. (2021). 

**Word tokenization.** The simplest tokenization method is splitting at white spaces, where each word is assigned its own token (Bengio et al., 2000). This approach, however, requires that all possible words are in the vocabulary which is not possible in practice. Therefore word tokenization often fails to handle cases like “won’t”, words spelled with accented characters like “naïve” or “açaí”, speling mistakes and named entities like “Cottonshopeburnfoot” (Sun et al., 2020). This makes it unsuitable for representing _open vocabularies_ , where the words encountered are not limited to a predetermined set. Furthermore, languages that do not use spaces to separate words, such as Chinese, Japanese and Burmese, pose additional challenges for this approach (Shao et al., 2018). 

**Subword tokenization.** Hence, most current models use _subword tokenization_ , where complex words are broken down into multiple tokens. Subword tokenization can efficiently handle complex terms by breaking them down into parts, _e.g._ , “Cottonshopeburnfoot” _→_ “Cotton”+“shop”+“e”+“burn”+“foot”. This approach can represent novel words, including misspelled ones, in an open vocabulary setting. 

Subword vocabularies are usually data-based approaches which use large corpora to learn which subword sequences occur frequently in practice. Schuster and Nakajima (2012) introduced one of the first subword tokenizers, WordPiece, as a way to handle Japanese and Korean. Sennrich et al. (2016) proposed using Byte-Pair Encoding (BPE) (Gage, 1994) for learning subwords by merging the most frequently occurring pairs. BPE has since been widely used for most of the popular tokenizers. Kudo (2018) proposed an alternative approach via gradually pruning a large vocabulary. It removes tokens that are less likely to improve the performance of a simple unigram language model. Both methods rely on pretokenization (splitting on whitespaces, when available), which is not an invertible process. SentencePiece (Kudo and Richardson, 2018) addresses this de-tokenization ambiguity by treating whitespace as a special symbol, including it in the vocabulary, and supports both methods. SentencePiece with BPE is by far the most popular tokenization method for the models considered in this paper. 

**Unicode support.** Even if subword tokenization ensures that individual characters are in the vocabulary, this still leaves the question of which characters are to be included. Simple solution is to take the ASCII characters. However, this means that words in other scripts or accented letters will fall out of it. A common workaround is to represent strings outside the vocabulary as a special `UNK` token. However, if there are too many `UNK` tokens in an input, the performance of the model tends to deteriorate (Pfeiffer et al., 2021). Therefore, it is desirable that the number of `UNK` tokens in the input is kept as low as possible. A simple and commonly used solution is to base the vocabulary building on Unicode. 

Unicode is a computing industry standard for representing text characters (The Unicode Consortium, 2022). Unicode supports virtually all languages (including many ancient ones, emojis and special characters) by assigning every grapheme, modifier, punctuation mark, control character or formatting character one of 1,114,112 integer _codepoints_ . The codepoints can be represented in binary as the variable-width encoding UTF-8, which encodes every codepoint with one to four bytes, or the fixed-width UTF-32 which encodes all codepoints with four bytes (see Figure 4). 

UTF-8 can therefore represent any string in any language as a string of bytes. As each byte can take only one out of 256 values, 256 tokens can be sufficient to encode all texts. In practice this is usually combined with the BPE tokenizer. At first, the corpus is en- 

18 



Figure 4: Comparison of variable width Unicode encoding (UTF-8) and fixed width encoding (UTF-32). Image adapted from (The Unicode Consortium, 2022). 

coded as UTF-8 bytes and then BPE is ran on top of it. As most characters occur frequently, BPE would assign them a dedicated token. If the model encounters a character that didn’t exist in the training corpus ( _e.g._ , the medium skin tone waving hand ), it can still represent it byte-by-byte ( `F0` + `9F` + `91` + `8B` for the waving hand and `F0` + `9F` + `8F` + `BD` for the skin tone modifier). This allows the vocabulary to efficiently represent frequently occurring words and rare characters. For example, the sentence “I love açaí” could be tokenized as “I ”+“love ”+“a”+ `C3` + `A7` +“a”+ `C3` + `AD` . 

**Byte-level and character-level tokenization.** If we can represent any input with just 256 characters, then why bother with subword tokens? A key consideration is sequence length. This is since transformers (Vaswani et al., 2017), the currently predominant deep learning architecture for language models, have attention layers with a quadratic complexity in the input length. Hence, as the number of characters is much longer than the sub-word tokenization, working on the character level has been traditionally considered computationally inefficient. However, Chung et al. (2016), Lee et al. (2017), Gao et al. (2020), Clark et al. (2022) and Xue et al. (2022) proposed various architectures working around this issue and operating directly on characters or UTF-8 bytes. 

## **B Parity for Linguistic Varieties** 

A language can vary according to factors such as geography, history, social class and culture. As a result, different dialects, pidgin and creole language variations emerge, each with its own distinct set of grammar, vocabulary and pronunciation rules.<sup>9</sup> Unequal treatment of certain dialects or languages can lead to social and economic disadvantages for those who speak them. Therefore, it is important to also study the tokenization differences between the “standard” language and its varieties.<sup>10</sup> Unfortunately, parallel corpora for dialects, pidgin and creole language variations are far and few in between. In this section, however, we show results on regional Swiss German varieties, Arabic and Japanese dialects, as well as Haitian and Mauritian creoles. 

**Swiss German dialects.** Swiss German is a dialect continuum which significantly differs from the formal High German. German-speaking Switzerland is diglossic:<sup>11</sup> High German is used alongside regional dialects (Hogg et al., 1984). In contrast to other dialects, the use of Swiss dialects is increasing (Sieber and Sitta, 1987) especially online (Lüdi, 2007). Swiss German dialects are often considered unintelligible to High German speakers and sometimes even speakers of different dialects may find difficulty understanding each other (Russ, 1990). Therefore, ensuring that German-targeting NLP applications can process Swiss German dialects is important. 

To this end, we compare the tokenization parity relative to High German of GottBERT (Scheible et al., 2020) on the regional dialects of Aargau, Bern, Basel, Graubünden, Luzern, 

> 9While no standard definitions exist, dialects are usually considered to be regional variations of a language, whereas pidgin and creole languages are contact languages that emerge from the interaction of speakers of different languages (Muysken and Smith, 1994). 

> 10We refer to the language that the datasets label as “standard”, “official” or “dominant” without necessarily endorsing this designation. 

> 11Diglossia is the situation of two dialects or languages being used by a single language community (Kaye, 2001). 

19 

Table 6: GottBERT tokenizer premiums on the SwissDial dataset for **Swiss German dialects** . The premium is computed with respect to High German. 

|Region|GottBERT|parity|
|---|---|---|
|High German<br>||1.00|
|Zürich||1.38|
|St. Gallen||1.40|
|Basel||1.41|
|Graubünden||1.44|
|Luzern||1.52|
|Aargau||1.53|
|Wallis||1.58|
|Bern||1.59|



Table 7: ArabicBERT tokenizer premiums on the MADAR dataset for **Arabic dialects** . The premium is computed relative to Standard Arabic. 

|City<br>Ara|bicBERT|City<br>ArabicBERT|
|---|---|---|
|Jeddah|0.91|Sanaa<br>1.01|
|Doha|0.92|Beirut<br>1.02|
|Riyadh|0.92|Benghazi<br>1.02|
|Muscat|0.94|Cairo<br>1.03|
|Basra|0.95|Sfax<br>1.03|
|Salt|0.95|Tripoli<br>1.05|
|Baghdad|0.96|Aswan<br>1.06|
|Damascus|0.97|Alexandria<br>1.06|
|Aleppo|0.97|Tunis<br>1.06|
|Jerusalem|0.97|Algiers<br>1.07|
|Khartoum|0.98|Mosul<br>1.10|
|Amman|0.99|Fes<br>1.11|
|Std. Arabic|1.00|Rabat<br>1.17|



St. Gallen, Wallis and Zürich. We use SwissDial, a parallel multidialectal corpus, as the basis of comparison (Dogan-Schönberger et al., 2021). It is worth noting, that the dialect of each city and its corresponding region may differ significantly. Therefore there might be large variations within regions as well. 

The results in Table 6 show a disparity between the tokenization lengths for High German and the Swiss dialects with a premium ranging from 1.38 for the Zürich dialect, or _Züritüütsch_ , to 1.59 for the Bernese _Bärndütsch_ . In fact, English has a lower premium than any Swiss dialect (1.35 on FLORES-200, Table 2) and the premium for Bernese German is close to the linguistically further Swedish (1.64) and Norwegian Bokmål (1.65). The following example from SwissDial shows how the sentence “Like he’s waiting for something” has almost twice as long tokenization in Bernese German compared to High German: 





The fact that the GottBERT tokenizer results in better parity for English, Swedish and Norwegian Bokmål than for Swiss German dialects highlights that it does not likely pick out stable linguistic constructs. 

**Arabic dialects.** Similarly to Swiss German, Arabic is usually spoken in diglossic speech communities, where Modern Standard Arabic is spoken alongside at least one prestigious vernacular particular to the country or region (Bassiouney, 2009). As both Standard Arabic 

20 

Table 8: BERT Japanese tokenizer premiums on the CPJD dataset for **Japanese dialects** . The premium is computed with respect to Standard Japanese. The CPJD dataset consists of two parallel corpora with the dialects split across the two. Hence, we have also indicated the corpus for each dialect. Nara-ben has two entries as the dataset has transcriptions for two separate speakers. The suffix “-ben” (弁) means “speech” or “dialect”. 

|Dialect<br>Co|rpus P|arity|Dialect<br>Co|rpus|Parity|
|---|---|---|---|---|---|
|Akita-ben|2|1.09|Miyazaki-ben|1|1.05|
|Awa-ben|2|1.09|Morokata-ben|1|1.15|
|Fukui-ben|2|1.04|Nara-ben|2|1.09|
|Fukuoka-ben|1|1.03|Nara-ben|2|1.03|
|Hiroshima-ben|1|1.02|Okayama-ben|1|1.15|
|Hokkaido-ben|2|1.06|Oosaka-ben|2|1.03|
|Iwaki-ben|2|1.08|Saitama-ben|1|1.01|
|Iyo-ben|1|1.05|Tosa-ben|1|1.03|
|Izumo-ben|1|1.10|Toshu-ben|1|1.06|
|Kanazawa-ben|2|1.11|Tsugaru-ben|1|1.09|
|Kyokotoba|2|1.07||||



and its dialects are commonly used in written communication, it is vital that tokenizers handle them equally well. 

To assess the performance of Arabic tokenizers, we compare the tokenization lengths of ArabicBERT (Safaya et al., 2020) across 25 Arabic dialects. To this end, we use the MADAR parallel corpus of Arabic dialects (Bouamor et al., 2018). 

Table 7 shows the premiums relative to Standard Modern Arabic. The premium varies from 0.91 for the Jeddah dialect to 1.17 for the Rabat dialect. This is significantly lower than the premium for English (1.83 on FLORES-200 Table 2). The range is also much smaller than for the Swiss German dialects and approximately half of the considered dialects have a lower premium than Standard Modern Arabic. Therefore, one could say that the tokenizer of ArabicBERT achieves tokenization parity for these 25 Arabic vernaculars. This is likely because the corpus and vocabulary set on which ArabicBERT was trained contained dialectical Arabic. It is also possible that Arabic dialects are closer to Modern Standard Arabic and more mutually intelligible than Swiss German dialects are to High German (Čéplö et al., 2016; Trentman and Shiri, 2020). Still, this difference between the parity for Swiss and Arabic dialects indicates that including a broader set of vernaculars and dialects in the corpus results in improved tokenization parity. 

**Japanese dialects.** Japanese also has a number of regional dialects (Hattori, 1973). We compare the tokenization parity of BERT Japanese (Tohoku NLP Group, 2019) across them. We employ the CPJD dataset by Takamichi and Saruwatari (2018) which contains transcriptions of the voice recordings of 250 sentences across 20 dialects. 

The results in Table 8 show that the premium compared to Standard Japanese (Tokyo dialect) ranges from 1.01 (for Saitama prefecture, neighbouring Tokyo) to 1.15 (for Morokataben and Okayama-ben). These all are significantly lower than the premium for English (1.49, as shown in Table 2). Therefore, similarly to ArabicBERT, this is an example of the tokenizer being relatively well-aligned with the dialects. This is likely because Japanese dialects are more closely related (and intelligible (Yamagiwa, 1967) to Standard Japanese speakers) than the Swiss dialects are to High German speakers. 

**Mauritian and Haitian Creoles.** While creoles often have some similarities with a highresource language (usually English or French), the differences are significant to necessitate special attention to their support (Lent et al., 2021, 2022). This is especially critical for emergency services and disaster management (Munro, 2010). 

Mauritian Creole is based on French as well as the languages of slaves imported from Madagascar and East Africa. As the British gained control of Mauritius, they brought indentured labourers from India who further had an effect on the formation of the modern Mauritian 

21 

Creole (Seuren, 1995). Similarly, Haitian Creole ( _Kreyòl_ ) emerged from the interaction of French and the various Niger-Congo languages spoken by the Africans brought as slaves (DeGraff, 2007). 

Considering that both languages have their basis in French, one would expect that tokenizers targeting French would have low tokenization parities for Mauritian and Haitian Creoles. However, taking the tokenizer of CamemBERT (Martin et al., 2020), the premium for Mauritian Creole is 1.20 using the MorisienMT parallel corpus (Dabre and Sukhoo, 2022). The premium for Haitian Creole is 1.64 when using the QEDv2 corpus (Tiedemann, 2012; Abdelali et al., 2014). Haitian Creole is also represented in the FLORES-200 dataset where the premium relative to French is 1.58. This is significantly larger than linguistically further languages such as English (1.20), Pangasinan (1.49) and Nigerian Fulfulde (1.54). Therefore, CamemBERT is not well-placed to tokenize French-related creoles despite the model being trained for French. 

## **C Extended Tables of Tokenization Premiums** 

In addition to the models presented in the main text, these extended tables also include LLAMA (Touvron et al., 2023), MBart50 (Liu et al., 2020; Tang et al., 2020), SeamlessM4T (Barrault et al., 2023) and Qwen-VL (Bai et al., 2023). 

22 

|Language|A||base|base|edit|_base|RTa|ERT|mBERT|RT|ert|RoBERTa|00|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
||LLAM|GPT-2|r50k_|p50k_|p50k_|cl100k|RoBE|GottB|Came|PhoBE|RoCB|XLM-|M2M1|
|Acehnese (Arabic script)|4.00|4.78|4.78|4.78|4.78|3.78|4.78|4.95|—|—|—|1.94|1.89|
|Acehnese (Latin script)<br>A|1.89<br>|2.16<br>|2.16<br>|2.16<br>|2.16<br>|1.98<br>|2.16<br>|1.56<br>|1.55|1.37|1.10|1.57<br>|1.47<br>|
|Mesopotamian rabic<br>Ta’izzi-Adeni Arabic|3.34<br>3.38|4.27<br>4.34|4.27<br>4.34|4.27<br>4.34|4.27<br>4.34|2.99<br>3.01|4.27<br>4.34|5.10<br>5.16|—<br>—|—<br>—|—<br>—|1.16<br>1.17|1.27<br>1.28|
|Tunisian Arabic|3.31|4.20|4.20|4.20|4.20|2.93|4.20|5.03|—|—|—|1.20|1.29|
|Afrikaans<br>|1.55<br>|1.94<br>|1.94<br>|1.94<br>|1.94<br>|1.69<br>|1.94<br>|1.25<br>|1.38|1.26|1.06|1.20<br>|1.22<br>|
|South Levantine Arabic<br>Akan|3.20<br>2.20|4.02<br>2.80|4.02<br>2.80|4.02<br>2.80|4.02<br>2.80|2.84<br>2.68|4.02<br>2.80|4.84<br>1.90|—<br>1.64|—<br>1.45|—<br>—|1.12<br>1.98|1.22<br>1.83|
|Tosk Albanian<br>|2.26<br>|2.65<br>|2.65<br>|2.65<br>|2.65<br>|2.25<br>|2.65<br>|1.77<br>|1.82|1.69|1.12|1.32<br>|1.36<br>|
|Amharic|7.32|7.79|7.79|7.79|7.79|7.68|7.79|5.19|—|—|—|1.34|1.42|
|North Levantine Arabic<br>StddAbi|3.19<br>342|4.04<br>440|4.04<br>440|4.04<br>440|4.04<br>440|2.83<br>304|4.04<br>440|4.83<br>521|—<br>—|—<br>—|—<br>—|1.15<br>118|1.24<br>129|
|anar rac<br>Standard Arabic (Romanize|.<br>d)<br>2.31|.<br>2.51|.<br>2.51|.<br>2.51|.<br>2.51|.<br>2.45|.<br>2.51|.<br>1.76|1.72|1.55|1.19|.<br>1.94|.<br>1.83|
|Najdi Arabic<br>Moroccan Arabic<br>|3.43<br>3.35<br>|4.41<br>4.21<br>|4.41<br>4.21<br>|4.41<br>4.21<br>|4.41<br>4.21<br>|3.04<br>2.96<br>|4.41<br>4.21<br>|5.22<br>5.08<br>|—<br>—|—<br>—|—<br>—|1.18<br>1.25<br>|1.30<br>1.33<br>|
|Egyptian Arabic<br>Assamese|3.36<br>6.14|4.23<br>9.79|4.23<br>9.79|4.23<br>9.78|4.23<br>9.78|2.96<br>6.20|4.23<br>9.79|5.10<br>8.32|—<br>—|—<br>—|—<br>—|1.17<br>1.90|1.27<br>2.24|
|Asturian<br>|1.48<br>|1.89<br>|1.89<br>|1.89<br>|1.89<br>|1.58<br>|1.89<br>|1.33<br>|1.31|1.24|1.04|1.27<br>|1.15<br>|
|Awadhi<br>Central Aymara|4.53<br>2.03|7.19<br>2.32|7.19<br>2.32|7.19<br>2.32|7.19<br>2.32|4.78<br>2.17|7.19<br>2.32|8.19<br>1.62|—<br>1.62|—<br>1.47|—<br>1.09|1.37<br>1.70|1.47<br>1.64|
|SouthAzerbaijani|376|516|516|516|516|334|516|532|—|—|—|143|150|
|<br>NhAbii|.<br>261|.<br>347|.<br>347|.<br>347|.<br>347|.<br>264|.<br>347|.<br>231||190||.<br>115|.<br>126|
|ort zerajan<br>Bashkir|.<br>2.91|.<br>6.01|.<br>6.01|.<br>6.01|.<br>6.01|.<br>4.28|.<br>6.01|.<br>3.97|—<br>—|.<br>—|—<br>—|.<br>2.06|.<br>1.23|
|Bambara<br>Bli|1.99<br>177|2.66<br>197|2.66<br>197|2.66<br>197|2.66<br>197|2.57<br>180|2.66<br>197|1.84<br>139|1.54<br>143|1.40<br>128|—<br>114|1.82<br>132|1.72<br>129|
|anese<br>Belarusian|.<br>2.38|.<br>6.56|.<br>6.56|.<br>6.56|.<br>6.56|.<br>3.55|.<br>6.56|.<br>4.17|.<br>—|.<br>2.88|.<br>—|.<br>1.46|.<br>1.56|
|Bemba<br>Bli|2.15<br>538|2.46<br>965|2.46<br>965|2.46<br>965|2.46<br>965|2.23<br>584|2.46<br>965|1.69<br>854|1.68<br>—|1.53<br>—|1.26<br>—|1.76<br>138|1.67<br>155|
|enga<br>Bhojpuri|.<br>4.52|.<br>7.18|.<br>7.18|.<br>7.18|.<br>7.18|.<br>4.69|.<br>7.18|.<br>8.08|—|—|—|.<br>1.47|.<br>1.54|
|Banjar (Arabic script)<br>Banjar(Latinscrit)|4.22<br>175|5.03<br>198|5.03<br>198|5.03<br>198|5.03<br>198|3.80<br>171|5.03<br>198|5.53<br>138|—<br>135|—<br>121|—<br>108|1.92<br>121|1.93<br>116|
|p<br>Standard Tibetan|.<br>6.67|.<br>14.93|.<br>14.93|.<br>14.93|.<br>14.93|.<br>11.27|.<br>14.93|.<br>10.87|.<br>—|.<br>—|.<br>—|.<br>—|.<br>—|
|Bosnian|1.69|2.19|2.19|2.19|2.19|1.87|2.19|1.47|1.46|1.35|1.02|1.12|1.17|
|Buinese|187|220|220|220|220|198|220|149|145|135|110|151|149|
|g<br>Bulgarian<br>|.<br>1.78<br>|.<br>5.51<br>|.<br>5.51<br>|.<br>5.51<br>|.<br>5.51<br>|.<br>2.64<br>|.<br>5.51<br>|.<br>3.51<br>|.<br>—<br>|.<br>2.57<br>|.<br>—<br>|.<br>1.16<br>|.<br>1.23<br>|
|Catalan|1.51|1.92|1.92|1.92|1.92|1.71|1.92|1.40|1.33|1.31|1.10|1.26|1.26|
|Cebuano|196|224|224|224|224|193|224|157|159|141|120|152|138|
|Czech<br>|.<br>1.69<br>|.<br>2.62<br>|.<br>2.62<br>|.<br>2.62<br>|.<br>2.62<br>|.<br>2.11<br>|.<br>2.62<br>|.<br>1.73<br>|.<br>—<br>|.<br>1.48<br>|.<br>0.99<br>|.<br>1.17<br>|.<br>1.23<br>|
|Chokwe|1.91|2.16|2.16|2.16|2.16|1.98|2.16|1.51|1.49|1.32|1.10|1.55|1.47|
|Central Kurdish<br>Crimean Tatar<br>|4.43<br>2.13<br>|6.49<br>2.49<br>|6.49<br>2.49<br>|6.49<br>2.49<br>|6.49<br>2.49<br>|4.80<br>2.12<br>|6.49<br>2.49<br>|5.82<br>1.67<br>|—<br>1.68<br>|—<br>1.54<br>|—<br>—<br>|2.30<br>1.38<br>|2.48<br>1.37<br>|
|Welsh|2.09|2.34|2.34|2.34|2.34|2.12|2.34|1.66|1.68|1.53|1.06|1.43|1.44|
|Danish|1.54|1.90|1.90|1.90|1.90|1.62|1.90|1.26|1.39|1.29|1.04|1.09|1.12|
|German<br>|1.41<br>|2.14<br>|2.14<br>|2.14<br>|2.14<br>|1.58<br>|2.14<br>|0.74<br>|1.55<br>|1.40<br>|1.20<br>|1.17<br>|1.24<br>|
|Southwestern Dinka|1.88|2.48|2.48|2.48|2.48|2.25|2.48|1.60|1.43|1.32|0.75|1.68|1.55|
|Dyula|1.88|2.20|2.20|2.20|2.20|2.05|2.20|1.54|1.43|1.30|0.98|1.65|1.53|
|Dzongkha<br>|7.42<br>|16.36<br>|16.36<br>|16.36<br>|16.36<br>|12.33<br>|16.36<br>|11.95<br>|—|—<br>|—<br>|—<br>|—<br>|
|Greek|4.99|6.54|6.54|6.54|6.54|5.15|6.54|4.99|—|3.11|1.15|1.45|1.58|
|English|1.00|1.00|1.00|1.00|1.00|1.00|1.00|1.00|1.00|1.00|1.00|1.00|1.00|
|Eseranto|167|203|203|203|203|187|203|137|135|126|101|120|138|
|p<br>Estonian|.<br>1.76|.<br>2.11|.<br>2.11|.<br>2.11|.<br>2.11|.<br>1.87|.<br>2.11|.<br>1.39|.<br>1.42|.<br>1.33|.<br>1.03|.<br>1.12|.<br>1.20|
|Basque<br>Ewe<br>|1.79<br>2.28<br>|2.10<br>2.90<br>|2.10<br>2.90<br>|2.10<br>2.90<br>|2.10<br>2.90<br>|1.88<br>2.75<br>|2.10<br>2.90<br>|1.39<br>1.97<br>|1.44<br>1.69<br>|1.33<br>1.46<br>|1.11<br>—|1.16<br>2.01<br>|1.23<br>1.86<br>|
|Faroese<br>|1.92<br>|2.38<br>|2.38<br>|2.38<br>|2.38<br>|2.07<br>|2.38<br>|1.66<br>|1.64<br>|1.46<br>|—<br>|1.44<br>|1.41<br>|
|Fijian<br>Finnish|2.02<br>191|2.30<br>228|2.30<br>228|2.30<br>228|2.30<br>228|2.15<br>199|2.30<br>228|1.67<br>146|1.52<br>156|1.39<br>147|1.13<br>113|1.72<br>114|1.62<br>123|
|Fon<br>French|.<br>2.83<br>1.47|.<br>4.08<br>2.00|.<br>4.08<br>2.00|.<br>4.08<br>2.00|.<br>4.08<br>2.00|.<br>3.67<br>1.60|.<br>4.08<br>2.00|.<br>2.75<br>1.47|.<br>—<br>0.84|.<br>—<br>1.38|.<br>—<br>1.20|.<br>2.51<br>1.30|.<br>2.31<br>1.33|
|Friulian|1.70|2.07|2.07|2.07|2.07|1.85|2.07|1.47|1.38|1.33|1.07|1.56|1.47|
|Nigerian Fulfulde<br>|1.72<br>|1.99<br>|1.99<br>|1.99<br>|1.99<br>|1.85<br>|1.99<br>|1.37<br>|1.29<br>|1.16<br>|0.86<br>|1.46<br>|1.27<br>|
|West Central Oromo|2.22|2.53|2.53|2.53|2.53|2.32|2.53|1.72|1.73|1.61|1.24|1.78|1.49|
|Scottish Gaelic<br>Iih|2.33<br>217|2.70<br>256|2.70<br>256|2.70<br>256|2.70<br>256|2.42<br>233|2.70<br>256|1.86<br>176|1.80<br>175|1.61<br>155|1.24<br>115|1.75<br>150|1.61<br>150|
|rs<br>Galician|.<br>1.48|.<br>1.91|.<br>1.91|.<br>1.91|.<br>1.91|.<br>1.56|.<br>1.91|.<br>1.39|.<br>1.36|.<br>1.30|.<br>1.11|.<br>1.13|.<br>1.14|
|Guarani|1.99|2.46|2.46|2.46|2.46|2.17|2.46|1.68|1.55|1.45|1.05|1.72|1.63|
|Gujarati<br>|9.98<br>|12.27<br>|12.27<br>|12.27<br>|12.27<br>|7.69<br>|12.27<br>|8.17<br>|—<br>|—<br>|—<br>|1.42<br>|1.58<br>|
|Haitian Creole|1.58|1.90|1.90|1.90|1.90|1.74|1.90|1.35|1.32|1.15|0.89|1.39|1.16|
|Hausa|1.89|2.15|2.15|2.15|2.15|2.00|2.15|1.49|1.47|1.26|1.02|1.40|1.29|



23 

|Language|Bart50|5|nT5|T5|NINE|OOM|abicBERT|RIL|F-32|RT Japanese|amlessM4T|LB|en|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
||M|mT|Fla|By|CA|BL|Ar|Mu|UT|BE|Se|NL|Qw|
|Acehnese (Arabic script)<br>Acehnese(Latinscript)|1.94<br>157|1.79<br>144|—<br>255|1.51<br>109|0.85<br>107|2.65<br>174|—<br>144|—<br>202|0.85<br>107|—<br>141|1.89<br>124|1.89<br>124|2.66<br>195|
|<br>MiAbi|.<br>116|.<br>128|.|.<br>156|.<br>086|.<br>115|.<br>055|.<br>193|.<br>086|.|.<br>137|.<br>137|.<br>163|
|esopotaman rac<br>Ta’izzi-Adeni Arabic|.<br>1.17|.<br>1.32|—<br>—|.<br>1.58|.<br>0.87|.<br>1.15|.<br>0.55|.<br>1.94|.<br>0.87|—<br>—|.<br>1.39|.<br>1.39|.<br>1.63|
|TunisianArabic|1.20|1.29|—|1.54|0.85|1.19|0.57|1.90|0.85|—|1.39|1.39|1.66|
|<br>Afik|120|120|215|107|106|169|133|184|106|127|122|122|167|
|raans<br>South Levantine Arabic|.<br>1.12|.<br>1.24|.<br>—|.<br>1.49|.<br>0.83|.<br>1.12|.<br>0.55|.<br>1.82|.<br>0.83|.<br>—|.<br>1.31|.<br>1.31|.<br>1.55|
|Akan|1.98|1.82|2.96|1.10|1.00|2.05|—|—|1.00|1.45|1.40|1.40|2.28|
|Tosk Albanian<br>|1.32<br>|1.48<br>|3.09|1.20<br>|1.12<br>|2.17<br>|1.46|2.52|1.12<br>|—|1.35<br>|1.35<br>|2.23<br>|
|Amharic|1.34|1.73|—|1.72|0.67|5.07|—|—|0.67|—|1.32|1.32|4.16|
|North Levantine Arabic|1.15|1.23|—|1.48|0.82|1.13|0.55|1.83|0.82|—|1.33|1.33|1.58|
|Standard Arabic<br>|1.18<br>|1.35<br>|—<br>|1.60<br>|0.88<br>|1.14<br>|0.55<br>|1.97<br>|0.88<br>|—<br>|1.40<br>|1.40<br>|1.63<br>|
|Standard Arabic (Romanized)|1.94|1.73|2.94|1.17|1.17|2.15|1.60|2.28|1.17|1.64|1.86|1.86|2.42|
|Najdi Arabic|1.18|1.35|—|1.60|0.88|1.15|0.55|1.97|0.88|—|1.40|1.40|1.63|
|MoroccanArabic|125|129|—|156|086|126|063|191|086|—|139|139|170|
|<br>Egyptian Arabic<br>Assamese|.<br>1.17<br>1.90|.<br>1.28<br>1.94|—<br>—|.<br>1.56<br>2.54|.<br>0.86<br>0.96|.<br>1.16<br>1.41|.<br>0.57<br>—|.<br>1.89<br>1.24|.<br>0.86<br>0.96|—<br>—|.<br>1.36<br>1.39|.<br>1.36<br>1.39|.<br>1.64<br>5.46|
|Asturian<br>|1.27<br>|1.28<br>|2.07|1.07<br>|1.03<br>|1.31<br>|1.24|1.81<br>|1.03<br>|1.26|1.17<br>|1.17<br>|1.56<br>|
|Awadhi<br>Central Aymara|1.37<br>1.70|1.62<br>1.57|—<br>2.71|2.50<br>1.07|0.98<br>1.05|1.43<br>1.94|—<br>1.44|1.29<br>1.98|0.98<br>1.05|—<br>1.45|1.22<br>1.32|1.22<br>1.32|4.36<br>2.15|
|SouthAzerbaijani|143|142|—|163|089|181|111|172|089|—|137|137|262|
|<br>North Azerbaijani<br>|.<br>1.15<br>|.<br>1.35<br>|—|.<br>1.26<br>|.<br>1.09<br>|.<br>2.30<br>|.<br>1.74|.<br>—|.<br>1.09<br>|—|.<br>1.33<br>|.<br>1.33<br>|.<br>2.49<br>|
|Bashkir|2.06|1.60|—|1.85|1.01|3.57|—|—|1.01|—|1.22|1.22|3.14|
|Bambara<br>Bl|1.82<br>12|1.65<br>12|2.70<br>2|1.04<br>111|0.96<br>111|1.89<br>14|—<br>14|—<br>1|0.96<br>111|1.34<br>1|1.27<br>1|1.27<br>1|2.14<br>1|
|ainese<br>Belarusian|.3<br>1.46|.9<br>1.59|.37<br>—|.<br>2.06|.<br>1.13|.6<br>3.24|.0<br>2.60|.83<br>—|.<br>1.13|.35<br>—|.08<br>1.72|.08<br>1.72|.79<br>3.00|
|Bemba|1.76|1.57|3.01|1.23|1.23|1.92|1.65|2.17|1.23|1.64|1.39|1.39|2.20|
|Bengali<br>|1.38<br>|1.58<br>|—|2.61<br>|0.98<br>|1.17<br>|—|1.01<br>|0.98<br>|—|1.28<br>|1.28<br>|5.09<br>|
|Bhojpuri<br>Banjar (Arabic script)|1.47<br>1.92|1.63<br>1.76|—<br>—|2.47<br>1.69|0.97<br>0.93|1.53<br>2.47|—<br>1.04|1.39<br>—|0.97<br>0.93|—<br>—|1.28<br>1.88|1.28<br>1.88|4.33<br>2.63|
|Banjar (Latin script)<br>|1.21|1.16<br>|2.20|1.05<br>|1.05<br>|1.30<br>|1.32|1.71|1.05<br>|1.29|1.08<br>|1.08<br>|1.70<br>|
|Standard Tibetan|—|3.68|—|3.31|1.13|6.66|—|—|1.13|—|1.44|1.44|7.33|
|Bosnian|1.12|1.33|2.48|1.03|1.01|1.84|1.39|—|1.01|1.30|1.19|1.19|1.86|
|Buinese|151|144|251|109|106|171|145|196|106|139|130|130|196|
|g<br>Bulgarian|.<br>1.16|.<br>1.28|.<br>—|.<br>1.89|.<br>1.04|.<br>2.49|.<br>2.35|.<br>—|.<br>1.04|.<br>—|.<br>1.31|.<br>1.31|.<br>2.20|
|Catalan|1.26|1.36|2.14|1.12|1.10|1.18|1.29|1.90|1.10|1.30|1.25|1.25|1.69|
|Cebuano<br>|1.52<br>|1.42<br>|2.86<br>|1.20<br>|1.20<br>|1.78<br>|1.51<br>|2.10|1.20<br>|1.53|1.29<br>|1.29<br>|1.91<br>|
|Czech<br>Chokwe|1.17<br>1.55|1.27<br>1.41|2.72<br>2.66|1.08<br>1.07|0.97<br>1.07|2.03<br>1.72|1.31<br>1.47|—<br>1.94|0.97<br>1.07|—<br>1.42|1.26<br>1.34|1.26<br>1.34|2.07<br>1.94|
|CentralKurdish|230|175|—|178|097|321|165|—|097|—|130|130|346|
|<br>Crimean Tatar<br>|.<br>1.38|.<br>1.32|2.80|.<br>1.13|.<br>1.03|.<br>2.07|.<br>1.45|—|.<br>1.03|—|.<br>1.25|.<br>1.25|.<br>1.95|
|Welsh<br>Danish|1.43<br>109|1.70<br>114|3.12<br>226|1.07<br>105|1.07<br>103|2.09<br>167|1.55<br>128|2.32<br>183|1.07<br>103|1.47<br>—|1.38<br>111|1.38<br>111|2.09<br>161|
|German<br>|.<br>1.17<br>|.<br>1.19<br>|.<br>1.37|.<br>1.18<br>|.<br>1.17<br>|.<br>1.68<br>|.<br>1.44|.<br>2.02|.<br>1.17<br>|1.37|.<br>1.29<br>|.<br>1.29<br>|.<br>1.55<br>|
|Southwestern Dinka<br>Dyula|1.68<br>1.65|1.58<br>1.55|—<br>2.68|0.96<br>1.07|0.86<br>1.01|1.82<br>1.80|—<br>1.30|—<br>2.06|0.86<br>1.01|—<br>1.39|1.25<br>1.44|1.25<br>1.44|2.01<br>1.96|
|Dkh||424||364|125|736|||125||148|148|819|
|zonga<br>|—<br>|.<br>|—|.<br>|.<br>|.<br>|—<br>|—|.<br>|—|.<br>|.<br>|.<br>|
|Greek|1.45|1.65|—|2.17|1.20|3.81|2.70|—|1.20|—|1.65|1.65|4.95|
|English<br>Et|1.00<br>120|1.00<br>119|1.00<br>219|1.00<br>102|1.00<br>100|1.00<br>165|1.00<br>124|1.00|1.00<br>100|1.00|1.00<br>123|1.00<br>123|1.00<br>180|
|sperano<br>Estonian|.<br>1.12|.<br>1.12|.<br>2.43|.<br>1.01|.<br>0.98|.<br>1.77|.<br>1.28|—<br>1.71|.<br>0.98|—<br>—|.<br>1.16|.<br>1.16|.<br>1.85|
|Basque<br>Ewe<br>|1.16<br>2.01<br>|1.22<br>1.82<br>|2.33<br>2.85<br>|1.07<br>1.07<br>|1.06<br>0.97<br>|1.14<br>2.11<br>|1.41<br>—<br>|1.90<br>—|1.06<br>0.97<br>|1.35<br>—|1.27<br>1.27<br>|1.27<br>1.27<br>|1.87<br>2.36<br>|
|Faroese<br>Fijian|1.44<br>1.72|1.40<br>1.59|2.73<br>3.02|1.09<br>1.17|1.02<br>1.17|1.95<br>1.99|1.41<br>1.65|—<br>2.01|1.02<br>1.17|—<br>1.53|1.31<br>1.32|1.31<br>1.32|2.04<br>2.13|
|Finnish<br>|1.14<br>|1.16<br>|2.61|1.11<br>|1.07<br>|1.89<br>|1.42|2.05|1.07<br>|1.45|1.21<br>|1.21<br>|1.97<br>|
|Fon<br>|2.51|2.36|—|1.26|1.02|2.21|—|—|1.02|—|1.59|1.59|2.87|
|French<br>Friulian|1.30<br>156|1.40<br>152|1.60<br>230|1.24<br>113|1.19<br>110|1.20<br>170|1.33<br>128|1.96<br>194|1.19<br>110|1.36<br>129|1.35<br>137|1.35<br>137|1.57<br>183|
|Nigerian Fulfulde<br>|.<br>1.46<br>|.<br>1.32|.<br>2.14|.<br>0.96|.<br>0.93|.<br>1.66|.<br>1.16|.<br>1.54<br>|.<br>0.93|.<br>1.21|.<br>1.24<br>|.<br>1.24<br>|.<br>1.75|
|West Central Oromo|1.78|1.69|3.16|1.20|1.19|2.19|1.63|2.17|1.19|1.63|1.42|1.42|2.29|
|Scottish Gaelic<br>h|1.75<br>|1.85<br>|3.24<br>|1.28<br>|1.24<br>|2.25<br>|1.57<br>|2.27<br>|1.24<br>|1.49<br>|1.56<br>|1.56<br>|2.38<br>|
|Iris<br>Galician|1.50<br>1.13|1.67<br>1.31|3.14<br>2.18|1.23<br>1.13|1.16<br>1.11|2.15<br>1.27|1.45<br>1.30|2.46<br>1.91|1.16<br>1.11|1.51<br>1.32|1.42<br>1.16|1.42<br>1.16|2.28<br>1.54|
|Guarani|1.72|1.62|2.57|1.09|1.01|1.87|1.40|1.99|1.01|—|1.34|1.34|2.09|
|Gujarati<br>|1.42<br>|1.73<br>|—<br>|2.50<br>|0.96<br>|1.35<br>|—<br>|1.19<br>|0.96<br>|—<br>|1.35<br>|1.35<br>|6.78<br>|
|Haitian Creole|1.39|1.22|2.32|0.95|0.92|1.56|1.18|1.68|0.92|1.19|1.11|1.11|1.72|
|Hausa|1.40|1.37|2.61|1.08|1.07|1.78|1.34|1.78|1.07|1.35|1.18|1.18|1.95|



24 

|Language|A||base|base|edit|_base|RTa|ERT|mBERT|RT|ert|RoBERTa|00|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
||LLAM|GPT-2|r50k_|p50k_|p50k_|cl100k|RoBE|GottB|Came|PhoBE|RoCB|XLM-|M2M1|
|Hebrew|3.29|4.39|4.39|4.39|4.39|3.66|4.39|4.52|—|—|—|1.12|1.22|
|Hindi|4.60|7.46|7.46|7.46|7.46|4.79|7.46|8.34|—|—|—|1.25|1.36|
|Chhattisgarhi<br>Croatian|4.44<br>1.67|7.21<br>2.15|7.21<br>2.15|7.21<br>2.15|7.21<br>2.15|4.69<br>1.85|7.21<br>2.15|8.05<br>1.46|—<br>1.43|—<br>1.33|—<br>1.00|1.41<br>1.10|1.51<br>1.15|
|Hungarian|1.79|2.66|2.66|2.66|2.66|2.15|2.66|1.79|1.78|1.57|1.09|1.18|1.28|
|Armenian<br>Igbo|5.11<br>2.32|10.01<br>3.42|10.01<br>3.42|10.01<br>3.42|10.01<br>3.42|9.98<br>2.44|10.01<br>3.42|6.67<br>2.33|—<br>1.77|—<br>1.48|—<br>0.99|1.38<br>2.12|1.50<br>1.47|
|Ilocano|2.01|2.26|2.26|2.26|2.26|2.05|2.26|1.59|1.61|1.41|1.21|1.61|1.33|
|Indonesian<br>|1.76<br>|1.98<br>|1.98<br>|1.98<br>|1.98<br>|1.55<br>|1.98<br>|1.37<br>|1.40|1.25<br>|1.12|0.94<br>|0.98<br>|
|Icelandic|1.98|2.43|2.43|2.43|2.43|2.15|2.43|1.72|—|1.50|—|1.23|1.29|
|Italian<br>Javanese<br>|1.46<br>1.72<br>|2.01<br>1.93<br>|2.01<br>1.93<br>|2.01<br>1.93<br>|2.01<br>1.93<br>|1.64<br>1.73<br>|2.01<br>1.93<br>|1.43<br>1.36<br>|1.36<br>1.39|1.33<br>1.21|1.19<br>1.06<br>|1.19<br>1.15<br>|1.25<br>1.10<br>|
|Japanese|2.24|3.00|3.00|3.00|3.00|2.30|3.00|3.23|—|—|0.52|1.11|1.20|
|Kabyle|2.00|2.50|2.50|2.50|2.50|2.47|2.50|1.74|1.59|1.43|0.90|1.84|1.71|
|Jingpho<br>|2.27<br>|2.65<br>|2.65<br>|2.65<br>|2.65<br>|2.35<br>|2.65<br>|1.89<br>|1.78<br>|1.54<br>|1.20<br>|1.94<br>|1.78<br>|
|Kamba<br>Kannada|1.91<br>10.83|2.32<br>13.69|2.32<br>13.69|2.32<br>13.68|2.32<br>13.68|2.17<br>8.90|2.32<br>13.69|1.62<br>9.27|1.48<br>—|1.30<br>—|0.98<br>—|1.62<br>1.36|1.52<br>1.53|
|Kashmiri(Arabicscript)|443|619|619|619|619|462|619|563|—|—|—|193|193|
|<br>Kashmiri (Devanagari script)<br>|.<br>4.44<br>|.<br>7.03<br>|.<br>7.03<br>|.<br>7.03<br>|.<br>7.03<br>|.<br>4.69<br>|.<br>7.03<br>|.<br>7.76<br>|—|—|—|.<br>1.82<br>|.<br>1.86<br>|
|Georgian|4.87|13.85|13.85|13.85|13.85|9.85|13.85|9.22|—|—|—|1.34|1.56|
|Kazakh|2.51|5.92|5.92|5.92|5.92|3.79|5.92|3.91|—|2.66|—|1.15|1.28|
|Kabiye<br>Kabuverdianu|3.48<br>1.58|4.87<br>1.93|4.87<br>1.93|4.87<br>1.93|4.87<br>1.93|4.74<br>1.72|4.87<br>1.93|3.28<br>1.32|—<br>1.30|—<br>1.21|—<br>0.98|2.98<br>1.35|2.71<br>1.30|
|Halh Mongolian<br>Kh|2.76<br>1026|6.42<br>1533|6.42<br>1533|6.42<br>1533|6.42<br>1533|3.77<br>888|6.42<br>1533|4.24<br>1022|—|2.72|—|1.21<br>162|1.34<br>187|
|mer<br>|.<br>|.<br>|.<br>|.<br>|.<br>|.<br>|.<br>|.<br>|—|—<br>|—<br>|.<br>|.<br>|
|Kikuyu<br>Kinyarwanda|2.52<br>2.04|3.44<br>2.37|3.44<br>2.37|3.44<br>2.37|3.44<br>2.37|3.29<br>2.14|3.44<br>2.37|2.36<br>1.61|—<br>1.59|1.66<br>1.47|1.18<br>1.15|2.31<br>1.72|2.17<br>1.63|
|K|244|574|574|574|574|351|574|379|—|267|—|116|166|
|yrgyz<br>Kimbundu<br>Northern Kurdish|.<br>2.02<br>2.05|.<br>2.33<br>2.45|.<br>2.33<br>2.45|.<br>2.33<br>2.45|.<br>2.33<br>2.45|.<br>2.13<br>2.20|.<br>2.33<br>2.45|.<br>1.64<br>1.66|1.58<br>1.65|.<br>1.43<br>1.40|1.12<br>0.99|.<br>1.64<br>1.38|.<br>1.54<br>1.66|
|CentralKanuri(Arabicscrit)|382|474|474|474|474|363|474|520|—|—|—|260|249|
|p<br>Central Kanuri (Latin script)|.<br>2.15|.<br>2.57|.<br>2.57|.<br>2.57|.<br>2.57|.<br>2.37|.<br>2.57|.<br>1.78|1.60|1.44|—|.<br>1.74|.<br>1.65|
|Kikongo|1.93|2.17|2.17|2.17|2.17|1.99|2.17|1.61|1.44|1.37|1.12|1.58|1.48|
|Korean|318|507|507|507|507|238|507|386|—|—|099|116|121|
|Lao<br>|.<br>11.47<br>|.<br>13.19<br>|.<br>13.19<br>|.<br>13.19<br>|.<br>13.19<br>|.<br>9.62<br>|.<br>13.19<br>|.<br>8.79<br>|—<br>|—<br>|.<br>—<br>|.<br>1.39<br>|.<br>1.61<br>|
|Ligurian|1.84|2.29|2.29|2.29|2.29|1.98|2.29|1.57|1.50|1.43|1.09|1.65|1.59|
|Limburgish|164|205|205|205|205|180|205|134|139|132|104|145|138|
|Lingala<br>|.<br>1.79<br>|.<br>2.03<br>|.<br>2.03<br>|.<br>2.03<br>|.<br>2.03<br>|.<br>1.86<br>|.<br>2.03<br>|.<br>1.47<br>|.<br>1.37<br>|.<br>1.26<br>|.<br>1.08<br>|.<br>1.52<br>|.<br>1.26<br>|
|Lithuanian<br>Lombard|1.89<br>185|2.45<br>237|2.45<br>237|2.45<br>237|2.45<br>237|2.21<br>204|2.45<br>237|1.63<br>158|1.53<br>152|1.42<br>141|1.04<br>104|1.17<br>171|1.25<br>156|
|Ltli|.<br>199|.<br>239|.<br>239|.<br>239|.<br>239|.<br>220|.<br>239|.<br>167|.<br>162|.<br>148|.<br>102|.<br>157|.<br>151|
|agaan<br>Luxembourgish|.<br>1.80|.<br>2.25|.<br>2.25|.<br>2.25|.<br>2.25|.<br>1.99|.<br>2.25|.<br>1.30|.<br>1.52|.<br>1.43|.<br>1.15|.<br>1.64|.<br>1.32|
|Luba-Kasai<br>Gd|1.89<br>190|2.13<br>217|2.13<br>217|2.13<br>217|2.13<br>217|1.94<br>196|2.13<br>217|1.50<br>148|1.44<br>147|1.31<br>136|1.09<br>107|1.54<br>155|1.43<br>138|
|ana<br>Luo<br>Mizo|.<br>1.76<br>1.86|.<br>2.04<br>2.09|.<br>2.04<br>2.09|.<br>2.04<br>2.09|.<br>2.04<br>2.09|.<br>1.82<br>1.96|.<br>2.04<br>2.09|.<br>1.40<br>1.53|.<br>1.39<br>1.52|.<br>1.27<br>1.29|.<br>1.03<br>1.06|.<br>1.52<br>1.65|.<br>1.43<br>1.54|
|Standard Latvian<br>|2.10<br>|2.54<br>|2.54<br>|2.54<br>|2.54<br>|2.35<br>|2.54<br>|1.76<br>|1.68|1.56|1.05|1.23<br>|1.29<br>|
|Magahi<br>Maithili|4.49<br>4.63|7.22<br>7.43|7.22<br>7.43|7.22<br>7.43|7.22<br>7.43|4.70<br>4.90|7.22<br>7.43|8.07<br>8.27|—<br>—|—<br>—|—<br>—|1.41<br>1.58|1.50<br>1.64|
|Malaalam|554|1524|1524|1524|1524|900|1524|1016|—|—|—|138|159|
|y<br>Marathi<br>|.<br>4.58<br>|.<br>7.87<br>|.<br>7.87<br>|.<br>7.87<br>|.<br>7.87<br>|.<br>5.07<br>|.<br>7.87<br>|.<br>8.76<br>|—|—|—|.<br>1.22<br>|.<br>1.38<br>|
|Minangkabau (Arabic script)<br>Minangkabau (Latin script)<br>|4.32<br>1.77<br>|5.25<br>1.97<br>|5.25<br>1.97<br>|5.25<br>1.97<br>|5.25<br>1.97<br>|3.97<br>1.77<br>|5.25<br>1.97<br>|5.71<br>1.40<br>|—<br>1.39|—<br>1.25<br>|—<br>1.09|2.02<br>1.31<br>|1.99<br>1.25<br>|
|Macedonian<br>Maltese|1.84<br>2.16|5.46<br>2.69|5.46<br>2.69|5.46<br>2.69|5.46<br>2.69|2.77<br>2.41|5.46<br>2.69|3.48<br>1.80|—<br>1.72|2.58<br>1.57|—<br>1.03|1.17<br>1.96|1.24<br>1.87|
|Meitei (Bengali script)<br>M|5.84<br>|10.22<br>|10.22<br>|10.22<br>|10.22<br>|6.71<br>|10.22<br>|9.06<br>|—<br>|—<br>|—<br>|2.56<br>|2.59<br>|
|ossi<br>|2.12<br>|2.54<br>|2.54<br>|2.54<br>|2.54<br>|2.32<br>|2.54<br>|1.74<br>|1.51<br>|1.38<br>|0.85<br>|1.78<br>|1.66<br>|
|Maori<br>Burmese|2.18<br>8.37|2.45<br>16.89|2.45<br>16.89|2.45<br>16.89|2.45<br>16.89|2.35<br>11.70|2.45<br>16.89|1.77<br>11.26|1.69<br>—|1.47<br>—|1.05<br>—|1.86<br>1.72|1.74<br>2.21|
|Dutch<br>|1.46<br>|1.97<br>|1.97<br>|1.97<br>|1.97<br>|1.59<br>|1.97<br>|1.28<br>|1.40<br>|1.32<br>|1.13<br>|1.14<br>|1.18<br>|
|Norwegian Nynorsk|1.54|1.93|1.93|1.93|1.93|1.64|1.93|1.25|1.40|1.29|1.02|1.17|1.17|
|Norwegian Bokmål|1.50|1.86|1.86|1.86|1.86|1.56|1.86|1.23|1.37|1.27|1.01|1.07|1.10|
|Nli|449|759|759|759|759|479|759|837|—|—|—|113|128|
|epa<br>Northern Sotho|.<br>2.02|.<br>2.32|.<br>2.32|.<br>2.32|.<br>2.32|.<br>2.18|.<br>2.32|.<br>1.63|1.58|1.48|1.12|.<br>1.75|.<br>1.52|
|Nuer|2.83|4.23|4.23|4.23|4.23|4.00|4.23|2.79|—|—|—|2.62|2.44|
|Nyanja<br>|2.02<br>|2.26<br>|2.26<br>|2.26<br>|2.26<br>|2.08<br>|2.26<br>|1.57<br>|1.55<br>|1.42<br>|1.17<br>|1.59<br>|1.55<br>|
|Occitan|1.66|2.07|2.07|2.07|2.07|1.83|2.07|1.47|1.40|1.38|1.14|1.50|1.31|
|Odia|11.59|13.38|13.38|13.38|13.38|12.48|13.38|8.94|—|—|—|1.45|1.56|



25 

|Language|Bart50|5|nT5|T5|NINE|OOM|abicBERT|RIL|F-32|RT Japanese|amlessM4T|LB|en|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
||M|mT|Fla|By|CA|BL|Ar|Mu|UT|BE|Se|NL|Qw|
|Hebrew|1.12|1.22|—|1.39|0.78|2.92|1.72|—|0.78|—|1.24|1.24|1.48|
|Hindi<br>Chhttirhi|1.25<br>141|1.59<br>160|—<br>—|2.55<br>246|1.00<br>097|1.28<br>144|—<br>—|1.16<br>134|1.00<br>097|—<br>—|1.22<br>126|1.22<br>126|4.47<br>426|
|asga<br>Croatian|.<br>1.10|.<br>1.30|2.43|.<br>1.01|.<br>0.98|.<br>1.80|1.36|.<br>—|.<br>0.98|1.27|.<br>1.17|.<br>1.17|.<br>1.83|
|Hungarian<br>Ai|1.18<br>138|1.26<br>158|2.99<br>—|1.16<br>204|1.05<br>111|2.07<br>431|1.40<br>—|2.31<br>—|1.05<br>111|—<br>—|1.27<br>151|1.27<br>151|2.12<br>534|
|rmenan<br>Igbo|.<br>2.12|.<br>1.79|3.17|.<br>1.21|.<br>1.02|.<br>1.72|1.50|—|.<br>1.02|—|.<br>1.32|.<br>1.32|.<br>2.37|
|Ilocano|1.61|1.61|2.82|1.21|1.21|1.90|1.55|2.01|1.21|1.55|1.33|1.33|2.03|
|Indonesian<br>|0.94<br>|1.08<br>|2.24<br>|1.08<br>|1.08<br>|0.96<br>|1.35<br>|1.74|1.08<br>|1.33|0.93<br>|0.93<br>|1.54<br>|
|Icelandic|1.23|1.32|2.81|1.09|0.99|1.99|1.34|—|0.99|—|1.29|1.29|2.11|
|Italian|1.19|1.34|2.18|1.19|1.18|1.62|1.41|1.92|1.18|1.37|1.25|1.25|1.62|
|Javanese<br>|1.15<br>|1.21<br>|2.21|1.04<br>|1.04<br>|1.40<br>|1.36<br>|1.74|1.04<br>|1.29<br>|1.03<br>|1.03<br>|1.72<br>|
|Japanese|1.11|0.90|—|1.27|0.44|1.81|1.01|—|0.44|0.67|1.01|1.01|1.46|
|Kabyle|1.84|1.82|2.83|1.06|0.99|2.02|1.29|—|0.99|—|1.56|1.56|2.14|
|Jingpho<br>|1.94<br>|1.79<br>|3.41<br>|1.27<br>|1.28<br>|2.14<br>|1.71<br>|2.32|1.28<br>|1.65|1.47<br>|1.47<br>|2.32<br>|
|Kamba|1.62|1.52|2.69|1.01|0.98|1.77|1.33|—|0.98|—|1.28|1.28|1.99|
|Kannada|1.36|1.44|—|2.83|1.05|1.31|—|1.06|1.05|—|1.37|1.37|6.98|
|Kashmiri (Arabic script)<br>|1.93<br>|2.00<br>|—|1.72<br>|0.96<br>|2.32<br>|1.26|1.75<br>|0.96<br>|—|1.81<br>|1.81<br>|3.48<br>|
|Kashmiri (Devanagari script)<br>|1.82<br>|1.79<br>|—|2.40<br>|0.96<br>|1.85<br>|—|1.75|0.96<br>|—|1.69<br>|1.69<br>|4.41<br>|
|Georgian<br>Kazakh|1.34<br>115|1.55<br>120|—<br>—|2.95<br>189|1.10<br>103|4.98<br>323|—<br>—|—<br>—|1.10<br>103|—<br>—|1.61<br>118|1.61<br>118|5.25<br>302|
|Kabiye<br>Kabuverdianu|.<br>2.98<br>1.35|.<br>2.83<br>1.28|—<br>2.21|.<br>1.37<br>1.02|.<br>1.09<br>0.99|.<br>3.34<br>1.51|—<br>1.25|—<br>1.81|.<br>1.09<br>0.99|—<br>1.29|.<br>1.56<br>1.28|.<br>1.56<br>1.28|.<br>3.35<br>1.70|
|Halh Mongolian<br>|1.21<br>|1.48<br>|—|1.91<br>|1.04<br>|3.38<br>|—|—|1.04<br>|—|1.36<br>|1.36<br>|3.10<br>|
|Khmer<br>Kikuyu|1.62<br>2.31|1.43<br>2.18|—<br>—|3.33<br>1.30|1.18<br>1.17|6.40<br>2.48|—<br>1.56|—<br>—|1.18<br>1.17|—<br>—|1.80<br>1.52|1.80<br>1.52|6.61<br>2.66|
|Kinyarwanda<br>K|1.72<br>11|1.51<br>12|2.76|1.13<br>1|1.11<br>12|1.58<br>2|1.54|2.15|1.11<br>12|1.50|1.30<br>12|1.30<br>12|2.12<br>274|
|yrgyz<br>Kimbundu|.6<br>1.64|.3<br>1.48|—<br>2.91|.88<br>1.11|.0<br>1.11|3.0<br>1.81|—<br>1.55|—<br>1.99|.0<br>1.11|—<br>1.52|.5<br>1.35|.5<br>1.35|.<br>2.10|
|Northern Kurdish|1.38|1.42|2.74|1.10|1.00|2.03|1.29|—|1.00|—|1.44|1.44|2.16|
|Central Kanuri (Arabic script)<br>|2.60<br>|2.43<br>|—<br>|1.60<br>|0.88<br>|2.10<br>|—|2.37|0.88<br>|—|2.54<br>|2.54<br>|3.15<br>|
|Central Kanuri (Latin script)<br>Kikongo|1.74<br>1.58|1.58<br>1.46|2.82<br>3.01|1.11<br>1.14|1.05<br>1.14|2.00<br>1.75|—<br>1.59|—<br>1.97|1.05<br>1.14|—<br>1.54|1.55<br>1.21|1.55<br>1.21|2.16<br>1.98|
|Korean<br>|1.16<br>|1.27<br>|—|1.20<br>|0.51<br>|2.79<br>|1.30|—|0.51<br>|—|1.03<br>|1.03<br>|1.64<br>|
|Lao|1.39|1.27|—|2.73|0.99|8.70|—|—|0.99|—|1.47|1.47|5.79|
|Ligurian|1.65|1.69|2.54|1.17|1.10|1.81|1.38|2.05|1.10|—|1.60|1.60|1.95|
|Limburish|145|138|225|107|104|175|132|192|104|128|144|144|178|
|g<br>Lingala|.<br>1.52|.<br>1.38|.<br>2.73|.<br>1.08|.<br>1.08|.<br>1.65|.<br>1.47|.<br>1.90|.<br>1.08|.<br>1.41|.<br>1.12|.<br>1.12|.<br>1.85|
|Lithuanian<br>Lombard|1.17<br>171|1.23<br>170|2.58<br>258|1.06<br>116|1.00<br>107|1.94<br>184|1.33<br>129|—<br>196|1.00<br>107|—<br>—|1.18<br>161|1.18<br>161|2.06<br>200|
|Latgalian|.<br>1.57|.<br>1.46|.<br>2.70|.<br>1.05|.<br>0.99|.<br>1.99|.<br>1.36|.<br>—|.<br>0.99|—|.<br>1.42|.<br>1.42|.<br>2.14|
|Luxembourgish|1.64|1.46|2.24|1.15|1.12|1.89|1.40|2.17|1.12|1.31|1.44|1.44|1.96|
|Luba-Kasai<br>Gd|1.54<br>155|1.37<br>140|2.48<br>265|1.08<br>103|1.08<br>102|1.68<br>167|1.44<br>146|1.89<br>194|1.08<br>102|1.41<br>141|1.21<br>126|1.21<br>126|1.92<br>194|
|ana<br>Luo|.<br>1.52|.<br>1.41|.<br>2.55|.<br>1.05|.<br>1.05|.<br>1.68|.<br>1.35|.<br>1.87|.<br>1.05|.<br>1.35|.<br>1.24|.<br>1.24|.<br>1.81|
|Mizo|1.65|1.57|2.76|1.10|1.10|1.83|1.43|1.92|1.10|1.37|1.31|1.31|1.94|
|Standard Latvian<br>|1.23<br>|1.30<br>|2.78|1.11<br>|1.02<br>|2.08<br>|1.35|—<br>|1.02<br>|—|1.20<br>|1.20<br>|2.29<br>|
|Magahi<br>Maithili|1.41<br>1.58|1.61<br>1.74|—<br>—|2.46<br>2.53|0.96<br>0.98|1.45<br>1.56|—<br>—|1.34<br>1.50|0.96<br>0.98|—<br>—|1.23<br>1.24|1.23<br>1.24|4.23<br>4.42|
|Mll|138|135|—|310|113|138|—|118|113|—|149|149|731|
|aayaam<br>Marathi|.<br>1.22|.<br>1.52|—|.<br>2.67|.<br>1.01|.<br>1.21|—|.<br>1.06|.<br>1.01|—|.<br>1.26|.<br>1.26|.<br>4.65|
|Minangkabau (Arabic script)<br>Mikb(Ltiit)|2.02<br>131|1.84<br>125|—<br>235|1.74<br>107|0.96<br>107|2.58<br>144|1.13<br>136|—<br>177|0.96<br>107|—<br>132|1.97<br>115|1.97<br>115|2.79<br>175|
|nangaau an scrp<br>|.<br>|.<br>|.|.<br>|.<br>|.<br>|.|.|.<br>|.|.<br>|.<br>|.<br>|
|Macedonian<br>Maltese|1.17<br>1.96|1.29<br>1.69|—<br>2.94|1.89<br>1.16|1.04<br>1.11|2.50<br>2.25|—<br>1.44|—<br>—|1.04<br>1.11|—<br>—|1.24<br>1.46|1.24<br>1.46|2.26<br>2.24|
|Meitei (Bengali script)<br>|2.56<br>|2.21<br>|—<br>|2.77<br>|1.03<br>|2.35<br>|—<br>|2.34|1.03<br>|—|1.73<br>|1.73<br>|5.64<br>|
|Mossi<br>Maori|1.78<br>1.86|1.80<br>1.69|2.90<br>3.28|1.03<br>1.16|0.96<br>1.11|1.99<br>2.12|1.19<br>1.49|—<br>2.12|0.96<br>1.11|—<br>1.45|1.36<br>1.38|1.36<br>1.38|2.06<br>2.33|
|Burmese|172|156|—|351|124|1005|—|—|124|—|159|159|899|
|Dutch|.<br>1.14|.<br>1.17|2.19|.<br>1.11|.<br>1.11|.<br>1.71|1.38|1.91|.<br>1.11|1.33|.<br>1.19|.<br>1.19|.<br>1.58|
|Norwegian Nynorsk|1.17|1.18|2.29|1.04|1.01|1.65|1.28|1.82|1.01|1.22|1.16|1.16|1.63|
|NorwegianBokmål|107|112|224|103|101|162|126|179|101|118|110|110|155|
|<br>Nepali<br>|.<br>1.13<br>|.<br>1.47<br>|.<br>—<br>|.<br>2.56<br>|.<br>0.96<br>|.<br>1.17<br>|.<br>—<br>|.<br>1.01<br>|.<br>0.96<br>|.<br>—<br>|.<br>1.18<br>|.<br>1.18<br>|.<br>4.45<br>|
|Northern Sotho|1.75|1.57|2.81|1.17|1.15|1.94|1.48|2.18|1.15|1.48|1.35|1.35|2.17|
|Nuer<br>|2.62<br>|2.42<br>|—<br>|1.32<br>|1.08<br>|2.79<br>|—<br>|—<br>|1.08<br>|—<br>|1.89<br>|1.89<br>|3.39<br>|
|Nyanja<br>Occitan|1.59<br>1.50|1.35<br>1.48|2.71<br>2.26|1.12<br>1.17|1.12<br>1.14|1.78<br>1.49|1.52<br>1.33|2.02<br>1.93|1.12<br>1.14|1.44<br>1.33|1.15<br>1.40|1.15<br>1.40|2.06<br>1.81|
|Odia|1.45|3.11|—|2.73|1.03|1.36|—|1.21|1.03|—|1.38|1.38|9.79|



26 

|Language|A||base|base|edit|_base|RTa|ERT|mBERT|RT|ert|RoBERTa|00|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
||LLAM|GPT-2|r50k_|p50k_|p50k_|cl100k|RoBE|GottB|Came|PhoBE|RoCB|XLM-|M2M1|
|Pangasinan|1.50|1.66|1.66|1.66|1.66|1.57|1.66|1.27|1.25|1.11|1.00|1.29|1.23|
|Eastern Panjabi<br>|9.44<br>|7.90<br>|7.90<br>|7.90<br>|7.90<br>|7.87<br>|7.90<br>|8.47<br>|—<br>|—<br>|—<br>|1.57<br>|1.68<br>|
|Papiamento<br>Southern Pashto|1.65<br>4.27|1.98<br>5.39|1.98<br>5.39|1.98<br>5.39|1.98<br>5.39|1.75<br>3.83|1.98<br>5.39|1.33<br>5.37|1.37<br>—|1.25<br>—|1.03<br>—|1.37<br>1.38|1.32<br>1.40|
|Western Persian|3.98|5.32|5.32|5.32|5.32|3.28|5.32|5.47|—|—|—|1.10|1.17|
|Plateau Malagasy<br>|2.12<br>|2.58<br>|2.58<br>|2.58<br>|2.58<br>|2.26<br>|2.58<br>|1.74<br>|1.69<br>|1.49<br>|1.26<br>|1.57<br>|1.49<br>|
|Polish|1.70|2.69|2.69|2.69|2.69|1.91|2.69|1.79|1.71|1.58|1.00|1.19|1.26|
|Portuguese|1.42|1.94|1.94|1.94|1.94|1.48|1.94|1.38|1.36|1.30|1.09|1.11|1.14|
|Dari<br>|3.88<br>|5.11<br>|5.11<br>|5.11<br>|5.11<br>|3.16<br>|5.11<br>|5.31<br>|—<br>|—<br>|—<br>|1.09<br>|1.15<br>|
|Ayacucho Quechua|1.96|2.20|2.20|2.20|2.20|2.08|2.20|1.61|1.54|1.40|1.14|1.59|1.54|
|Romanian<br>Rundi|1.70<br>205|2.48<br>233|2.48<br>233|2.48<br>233|2.48<br>233|1.88<br>213|2.48<br>233|1.69<br>163|1.54<br>159|1.46<br>147|1.13<br>115|1.24<br>171|1.29<br>163|
|Russian|.<br>1.64|.<br>5.74|.<br>5.74|.<br>5.74|.<br>5.74|.<br>2.49|.<br>5.74|.<br>3.67|.<br>—|.<br>2.71|.<br>1.03|.<br>1.17|.<br>1.22|
|Sango|1.95|2.23|2.23|2.23|2.23|2.08|2.23|1.54|1.50|1.32|1.02|1.66|1.53|
|Sanskrit<br>|4.59<br>|7.94<br>|7.94<br>|7.94<br>|7.94<br>|5.00<br>|7.94<br>|8.60<br>|—|—|—|1.43|1.69|
|Santali<br>Sicilian|11.92<br>1.81|12.86<br>2.27|12.86<br>2.27|12.86<br>2.27|12.86<br>2.27|12.80<br>2.01|12.86<br>2.27|8.56<br>1.57|—<br>1.43|—<br>1.37|—<br>1.06|—<br>1.58|—<br>1.53|
|Shan<br>|11.85<br>|18.76<br>|18.76<br>|18.76<br>|18.76<br>|15.05<br>|18.76<br>|12.51<br>|—|—|—|4.43<br>|4.63<br>|
|Sinhala<br>|7.86<br>|12.86<br>|12.86<br>|12.86<br>|12.86<br>|8.83<br>|12.86<br>|8.59<br>|—<br>|—<br>|—<br>|1.35<br>|1.53<br>|
|Slovak|1.82|2.52|2.52|2.52|2.52|2.14|2.52|1.65|1.60|1.46|1.02|1.18|1.24|
|Slovenian|1.67|2.11|2.11|2.11|2.11|1.88|2.11|1.46|1.44|1.32|1.01|1.13|1.19|
|Samoan<br>|2.14<br>|2.57<br>|2.57<br>|2.57<br>|2.57<br>|2.29<br>|2.57<br>|1.69<br>|1.63<br>|1.50<br>|1.09<br>|1.92<br>|1.80<br>|
|Shona|2.01|2.29|2.29|2.29|2.29|2.13|2.29|1.58|1.58|1.44|1.18|1.63|1.58|
|Sindhi|4.20|5.00|5.00|5.00|5.00|4.00|5.00|5.22|—|—|—|1.28|1.30|
|Sli|214|236|236|236|236|218|236|166|169|148|116|139|137|
|oma<br>Southern Sotho|.<br>2.07|.<br>2.34|.<br>2.34|.<br>2.34|.<br>2.34|.<br>2.21|.<br>2.34|.<br>1.64|.<br>1.63|.<br>1.48|.<br>1.18|.<br>1.78|.<br>1.60|
|Spanish|1.45|1.99|1.99|1.99|1.99|1.55|1.99|1.45|1.44|1.36|1.19|1.20|1.21|
|Sardinian<br>|1.82<br>|2.26<br>|2.26<br>|2.26<br>|2.26<br>|1.99<br>|2.26<br>|1.53<br>|1.48|1.40<br>|1.16|1.61<br>|1.51<br>|
|Serbian|1.73|5.34|5.34|5.34|5.34|2.92|5.34|3.41|—|2.45|—|1.18|1.26|
|Swati|2.03|2.31|2.31|2.31|2.31|2.16|2.31|1.59|1.60|1.45|1.21|1.61|1.44|
|Sundanese|176|202|202|202|202|182|202|139|139|124|107|122|110|
|Swedish|.<br>1.44|.<br>1.95|.<br>1.95|.<br>1.95|.<br>1.95|.<br>1.58|.<br>1.95|.<br>1.22|.<br>1.41|.<br>1.31|.<br>1.02|.<br>1.07|.<br>1.10|
|Swahili|1.86|2.13|2.13|2.13|2.13|1.95|2.13|1.49|1.42|1.32|1.06|1.16|1.20|
|Silesian|195|260|260|260|260|218|260|174|170|159|099|165|159|
|Tamil<br>|.<br>5.87<br>|.<br>15.58<br>|.<br>15.58<br>|.<br>15.58<br>|.<br>15.58<br>|.<br>7.65<br>|.<br>15.58<br>|.<br>10.38<br>|.<br>—<br>|.<br>—<br>|.<br>—|.<br>1.35<br>|.<br>1.55<br>|
|Tamasheq (Latin script)|1.93|2.39|2.39|2.39|2.39|2.22|2.39|1.62|1.50|1.29|—|1.71|1.57|
|Tamasheq(Tifinaghscript)|842|1043|1043|1043|1043|1013|1043|695|—|—|—|—|—|
|i <br>Tatar<br>Telugu|.<br>2.53<br>10.71|.<br>5.82<br>13.09|.<br>5.82<br>13.09|.<br>5.82<br>13.09|.<br>5.82<br>13.09|.<br>3.75<br>8.34|.<br>5.82<br>13.09|.<br>3.84<br>8.73|—<br>—|—<br>—|—<br>—|1.81<br>1.33|1.54<br>—|
|Tajik|2.70|6.09|6.09|6.09|6.09|3.64|6.09|4.00|—|2.82|—|2.14|2.06|
|Tagalog<br>|2.00<br>|2.28<br>|2.28<br>|2.28<br>|2.28<br>|2.06<br>|2.28<br>|1.63<br>|1.67|1.45<br>|1.27|1.43<br>|1.43<br>|
|Thai|4.35|9.05|9.05|9.05|9.05|4.39|9.05|6.59|—|2.83|—|1.08|1.27|
|Tigrinya|7.47|7.88|7.88|7.88|7.88|7.80|7.88|5.25|—|—|—|1.97|1.91|
|Tok Pisin<br>|1.95<br>|2.21<br>|2.21<br>|2.21<br>|2.21<br>|2.04<br>|2.21<br>|1.55<br>|1.66<br>|1.45<br>|1.25<br>|1.73<br>|1.65<br>|
|Tswana|2.12|2.39|2.39|2.39|2.39|2.28|2.39|1.68|1.67|1.55|1.21|1.85|1.68|
|Tsonga|2.16|2.45|2.45|2.45|2.45|2.26|2.45|1.70|1.70|1.46|1.19|1.79|1.69|
|Turkmen<br>|2.23<br>|2.82<br>|2.82<br>|2.82<br>|2.82<br>|2.40<br>|2.82<br>|1.76<br>|1.78<br>|1.62<br>|1.11<br>|1.78<br>|1.71<br>|
|Tumbuka|2.46|2.78|2.78|2.78|2.78|2.57|2.78|1.93|1.85|1.67|1.34|1.92|1.88|
|Turkish|2.09|2.43|2.43|2.43|2.43|1.91|2.43|1.61|1.65|1.51|—|1.04|1.15|
|Twi|201|262|262|262|262|251|262|180|157|138|—|188|174|
|Central Atlas Tamazight<br>|.<br>8.86|.<br>10.39|.<br>10.39|.<br>10.39|.<br>10.39|.<br>10.04|.<br>10.39|.<br>6.92|.<br>—|.<br>—|—|.<br>—|.<br>—|
|Uyghur|4.89|7.16|7.16|7.16|7.16|5.19|7.16|6.44|—|—|—|1.41|3.00|
|Ukrainian<br>Umbundu<br>|1.72<br>1.89<br>|5.75<br>2.24|5.75<br>2.24|5.75<br>2.24|5.75<br>2.24|3.00<br>2.01<br>|5.75<br>2.24|3.69<br>1.53<br>|—<br>1.48|2.58<br>1.36|—<br>1.05|1.21<br>1.57|1.28<br>1.49|
|Urdu<br>NorthernUzbek|4.37<br>203|6.30<br>230|6.30<br>230|6.30<br>230|6.30<br>230|4.39<br>217|6.30<br>230|5.74<br>163|—<br>159|—<br>148|—<br>119|1.23<br>133|1.30<br>137|
|<br>Vi|.<br>156|.<br>200|.<br>200|.<br>200|.<br>200|.<br>170|.<br>200|.<br>138|.<br>134|.<br>123|.|.<br>136|.<br>131|
|enetan<br>|.<br>|.<br>|.<br>|.<br>|.<br>|.<br>|.<br>|.<br>|.|.<br>|—<br>|.<br>|.<br>|
|Vietnamese|2.92|4.54|4.54|4.54|4.54|2.45|4.54|3.06|—|0.83|0.98|1.18|1.15|
|Waray|2.02|2.38|2.38|2.38|2.38|1.95|2.38|1.61|1.66|1.42|1.25|1.55|1.45|
|Wolof<br>|1.80<br>|2.14<br>|2.14<br>|2.14<br>|2.14<br>|1.92<br>|2.14<br>|1.49<br>|1.43<br>|1.28<br>|0.93<br>|1.60<br>|1.40<br>|
|Xhosa|1.97|2.26|2.26|2.26|2.26|2.06|2.26|1.57|1.57|1.40|1.13|1.50|1.37|
|Eastern Yiddish<br>Yrb|4.57<br>270|6.63<br>389|6.63<br>389|6.63<br>389|6.63<br>389|5.57<br>296|6.63<br>389|6.34<br>263|—<br>—|—<br>166|—<br>088|1.58<br>227|1.61<br>174|
|oua<br>Yue Chinese<br>i|.<br>2.11|.<br>3.09|.<br>3.09|.<br>3.09|.<br>3.09|.<br>2.12|.<br>3.09|.<br>2.78|—|.<br>—|.<br>0.36|.<br>0.93|.<br>1.03|
|Chinese (Simplified)|2.00|3.21|3.21|3.21|3.21|1.91|3.21|2.93|—|—|0.39|0.97|1.05|
|i<br>Chinese (Traditional)<br>|2.16<br>|3.16<br>|3.16<br>|3.16<br>|3.16<br>|2.18<br>|3.16<br>|2.83<br>|—<br>|—<br>|0.36<br>|0.96<br>|1.06<br>|
|Standard Malay|1.83|2.05|2.05|2.05|2.05|1.62|2.05|1.42|1.45|1.28|1.15|0.95|1.00|
|Zulu|2.09|2.41|2.41|2.41|2.41|2.20|2.41|1.65|1.64|1.47|1.20|1.55|1.35|



27 

|Language|50||5||NE|M|BERT|L|2|Japanese|ssM4T|||
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
||art|5|nT|T5|NI|OO|abic|RI|F-3|RT|mle|LB|en|
||MB|mT|Fla|By|CA|BL|Ar|Mu|UT|BE|Sea|NL|Qw|
|Pangasinan|1.29|1.22|2.18|1.00|1.00|1.45|1.24|1.54|1.00|1.21|1.11|1.11|1.56|
|Eastern Panjabi<br>Pit|1.57<br>137|2.11<br>136|—<br>228|2.59<br>108|1.01<br>105|1.43<br>154|—<br>125|1.35<br>180|1.01<br>105|—<br>130|1.50<br>127|1.50<br>127|7.30<br>173|
|apameno<br>Southern Pashto|.<br>1.38|.<br>1.64|.<br>—|.<br>1.66|.<br>0.95|.<br>2.55|.<br>—|.<br>—|.<br>0.95|.<br>—|.<br>1.45|.<br>1.45|.<br>2.87|
|WesternPersian|1.10|1.34|—|1.70|0.94|1.78|1.11|1.62|0.94|—|1.13|1.13|2.60|
|<br>Plateau Malagasy<br>Polish|1.57<br>1.19|1.59<br>1.31|3.00<br>2.82|1.26<br>1.13|1.22<br>1.06|2.07<br>2.14|1.64<br>1.52|2.33<br>—|1.22<br>1.06|1.59<br>—|1.39<br>1.37|1.39<br>1.37|2.23<br>1.76|
|Portuguese|1.11|1.29|2.21|1.12|1.09|1.12|1.30|1.88|1.09|1.24|1.17|1.17|1.45|
|Di|109|131|—|163|092|164|109|158|092|—|111|111|250|
|ar<br>Ayacucho Quechua<br>Romanian|.<br>1.59<br>1.24|.<br>1.42<br>1.37|2.59<br>1.50|.<br>1.08<br>1.19|.<br>1.07<br>1.13|.<br>1.83<br>1.91|.<br>1.47<br>1.33|.<br>1.95<br>—|.<br>1.07<br>1.13|1.42<br>—|.<br>1.28<br>1.35|.<br>1.28<br>1.35|.<br>2.06<br>1.86|
|Rundi<br>|1.71<br>|1.52<br>|2.78|1.12<br>|1.12<br>|1.64<br>|1.54<br>|2.13|1.12<br>|1.50|1.33<br>|1.33<br>|2.11<br>|
|Russian|1.17|1.27|—|1.98|1.09|2.48|2.50|—|1.09|—|1.34|1.34|1.75|
|Sango|1.66|1.63|3.14|1.12|1.09|1.80|1.45|2.05|1.09|1.49|1.39|1.39|2.04|
|Sanskrit<br>|1.43|1.65|—|2.63<br>|0.98<br>|1.63<br>|—|1.21|0.98<br>|—|1.40<br>|1.40<br>|4.58<br>|
|Santali<br>Sicilian|—<br>1.58|—<br>1.53|—<br>2.46|2.79<br>1.11|1.06<br>1.05|12.71<br>1.80|—<br>1.41|—<br>1.84|1.06<br>1.05|—<br>—|2.49<br>1.44|2.49<br>1.44|8.99<br>1.95|
|Shan<br>|4.43<br>|3.28<br>|—|3.94<br>|1.42<br>|12.06<br>|—|—|1.42<br>|—|1.94<br>|1.94<br>|10.51<br>|
|Sinhala<br>Slovak|1.35<br>1.18|1.66<br>1.30|—<br>2.74|2.64<br>1.09|1.00<br>1.00|8.21<br>2.01|—<br>1.35|—<br>—|1.00<br>1.00|—<br>—|1.68<br>1.21|1.68<br>1.21|7.02<br>2.08|
|Slovenian|113|120|242|102|100|181|137|—|100|130|117|117|187|
|Samoan<br>|.<br>1.92<br>|.<br>1.92<br>|.<br>3.09<br>|.<br>1.22<br>|.<br>1.16<br>|.<br>2.13<br>|.<br>1.57<br>|2.22<br>|.<br>1.16<br>|.<br>1.55<br>|.<br>1.60<br>|.<br>1.60<br>|.<br>2.26<br>|
|Shona|1.63|1.35|2.79|1.12|1.12|1.80|1.55|2.06|1.12|1.48|1.23|1.23|2.11|
|Sindhi<br>Sli|1.28<br>1|1.74<br>14|—<br>|1.60<br>114|0.91<br>114|2.51<br>2|—<br>12|1.22<br>2|0.91<br>114|—<br>12|1.33<br>1|1.33<br>1|2.87<br>21|
|oma<br>Southern Sotho|.39<br>1.78|.8<br>1.59|3.06<br>2.92|.<br>1.21|.<br>1.20|.03<br>1.96|.5<br>1.61|.05<br>2.16|.<br>1.20|.5<br>1.54|.39<br>1.39|.39<br>1.39|.6<br>2.19|
|Spanish|1.20|1.31|2.23|1.21|1.19|1.21|1.38|1.98|1.19|1.41|1.24|1.24|1.52|
|Sardinian<br>Serbian|1.61<br>1.18|1.57<br>1.30|2.46<br>—|1.19<br>1.80|1.16<br>0.99|1.73<br>2.57|1.38<br>—|1.98<br>—|1.16<br>0.99|1.36<br>—|1.44<br>1.24|1.44<br>1.24|1.97<br>2.34|
|Swati<br>Sundanese|1.61<br>122|1.41<br>122|2.80<br>232|1.12<br>105|1.13<br>104|1.83<br>148|1.55<br>133|2.09<br>180|1.13<br>104|1.52<br>131|1.28<br>104|1.28<br>104|2.14<br>180|
|Swedish|.<br>1.07|.<br>1.11|.<br>2.22|.<br>1.04|.<br>1.01|.<br>1.65|.<br>1.21|.<br>1.90|.<br>1.01|.<br>1.20|.<br>1.13|.<br>1.13|.<br>1.57|
|Swahili|1.16|1.25|2.66|1.05|1.05|1.24|1.45|1.86|1.05|1.43|1.13|1.13|1.93|
|Silesian<br>|1.65<br>|1.57<br>|2.87|1.10<br>|1.04<br>|2.16<br>|1.52|—<br>|1.04<br>|—|1.52<br>|1.52<br>|2.09<br>|
|Tamil|1.35|1.26|—|3.17|1.17|1.27|—|1.06|1.17|—|1.42|1.42|6.15|
|Tamasheq (Latin script)|1.71|1.64|2.55|1.01|0.95|1.90|—|—|0.95|—|1.52|1.52|1.99|
|Tamashe(Tifinahscrit)|—|359|—|229|094|774|—|—|094|—|243|243|537|
|q ig p<br>Tatar<br>|1.81|.<br>1.41|—|.<br>1.85|.<br>1.01|.<br>3.15|—|—|.<br>1.01|—|.<br>1.21|.<br>1.21|.<br>2.88|
|Telugu|1.33|1.42|—|2.68|1.01|1.33|—|1.21|1.01|—|1.34|1.34|7.06|
|Tajik<br>|2.14<br>|1.62<br>|—<br>|2.01<br>|1.11<br>|3.29<br>|2.39<br>|—<br>|1.11<br>|—<br>|1.57<br>|1.57<br>|2.90<br>|
|Tagalog<br>Thai|1.43<br>1.08|1.46<br>0.99|2.85<br>—|1.26<br>2.75|1.26<br>0.96|1.85<br>4.63|1.56<br>—|2.08<br>—|1.26<br>0.96|1.60<br>—|1.34<br>1.52|1.34<br>1.52|2.04<br>2.59|
|Tigrinya|197|203|—|175|069|516|—|—|069|—|144|144|424|
|Tok Pisin<br>|.<br>1.73<br>|.<br>1.65<br>|2.76<br>|.<br>1.28<br>|.<br>1.28<br>|.<br>1.92<br>|1.61<br>|2.10<br>|.<br>1.28<br>|1.57<br>|.<br>1.39<br>|.<br>1.39<br>|.<br>2.02<br>|
|Tswana<br>Tsonga|1.85<br>1.79|1.68<br>1.61|3.01<br>3.13|1.25<br>1.20|1.25<br>1.20|2.02<br>2.01|1.62<br>1.65|2.25<br>2.19|1.25<br>1.20|1.57<br>1.64|1.45<br>1.30|1.45<br>1.30|2.26<br>2.23|
|Turkmen<br>|1.78<br>|1.68<br>|2.87<br>|1.17<br>|1.06<br>|2.19<br>|1.44<br>|—|1.06<br>|—|1.36<br>|1.36<br>|2.20<br>|
|Tumbuka|1.92|1.61|3.29|1.32|1.30|2.19|1.79|—|1.30|—|1.43|1.43|2.51|
|Turkish|1.04|1.12|2.67|1.12|1.03|1.96|1.45|—|1.03|—|1.14|1.14|1.61|
|Ti|188|171|285|105|098|181|||098|140|125|125|215|
|w<br>Central Atlas Tamazight|.<br>—|.<br>3.48|.<br>—|.<br>2.28|.<br>0.89|.<br>7.69|—<br>—|—<br>—|.<br>0.89|.<br>—|.<br>2.06|.<br>2.06|.<br>5.09|
|Uyghur|1.41|2.57|—|1.97|1.07|3.67|—|—|1.07|—|1.40|1.40|3.74|
|Ukrinin|121|133|—|186|102|275|235|—|102|—|128|128|251|
|aa<br>Umbundu|.<br>1.57|.<br>1.47|2.72|.<br>1.05|.<br>1.01|.<br>1.74|.<br>1.46|1.94|.<br>1.01|1.33|.<br>1.29|.<br>1.29|.<br>1.97|
|Urdu|1.23|1.52|—|1.76|0.99|1.36|1.45|1.26|0.99|—|1.30|1.30|3.19|
|NorthernUzbek|133|138|280|113|113|198|158|212|113|153|132|132|215|
|<br>Venetian|.<br>1.36|.<br>1.36|.<br>2.21|.<br>1.06|.<br>1.01|.<br>1.57|.<br>1.24|.<br>1.84|.<br>1.01|.<br>1.23|.<br>1.29|.<br>1.29|.<br>1.68|
|Vietnamese|1.18|1.95|—|1.39|1.05|1.27|1.38|—|1.05|—|1.18|1.18|1.41|
|Wara|155|145|266|125|125|180|160|215|125|152|136|136|193|
|y<br>Wolof<br>Xhosa|.<br>1.60<br>1.50|.<br>1.44<br>1.35|.<br>2.62<br>2.73|.<br>1.00<br>1.06|.<br>0.96<br>1.06|.<br>1.68<br>1.67|.<br>1.28<br>1.52|.<br>1.93<br>2.05|.<br>0.96<br>1.06|.<br>1.26<br>1.45|.<br>1.31<br>1.21|.<br>1.31<br>1.21|.<br>1.89<br>2.04|
|Eastern Yiddish<br>|1.58<br>|1.66<br>|—|1.94<br>|1.08<br>|4.42<br>|2.41<br>|—|1.08<br>|—|1.69<br>|1.69<br>|2.77<br>|
|Yoruba<br>|2.27<br>|2.06<br>|—|1.28<br>|0.97<br>|1.64<br>|1.24|—|0.97<br>|—<br>|1.52<br>|1.52<br>|2.69<br>|
|Yue Chinese|0.93|0.95|—|0.87|0.31|0.93|—|—|0.31|0.55|1.05|1.05|1.17|
|Chinese(Simplified)|097|092|—|093|034|095|—|—|034|055|111|111|107|
|i<br>Chinese (Traditional)<br>|.<br>0.96<br>|.<br>0.98<br>|—<br>|.<br>0.89<br>|.<br>0.32<br>|.<br>0.97<br>|—<br>|—<br>|.<br>0.32<br>|.<br>0.57<br>|.<br>1.08<br>|.<br>1.08<br>|.<br>1.21<br>|
|Standard Malay|0.95|1.11|2.32|1.12|1.11|1.07|1.39|1.80|1.11|1.36|0.96|0.96|1.61|
|Zulu|1.55|1.40|2.84|1.12|1.12|1.76|1.62|2.15|1.12|1.54|1.24|1.24|2.18|



28 

