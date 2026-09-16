# Comparing the Level of Code-Switching in Corpora

Björn Gambäck<sup>∗</sup>, Amitava Das<sup>†</sup>

<sup>∗</sup>Department of Computer and Information Science

Norwegian University of Science and Technology

NO–7491 Trondheim, Norway

gamback@idi.ntnu.no

<sup>†</sup>Indian Institute of Information Technology

Sri City, Satyavedu Mandal, Chittoor District

Andhra Pradesh – 517588, India

amitava.das@iiits.in

## Abstract

Social media texts are often fairly informal and conversational, and when produced by bilinguals tend to be written in several different languages simultaneously, in the same way as conversational speech. The recent availability of large social media corpora has thus also made large-scale code-switched resources available for research. The paper addresses the issues of evaluation and comparison these new corpora entail, by defining an objective measure of corpus level complexity of code-switched texts. It is also shown how this formal measure can be used in practice, by applying it to several code-switched corpora.

Keywords: Code-switching, Evaluation, Corpora, Social Media Text

## 1. Introduction

When two individuals who are bi- or multi-lingual in an overlapping set of languages communicate, they tend to switch seemlessly and effortlessly between the languages (codes) they share. When this code alternation occurs at or above the utterance level, the phenomenon is referred to as code-switching; when the alternation is utteranceinternal, the term ‘code-mixing’ is common, even though ‘code-switching’ is frequently used in those cases as well. Code-mixing in itself is often an effect of what recently (particularly within language teaching) has started to be called ‘translanguaging’, that is, when truly bi-lingual individuals are creating new meanings based on their full and double language repertoire (Lewis et al., 2012).

Code-switching is most prominent in spoken language conversations and has thus traditionally mainly been studied by psycho- and sociolinguists (Auer, 1999; Muysken, 2000; Gafaranga and Torras, 2002; Bullock et al., 2014) and by speech researchers (Lyu et al., 2015), while the lack of large-scale textual corpora has made code-switching less attractive as a subject of study in computational or corpora linguistics. However, this started changing in 2003 with the advent of social media, where large amounts of texts are written that are more informal and more conversational in nature, and hence when produced by bilinguals tend to contain more code-switching (Paolillo, 1996).

This new availability of large-scale code-switched resources in turn raises questions of evaluation: how do we compare the results of applying language processing tools to one code-switched corpus to those on another? Or more specifically: how can we compare the level of codeswitching in corpora? And for corpora containing a mix of a specific set of languages or across corpora from different sets of languages? These are the issues that the present paper aims to address.

That two texts come from social media does not in itself imply that they belong to one, delimited textual domain. Rather, there is a wide spectrum of different types of texts that are transmitted through social media, and the level of formality of the language in addition depends more on the style of the writer than on the actual media (Eisenstein, 2013; Androutsopoulos, 2011). They both argue that the common denominator of social media text is not that it is ‘noisy’ and informal per se, but that it describes language in (rapid) change. Furthermore, although social media often convey more ungrammatical text than more formal writings, Baldwin et al. (2013) have shown that the relative occurrence of non-standard syntax is fairly constant among many types of media, such as mails, tweets, forums, comments, and blogs.

Due to the ease of availability of Twitter, most research on social media text has so far focused on tweets (Twitter messages). Lui and Baldwin (2014) note that users that mix languages in their writing still tend to avoid code-switching inside a specific tweet, a fact that has been utilized to investigate which language is dominant in a tweet (Carter, 2012; Lignos and Marcus, 2013; Voss et al., 2014). However, tweets still tend to be somewhat formal by more often following grammatical norms and using standard lexical items (Hu et al., 2013), while chats are more conversational (Paolillo, 1999), and hence less formal, which tend to increase their level of code-switching (Cárdenas-Claros and Isharyanti, 2009; Paolillo, 2011; Nguyen and Dogruöz,˘ 2013; Das and Gambäck, 2014).

The paper is organized as follows: Section 2. describes a formal measure that can be used to compare the complexity of code-switched corpora. Section 3. then uses this corpus level switching measure in practise, applying it to a set of recently produced code-switched corpora. Finally, Section 4. sums up and elaborates on the results.

## 2. Measuring Code-Switching in Corpora

When comparing different code-switched corpora to each other, it is desirable to have a measurement of the level of mixing between languages, in particular since error rates for various language processing application would be expected to increase as the level of code-switching increases. Both Kilgarriff (2001) and Pinto et al. (2011) discussed several statistical measures that can be used to compare corpora more objectively, but those measures presume that the corpora are essentially monolingual.

Debole and Sebastiani (2005) analysed the complexity of the different subsets of the Reuters-21578 corpus in terms of the relative hardness of learning classifiers on the subcorpora, a strategy which does not assume monolinguality in the corpora. However, they were only interested in the relative difficulty and give no measure of the complexity as such. In Gambäck and Das (2014) we instead suggested an initial Code-Mixing Index to assess the level of codeswitching in an utterance. This measure will be taken as the starting point, and elaborated on here.

## 2.1. Utterance Level Switching

If an utterance x only contains language independent tokens, its code-mixing is zero; for other utterances, the level of mixing depends on the fraction of language dependent tokens that belong to the matrix language (the most frequent language in the utterance) and on $N _ { \ast }$ the number of tokens in x except the language independent ones (i.e., all tokens that belong to any language $L _ { i } ) \colon ^ { 1 }$

$$
C _ {u} (x) = \left\{ \begin{array}{l l} \frac {N (x) - \max _ {L _ {i} \in \mathbb {L}} \left\{t _ {L _ {i}} \right\} (x)}{N (x)} & : N (x) > 0 \\ 0 & : N (x) = 0 \end{array} \right. \tag {1}
$$

$( L _ { i } { \in } \mathbb { L }$ , the set of all languages in the corpus; $1 \leq$ max $\{ t _ { L _ { i } } \} ~ \leq ~ N )$ . Notably, for mono-lingual utterances $C _ { u } = 0$ (since then max $\{ t _ { L _ { i } } \} = N ) . ^ { 2 }$

This initial measure has several short-comings. In particular, it does not reflect what fraction of a corpus’ utterances contain code-switching, nor take into account the number of code alternation points: arguably, a higher number of language switches in an utterance increases its complexity, while a corpus with a larger fraction of mixed utterances is (on average) more complex.<sup>3</sup>

Two main sources of information will be utilized to fully account for the code alternation at utterance level: the ratio of tokens belonging to the matrix language $( f _ { m } = [ N -$ max $\{ t _ { L _ { i } } \} ] / N$ as in Equation 1) and the number of code alternation points per token $( f _ { p } = P / N$ , where $P$ is the number of code alternation points; $0 \leq P < N )$ .

There are many ways to combine two (or several) information sources, in particular if they are independent; see, e.g., Genest and McConway (1990) for an overview. However, $P$ partially depends on max $\{ t _ { L _ { i } } \} , ^ { 4 }$ which, for example, rules out the common logarithmic opinion poll:

$$
p (x) = \prod_ {k = 1} ^ {n} p _ {k} (x) ^ {w _ {k}} \quad : \sum_ {k} w _ {k} = 1 \tag {2}
$$

Instead we will use the linear opinion poll:

$$
p (x) = \sum_ {k = 1} ^ {n} w _ {k} \times p _ {k} (x) \quad : \sum_ {k} w _ {k} = 1 \tag {3}
$$

Combining $f _ { m } ( x )$ and $f _ { p } ( x )$ gives a revised utterance level measure for $N ( x ) > 0 { : }$

$$
C _ {u} (x) = w _ {m} f _ {m} (x) + w _ {p} f _ {p} (x) \tag {4}
$$

$$
\begin{array}{l} = w _ {m} \frac {N (x) - \max _ {L _ {i} \in \mathbb {L}} \left\{t _ {L _ {i}} \right\} (x)}{N (x)} \cdot 1 0 0 + w _ {p} \frac {P (x)}{N (x)} \cdot 1 0 0 \\ = 1 0 0 \cdot \frac {w _ {m} \Big (N (x) - \max _ {L _ {i} \in \mathbb {L}} \{t _ {L _ {i}} \} (x) \Big) + w _ {p} P (x)}{N (x)} \\ \end{array}
$$

where $w _ { m }$ and $w _ { p }$ are weights $\begin{array} { r c l } { ( w _ { m } \ + \ w _ { p } } & { = } & { 1 ) } \end{array}$ Again, $C _ { u } ~ = ~ 0$ for mono-lingual utterances (since then max $\{ t _ { L _ { i } } \} = N$ and $P = 0 )$ .

## 2.2. Corpus Level Switching

Moving to corpus level, the measure could be defined sim ply as average utterance level switching, as in Equation 5

$$
C _ {a v g} = \frac {1}{U} \sum_ {x = 1} ^ {U} C _ {u} (x) \tag {5}
$$

where U is the number of utterances in the corpus.

However, that would ignore two important points: that $C _ { u }$ does not account for code-alternation between two utterances, and that the frequency of code-switched utterances in a corpus increases its complexity.

Hence, when combining several utterances, an utterance’s matrix language and the matrix language of the previous utterance need to be represented (if they differ, that implies adding a code-alternation point between the two utterances).<sup>5</sup> For each pair of utterances, a factor must be included to account for this, as shown in Equation 6:

$$
\begin{array}{l} C _ {u} (x - 1, x) = C _ {u} (x - 1) + C _ {u} (x) + w _ {p} \delta (x): \\ \left\{ \begin{array}{l} L _ {x - 1} = \max _ {L _ {i} \in \mathbb {L}} \left\{t _ {L _ {i}} (x - 1) \right\} \\ L _ {x} = \max _ {L _ {i} \in \mathbb {L}} \left\{t _ {L _ {i}} (x) \right\} \\ \delta (x) = \left\{ \begin{array}{l} 0: x = 1 \lor L _ {x - 1} = L _ {x} \\ 1: x \neq 1 \land L _ {x - 1} \neq L _ {x} \end{array} \right. \end{array} \right\} \tag {6} \\ \end{array}
$$

For combining a corpus’ all utterances, we take inspiration from readability indices that are purely word frequencybased and (as $C _ { u } )$ , e.g., make no distinction between different word classes. Those are calculated using the average sentence length and another factor, e.g., the average number of syllables per word as in the ‘Reading Ease’ score (Flesch, 1948), the frequency of multi-syllabic words in ‘Fog’ (Gunning, 1952), or the frequency of long words in ‘LIX’ (Björnsson, 1968).

Flesch’ Reading Ease score is based on the average number of words per sentence and average number of syllables per word:

$$
\mathrm{RE} = 2 0 6. 8 3 5 - [ 1. 0 1 5 \cdot (\frac {\mathrm{W}}{\mathrm{S}}) + 8 4. 6 \cdot (\frac {\mathrm{L}}{\mathrm{W}}) ] \tag {7}
$$

where W is the number of words in the text, S the total number of sentences, and L the total number of syllables (hence words/sentence are weighted as 1.2·syllables/word).

The Fog Index is the number of words per sentence plus the percentage of multi-syllabic words:

$$
\mathrm{Fog} = 0. 4 \cdot [ \frac {\mathrm{W}}{\mathrm{S}} + 1 0 0 \cdot (\frac {\mathrm{F}}{\mathrm{W}}) ] \tag {8}
$$

where W is the number of words in the text, S the number of sentences, and F the number of “foggy” words, that is, mainly words with more than three syllables.

The LIX measurement is the number of words per sentence plus the percentage of long words:

$$
\mathrm{LIX} = \frac {\mathrm{W}}{\mathrm{S}} + 1 0 0 \cdot (\frac {\mathrm{L}}{\mathrm{W}}) \tag {9}
$$

where W is the number of words in the text, S the number of sentences, and L the number of long words (defined as words with more than five characters).

In the case of code-switching, the first factor is the average switching level per utterance, as calculated by inserting the $C _ { u }$ given by Equation 6 into the average of Equation 5, while the second factor is the frequency of utterances that contain any code-switching (i.e., utterances with $C _ { u } > 0 )$ . Thus arriving at Equation 10:

$$
\begin{array}{l} C _ {c} = \frac {\sum_ {x = 1} ^ {U} C _ {u} (x) + w _ {p} \delta (x)}{U} + w _ {s} \frac {S}{U} \cdot 1 0 0 \tag {10} \\ = \frac {1 0 0}{U} \left[ \sum_ {x = 1} ^ {U} \left(w _ {m} f _ {m} (x) + w _ {p} \left[ f _ {p} (x) + \delta (x) \right]\right) + w _ {s} S \right] \\ \end{array}
$$

where S is the number of utterances that contain codeswitching $( 0 ~ \leq ~ S ~ \leq ~ U )$ , and $w _ { s }$ the relative weight attached to the switching frequency.

## 3. Comparing Corpora Level Switching

The main issue when applying an information source combination method (e.g., Equation 2 or 3) is how to choose the weights, and several strategies have been proposed. We tried a number of them experimentally at the utterance level, but the only combination giving reliable and intuitive values was the average (equal weights: $\begin{array} { r } { w _ { k } = \frac { 1 } { n } ) } \end{array}$ reflecting an observation also made by Clemen (2008, p.765): “Having spent much ofmy career studying various combination methods, it has been somewhat frustrating to consistently find that the simple average performs so well empirically.”

With two information sources, the weights are $w _ { m } = w _ { p }$ = $\textstyle { \frac { 1 } { 2 } }$ and Equation 4 (for N >0) reads:

$$
C _ {u} (x) = 1 0 0 \cdot \frac {N (x) - \max _ {L _ {i} \in \mathbb {L}} \{t _ {L _ {i}} \} (x) + P (x)}{2 N (x)} \tag {11}
$$

Similarly, for determing the relative weight attached to the switching frequency at the corpus level (w in Equation 10), we again compare to readability indices. Fog and LIX combine two information sources without weighting (i.e., indirectly use the average); however, Reading Ease applies a weighting which treats the {words/sentence} factor as 1.2 × {syllables/word}. Flesch (1948) derived this through regression based on correlations between the average grade of children and those who could answer 50% and 75% of some test questions.

<table><tr><td>Language Pair</td><td>words</td><td>utterances (U)</td><td colspan="2">switched</td><td colspan="2"> $C_{avg}$ (U) (S)</td><td colspan="2"> $P_{avg}$ (U) (S)</td><td>δ(U)</td><td> $C_c$ </td></tr><tr><td>DU-TR</td><td>70,874</td><td>3,065</td><td>382</td><td>12.46</td><td>4.11</td><td>33.02</td><td>0.23</td><td>1.88</td><td>48.87</td><td>14.50</td></tr><tr><td>EN-HI</td><td>27,167</td><td>2,583</td><td>570</td><td>22.07</td><td>1.87</td><td>8.46</td><td>0.57</td><td>2.56</td><td>17.81</td><td>20.26</td></tr><tr><td>EN-ES</td><td>140,746</td><td>11,400</td><td>2,335</td><td>20.48</td><td>4.91</td><td>23.95</td><td>0.38</td><td>1.84</td><td>13.83</td><td>21.97</td></tr><tr><td>EN-ZH</td><td>17,430</td><td>999</td><td>322</td><td>32.23</td><td>4.19</td><td>13.01</td><td>0.70</td><td>2.18</td><td>22.32</td><td>31.06</td></tr><tr><td>EN-NE</td><td>146,056</td><td>9,993</td><td>4,926</td><td>49.29</td><td>7.98</td><td>16.19</td><td>1.52</td><td>3.08</td><td>35.18</td><td>49.06</td></tr><tr><td>ARB-ARZ</td><td>119,317</td><td>5,839</td><td>931</td><td>15.94</td><td>3.77</td><td>23.67</td><td>0.19</td><td>1.20</td><td>13.29</td><td>17.06</td></tr></table>

Table 1: Code-switching levels in some corpora

Using these weights, Equation 10 becomes

$$
C _ {c} = \frac {1 0 0}{U} \left[ \frac {1}{2} \sum_ {x = 1} ^ {U} \left(f _ {m} (x) + f _ {p} (x) + \delta (x)\right) + \frac {5}{6} S \right] \tag {12}
$$

$$
= \frac{100}{U}\biggl [\frac{1}{2}\sum_{x = 1}^{U}\Bigl (1 - \frac{\max\limits_{L_i\in\mathbb{L}}\{t_{L_i}\}(x) + P(x)}{N(x)} +\delta(x)\Bigr)\Bigr)+\frac{5}{6} S\biggr ]
$$

To show how the measure can be used in practice to objectively compare the complexity of code-switching, $C _ { c }$ values as in Equation 12 were calculated for some recently produced code-switched corpora: the Dutch-Turkish chat corpus of Nguyen and Dogruöz (2013), the English-Hindi˘ Twitter and Facebook chat corpus of Jamatia et al. (2015), and the four corpora<sup>6</sup> used in the shared task on word-level language detection in code-switched text (Solorio et al., 2014) organized by the workshop on Computational $\mathsf { A p - }$ proaches to Code Switching at the 2014 Conference on Empirical Methods in Natural Language Processing (EMNLP).

When comparing $C _ { c }$ values for different corpora, it is necessary to consider their respective tagsets and annotation guidelines. So does the annotation strategy chosen for the EMNLP corpora prescribe that elements such as abbreviations should be tagged with the language they belong to, while other annotations schemes treat them as language independent. Another potential problem can be to decide whether a tag is directly language related or not. The EMNLP tagset includes the tags ‘mixed’ and ‘ambiguous’, that here are treated as language items in the calculations, but without being assigned to any specific language (when selecting an utterance’s matrix language), which follows the EMNLP annotation guidelines.

Table 1 shows statistics and $C _ { c }$ values for the corpora. The first columns give the number of words and total number of utterances (U) in each corpus, followed by the number and percentage of the utterances that really contain any codeswitching (S). The second set of columns provide the $C _ { a v g }$ values over both all the utterances and over only the utterances that actually contain code-switching, followed by the average number of intra-utterance code-alternation points (P) for the same two sets of utterances (the total and those containing switching), and finally the frequency of interutterance switching (δ), i.e., switching of matrix language between two utterances. The last column gives the actual $C _ { c }$ value for each corpus.

It is noticable that the EN-NE corpus from EMNLP exhibits the highest level of code-switching, both at corpus level $( C _ { c } = 4 9 . 0 6 )$ and on average at utterance level $( C _ { a v g } ~ = ~ 7 . 9 8$ for all utterances, U), as well as highest average number of code-alternation points per utterance $( P _ { a v g } ~ = ~ 1 . 5 2 ;$ for those utterances that contain switching: $P _ { a v g } = 3 . 0 8 )$ , while DU-TR has the lowest $C _ { c }$ value, but the highest frequency of matrix language switching between utterances (there are 1, 498 switches for 3, 065 utterances), and also the highest utterance level switching $( C _ { a v g } = 3 3 . 0 2 )$ if counting the average over those utterances that contain switching (S).

## 4. Discussion and Conclusion

The paper has defined an objective measure of the complexity of code-switched texts, i.e., texts written in several different languages, something which is particularly common in social media. Certainly, though, no such measure will ever be able to capture all types of differences between corpora. In particular, the ways corpora were collected and annotated, and their intended usage also need to be taken into account. However, levelling out such differences should arguably not be the aim of the code-switching measure itself, but rather be left to the users: when comparing corpora with widely different scopes, the users themselves need to be aware of the potential variation and consider this when deciding on whether a straight-forward comparison really makes sense.

The English-Nepalese EMNLP corpus showed an extremely high level of switching and the Mandarin Chinese a fairly high level. This could of course possibly have been caused by errors and problems in tagging, but in contrast to other corpora, the tagset used in the EMNLP shared task included a tag for words that are ambiguous in a context (i.e., words that even given the contextual information could potentially belong to two or more of the languages in the corpus), which potentially should ease the annotation task.

It is important to keep in mind that the code-switching corpus complexity measurement is intended to be independent of the languages contained in the corpus, while the performance of a language processing system of course also will depend on the actual languages, their relationship, and their annotation schemes. Thus the reported level of mixing in the ARB-ARZ corpus is quite low (only 15.94%, with $C _ { c } = 1 7 . 0 6 )$ , but the “languages” involved are both Arabic dialects, so very closely-related, and hence the potential overlap between them is high, even if that has not been reflected in the annotation. (Notably, almost all words of the “standard” language will also belong to a dialect, while the opposite relation does not hold. So only utterances containing strictly Egyptian Arabic words and expressions would be expected to be annotated as ARZ in this case, and all others as ARB, Modern Standard Arabic.)

For this reason, the dialectal Arabic corpus actually was the one causing most problems for the processors in the EMNLP 2014 shared task on code-switched language identification. On the other hand, the corpus with the highest $C _ { c }$ (EN-NE) was the second easiest one to label for the systems participating in the shared task (the language pair which was the easiest to separate was Mandarin–English, although not for linguistic reasons, but simply since the two languages were written in different scripts).

## 5. Acknowledgements

The EMNLP shared task corpora are public domain, as tweet IDs. The Dutch-Turkish corpus has gratefully been made available by the researchers involved. Interested readers should contact: Dong Nguyen (U Twente) or Seza Dogruöz (Tilburg U). Many thanks to them and to the˘ EMNLP shared task organisers (Solorio et al., 2014), as well as to Anupam Jamatia, Kunal Chakma and Dwijen Rudrapal (all at NIT Agartala).

## 6. Bibliographical References

Androutsopoulos, J. (2011). Language change and digital media: a review of conceptions and evidence. In Tore Kristiansen et al., editors, Standard Languages and Language Standards in a Changing Europe, pages 145–159. Novus, Oslo, Norway, February.  
Auer, P. (1999). From codeswitching via language mixing to fused lects: Toward a dynamic typology of bilingual speech. International Journal ofBilingualism, 3(4):309– 332.  
Baldwin, T., Cook, P., Lui, M., MacKinlay, A., and Wang, L. (2013). How noisy social media text, how diffrnt social media sources? In Proceedings of the 6th International Joint Conference on Natural Language Processing, pages 356–364, Nagoya, Japan, October. AFNLP.  
Björnsson, C.-H. (1968). Läsbarhet. Liber, Stockholm, Sweden. (in Swedish).  
Bullock, B. E., Hinrichs, L., and Toribio, A. J. (2014). World Englishes, code-switching, and convergence. In Markku Filppula, et al., editors, The Oxford Handbook of World Englishes. Oxford University Press, Oxford, England. Forthcoming. Online publication: March 2014.  
Cárdenas-Claros, M. S. and Isharyanti, N. (2009). Code switching and code mixing in internet chatting: between  
‘yes’, ‘ya’, and ‘si’ a case study. Journal of Computer-Mediated Communication, 5(3):67–78.  
Carter, S. (2012). Exploration and Exploitation of Multilingual Data for Statistical Machine Translation. PhD Thesis, University of Amsterdam, Informatics Institute, Amsterdam, The Netherlands, December.  
Clemen, R. T. (2008). Comment on Cooke’s classical method. Reliability Engineering & System Safety, 93(5):760–765, May.  
Das, A. and Gambäck, B. (2014). Identifying languages at the word level in code-mixed Indian social media text. In Proceedings of the 11th International Conference on Natural Language Processing, pages 169–178, Goa, India, December.  
Debole, F. and Sebastiani, F. (2005). An analysis of the relative hardness of Reuters-21578 subsets. Journal of the American Society for Information Science and Technology, 58(6):584–596, April.  
Eisenstein, J. (2013). What to do about bad language on the internet. In Proceedings of the 2013 Conference of the North American Chapter ofthe Associationfor Computational Linguistics: Human Language Technologies, pages 359–369, Atlanta, Georgia, June. ACL.  
Flesch, R. (1948). A new readability yardstick. Journal of Applied Psychology, 32(3):221–233, June.  
Gafaranga, J. and Torras, M.-C. (2002). Interactional otherness: Towards a redefinition of codeswitching. International Journal ofBilingualism, 6(1):1–22.  
Gambäck, B. and Das, A. (2014). On measuring the complexity of code-mixing. In Proceedings ofthe 11th International Conference on Natural Language Processing, pages 1–7, Goa, India, December. 1st Workshop on Language Technologies for Indian Social Media.  
Genest, C. and McConway, K. J. (1990). Allocating the weights in the linear opinion pool. Journal of Forecasting, 9(1):53–73, Jan/Feb.  
Gunning, R. (1952). The Technique of Clear Writing. McGraw-Hill, New York, New York.  
Hu, Y., Talamadupula, K., and Kambhampati, S. (2013). Dude, srsly?: The surprisingly formal nature of Twitter’s language. In Proceedings of the 7th International Conference on Weblogs and Social Media, Boston, Massachusetts, July. AAAI.  
Jamatia, A., Gambäck, B., and Das, A. (2015). Partof-speech tagging for code-mixed English-Hindi Twitter and Facebook chat messages. In Proceedings ofthe 10th International Conference on Recent Advances in Natural Language Processing, pages 239–248, Hissar, Bulgaria, September.  
Kilgarriff, A. (2001). Comparing corpora. International Journal ofCorpus Linguistics, 6(1):97–133.  
Lewis, G., Jones, B., and Baker, C. (2012). Translanguaging: origins and development from school to street and beyond. Educational Research and Evaluation, 18(7):641–654, October.  
Lignos, C. and Marcus, M. (2013). Toward web-scale analysis of codeswitching. In 87th Annual Meeting ofthe Linguistic Society of America, Boston, Massachusetts, January. Poster.  
Lui, M. and Baldwin, T. (2014). Accurate language identification of twitter messages. In Proceedings of the 14th Conference of the European Chapter of the Association for Computational Linguistics, pages 17–25, Göteborg, Sweden, April. ACL. 5th Workshop on Language Analysis for Social Media.  
Lyu, D.-C., Tan, T.-P., Chng, E.-S., and Li, H. (2015). Mandarin–English code-switching speech corpus in South-East Asia: SEAME. Language Resources and Evaluation, 49(3):581–600, September.  
Muysken, P. (2000). Bilingual speech: A typology ofcodemixing. Cambridge University Press, Cambridge, England.  
Nguyen, D. and Dogruöz, A. S. (2013). Word level lan-˘ guage identification in online multilingual communication. In Proceedings of the 2013 Conference on Empirical Methods in Natural Language Processing, pages 857–862, Seattle, Washington, October. ACL.  
Paolillo, J. (1996). Language choice on soc.culture.punjab. Electronic Journal ofCommunication, 6(3), June.  
Paolillo, J. (1999). The virtual speech community: Social network and language variation on IRC. Journal of  
Computer-Mediated Communication, 4(4), June.  
Paolillo, J. (2011). “conversational” codeswitching on usenet and internet relay chat. Language@Internet, 8(article 3), June.  
Pinto, D., Rosso, P., and Jiménez-Salazar, H. (2011). A self-enriching methodology for clustering narrow domain short texts. The Computer Journal, 54(7):1148– 1165, July.  
Solorio, T., Blair, E., Maharjan, S., Bethard, S., Diab, M., Gohneim, M., Hawwari, A., AlGhamdi, F., Hirschberg, J., Chang, A., and Fung, P. (2014). Overview for the first shared task on language identification in code-switched data. In Proceedings ofthe 2014 Conference on Empirical Methods in Natural Language Processing, pages 62– 72, Doha, Qatar, October. ACL. 1st Workshop on Computational Approaches to Code Switching.  
Voss, C., Tratz, S., Laoudi, J., and Briesch, D. (2014). Finding romanized Arabic dialect in code-mixed tweets. In Proceedings of the 9th International Conference on Language Resources and Evaluation, pages 188–199, Reykjavík, Iceland, May. ELRA.