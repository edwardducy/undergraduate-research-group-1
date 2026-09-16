![](images/307ba365fbb36ea16dd539904e266274999f36970e832fae593b4b00c4897d0e.jpg)

<details>
<summary>text_image</summary>

I
B
S
</details>

The Measurement of Observer Agreement for Categorical Data

Author(s): J. Richard Landis and Gary G. Koch

Source: Biometrics<sub>,</sub> Vol. 33, No. 1 (Mar., 1977), pp. 159-174

Published by: International Biometric Society

Stable URL: http://www.jstor.org/stable/2529310

Accessed: 19/07/2013 07:36

Your use of the JSTOR archive indicates your acceptance of the Terms & Conditions of Use, available at http://www.jstor.org/page/info/about/policies/terms.jsp

JSTOR is a not-for-profit service that helps scholars, researchers, and students discover, use, and build upon a wide range of content in a trusted digital archive. We use information technology and tools to increase productivity and facilitate new forms of scholarship. For more information about JSTOR, please contact support@jstor.org.

![](images/c4e866e20e101df8e3252a7473e76317e232b14c3ab3675b197351e89bf12f6e.jpg)

http://www.jstor.org

International Biometric Society is collaborating with JSTOR to digitize, preserve and extend access to Biometrics.

# The Measurement of Observer Agreement for Categorical Data

J. RICHARD LANDIS

Department of Biostatistics, University of Michigan, Ann Arbor, Michigan 48109 U.S.A.

GARY G. KOCH

Department of Biostatistics, University of North Carolina, Chapel Hill, North Carolina 27514 U.S.A.

## Summar

This paper presents a general statistical methodology for the analysis of multivariate cate gorical data arising from observer eliability studies. The proceduressentially involves the con struction of functions of the observed proportions which are directed at the extent to which th observers agree among themselves and the construction of test statistics for hypotheses involvi these functions. Tests for interobserver bias are presented in terms of first-order marginal hom geneity and measures of interobserver agreement are developed as generalized kappa-type statistic These procedures are illustrated with a clinical diagnosis example from the epidemiological liter ture

## 1. Introducti

Researchers in manv fields have become increasingly aware of the observer (rater or interviewer) as an important source of measurement error. Consequently, reliability studie are conducted in experimental or survey situations to assess the level of observer variabilit in the measurement procedures to be used in data acquisition. When the data arisin from such studies are quantitative, tests for interobserver bias and measures of inter observer agreement are usually obtained from standard ANOVA mixed models or random effects models such as those discussed in Anderson and Bancroft [1952], Scheffe [1959] and Searle [1971]. As a result, hypothesis tests of observer effects are used to investigat interobserver bias, i.e., differences in the mean response among observers, and estimate of intraclass correlation coefficients are used to measure interobserver eliability. rVlodifi tions and extensions of these standard ANOVA models have been proposed by Grubb [1948, 1973], Mandel [1959], Fleiss [1966], Overall [1968], and Loewenson, Bearman and Resch [1972] to evaluate the measurement error in various types of applications. Althoug assumptions of normality for these models may not be warranted in certain cases, th ANOVA procedures discussed in Searle [1971] and the symmetric square difference pro cedure in Koch [1967, 1968] still permit the estimation of the appropriate componen of variance and the reliability coefficien

On the other hand, many observer reliability studies involve categorical data in whic the response variable is classified into nominal (or possibly ordinal) multinomial categorie As reviewed in Landis and Koch [1975a, 1975b], a wide variety of estimation and testin procedures have been recommended for the assessment of observer variability in thes cases. In this paper we propose a unified approach to the evaluation of observer agreemen for categorical data by expressing the quantities which reflect the extent to which th observers agree among themselves as functions of observed proportions obtained fro underlying multidimensional contingency tables. These functions are then used to produc test statistics for the relevant hypotheses concerning interobserver bias in the overal usage of the measurement scale and interobserver agreement on the classification of in dividual subjects. For illustrative purposes, this general methodology is developed withi the context of a typical data set which resulted from an investigation of observer vari ability in the clinical diagnosis of multiple sclerosi

## 2. A Clinical Diagnois Exampl

Let us consider the data arising from the diagnosis of multiple sclerosis reported in Westlund and Kurland [1953]. Among other things, the investigators were interested in comparing patient groups to study possible differences in the geographical distributio of the disease. For this purpose, a series of patients in Winnipeg, Manitoba and a separat series of patients in New Orleans, Louisiana were selected and were examined by a neuro ogist in their respective locations. After the completion of all the examinations, each neurologist was requested to review all the records without seeing his earlier summar and diagnosis and to classify them into one of the following diagnostic classes

1. Certain multiple sclerosi  
2. Probable multiple sclerosis  
3. Possible multiple sclerosis (odds 50: 50);  
4. Doubtful, unlikely, or definitely not multiple sclerosi

In order to evaluate agreement between the diagnosticians, the Winnipeg neurologist he reviewed and classified each of the New Orleans patient records, and vice versa. The data resulting from these review diagnoses are presented in Table 1.

A preliminary inspection of the Winnipeg data indicates that the Winnipeg neurologi tended to diagnose more of the patients as certain (1) or probable (2) multiple sclerosi than did his counterpart in New Orleans. As a result, they agreed on the diagnosis of only 64/149 (43 percent) of the patients. Although the differences in the overall crud distributions of the diagnoses seem to be less prominent within the New Orleans patient the neurologists diagnosed only 33/69 (48 percent) of them into identically the sam category. The statistical issues concerning these differences in diagnosis can be summarize within the framework of the following basic questions

(1) Are there any differences between the two patient populations with respect to the overall crude distribution of the diagnoses by each of the two neurologist  
(2) Are there any differences between the overall crude distributions of the diagnose by the two neurologists within each of the respective patient populations  
(3) Is there any neurologist X sub-population interaction in the overall crude distribu tion of the diagnoses  
(4) Is there any difference between the two patient populations with respect to th overall agreement of the two neurologists on the specific diagnosis of individua patients?  
(5) Is the agreement of the two neurologists on the specific diagnosis of individua patients significantly different from dance agreement based on their overall crud distributions of diagnoses

Table 1 DIAGNOSTIC CLASSIFICATION REGARDING MIULTIPLE SCLEROSIS

<table><tr><td colspan="2">Sub-population</td><td colspan="6">Winnipeg Patients (1)</td></tr><tr><td>Observer</td><td></td><td colspan="4">Winnipeg Neurologist</td><td>(2)</td><td></td></tr><tr><td></td><td>Diagnostic Class</td><td>1</td><td>2</td><td>3</td><td>4</td><td>Total</td><td>Proportion</td></tr><tr><td></td><td>1</td><td>38</td><td>5</td><td>0</td><td>1</td><td>44</td><td>0.295</td></tr><tr><td rowspan="3">New Orleans Neurologist (1)</td><td>2</td><td>33</td><td>11</td><td>3</td><td>0</td><td>47</td><td>0.315</td></tr><tr><td>3</td><td>10</td><td>14</td><td>5</td><td>6</td><td>35</td><td>0.235</td></tr><tr><td>4</td><td>3</td><td>7</td><td>3</td><td>10</td><td>23</td><td>0.154</td></tr><tr><td></td><td>Total</td><td>84</td><td>37</td><td>11</td><td>17</td><td>149</td><td></td></tr><tr><td colspan="2">Proportion</td><td>0.564</td><td>0.248</td><td>0.074</td><td>0.114</td><td></td><td></td></tr></table>

<table><tr><td colspan="4">Sub-population</td><td colspan="4">New Orleans Patients (2)</td></tr><tr><td colspan="4">Observer</td><td colspan="4">Winnipeg Neurologist (2)</td></tr><tr><td></td><td>Diagnostic Class</td><td>1</td><td>2</td><td>3</td><td>4</td><td>Total</td><td>Proportion</td></tr><tr><td></td><td>1</td><td>5</td><td>3</td><td>0</td><td>0</td><td>8</td><td>0.116</td></tr><tr><td></td><td>2</td><td>3</td><td>11</td><td>4</td><td>0</td><td>18</td><td>0.261</td></tr><tr><td rowspan="2">New Orleans Neurologist (1)</td><td>3</td><td>2</td><td>13</td><td>3</td><td>4</td><td>22</td><td>0.319</td></tr><tr><td>4</td><td>1</td><td>2</td><td>4</td><td>14</td><td>21</td><td>0.304</td></tr><tr><td></td><td>Total</td><td>11</td><td>29</td><td>11</td><td>18</td><td>69</td><td></td></tr><tr><td></td><td>Proportion</td><td>0.159</td><td>0.420</td><td>0.159</td><td>0.261</td><td></td><td></td></tr></table>

(6) Are there certain patterns of disagreement which may reflect significant imprecisio in the diagnostic criteri

As stated in Koch et al. [1977], questions (1)-(3) are directly analogous to the hypothese of "no whole-plot effects," "no split-plot effects," and "no whole-plot X split-plot inter action" in standard split-plot experiments. In this context, question (1) addresses differenc among the sub-populations, question (2) involves the issue of interobserver bias, and question (3) is concerned with the observer X sub-population interaction. Thus, the first-order marginal distributions of response for each of the neurologists within each sub-population contain the relevant information for dealing with these questions. In contrast to overall crude differences, questions (4)-(6) are addressed at interobserv agreement on a subject-to-subject basis; and, as such they are directly analogous to hypotheses concerning intraclass correlation coefficients in random effects models. Hence, certain functions of the diagonal cells of various subtables are used to provide informati for estimating and testing the significance of agreement on the classification of individua subj ects

In the following sections a general methodology for answering these questions is developed in terms of specific hypotheses. These procedures are then illustrated with an analysis of the data in Table 1.

## 3. Methodolo

Let $i = 1 , 2 , \cdots ,$ s index a set of sub-populations from which random samples have been selected. Suppose that the same response variable is measured separately by each of d observers using an L-point scale. Let th $r ~ = ~ L ^ { d }$ response profiles be indexed by a vector subscrip ${ \bf j } = ( j _ { 1 } , j _ { 2 } , \cdots , j _ { d } )$ , wher $j _ { \boldsymbol { \imath } } = 1 , 2 , \cdot \cdot \cdot$ , L fo $g = 1 , 2 , \cdots , d .$ Further more, let $\pi _ { i \textbf { j } } = \pi _ { i j _ { \uparrow } , j _ { 2 } , \cdots , j _ { d } }$ represent the joint probability of response profile j for randoml selected subjects from the ith sub-population. Then let the first-order marginal probabilit

$$
\phi_ {i g k} = \sum_ {\mathrm{jwith} j _ {g} = k} \dots \sum_ {\pi_ {i j _ {1}, j _ {2}}, \dots , i _ {d}} \quad \text {for} \quad \begin{array}{l} i = 1, 2, \dots , s \\ g = 1, 2, \dots , d \\ k = 1, 2, \dots , L \end{array} \tag {3.1}
$$

represent the probability of the kth response category for the gth observer ill the ith sub population

## 3.1 Hypotheses Involving Marginal Distributio

Hypotheses directed at the questions of differences among sub-populations and inter observer bias involve distributions of the response profiles and can be expressed in term of constraints on the first-order marginal probabilitie $\left\{ \phi _ { i g k } \right\}$ . As a result, the specifi hypotheses associated with questions (1)-(3) are directly analogous to $H _ { S M } \mathrm { ~ , ~ } H _ { C M }$ , and $H _ { A M }$ outlined in Koch et al. [1977] in expressions (2.4), (2.5), and (2.9), respectively. In particular, the d observers correspond to the d conditions, and thus the hypothesis of first order marginal symmetry (homogeneity) addresses the issue of interobserver bias. These hypotheses can also be expressed in terms of constraints on mean score functions associated with each observer such as the $\left\{ \eta _ { i \ g } \right\}$ summary indexes specified in (2.14) in Koch et al. [1977] Further discussion of hypotheses involving marginal distributions within the contex of observer agreement studies is given in Landis [1975]

## 3.2 Hypotheses Involving Generalized Kappa-Type Measure

Whereas the previous hypotheses concerning differences among sub-populations and interobserver bias involved only the first-order marginal probabilities, hypotheses directe at the extent to which observers agree among themselves on the classification of individua subjects must be formulated in terms of the internal elements of the table. For example the estimate of the crude proportion of agreement between two observers is simply the sum of the observed proportions on the main diagonal of the corresponding two-way table In addition, if partial credit is permitted for certain types of disagreement, an estimat of the weighted proportion of agreement will involve the weighted inclusion of the of diagonal cells

As reviewed in Landis and! Koch [1975a, 1975b], numerous measures of observer agree ment have been proposed for categorical data, e.g., Goodman and Kruskal [1954], Cohen [1960, 1968], Fleiss [1971], Light [1971], and Cicchetti [1972]. Most of these quantitie are of the for

$$
\kappa = \frac {\pi_ {0} - \pi_ {e}}{1 - \pi_ {e}}, \tag {3.2}
$$

where $\pi _ { 0 }$ 0 is an observational probability of agreement and $\pi _ { e }$ re is a hypothetical expecte probability of agreement under an appropriate set of baseline constraints such as tota independence of observer classifications. Ranging fro $[ - \pi _ { e } / ( 1 - \pi _ { e } ) ]$ to +1, K indicate the extent to which the observational probability of agreement is in excess of the prob ability of agreement hypothetically expected under the baseline constraints. Furthermor as shown in Fleiss and Cohen [1973] and Fleiss [1975], K is directly analogous to the intra class correlation coefficient obtained from ANOVA models for quantitative measuremen and can be used as a measure of the reliability of multiple determinations on the sam subj ects.

Several kappa-type measures of interobserver agreement can be formulated to in vestigate selected patterns of disagreement simultaneously by choosing correspondi sets of weights which reflect the role of each response category in a given agreement index For example, a set of weights can be chosen so that the resulting agreement measur indicates the combined performance of all the observers, such as majority or consensu agreement, or sets of weights can be directed at subsets of observers, such as all possibl pairwise agreement measures. Alternatively, these weights can be chosen so that th associated kappa measures indicate the increments in agreement which result by succes sively combining relevant categories of the response variable. Such kappa measures are said to be in a hierarachical relationship with each other. Thus, in general, let $w _ { 1 { \bf j } } ~ , ~ w _ { 2 { \bf j } }$ $\cdots , w _ { u } \mathbf { j }$ be u sets of weights assigned to the response profiles indexed by ${ \bf j } = ( j _ { 1 } , j _ { 2 } , \cdots , j _ { d } )$ Moreover let $0 \le w _ { h \textbf { j } } \le \textbf { l }$ fo $h = 1 , 2 , \cdots$ , u over all ${ \bf j } ,$ so that the resulting estimate are interpretable as probabilities of agreement. Then the observational probability o agreement associated with the hth set of weights in the ith sub-population is the weighte sum

$$
\lambda_ {i h} = \sum \dots \sum_ {\mathrm{j}} w _ {h \mathrm{j}} \pi_ {i \mathrm{j}} \quad \text {for} \quad \begin{array}{l} i = 1, 2, \dots , s \\ h = 1, 2, \dots , u. \end{array} \tag {3.3}
$$

Correspondingly, the expected proportion of agreement associated with (3.3) is the weighte sum

$$
\gamma_ {i h} = \sum \dots \sum_ {\mathrm{j}} w _ {h \mathrm{j}} \pi_ {i \mathrm{j}} ^ {(e)} \quad \text {for} \quad \begin{array}{l} i = 1, 2, \dots , s \\ h = 1, 2, \dots , u, \end{array} \tag {3.4}
$$

wher ${ \pi _ { i } } _ { \bf j } ^ { ( e ) }$ represents the joint hypothetical expected probability of response profile j for randomly selected subjects from the ith sub-populatio

These expected probabilities are determined by the choice of a particular set of baselin constraints assumed for the response profiles. For this purpose, let $\underline { { { E } } } ~ = ~ \{ \underline { { { E } } } _ { 1 } ~ , ~ \underline { { { E } } } _ { 2 } ~ , ~ \cdots \}$ represent such underlying constraints on the marginal probabilitie $\left\{ \phi _ { i g k } \right\}$ of (3.1). In this context, the following sets of constraints are of interest in creating interobserv agreement measure

(i) Under the assumption of total independence among the response variables fro the d observers, th $\left\{ \pi _ { i \mathbf { j } } ^ { \quad ( e ) } \right\}$ satisf

$$
\begin{array}{l} \underline {{E}} _ {1}: \pi_ {i j _ {1} j _ {2}} \dots_ {j _ {d}} ^ {(e)} = \phi_ {i 1 j _ {1}} \phi_ {i 2 j _ {2}} \dots \phi_ {i d j _ {d}} \\ = \prod_ {k = 1} ^ {d} \phi_ {i k i _ {k}} \quad \text {for} \quad i = 1, 2, \dots , s. \tag {3.5} \\ \end{array}
$$

(ii) Under the assumption of $^ { \mathfrak { s } } \mathrm { n o }$ interobserver bias" the hypothesis of first-ord marginal homogeneit $( H _ { c M }$ in Koch et al. $[ 1 9 7 7 ]$ holds. In this situation, let the common probability of classification into the kth category be

$$
\psi_ {i k} = \phi_ {i 1 k} = \phi_ {i 2 k} = \dots = \phi_ {i d k} \tag {3.6}
$$

for $i = 1 , 2 , \cdots , s$ and $k = 1 , 2 , \cdots ,$ L. Then under the baseline constraints of total independence and marginal homogeneity the $\left\{ \pi _ { i \mathbf { j } } ^ { \quad ( e ) } \right\}$ satisf

$$
\begin{array}{l} \underline {{E}} _ {2}: \pi_ {i j _ {1} i _ {2}} \dots_ {j _ {d}} ^ {(e)} = \psi_ {i j _ {1}} \psi_ {i j _ {2}} \dots \psi_ {i j _ {d}} \\ = \prod_ {g = 1} ^ {d} \psi_ {i j _ {g}} \quad \text {for} \quad i = 1, 2, \dots , s. \tag {3.7} \\ \end{array}
$$

Consequently, a generalized kappa-type measure of agreement directly analogous to (3.2) can be formulated b

$$
\kappa_ {i h} = \frac {\lambda_ {i h} - \gamma_ {i h}}{1 - \gamma_ {i h}} \quad \text {for} \quad \begin{array}{l} i = 1, 2, \dots , s \\ h = 1, 2, \dots , u, \end{array} \tag {3.8}
$$

under a set of specified constraints in $E .$ Here $\kappa _ { i h }$ represents an agreement measure amon the d observers in the ith sub-population with respect to the hth set of weight

Within this framework, the specific hypotheses associated with questions (4)-(6) can now be formulated as follow

(4) If there are no differences among the s sub-populations with respect to the measure of overall specific agreement among the d observers unde $E _ { z }$ then th $\left\{ \kappa _ { i h } \right\}$ satisf the hypothes

$$
H _ {S A \mid \underline {{E}} _ {z}}: \kappa_ {1 h} = \kappa_ {2 h} = \dots = \kappa_ {s h} \quad \text {for} \quad h = 1, 2, \dots , u, \tag {3.9}
$$

where SA denotes sub-population agreemen

(5) If the level of observed agreement is equal to that expected under $E _ { z }$ then th $\left\{ \kappa _ { i h } \right\}$ satisfy the hypothe

$$
H _ {N A \mid E _ {z}}: \kappa_ {i h} = 0 \quad \text {for} \quad \begin{array}{l} i = 1, 2, \dots , s \\ h = 1, 2, \dots , u, \end{array} \tag {3.10}
$$

where NA denotes no agreemen

(6) In some cases the weights for the kappa measures are chosen to be in a hierarchic relationship with each other in order to investigate specific disagreement pattern In these situations, if the extent of disagreement is the same for the categorie combined by the (h + 1)-st set of weights as for those combined by the hth set, then the $\left\{ \kappa _ { i h } \right\}$ satisfy the hypothes

$$
H _ {H A \mid \underline {{E}} z}: \kappa_ {i, h + 1} = \kappa_ {i, h} \quad \text {for} \quad i = 1, 2, \dots , s, \tag {3.11}
$$

where HA denotes hierarchical agreemen

In order to maintain consistent nomenclature when describing the relative strengt of agreement associated with kappa statistics, the following labels will be assigned to the corresponding ranges of kappa:

<table><tr><td>Kappa Statistic</td><td>Strength of Agreement</td></tr><tr><td>&lt;0.00</td><td>Poor</td></tr><tr><td>0.00–0.20</td><td>Slight</td></tr><tr><td>0.21–0.40</td><td>Fair</td></tr><tr><td>0.41–0.60</td><td>Moderate</td></tr><tr><td>0.61–0.80</td><td>Substantial</td></tr><tr><td>0.81–1.00</td><td>Almost Perfect</td></tr></table>

Although these divisions are clearly arbitrary, they do provide useful "benchmarks" fo the discussion of the specific example in Table 1

## 3.3 Estimation and Hypothesis Testin

Test statistics for the hypotheses considered in the previous sections as well as estimator for corresponding model parameters can be obtained by using the general approach fo the analysis of multivariate categorical data proposed by Grizzle, Starmer and Koch [1969 (hereafter abbreviated GSK) as outlined in Appendix 1 in Koch et al. [1977]. The hypothese in Section 3.1 involving constraints on the first-order marginal probabilities can be tested by expressing the estimates of th $\left\{ \phi _ { i \theta k } \right\}$ or the $\left\{ \eta _ { i \pmb { \varrho } } \right\}$ d as linear functions of the type give in Appendix 1 (A.14) in Koch et al. [1977]. These particular matrix expressions have alread been discussed in considerable detail in Koch and Reinfurt [1971] and Koch et at. [1977] and thus they will not be elaborated here. Otherwise, their specific construction for thes hypotheses in observer agreement studies is documented in Landis [1975]

In contrast to the linear functions which pertain to the hypotheses in Section 3.1, all the hypotheses involving generalized kappa-type measures require the expression of the ratio estimates of th $\left\{ \kappa _ { i h } \right\}$ as compounded logarithmic-exponential-linear functio of the observed proportions as formulated in Appendix 1 (A.20) in Koch et al. [1977] As a result, the test statistics for the hypotheses in Section 3.2 can also be generated by the corresponding expression given in Appendix 1 (A.11) in Koch et al. [1977]

## 4. Analysis of Multiple Sclerosis Data

This section is concerned with the analysis of the multiple sclerosis data ill Table 1 with primary emphasis given to illustrating the methodology in Section 3. Tests of sig nificance are used in a descriptive context to identify important sources of variation as opposed to a rigorous inferential context; and thus issues pertaining to multiple compar isons are ignored here. These, however, can be handled by the Scheffe type procedure given in Grizzle, Starmer and Koch [1969]. The design for this example involves s = 2 sub-populations, d = 2 observers, and $L = 4$ response categories. Thus, there are $r = L ^ { d } = 1 6$ possible multivariate response profiles within each of the sub-population

## 4.1 Marginal Homogeneity Test

The functions required to test the hypotheses involving marginal distributions can be generated in the formulation of (A.14) in Appendix 1 in Koch et al. [1977] with th function vecto $\bar { \mathbf { F } } ^ { \prime } = ( \bar { \mathbf { F } } _ { 1 } ^ { \prime } , \bar { \mathbf { F } } _ { 2 } ^ { \prime } )$ wher

$$
\mathbf {F} _ {1} ^ {\prime} = (0. 2 9 5, 0. 3 1 5, 0. 2 3 5, 0. 5 6 4, 0. 2 4 8, 0. 0 7 4) \tag {4.1}
$$

$$
\mathbf {F} _ {2} ^ {\prime} = (0. 1 1 6, 0. 2 6 1, 0. 3 1 9, 0. 1 5 9, 0. 4 2 0, 0. 1 5 9),
$$

Table 2HIERARCHICAL WEIGHTS FOR AGREEMENT MEASURES

<table><tr><td colspan="2">Weights</td><td colspan="4"> $w_{1j}$ </td><td colspan="4"> $w_{2j}$ </td><td colspan="4"> $w_{3j}$ </td><td colspan="4"> $w_{4j}$ </td></tr><tr><td colspan="2">Observer</td><td colspan="4">2</td><td colspan="4">2</td><td colspan="4">2</td><td colspan="4">2</td></tr><tr><td colspan="2">Diagnostic Class</td><td>1</td><td>2</td><td>3</td><td>4</td><td>1</td><td>2</td><td>3</td><td>4</td><td>1</td><td>2</td><td>3</td><td>4</td><td>1</td><td>2</td><td>3</td><td>4</td></tr><tr><td rowspan="4">Observer 1</td><td>1</td><td>1</td><td>0</td><td>0</td><td>0</td><td>1</td><td>1</td><td>0</td><td>0</td><td>1</td><td>1</td><td>0</td><td>0</td><td>1</td><td>1</td><td>0</td><td>0</td></tr><tr><td>2</td><td>0</td><td>1</td><td>0</td><td>0</td><td>1</td><td>1</td><td>0</td><td>0</td><td>1</td><td>1</td><td>0</td><td>0</td><td>1</td><td>1</td><td>1</td><td>0</td></tr><tr><td>3</td><td>0</td><td>0</td><td>1</td><td>0</td><td>0</td><td>0</td><td>1</td><td>0</td><td>0</td><td>0</td><td>1</td><td>1</td><td>0</td><td>1</td><td>1</td><td>1</td></tr><tr><td>4</td><td>0</td><td>0</td><td>0</td><td>1</td><td>0</td><td>0</td><td>0</td><td>1</td><td>0</td><td>0</td><td>1</td><td>1</td><td>0</td><td>0</td><td>1</td><td>1</td></tr></table>

which contain the marginal proportions for diagnostic classe $" 1 , " \ \cdots 2 "$ and $" 3 "$ for the two observers within the two sub-populations. The test statistic fo $H _ { s M }$ is $Q _ { \bf c } ~ = ~ 4 6 . 3 7$ wit $\mathrm { d . f . } \ = \ 6 _ { \mathrm { ; } }$ , which implies that there are significa $( \alpha ~ = ~ 0 . 0 1 )$ differences in the dis tributions of the observed response profiles between the Winnipeg and New Orleans patient The tests of this hypothesis within each of the observers also indicate statistically sig nifican $( \alpha ~ = ~ 0 . 0 1 )$ differences between the two sub-populations, although the Winnipe neurologist represents the more dominant component. Similarly, the test statistic fo $H _ { c M }$ is $Q \mathbf { c } \ = \ 6 9 . 0 1$ with d $\mathrm { f } , \ l = \ 6$ , which implies that there are significa $( \alpha = 0 . 0 1 )$ differenc in the response profiles between the two neurologists within each sub-population. Moreover the dominant component of these observer differences is within the Winnipeg patien group. These results suggest that significant interobserver bias exists between the two neurologists in their overall usage of the diagnostic classification scale. In addition, the goodness-of-fit statistic for testing the interaction hypothesi $H _ { A M }$ r is $Q ~ = ~ 1 4 . 0 9$ wit $\mathrm { { d . f . ~ = ~ 3 } }$ . This significa $( \alpha = 0 . 0 1 )$ ) observer X sub-population interaction is consisten with the result that the observer differences are more substantial in the Winnipeg patien grou $( Q _ { \bf c } = 5 8 . 4 7 )$ ) than in the New Orleans patient grou $( Q _ { \bf c } = 1 0 . 5 4 )$

Table 3OF HIERARCHICAL WEIGHTS

<table><tr><td>Set of Weights</td><td>Disagreement Permitted for Agreement Statistic</td></tr><tr><td>1</td><td>None; requires perfect agreement.</td></tr><tr><td>2</td><td>Certain (1) with Probable (2).</td></tr><tr><td>3</td><td>Certain (1) with Probable (2);Possible (3) with Doubtful (4).</td></tr><tr><td>4</td><td>Certain (1) with Probable (2);Probable (2) with Possible (3);Possible (3) with Doubtful (4).</td></tr></table>

## 4.2 Hierarchical Kappa-Type Measures of Agreeme

Specific patterns of disagreement between the neurologists on the diagnostic classifica tion of individual subjects can be investigated by selecting a hierarchy of weights whic successively combine adjoining categories of diagnosis in order to create potentially les stringent reliability measures. For example, the four sets of weights in Table 2 can be used to investigate the sources of imprecise diagnostic criteria. As indicated in Table 3, these weights are chosen so that specific disagreement patterns are successively tolerate in the corresponding estimates of agreement. In particula $w _ { 1 \mathrm { j } }$ j represents the set of weight which generate the kappa measure of perfect agreement proposed in Cohen [1960]

The sequence of hierarchical kappa-type statistics within each of the two patien populations associated with the weights given in Table 2 can be expressed in the formul tion (A.20) in Appendix 1 in Koch et al. [1977] under the baseline constraints of tota independenc $E _ { 1 }$ in (3.5) by lettin

$$
\mathbf {A} _ {1} = \left[ \begin{array}{c c c c c c c c c c c c c c c} 1 & 1 & 1 & 1 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\ 0 & 0 & 0 & 0 & 1 & 1 & 1 & 1 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\ 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 1 & 1 & 1 & 1 & 0 & 0 & 0 \\ 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 1 & 1 & 1 & 1 \\ 1 & 0 & 0 & 0 & 1 & 0 & 0 & 0 & 1 & 0 & 0 & 0 & 1 & 0 & 0 \\ 0 & 1 & 0 & 0 & 0 & 1 & 0 & 0 & 0 & 1 & 0 & 0 & 0 & 1 & 0 \\ 0 & 0 & 1 & 0 & 0 & 0 & 1 & 0 & 0 & 0 & 1 & 0 & 0 & 1 & 0 \\ 0 & 0 & 0 & 1 & 0 & 0 & 0 & 1 & 0 & 0 & 0 & 1 & 0 & 0 & 1 \\ 1 & 0 & 0 & 0 & 0 & 1 & 0 & 0 & 0 & 0 & 1 & 0 & 0 & 0 & 1 \\ 1 & 1 & 0 & 0 & 1 & 1 & 0 & 0 & 0 & 0 & 1 & 0 & 0 & 0 & 1 \\ 1 & 1 & 0 & 0 & 1 & 1 & 0 & 0 & 0 & 0 & 1 & 1 & 0 & 0 & 1 \\ 1 & 1 & 0 & 0 & 1 & 1 & 1 & 0 & 0 & 1 & 1 \end{array} \right] \otimes \mathbf {I} _ {2}; \tag {4.2}
$$

$$
\mathbf {A} _ {2} = \left[ \begin{array}{c c c c c c c c c c c} 1 & 0 & 0 & 0 & 1 & 0 & 0 & 0 & 0 & 0 & 0 \\ 1 & 0 & 0 & 0 & 0 & 1 & 0 & 0 & 0 & 0 & 0 \\ 1 & 0 & 0 & 0 & 0 & 0 & 1 & 0 & 0 & 0 & 0 \\ 1 & 0 & 0 & 0 & 0 & 0 & 0 & 1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 & 1 & 0 & 0 & 0 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 & 0 & 1 & 0 & 0 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 & 0 & 0 & 1 & 0 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 & 0 & 0 & 0 & 1 & 0 & 0 & 0 \\ 0 & 0 & 1 & 0 & 1 & 0 & 0 & 0 & 0 & 0 & 0 \\ 0 & 0 & 1 & 0 & 0 & 1 & 0 & 0 & 0 & 0 & 0 \end{array} \right] \otimes \mathbf {I} _ {2}; \tag {4.3}
$$

$$
\mathbf {A} _ {2} = \left[ \begin{array}{c c c c c c c c c c c} 0 & 0 & 1 & 0 & 0 & 0 & 1 & 0 & 0 & 0 & 0 \\ 0 & 0 & 1 & 0 & 0 & 0 & 0 & 1 & 0 & 0 & 0 \\ 0 & 0 & 0 & 1 & 1 & 0 & 0 & 0 & 0 & 0 & 0 \\ 0 & 0 & 0 & 1 & 0 & 1 & 0 & 0 & 0 & 0 & 0 \\ 0 & 0 & 0 & 1 & 0 & 0 & 1 & 0 & 0 & 0 & 0 \\ 0 & 0 & 0 & 1 & 0 & 0 & 0 & 1 & 0 & 0 & 0 \\ 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 1 & 0 & 0 \\ 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 1 & 0 \\ 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 1 \\ 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 1 \end{array} \right] \otimes \mathbf {I} _ {2}; \tag {4.3}
$$

$$
\begin{array}{r} \mathbf {A} _ {3} = \left[ \begin{array}{c c c c c c c c c c c c c c c c c c} - 1 & 0 & 0 & 0 & 0 & - 1 & 0 & 0 & 0 & 0 & - 1 & 0 & 0 & 0 & 0 & - 1 & 1 & 0 & 0 & 0 \\ - 1 & - 1 & 0 & 0 & - 1 & - 1 & 0 & 0 & 0 & 0 & - 1 & 0 & 0 & 0 & 0 & - 1 & 0 & 1 & 0 & 0 \\ - 1 & - 1 & 0 & 0 & - 1 & - 1 & 0 & 0 & 0 & 0 & - 1 & - 1 & 0 & 0 & - 1 & - 1 & 0 & 0 & 1 & 0 \\ - 1 & - 1 & 0 & 0 & - 1 & - 1 & - 1 & 0 & 0 & - 1 & - 1 & - 1 & 0 & 0 & - 1 & - 1 & 0 & 0 & 0 & 1 \\ 0 & 1 & 1 & 1 & 1 & 0 & 1 & 1 & 1 & 1 & 0 & 1 & 1 & 1 & 1 & 0 & 0 & 0 & 0 & 0 \\ 0 & 0 & 1 & 1 & 0 & 0 & 1 & 1 & 1 & 1 & 0 & 1 & 1 & 1 & 1 & 0 & 0 & 0 & 0 & 0 \\ 0 & 0 & 1 & 1 & 0 & 0 & 1 & 1 & 1 & 1 & 0 & 0 & 1 & 1 & 0 & 0 & 0 & 0 & 0 \\ 0 & 0 & 1 & 1 & 0 & 0 & 0 & 1 & 1 & 0 & 0 & 0 & 1 & 1 & 0 & 0 & 0 & 0 & 0 \end{array} \right] \otimes \mathbf {I} _ {2}; \\ \mathbf {A} _ {4} = [ \mathbf {I} _ {4} - \mathbf {I} _ {4} ] \otimes \mathbf {I} _ {2}; \end{array} \tag {4.5}
$$

For the data in Table 1, these estimates are given by

$$
\mathbf {F} = \left[ \begin{array}{l} \hat {\kappa} _ {1 1} \\ \hat {\kappa} _ {1 2} \\ \hat {\kappa} _ {1 3} \\ \hat {\kappa} _ {1 4} \\ \hat {\kappa} _ {2 1} \\ \hat {\kappa} _ {2 2} \\ \hat {\kappa} _ {2 3} \\ \hat {\kappa} _ {2 4} \end{array} \right] = \left[ \begin{array}{l} 0. 2 0 8 \\ 0. 3 2 8 \\ 0. 4 0 8 \\ 0. 5 9 6 \\ 0. 2 9 7 \\ 0. 3 3 2 \\ 0. 3 8 6 \\ 0. 7 8 9 \end{array} \right], \tag {4.6}
$$

wher $\hat { \kappa } _ { i h }$ is the estimate of the agreement measure in the ith sub-population associated with the hth set of weights shown in Table 2. In addition, the estimated covariance matri of F is given by

$$
\mathbf {V} _ {\mathrm{F}} = \left[ \begin{array}{c c c c c c c} 0. 2 5 4 6 & 0. 2 1 2 2 & 0. 1 8 6 8 & 0. 1 4 4 2 \\ 0. 2 1 2 2 & 0. 4 0 0 5 & 0. 3 8 6 2 & 0. 2 9 1 2 \\ 0. 1 8 6 8 & 0. 3 8 6 2 & 0. 5 2 0 0 & 0. 3 8 3 2 \\ 0. 1 4 4 2 & 0. 2 9 1 2 & 0. 3 8 3 2 & 0. 5 7 0 0 \end{array} \quad \begin{array}{c c c c} & & & \\ & & & \\ & & & \\ & & & \\ & & & \\ & & & \end{array} \right] \times 1 0 ^ {- 2}. \tag {4.7}
$$

The test statistics for the hierarchical hypotheses in (3.11) are displayed in Table 4. These results indicate that all increases in successive agreement measures within the Winnipeg patient group are significa $( \alpha = 0 . 0 5 )$ ; but for the New Orleans patient group the only significa $( \alpha = 0 . 0 5 )$ increase in agreement pertained to the final set of weight Thus, the neurologists are exhibiting significant disagreement between diagnoses (1,2), (2,3) and (3,4) in the Winnipeg group and significant disagreement between diagnose (2,3) in the New Orleans group, as evidenced by the inflated frequencies in these off-diagon cells in Table 1. On the other hand, the estimates in (4.6) suggest that the hierarchic kappa measures within both patient groups exhibit the same increasing trend. Since the estimated variances of the kappa statistics are much larger for the New Orleans patien group (due to the smaller sample size), the agreement patterns may indeed be essentiall the same in both patient group

Table 4 STATISTICAL TESTS FOR HIERARCHICAL HYPOTHESES

<table><tr><td>Hypothesis</td><td>D.F.</td><td> $Q_C$ </td></tr><tr><td colspan="3">Combined Patient Groups</td></tr><tr><td> $K_{12} = K_{11}; K_{22} = K_{21}$ </td><td>2</td><td>6.89*</td></tr><tr><td> $K_{13} = K_{12}; K_{23} = K_{22}$ </td><td>2</td><td>5.15</td></tr><tr><td> $K_{14} = K_{13}; K_{24} = K_{23}$ </td><td>2</td><td>28.13**</td></tr><tr><td colspan="3">Winnipeg Patients (1)</td></tr><tr><td> $K_{12} = K_{11}$ </td><td>1</td><td>6.20**</td></tr><tr><td> $K_{13} = K_{12}$ </td><td>1</td><td>4.38*</td></tr><tr><td> $K_{14} = K_{13}$ </td><td>1</td><td>10.96**</td></tr><tr><td colspan="3">New Orleans Patients (2)</td></tr><tr><td> $K_{22} = K_{21}$ </td><td>1</td><td>0.69</td></tr><tr><td> $K_{23} = K_{22}$ </td><td>1</td><td>0.76</td></tr><tr><td> $K_{24} = K_{23}$ </td><td>1</td><td>17.17**</td></tr></table>

\* means significant at a = 0.05;  
\*\* means significant at a = 0.01.

If the two neurologists are indeed exhibiting the same agreement patterns with respec to the weights given in Table 2 within the two groups of patients, then under (3.5) th $\left\{ \kappa _ { i h } \right\}$ satisfy the following hypotheses from (3.9)

$$
H _ {S A \mid E _ {1}}: \kappa_ {1 h} = \kappa_ {2 h} \quad \text {for} \quad h = 1, 2, 3, 4. \tag {4.8}
$$

Test statistics for these hypotheses both individually and jointly are presented in Tfable 5.

The results in Tables 4 and 5 suggest that a reduced model can be used to combin parameters which are essentially equivalent. For this purpose, the agreement statistic in (4.6) can be modeled by

$$
\mathbf {E} _ {\mathrm{A}} \{\mathbf {F} \} = \mathbf {X} \boldsymbol {\beta} = \left[ \begin{array}{l l l l l} 1 & 0 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 & 0 \\ 0 & 0 & 1 & 0 & 0 \\ 0 & 0 & 0 & 1 & 0 \\ 1 & 0 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 & 0 \\ 0 & 0 & 1 & 0 & 0 \\ 0 & 0 & 0 & 0 & 1 \end{array} \right] \left[ \begin{array}{c} \kappa_ {1} \\ \kappa_ {2} \\ \kappa_ {3} \\ \kappa_ {4} \\ \kappa_ {5} \end{array} \right], \tag {4.9}
$$

wher $\mathbf { \Sigma ^ { \scriptscriptstyle ( \textit { \iota } ) } } \mathbf { E } _ { \mathbf { A } } ^ { \mathbf { \Upsilon } , \mathfrak { \prime } } $ denotes "asymptotic expectation." For this model, the goodness-of-fit statisti is $Q = 2 . 2 7$ wit $\mathrm { d . f . } = 3$ . Thus, this reduced model provides a satisfactory characterizatio of the variation among these agreement measures. Specific test statistics for the cor responding hypotheses in (3.10) and (3.11) pertaining to the model X in (4.9) are given in Table 6. These results suggest that all the parameters are significant $( \alpha = 0 . 0 1 )$ ) differe from zero, and moreover, are significant $( \alpha = 0 . 0 5 )$ different from each other. Further more, by reducing the model to these smoothed estimates, the marginally significa $( \alpha = 0 . 1 0 )$ difference between $\kappa _ { 1 4 }$ and $\kappa _ { 2 4 }$ in Table 5 is now significa $( \alpha = 0 . 0 5 )$ for the comparison of $\kappa _ { 4 }$ and $\kappa _ { 5 }$ in this final model. Finally, the predicted values for the $\left\{ \kappa _ { i h } \right\}$ based on the fitted model (4.9) are displayed in Table 7 together with their corresponding estimate standard error

Table 5 STATISTICAL TESTS BETWEEN PATIENT SUB-POPULATIONS

<table><tr><td>Hypothesis</td><td>D.F.</td><td> $Q_{G}$ </td></tr><tr><td> $\kappa_{1h} = \kappa_{2h}$  for h = 1,2,3,4.</td><td>4</td><td>7.15</td></tr><tr><td> $\kappa_{11} = \kappa_{21}$ </td><td>1</td><td>0.90</td></tr><tr><td> $\kappa_{12} = \kappa_{22}$ </td><td>1</td><td>0.00</td></tr><tr><td> $\kappa_{13} = \kappa_{23}$ </td><td>1</td><td>0.03</td></tr><tr><td> $\kappa_{14} = \kappa_{24}$ </td><td>1</td><td>2.77</td></tr></table>

Table 6 STATISTICAL TESTS FOR MODEL X

<table><tr><td>Hypothesis</td><td>D,F,</td><td> $Q_{C}$ </td><td>Hypothesis</td><td>D.F.</td><td> $Q_{C}$ </td></tr><tr><td> $\kappa_2 = \kappa_1$ </td><td>1</td><td>5.40*</td><td> $\kappa_1 = 0$ </td><td>1</td><td>31.05**</td></tr><tr><td> $\kappa_3 = \kappa_2$ </td><td>1</td><td>4.92*</td><td> $\kappa_2 = 0$ </td><td>1</td><td>40.71**</td></tr><tr><td> $\kappa_4 = \kappa_3$ </td><td>1</td><td>12.33**</td><td> $\kappa_3 = 0$ </td><td>1</td><td>45.49**</td></tr><tr><td> $\kappa_5 = \kappa_4$ </td><td>1</td><td>4.88*</td><td> $\kappa_4 = 0$ </td><td>1</td><td>72.44**</td></tr><tr><td></td><td></td><td></td><td> $\kappa_5 = 0$ </td><td>1</td><td>94.97**</td></tr><tr><td colspan="6">* means significant at  $\alpha = 0.05$ ;** means significant at  $\alpha = 0.01$ .</td></tr></table>

Thus, these results suggest that the diagnostic criteria are not very distinct with respec to their usage by these two neurologists. In addition to bias at the macro stage, $\mathrm { i . e . , }$ con sidering only the overall marginal proportions, these observers exhibited significant dis agreement at the micro state, i.e., considering each individual subject, in specifyinga diagnosis. Only with respect to the relatively relaxed criterion corresponding to the fourt set of weights do the kappa statistics indicate a "moderate" to "substantial" level of interobservereliabilit

## 5. Discussion

In some applications, one may also be interested in a set of weights which assign varyin degrees of partial credit to the off-diagonal cells depending on the extent of the disagree ment, rather than successively combining adjoining categories as shown in Table 2. For example, the weight $w _ { \textrm { 2 j } }$ j in Table 8 are directly analogous to those discussed in Cohen [1968], Fleiss, Cohen and Everitt [1969] and Cicchetti [1972], which were used to generat weighted kappa and C statistics. For the data in Table 1, these estimates are given by

Table 7 SMOOTHED ESTIMATES OF AGREEMENT UNDER MODEL X

<table><tr><td colspan="2">Sub-population</td><td colspan="2">1</td><td colspan="2">2</td></tr><tr><td>Weights</td><td>Agreement Statistic</td><td>Estimate Under X</td><td>Estimated Standard Error</td><td>Estimate Under X</td><td>Estimated Standard Error</td></tr><tr><td> $w_{1j}$ </td><td> $\kappa_{i1}$ </td><td>0.236</td><td>0.042</td><td>0.236</td><td>0.042</td></tr><tr><td> $w_{2j}$ </td><td> $\kappa_{i2}$ </td><td>0.311</td><td>0.049</td><td>0.311</td><td>0.049</td></tr><tr><td> $w_{3j}$ </td><td> $\kappa_{i3}$ </td><td>0.383</td><td>0.057</td><td>0.383</td><td>0.057</td></tr><tr><td> $w_{4j}$ </td><td> $\kappa_{i4}$ </td><td>0.579</td><td>0.068</td><td>0.790</td><td>0.081</td></tr></table>

Table 8ALTERNATIVE WEIGHTS FOR OVERALL AGREEMENT MEASURES

<table><tr><td colspan="2">Weights</td><td colspan="4"> $^{W_{1j}}$ </td><td colspan="4"> $^{W_{2j}}$ </td></tr><tr><td>Observer</td><td></td><td colspan="4">2</td><td colspan="4">2</td></tr><tr><td></td><td>Diagnostic Class</td><td>1</td><td>2</td><td>3</td><td>4</td><td>1</td><td>2</td><td>3</td><td>4</td></tr><tr><td rowspan="4">Observer 1</td><td>1</td><td>1</td><td>0</td><td>0</td><td>0</td><td>1</td><td> $\frac{1}{2}$ </td><td> $\frac{1}{4}$ </td><td>0</td></tr><tr><td>2</td><td>0</td><td>1</td><td>0</td><td>0</td><td> $\frac{1}{2}$ </td><td>1</td><td> $\frac{1}{2}$ </td><td> $\frac{1}{4}$ </td></tr><tr><td>3</td><td>0</td><td>0</td><td>1</td><td>0</td><td> $\frac{1}{4}$ </td><td> $\frac{1}{2}$ </td><td>1</td><td> $\frac{1}{2}$ </td></tr><tr><td>4</td><td>0</td><td>0</td><td>0</td><td>1</td><td>0</td><td> $\frac{1}{4}$ </td><td> $\frac{1}{2}$ </td><td>1</td></tr></table>

$$
\mathbf {F} = \left[ \begin{array}{l} \hat {\kappa} _ {1 1} \\ \hat {\kappa} _ {1 2} \\ \hat {\kappa} _ {2 1} \\ \hat {\kappa} _ {2 2} \end{array} \right] = \left[ \begin{array}{l} 0. 2 0 8 \\ 0. 3 1 5 \\ 0. 2 9 7 \\ 0. 4 0 7 \end{array} \right], \tag {5.1}
$$

where the $\left\{ \hat { \kappa } _ { i 1 } \right\}$ estimate the perfect agreement kappa measure and the $\left\{ \hat { \kappa } _ { i 2 } \right\}$ estimat the partial-credit weighted kappa agreement measure between the two neurologists in the two patient populations. A more extensive analysis of these data under the weights in Table 8 is given in Landis [1975] and Landis et al. [1976]

Although the methodology for the assessment of observer agreement developed ill thi paper is quite general, these procedures have been illustrated with an example involvin only two observers. However, for situations in which either the number of observers d or the number of response categories L is moderately large, the number of possible mult variate response profil $r ~ = ~ \bar { L } ^ { d }$ becomes extremely large. Consequently, the matrice required to implement the GSK procedures directly may be outside the scope of computa tional feasibility. In addition, for each of the s sub-populations many of the r possibl response profiles will not necessarily be observed in the respective samples so that cor responding cell frequencies are zero. Thus, in such cases, specialized computing procedur are required to obtain the estimates of the pertinent function

One alternative approach for handling such very large contingency tables in whic most of the observed cell frequencies are zero is discussed in Landis and Koch [1977] In this regard, the same estimators which would need to be obtained from the conceptua multidimensional contingency table can be generated by first forming appropriate indicato variables of the raw data from each subject and then computing the across-subject arith metic means. Subsequent to these preliminary steps, the usual matrix operations discusse in Appendix 1 in Koch et al. [1977] can then be applied to these indicator variable mean to determine the required measures of observer agreement. These alternative computa tions involving raw data, as well as the extended GSK procedures ummarized in Appendix1 in Koch et al. [1977] can all be performed by a recently developed computer progra (GENCAT) discussed in Landis, et al. [1976]

## Acknowledgme

This research was partially supported by Research Grants GM\1-00038-20 and GM-70004-05 from the National Institute of General Medical Sciences and by the U. S. Bureau of the Census through Joint Statistical Agreements JSA 74-2 and JSA 75-2. The author would like to thank the referees for their helpful comments on an earlier draft of this paper In addition, the authors are grateful to 1\ls. Rebecca Wesson and Ms. Lynn Wilkinso for their conscientious typing of previous drafts of this paper, and to 1\ls. Linda L. Blakle and 1\ls. Connie M\'Jassey for their efficient yping of the final version of this manuscrip

La Mlesure de la Concordance Entre Observations pour des Donnees en Categori

## Resume

Particle expose une methodologie statistique generate pour l'analyse de donnees mult variates en categories provenant d'etudes de fiabilite d'observateurs. La procedure fait principal ment appel a la construction de fonctions des proportions observees traduisant la concordan des observateurs entre eux et a la construction de statistiques de tests pour des hypotheses impli quant ces fonctions. On present des tests pour des biais entre observateurs en fonction de l'hom geneit' marginale du premier ordre et on construit des mesures de concordance entre observateu comme des statistiques generalisant celies du type kappa. On illustre ces procedures avec an exemple de diagnostic linique provenant de la litterature epidemiologiq

## Referenc

Anderson, R. L. and Bancroft, T. A. [1952]. Statistical Theory in Research. McGraw Hill, New York  
Bhapkar, V. P. [1966]. A note on the equivalence of two test criteria for hypotheses in categori data. Journal of the American Statistical Association 61, 228-23  
Bhapkar, V. P. [1968]. On the analysis of contingency tables with a quantitative response. Biometri 24, 329-33  
Bhapkar, V. P. and Koch, G. G. [1968a]. Hypotheses of "no interaction" in multidimensional con tingency tables. Technometrics 10, 107-12  
Bhapkar, V. P. and Koch, G. G. [1968b]. On the hypotheses of "no interaction" in contingency table Biometrics 24, 567-59  
Ciechetti, D. V. [1972]. A new measure of agreement between rank-ordered variables. Proceedin 80th Annual Convention, APA, 17-18  
Cohen, J. [1960]. A coefficient of agreement for nominal scales. Educational and Psychological Measur ment 20, 37-46  
Cohen, J. [1968]. Weighted kappa: nominal scale agreement with provision for scaled disagreeme or partial credit. Psychological Bulletin 70, 213-22  
Fleiss, J. L. [1966]. Assessing the accuracy of multivariate observations. Journal of the Americ Statistical Association 61, 403-41  
Fleiss, J. L., Cohen, J. and Everitt, B. S. [1969]. Large sample standard errors of kappa and weight kappa. Psychological Bulletin 72, 323-33  
Fleiss, J. L. [1971]. Measuring nominal scale agreement among many raters. Psychological Bulleti 76, 378-38  
Fleiss, J. L. and Cohen, J. [1973]. The equivalence of weighted kappa and the intraclass correlati coefficient as measures of reliability. Educational and Psychological Measurement 33, 613-61  
Fleiss, J. L. [1975]. Measuring agreement between two judges on the presence or absence of a trai Biometrics 31, 651-65  
Forthofer, R. N. and Koch, G. G. [1973]. An analysis for compounded functions of categorical data Biometrics 29, 143-15  
Grizzle, J. E., Starmer, C. F. and Koch, G. G. [1969]. Analysis of categorical data by linear model Biometrics 25, 489-50  
Grubbs, F. E. [1948]. On estimating precision of measuring instruments and product variability Journal of the American Statistical Association 43, 243-264  
Grubbs, F. E. [1973]. Errors of measurement, precision, accuracy and the statistical comparison of measuring instruments. Technometrics 15, 53-66.  
Goodman, L. A. and Kruskal, W. H. [1954]. Measures of association for cross classification. Journa of the American Statistical Association 49, 732-764  
Koch, G. G. [1967]. A general approach to the estimation of variance components. Technometrics9, 93-118.  
Koch, G. G. [1968]. Some further remarks concerning "A general approach to the estimation of variance components." Technometrics 10, 551-558.  
Koch, G. G. and Reinfurt, D. W. [1971]. The analysis of categorical data from mixed models. Biometrics 27, 157-173  
Koch, G. G., Landis, J. R., Freeman, J. L., Freeman, D. H., Jr. and Lehnen, R. G. [1977]. A genera methodology for the analysis of experiments with repeated measurement of categorical data. Biometrics 33, 133-158  
Landis, J. R. [1975]. A general methodology for the measurement of observer agreement when the data are categorical. Ph.D. Dissertation, University of North Carolina, Institute of Statistic Mimeo Series No. 1022  
Landis, J. R. and Koch, G. G. [1975a]. A review of statistical methods in the analysis of data arisin from observer reliability studies (Part I). Statistica Neerlandica 29, 101-123  
Landis, J. R. and Koch, G. G. [1975b]. A review of statistical methods in the analysis of data arisin from observer reliability studies (Part II). Statistica Neerlandica 29, 151-161  
Landis, J. R. and Koch, G. G. [1977]. An application of hierarchical kappa-type statistics in the assessment of majority agreement among multiple observers. Accepted for publication in Bio metric  
Landis, J. R., Stanish, W. M., Freeman, J. L. and Koch, G. G. [1976]. A computer program for the generalized chi-square analysis of categorical data using weighted least squares (GENCAT). University of Michigan Biostatistics Technical Report No. 8. Accepted for publication in Com puter Programs in Biomedicin  
Light, R. J. [1971]. Measures of response agreement for qualitative data: some generalizations and alternatives. Psychological Bulletin 76, 365-377.  
Loewenson, R. B., Bearman, J. E. and Resch, J. A. [1972]. Reliability of measurements for studie of cardiovascular atherosclerosis. Biometrics 28, 557-569  
Mandel, J. [1959]. The measuring process. Technometrics 1, 251-267.  
Neyman, J. [1949]. Contribution to the theory of the X2 test. Proceedings of the Berkeley Symposiu on mathematical statistics and probability, Berkeley and Los Angeles, University of Californi Press, 239-272  
Overall, J. E. [1968]. Estimating individual rater reliabilities from analysis of treatment effect Educational and Psychological Measurement 28, 255-264  
Scheff6, H. [1959]. The Analysis of Variance. Wiley, New York  
Searle, S. R. [1971]. Linear Models. Wiley, New York.  
Wald, A. [1943]. Tests of statistical hypotheses concerning general parameters when the number of observations is large. Transactions of the American Mathematical Society 54, 426-482  
Westlund, K. B. and Kurland, L. T. [1953]. Studies on multiple sclerosis in Winnipeg. Manitoba and New Orleans, Louisiana. American Journal of Hygiene 57, 380-396.

Received April 1975, Revised November 1975