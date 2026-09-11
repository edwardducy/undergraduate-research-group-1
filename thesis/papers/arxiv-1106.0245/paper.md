# **A Model of Inductive Bias Learning** 

### **Jonathan Baxter** 

JONATHAN.BAXTER@ANU.EDU.AU 

_Research School of Information Sciences and Engineering Australian National University, Canberra 0200, Australia_ 

## **Abstract** 

A major problem in machine learning is that of inductive bias: how to choose a learner’s hypothesis space so that it is large enough to contain a solution to the problem being learnt, yet small enough to ensure reliable generalization from reasonably-sized training sets. Typically such bias is supplied by hand through the skill and insights of experts. In this paper a model for _automatically learning_ bias is investigated. The central assumption of the model is that the learner is embedded within an _environment_ of related learning tasks. Within such an environment the learner can sample from multiple tasks, and hence it can search for a hypothesis space that contains good solutions to many of the problems in the environment. Under certain restrictions on the set of all hypothesis spaces available to the learner, we show that a hypothesis space that performs well on a sufficiently large number of training tasks will also perform well when learning novel tasks in the same environment. Explicit bounds are also derived demonstrating that learning multiple tasks within an environment of related tasks can potentially give much better generalization than learning a single task. 

## **1. Introduction** 

Often the hardest problem in any machine learning task is the initial choice of hypothesis space; it has to be large enough to contain a solution to the problem at hand, yet small enough to ensure good generalization from a small number of examples (Mitchell, 1991). Once a suitable bias has been found, the actual learning task is often straightforward. Existing methods of bias generally require the input of a human expert in the form of heuristics and domain knowledge (for example, through the selection of an appropriate set of features). Despite their successes, such methods are clearly limited by the accuracy and reliability of the expert’s knowledge and also by the extent to which that knowledge can be transferred to the learner. Thus it is natural to search for methods for _automatically learning_ the bias. 

Of course, the best way to bias the learner is to supply it with an H containing just a single optimal hypothesis. But finding such a hypothesis is precisely the original learning problem, so in the 

> � c 2000 AI Access Foundation and Morgan Kaufmann Publishers. All rights reserved. 

In this paper we introduce and analyze a formal model of _bias learning_ that builds upon the PAC model of machine learning and its variants (Vapnik, 1982; Valiant, 1984; Blumer, Ehrenfeucht, Haussler, & Warmuth, 1989; Haussler, 1992). These models typically take the following general form: the learner is supplied with a hypothesis space H and training data z = f(x1 ; y1 ); : : : ; (xm ; ym )g drawn independently according to some underlying distribution P on X � Y . Based on the information contained in z , the learner’s goal is to select a hypothesis h : X ! Y from H minimizing some measure erP (h) of expected loss with respect to P (for example, in the case of squared loss erP (h) := E ( x;y )�P ( h(x) � y )2 ). In such models the learner’s bias is represented by the choice of H ; if H does not contain a good solution to the problem, then, regardless of how much data the learner receives, it cannot learn. 

PAC model there is no distinction between bias learning and ordinary learning. Or put differently, the PAC model does not model the process of inductive bias, it simply takes the hypothesis space H as given and proceeds from there. To overcome this problem, in this paper we assume that instead of being faced with just a single learning task, the learner is embedded within an _environment_ of related learning tasks. The learner is supplied with a _family_ of hypothesis spaces H = fH g , and its goal is to find a bias (i.e. hypothesis space H 2 H ) that is appropriate for the entire environment. A simple example is the problem of handwritten character recognition. A preprocessing stage that identifies and removes any (small) rotations, dilations and translations of an image of a character will be advantageous for recognizing all characters. If the set of all individual character recognition problems is viewed as an environment of learning problems (that is, the set of all problems of the form “distinguish ‘A’ from all other characters”, “distinguish ‘B’ from all other characters”, and so on), this preprocessor represents a bias that is appropriate for all problems in the environment. It is likely that there are many other currently unknown biases that are also appropriate for this environment. We would like to be able to learn these automatically. 

There are many other examples of learning problems that can be viewed as belonging to environments of related problems. For example, each individual face recognition problem belongs to an (essentially infinite) set of related learning problems (all the other individual face recognition problems); the set of all individual spoken word recognition problems forms another large environment, as does the set of all fingerprint recognition problems, printed Chinese and Japanese character recognition problems, stock price prediction problems and so on. Even medical diagnostic and prognostic problems, where a multitude of diseases are predicted from the same pathology tests, constitute an environment of related learning problems. 

In many cases these “environments” are not normally modeled as such; instead they are treated as single, multiple category learning problems. For example, recognizing a group of faces would normally be viewed as a single learning problem with multiple class labels (one for each face in the group), not as multiple individual learning problems. However, if a reliable classifier for each individual face in the group can be constructed then they can easily be combined to produce a classifier for the whole group. Furthermore, by viewing the faces as an environment of related learning problems, the results presented here show that bias can be learnt that will be good for learning novel faces, a claim that cannot be made for the traditional approach. 

This point goes to the heart of our model: we are not not concerned with adjusting a learner’s bias so it performs better on some _fixed_ set of learning problems. Such a process is in fact just ordinary learning but with a richer hypothesis space in which some components labelled “bias” are also able to be varied. Instead, we suppose the learner is faced with a (potentially infinite) stream of tasks, and that by adjusting its bias on some subset of the tasks it improves its learning performance on future, as yet unseen tasks. 

Bias that is appropriate for all problems in an environment must be learnt by sampling from many tasks. If only a single task is learnt then the bias extracted is likely to be specific to that task. In the rest of this paper, a general theory of bias learning is developed based upon the idea of learning multiple related tasks. Loosely speaking (formal results are stated in Section 2), there are two main conclusions of the theory presented here: 

- Learning multiple related tasks reduces the sampling burden required for good generalization, at least on a number-of-examples-required-per-task basis. 

150 

- Bias that is learnt on sufficiently many training tasks is likely to be good for learning novel tasks drawn from the same environment. 

The second point shows that a form of _meta-generalization_ is possible in bias learning. Ordinarily, we say a learner generalizes well if, after seeing sufficiently many training examples, it produces a hypothesis that with high probability will perform well on future examples of the same task. However, a bias learner generalizes well if, after seeing sufficiently many training _tasks_ it produces a _hypothesis space_ that with high probability contains good solutions to novel tasks. Another term that has been used for this process is _Learning to Learn_ (Thrun & Pratt, 1997). 

Our main theorems are stated in an agnostic setting (that is, H does not necessarily contain a hypothesis space with solutions to all the problems in the environment), but we also give improved bounds in the realizable case. The sample complexity bounds appearing in these results are stated in terms of combinatorial parameters related to the complexity of the set of all hypothesis spaces H available to the bias learner. For Boolean learning problems (pattern classification) these parameters are the bias learning analogue of the _Vapnik-Chervonenkis dimension_ (Vapnik, 1982; Blumer et al., 1989). As an application of the general theory, the problem of learning an appropriate set of neuralnetwork features for an environment of related tasks is formulated as a bias learning problem. In the case of continuous neural-network features we are able to prove upper bounds on the number of training tasks and number of examples of each training task required to ensure a set of features that works well for the training tasks will, with high probability, work well on novel tasks drawn from the same environment. The upper bound on the number of tasks scales as O (b) where b is a measure of the complexity of the possible feature sets available to the learner, while the upper bound on the number of examples of each task scales as O (a + b=n) where O (a) is the number of examples required to learn a task if the “true” set of features (that is, the correct bias) is already known, and n is the number of tasks. Thus, in this case we see that as the number of related tasks learnt increases, the number of examples required of each task for good generalization decays to the minimum possible. For Boolean neural-network feature maps we are able to show a matching lower bound on the number of examples required per task of the same form. 

### **1.1 Related Work** 

There is a large body of previous algorithmic and experimental work in the machine learning and statistics literature addressing the problems of inductive bias learning and improving generalization through multiple task learning. Some of these approaches can be seen as special cases of, or at least closely aligned with, the model described here, while others are more orthogonal. Without being completely exhaustive, in this section we present an overview of the main contributions. See Thrun and Pratt (1997, chapter 1) for a more comprehensive treatment. 

- **Hierarchical Bayes.** The earliest approaches to bias learning come from Hierarchical Bayesian methods in statistics (Berger, 1985; Good, 1980; Gelman, Carlin, Stern, & Rubim, 1995). In contrast to the Bayesian methodology, the present paper takes an essentially empirical process approach to modeling the problem of bias learning. However, a model using a mixture of hierarchical Bayesian and information-theoretic ideas was presented in Baxter (1997a), with similar conclusions to those found here. An empirical study showing the utility of the hierarchical Bayes approach in a domain containing a large number of related tasks was given in Heskes (1998). 

151 

- **Early machine learning work.** In Rendell, Seshu, and Tcheng (1987) “VBMS” or _Variable Bias Management System_ was introduced as a mechanism for selecting amongst different learning algorithms when tackling a new learning problem. “STABB” or _Shift To a Better Bias_ (Utgoff, 1986) was another early scheme for adjusting bias, but unlike VBMS, STABB was not primarily focussed on searching for bias applicable to large problem domains. Our use of an “environment of related tasks” in this paper may also be interpreted as an “environment of analogous tasks” in the sense that conclusions about one task can be arrived at by analogy with (sufficiently many of) the other tasks. For an early discussion of analogy in this context, see Russell (1989, S4.3), in particular the observation that for analogous problems the sampling burden _per task_ can be reduced. 

- **Metric-based approaches.** The metric used in nearest-neighbour classification, and in vector quantization to determine the nearest code-book vector, represents a form of inductive bias. Using the model of the present paper, and under some extra assumptions on the tasks in the environment (specifically, that their marginal input-space distributions are identical and they only differ in the conditional probabilities they assign to class labels), it can be shown that there is an _optimal_ metric or distance measure to use for vector quantization and onenearest-neighbour classification (Baxter, 1995a, 1997b; Baxter & Bartlett, 1998). This metric can be learnt by sampling from a subset of tasks from the environment, and then used as a distance measure when learning novel tasks drawn from the same environment. Bounds on the number of tasks and examples of each task required to ensure good performance on novel tasks were given in Baxter and Bartlett (1998), along with an experiment in which a metric was successfully trained on examples of a subset of 400 Japanese characters and then used as a fixed distance measure when learning 2600 as yet unseen characters. 

A similar approach is described in Thrun and Mitchell (1995), Thrun (1996), in which a neural network’s output was trained to match labels on a novel task, while simultaneously being forced to match its gradient to _derivative_ information generated from a distance metric trained on previous, related tasks. Performance on the novel tasks improved substantially with the use of the derivative information. 

Note that there are many other adaptive metric techniques used in machine learning, but these all focus exclusively on adjusting the metric for a fixed set of problems rather than learning a metric suitable for learning novel, related tasks (bias learning). 

- **Feature learning or learning internal representations.** As with adaptive metric techniques, there are many approaches to feature learning that focus on adapting features for a fixed task rather than learning features to be used in novel tasks. One of the few cases where features have been learnt on a subset of tasks with the explicit aim of using them on novel tasks was Intrator and Edelman (1996) in which a low-dimensional representation was learnt for a set of multiple related image-recognition tasks and then used to successfully learn novel tasks of the same kind. The experiments reported in Baxter (1995a, chapter 4) and Baxter (1995b), Baxter and Bartlett (1998) are also of this nature. 

- **Bias learning in Inductive Logic Programming (ILP).** Predicate invention refers to the process in ILP whereby new predicates thought to be useful for the classification task at hand are added to the learner’s domain knowledge. By using the new predicates as background domain knowledge when learning novel tasks, predicate invention may be viewed as a form of 

152 

inductive bias learning. Preliminary results with this approach on a chess domain are reported in Khan, Muggleton, and Parson (1998). 

- **Improving performance on a fixed reference task.** “Multi-task learning” (Caruana, 1997) trains extra neural network outputs to match related tasks in order to improve generalization performance on a fixed reference task. Although this approach does not explicitly identify the extra bias generated by the related tasks in a way that can be used to learn novel tasks, it is an example of exploiting the bias provided by a set of related tasks to improve generalization performance. Other similar approaches include Suddarth and Kergosien (1990), Suddarth and Holden (1991), Abu-Mostafa (1993). 

- **Bias as computational complexity.** In this paper we consider inductive bias from a samplecomplexity perspective: how does the learnt bias decrease the number of examples required of novel tasks for good generalization? A natural alternative line of enquiry is how the runningtime or computational complexity of a learning algorithm may be improved by training on related tasks. Some early algorithms for neural networks in this vein are contained in Sharkey and Sharkey (1993), Pratt (1992). 

- **Reinforcement Learning.** Many control tasks can appropriately be viewed as elements of sets of related tasks, such as learning to navigate to different goal states, or learning a set of complex motor control tasks. A number of papers in the reinforcement learning literature have proposed algorithms for both sharing the information in related tasks to improve average generalization performance across those tasks Singh (1992), Ring (1995), or learning bias from a set of tasks to improve performance on future tasks Sutton (1992), Thrun and Schwartz (1995). 

### **1.2 Overview of the Paper** 

In Section 2 the bias learning model is formally defined, and the main sample complexity results are given showing the utility of learning multiple related tasks and the feasibility of bias learning. These results show that the sample complexity is controlled by the size of certain covering numbers associated with the set of all hypothesis spaces available to the bias learner, in much the same way as the sample complexity in learning Boolean functions is controlled by the _Vapnik-Chervonenkis_ dimension (Vapnik, 1982; Blumer et al., 1989). The results of Section 2 are upper bounds on the sample complexity required for good generalization when learning multiple tasks and learning inductive bias. 

The general results of Section 2 are specialized to the case of feature learning with neural networks in Section 3, where an algorithm for training features by gradient descent is also presented. For this special case we are able to show matching lower bounds for the sample complexity of multiple task learning. In Section 4 we present some concluding remarks and directions for future research. Many of the proofs are quite lengthy and have been moved to the appendices so as not to interrupt the flow of the main text. 

The following tables contain a glossary of the mathematical symbols used in the paper. 

153 

Symbol Description First Referenced<br>X Input Space 155<br>Y Output Space 155<br>P Distribution on X � Y (learning task) 155<br>l Loss function 155<br>H Hypothesis Space 155<br>h Hypothesis 155<br>erP (h) Error of hypothesis h  on distribution P 156<br>z Training set 156<br>A Learning Algorithm 156<br>er^ z (h) Empirical error of h  on training set z 156<br>P Set of all learning tasks P 157<br>Q Distribution over learning tasks 157<br>H Family of hypothesis spaces 157<br>erQ (H ) Loss of hypothesis space H  on environment Q 158<br>z (n; m) -sample 158<br>er^ z (H ) Empirical loss of H  on z 158<br>A Bias learning algorithm 159<br>hl Function induced by h  and l 159<br>Hl Set of hl 159<br>(h1 ; : : : ; hn )l Average of h1;l ; : : : ; hn;l 159<br>hl Same as (h1 ; : : : ; hn )l 159<br>n<br>Hl Set of (h1 ; : : : ; hn )l 159<br>H nl Set of Hln 159<br>H � Function on probability distributions 160<br>H � Set of H � 160<br>dP Pseudo-metric on Hln 160<br>d Pseudo-metric on H � 160<br>Q<br>N ("; H � ; dQ ) Covering number of H � 160<br>C ("; H � ) Capacity of H � 160<br>N ("; H nl ; dP ) Covering number of H nl 160<br>C ("; H nl ) Capacity of H nl 160<br>h Sequence of n  hypotheses (h1 ; : : : ; hn ) 163<br>P Sequence of n  distributions (P1 ; : : : ; Pn ) 163<br>erP (h) Average loss of h  on P 164<br>er^ z (h) Average loss of h  on z 164<br>F Set of feature maps 166<br>G Output class composed with feature maps f 166<br>G Æ f Hypothesis space associated with f 166<br>Gl Loss function class associated with G 166<br>N ("; Gl ; dP ) Covering number of Gl 166<br>C ( "; Gl ) Capacity of Gl 166<br>d[P ;Gl ℄ (f ; f 0 ) Pseudo-metric on feature maps f ; f 0 166<br>N ("; F ; d[P ;Gl ℄ ) Covering number of F 166<br>
154 



A MODEL OF INDUCTIVE BIAS LEARNING<br>


Symbol Description First Referenced<br>N ("; F ; d[P ;Gl ℄ ) Covering number of F 166<br>CGl ("; F ) Capacity of F 166<br>Hw Neural network hypothesis space 167<br>H H  restricted to vector x 172<br>jx<br>�H (m) Growth function of H 172<br>VCdim(H ) Vapnik-Chervonenkis dimension of H 172<br>H H  restricted to matrix x 173<br>jx<br>H H restricted to matrix x 173<br>jx<br>� H (n; m) Growth function of H 173<br>d H (n) Dimension function of H 173<br>d( H ) Upper dimension function of H 173<br>d( H ) Lower dimension function of H 173<br>optP ( H n ) Optimal performance of H n  on P 175<br>d� Metric on R + 179<br>h1 � � � � � hn Average of h1 , : : : , hn 179<br>H1 � � � � � Hn Set of h1 � � � � � hn 180<br>�(2m;n) Permutations on integer pairs 182<br>z� Permuted z 182<br>dz (h; h0 ) Empirical l1 metric on functions h 182<br>er^ P (H ) Optimal average error of H  on P 185<br>
## **2. The Bias Learning Model** 



In this section the bias learning model is formally introduced. To motivate the definitions, we first<br>describe the main features of ordinary (single-task) supervised learning models.<br>
### **2.1 Single-Task Learning** 



Computational learning theory models of supervised learning usually include the following ingre-<br>dients:<br>


� An  input space X and an  output space Y  ,<br>� a  probability distribution P on X � Y  ,<br>� a  loss function l : Y � Y ! R , and<br>� a  hypothesis space H  which is a set of  hypotheses  or functions h : X ! Y  .<br>


As an example, if the problem is to learn to recognize images of Mary’s face using a neural network,<br>then X  would be the set of all images (typically represented as a subset of R d where each component<br>is a pixel intensity), Y would be the set f0; 1g , and the distribution P would be peaked over images<br>of different faces and the correct class labels. The learner’s hypothesis space H  would be a class of<br>neural networks mapping the input space R d to f0;0;; 1gg . The loss in this case would be discrete loss:<br>


R d to f0;0;; 1gg . The loss in this case would be discrete loss:<br>0 1 if y 6= y 0<br>l (y ; y ) := � 0 if y = y 0 (1)<br>155<br>
BAXTER 

= Using the loss function allows us to present a unified treatment of both pattern recognition ( Y f0; 1g , l as above), and real-valued function learning ( _e.g._ regression) in which Y = R and usually l (y ; y 0 ) = (y � y 0 )2 . The goal of the learner is to select a hypothesis h 2 H with minimum _expected loss_ : erP (h) := l (h(x); y ) dP (x; y ): (2) ZX �Y Of course, the learner does not know P and so it cannot search through H for an h minimizing erP (h) . In practice, the learner samples repeatedly from X � Y according to the distribution P to generate a _training set_ z := f(x1 ; y1 ); : : : ; (xm ; ym )g: (3) Based on the information contained in z the learner produces a hypothesis h 2 H . Hence, in general a learner is simply a map A from the set of all training samples to the hypothesis space H : A : [ (X � Y )m ! H m>0 (stochastic learner’s can be treated by assuming a distribution-valued A .) Many algorithms seek to minimize the _empirical_ loss of h on z , where this is defined by: er^ z (h) := m1 Xm l (h(xi ); yi ): (4) i=1 Of course, there are more intelligent things to do with the data than simply minimizing empirical error—for example one can add regularisation terms to avoid over-fitting. However the learner chooses its hypothesis h , if we have a _uniform_ bound (over all h 2 H ) on the probability of large deviation between er^ z (h) and erP (h) , then we can bound the learner’s generalization error erP (h) as a function of its empirical loss on the training set er^ z (h) . Whether such a bound holds depends upon the “richness” of H . The conditions ensuring convergence between ^ erz (h) and erP (h) are by now well understood; for Boolean function learning ( Y = f0; 1g , discrete loss), convergence is controlled by the _VC-dimension_<sup>1</sup> of H : **Theorem 1.** _Let_ P _be any probability distribution on_ X � f0; 1g _and suppose_ z = f(x1 ; y1 ); : : : ; (xm ; ym )g _is generated by sampling_ m _times from_ X � f0; 1g _according to_ P _. Let_ d := VCdim(H ) _. Then with probability at least_ 1 � Æ _(over the choice of the training set_ z _),_ **all** h 2 H _will satisfy_ 1=2 erP (h) � er^ z (h) + 32 d log 2em + log 4 (5) � m � d Æ �� Proofs of this result may be found in Vapnik (1982), Blumer et al. (1989), and will not be reproduced here. 1. The VC dimension of a class of Boolean functions H is the largest integer d such that there exists a subset S := fx1 ; : : : ; xd g � X such that the restriction of H to S contains all 2d Boolean functions on S . 

156 

Theorem 1 only provides conditions under which the deviation between erP (h) and er^ z (h) is likely to be small, it does not guarantee that the true error erP (h) will actually be small. This is governed by the choice of H . If H contains a solution with small error and the learner minimizes error on the training set, then with high probability erP (h) will be small. However, a bad choice of H will mean there is no hope of achieving small error. Thus, the _bias_ of the learner in this model<sup>2</sup> is represented by the choice of hypothesis space H . 

### **2.2 The Bias Learning Model** 

The main extra assumption of the bias learning model introduced here is that the learner is embedded in an _environment_ of related tasks, and can sample from the environment to generate multiple training sets belonging to multiple different tasks. In the above model of ordinary (single-task) learning, a learning task is represented by a distribution P on X � Y . So in the bias learning model, an environment of learning problems is represented by a pair (P ; Q) where P is the set of all probability distributions on X � Y (i.e., P is the set of all possible learning problems), and Q is a distribution on P . Q controls which learning problems the learner is likely to see<sup>3</sup> . For example, if the learner is in a face recognition environment, Q will be highly peaked over face-recognition-type problems, whereas if the learner is in a character recognition environment Q will be peaked over character-recognition-type problems (here, as in the introduction, we view these environments as sets of individual classification problems, rather than single, multiple class classification problems). Recall from the last paragraph of the previous section that the learner’s bias is represented by its choice of hypothesis space H . So to enable the learner to learn the bias, we supply it with a _family_ or set of hypothesis spaces H := fH g . 

Putting all this together, formally a _learning to learn_ or _bias learning_ problem consists of: 

- an _input space_ X and an _output space_ Y (both of which are separable metric spaces), 

- � a _loss function_ l : Y � Y ! R , � an _environment_ (P ; Q) where P is the set of all probability distributions on X � Y and Q is a distribution on P , 

- � a _hypothesis space family_ H = fH g where each H 2 H is a set of functions h : X ! Y . From now on we will assume the loss function l has range [0; 1℄ , or equivalently, with rescaling, 

- we assume that l is bounded. 

2. The bias is also governed by how the learner uses the hypothesis space. For example, under some circumstances the learner may choose not to use the full power of H (a neural network example is early-stopping). For simplicity in this paper we abstract away from such features of the algorithm A and assume that it uses the entire hypothesis space H . 

3. Q ’s domain is a � -algebra of subsets of P . A suitable one for our purposes is the Borel � -algebra B (P ) generated by the topology of weak convergence on P . If we assume that X and Y are separable metric spaces, then P is also a separable metric space in the Prohorov metric (which metrizes the topology of weak convergence) (Parthasarathy, 1967), so there is no problem with the existence of measures on B (P ) . See Appendix D for further discussion, particularly the proof of part 5 in Lemma 32. 

157 

We define the goal of a bias learner to be to find a hypothesis space H 2 H minimizing the following loss: erQ (H ) := inf erP (h) dQ(P ) (6) ZP h2H = inf l (h(x); y ) dP (x; y ) dQ(P ): ZP h2H ZX �Y The only way erQ (H ) can be small is if, with high Q -probability, H contains a good solution h to any problem P drawn at random according to Q . In this sense erQ (H ) measures how appropriate the bias embodied by H is for the environment (P ; Q) . In general the learner will not know Q , so it will not be able to find an H minimizing erQ (H ) directly. However, the learner can sample from the environment in the following way: � Sample n times from P according to Q to yield: P1 ; : : : ; Pn<sup>.</sup> � Sample m times from X � Y according to each Pi<sup>to yield:</sup> zi = f(xi1 ; yi1 ) : : : ; (xim ; yim )g . � The resulting n training sets—henceforth called an (n; m) _-sample_ if they are generated by the above process—are supplied to the learner. In the sequel, an (n; m) -sample will be denoted by z and written as a matrix: (x11 ; y11 ) � � � (x1m ; y1m ) = z1 z := ... ... ... ... (7) (xn1 ; yn1 ) � � � (xnm ; ynm ) = zn An (n; m) -sample is simply n training sets z1 ; : : : ; zn<sup>sampledfrom</sup> n different learning tasks P1 ; : : : ; Pn<sup>, where eachtask is selectedaccordingto the environmentalprobabilitydistribution</sup> Q . The size of each training set is kept the same primarily to facilitate the analysis. Based on the information contained in z , the learner must choose a hypothesis space H 2 H . One way to do this would be for the learner to find an H minimizing the _empirical loss_ on z , where this is defined by: er^ z (H ) := n1 Xn hinf2H er^ zi (h) (8) i=1 Note that er^ z (H ) is simply the average of the best possible empirical error achievable on each training set zi<sup>,usingafunctionfrom</sup> H . It is a biased estimate of erQ (H ) . An unbiased estimate of erQ (H ) would require choosing an H with minimal average error over the n distributions P1 ; : : : ; Pn<sup>, where this is defined by</sup> n1 Pni=1 inf h2H erPi (h) . As with ordinary learning, it is likely there are more intelligent things to do with the training data z than minimizing (8). Denoting the set of all (n; m) -samples by ( X � Y )(n;m) , a general “bias learner” is a map A that takes (n; m) -samples as input and produces hypothesis spaces H 2 H as output: A : [ (X � Y ) (n;m) ! H : (9) n>0 m>0 

158 



A MODEL OF INDUCTIVE BIAS LEARNING<br>


(as stated, A  is a deterministic bias learner, however it is trivial to extend our results to stochastic<br>learners).<br>


Note that in this paper we are concerned only with the sample complexity properties of a bias<br>learner A ; we do not discuss issues of the computability of A .<br>


Since A  is searching for entire hypothesis spaces H  within a family of such hypothesis spaces<br>H , there is an extra representational question in our model of bias learning that is not present in<br>ordinary learning, and that is how the family H is represented and searched by A . We defer this<br>discussion until Section 2.5, after the main sample complexity results for this model of bias learning<br>have been introduced. For the specific case of learning a set of features suitable for an environment<br>of related learning problems, see Section 3.<br>Regardless of how the learner chooses its hypothesis space H , if we have a uniform bound (over<br>all H 2 H ) on the probability of large deviation between er^ z (H )  and erQ (H ) , and we can compute<br>an upper bound on er^ z (H ) , then we can bound the bias learner’s “generalization error” erQ (H ) .<br>With this view, the question of generalization within our bias learning model becomes: how many<br>tasks ( n ) and how many examples of each task ( m ) are required to ensure that er^ z (H )  and erQ (H )<br>are close with high probability, uniformly over all H 2 H ? Or, informally, how many tasks and how<br>many examples of each task are required to ensure that a hypothesis space with good solutions to<br>all the training tasks will contain good solutions to novel tasks drawn from the same environment?<br>


It turns out that this kind of uniform convergence for bias learning is controlled by the “size”<br>of certain function classes derived from the hypothesis space family H , in much the same way as<br>the VC-dimension of a hypothesis space H controls uniform convergence in the case of Boolean<br>function learning (Theorem 1). These “size” measures and other auxiliary definitions needed to<br>state the main theorem are introduced in the following subsection.<br>2.3 Covering Numbers<br>Definition 1. For any hypothesis h : X ! Y  , define hl : X � Y ! [0; 1℄ by<br>hl (x; y ) := l (h(x); y ) (10)<br>For any hypothesis space H  in the hypothesis space family H , define<br>Hl := fhl : h 2 H g: (11)<br>For any sequence of n  hypotheses (h1 ; : : : ; hn ) , define (h1 ; : : : ; hn )l : (X � Y )n ! [0; 1℄ by<br>1 n<br>(h1 ; : : : ; hn )l (x1 ; y1 ; : : : ; xn ; yn ) := n X l (hi (xi ); yi ): (12)<br>i=1<br>We will also use hl to denote (h1 ; : : : ; hn )l . For any H  in the hypothesis space family H , define<br>n<br>Hl := f(h1 ; : : : ; hn )l : h1 ; : : : ; hn 2 H g: (13)<br>Define<br>n n<br>H l := [ Hl : (14)<br>H2 H<br>


159<br>
BAXTER 



where now the supremum is over all sequences of n  probability measures on<br>


X � Y  .<br>


4. A pseudo-metric d  is a metric without the condition that<br>
In the first part of the definition above, hypotheses h : X ! Y are turned into functions hl mapping X � Y ! [0; 1℄ by composition with the loss function. Hl<sup>is then just the collection of all</sup> such functions where the original hypotheses come from H . Hl<sup>is often called a</sup><sup>_loss-function class_.</sup> In our case we are interested in the average loss across n tasks, where each of the n hypotheses is chosen from a fixed hypothesis space H . This motivates the definition of hl<sup>and</sup> Hln<sup>.Finally,</sup> H nl<sup>is thecollectionof all</sup> (h1 ; : : : ; hn )l<sup>,withthe restrictionthat all</sup> h1 ; : : : ; hn<sup>belongto a single</sup> hypothesis space H 2 H . � **Definition 2.** _For each_ H 2 H _, define_ H : P ! [0; 1℄ _by_ � H (P ) := inf erP (h): (15) h2H _For the hypothesis space family_ H _, define_ H � := fH � : H 2 H g: (16) It is the “size” of H nl<sup>and</sup> H � that controls how large the (n; m) -sample z must be to ensure er^ z (H ) and erQ (H ) are close uniformly over all H 2 H . Their size will be defined in terms of certain covering numbers, and for this we need to define how to measure the distance between elements of H nl<sup>and also between elements of</sup> H � . **Definition 3.** _Let_ P = (P1 ; : : : ; Pn ) _be any sequence of_ n _probability distributions on_ X � Y _. For any_ hl ; h0l 2 H nl<sup>_, define_</sup> dP (hl ; h0l ) := jhl (x1 ; y1 ; : : : ; xn ; yn ) � h0l (x1 ; y1 ; : : : ; xn ; yn )j Z(X �Y )n (17) dP1 (x1 ; y1 ) : : : dPn (xn ; yn ) _Similarly, for any distribution_ Q _on_ P _and any_ H1� ; H2� 2 H � _, define_ dQ (H1� ; H2� ) := jH1� (P ) � H2� (P )j dQ(P ) (18) ZP It is easily verified that dP<sup>and</sup> dQ<sup>are pseudo-metrics4on</sup> H nl<sup>and</sup> H � respectively. **Definition 4.** _An_ " _-cover of_ ( H � ; dQ ) _is a set_ fH1� ; : : : ; HN� g _such that for all_ H � 2 H � _,_ dQ (H � ; Hi� ) � " _for some_ i = 1 : : : N _. Note that we do not require the_ Hi�<sup>_tobecontainedin_</sup> H � _, just that they be measurable functions on_ P _. Let_ N ("; H � ; dQ ) _denote the size of the smallest such cover. Define the_ capacity _of_ H � _by_ C ("; H � ) := sup N ("; H � ; dQ ) (19) Q _where the supremum is over all probability measures on_ P _._ N ("; H nl ; dP ) _is defined in a similar way, using_ dP<sup>_in place of_</sup> dQ<sup>_.Define the_capacity</sup><sup>_of_</sup> H nl<sup>_by:_</sup> C ("; H nl ) := sup N ("; H nl ; dP ) (20) P 

d(x; y ) = 0 ) x = y . 

160 

### **2.4 Uniform Convergence for Bias Learners** 



Now we have enough machinery to state the main theorem. In the theorem the hypothesis space<br>family is required to be permissible . Permissibility is discussed in detail in Appendix D, but note<br>that it is a weak measure-theoretic condition satisfied by almost all “real-world” hypothesis space<br>families. All logarithms are to base e .<br>
3. Once the learner has found an H 2 H with a small value of er^ z (H ) , it can then use H to learn novel tasks P drawn according to Q . One then has the following theorem bounding the sample complexity required for good generalisation when learning with H (the proof is very similar to the proof of the bound on m in Theorem 2). 



161<br>
**Theorem 2.** _Suppose_ X _and_ Y _are separable metric spaces and let_ Q _be any probability distribution on_ P _, the set of all distributions on_ X � Y _. Suppose_ z _is an_ (n; m) _-sample generated by sampling_ n _times from_ P _according to_ Q _to give_ P1 ; : : : ; Pn<sup>_, and then sampling_</sup> m _times from each_ Pi<sup>_to generate_</sup> zi = f(xi1 ; yi1 ); : : : ; (xim ; yim )g _,_ i = 1; : : : ; n _. Let_ H = fH g _be any permissible hypothesis space family. If the number of tasks_ n _satisfies_ 256 8C � 32" ; H � � 64 n � max "2 log Æ ; "2 ; (21) ( ) _and the number of examples_ m _of each task satisfies_ 256 8C ( 32" ; H nl ) 64 m � max � n"2 log Æ ; "2 � ; (22) _then with probability at least_ 1 � Æ _(over the_ (n; m) _-sample_ z _), all_ H 2 H _will satisfy_ erQ (H ) � er^ z (H ) + " (23) _Proof._ See Appendix A. There are several important points to note about Theorem 2: 1. Provided the capacities C ("; H � ) and C ("; H nl ) are finite, the theorem shows that _any_ bias learner that selects hypothesis spaces from H can bound its generalisation error erQ (H ) in terms of er^ z (H ) for sufficiently large (n; m) -samples z . Most bias learner’s will not find the exact value of er^ z (H ) because it involves finding the smallest error of any hypothesis h 2 H on each of the n training sets in z . But any upper bound on er^ z (H ) (found, for example by gradient descent on some error function) will still give an upper bound on er Q (H ) . See Section 3.3.1 for a brief discussion on how this can be achieved in a feature learning setting. 2. In order to learn bias (in the sense that erQ (H ) and er^ z (H ) are close uniformly over all H 2 H ), both the number of tasks n and the number of examples of each task m must be sufficiently large. This is intuitively reasonable because the bias learner must see both sufficiently many tasks to be confident of the nature of the environment, and sufficiently many examples of each task to be confident of the nature of each task. 



BAXTER<br>


Theorem 3. Let z = f(x1 ; y1 ); : : : ; (xm ; ym )g  be a training set generated by sampling from<br>X � Y according to some distribution P . Let H  be a permissible hypothesis space. For all<br>"; Æ with 0 < "; Æ < 1 , if the number of training examples m  satisfies<br>64 4C � 16" ; Hl � 16<br>m � max "2 log Æ ; "2 (24)<br>( )<br>then with probability at least 1 � Æ , all h 2 H  will satisfy<br>erP (h) � er^ z (h) + ":<br>The capacity C ("; H ) appearing in equation (24) is defined in an analogous fashion to the<br>capacities in Definition 4 (we just use the pseudo-metric dP (hl ; h0l ) := RX �Y jhl (x; y ) �<br>h0l (x; y )j dP (x; y ) ). The important thing to note about Theorem 3 is that the number of ex-<br>amples required for good generalisation when learning novel tasks is proportional to the log-<br>arithm of the capacity of the learnt hypothesis space H . In contrast, if the learner does not<br>do any bias learning, it will have no reason to select one hypothesis space H 2 H over any<br>other and consequently it would have to view as a candidate solution any hypothesis in any<br>of the hypothesis spaces H 2 H . Thus, its sample complexity will be proportional to the<br>capacity of [H2 H fHl g = H 1l , which in general will be considerably larger than the capacity<br>of any individual H 2 H . So by learning H  the learner has  learnt to learn  in the environment<br>(P ; Q)  in the sense that it needs far smaller training sets to learn novel tasks.<br>4. Having learnt a hypothesis space H with a small value of er^ z (H ) , Theorem 2 tells us that<br>with probability at least 1 � Æ , the expected value of inf h2H erP (h)  on a novel task P will be<br>less than er^ z (H ) + " . Of course, this does not rule out really bad performance on some tasks<br>P . However, the probability of generating such “bad” tasks can be bounded. In particular,<br>note that erQ (H ) is just the expected value of the function H � over P , and so by Markov’s<br>inequality, for � > 0 ,<br>�<br>Pr P : inf erP (h) � � = Pr fP : H (P ) � � g<br>� h2H �<br>EQ H �<br>�<br>�<br>= erQ (H )<br>�<br>� er^ z (H ) + " (with probability 1 � Æ ).<br>�<br>5. Keeping the accuracy and confidence parameters "; Æ  fixed, note that the number of examples<br>required of each task for good generalisation obeys<br>1<br>m = O log C ( "; H nl ) : (25)<br>� n �<br>


So provided log C ("; H nl ) increases sublinearly with n , the upper bound on the number of<br>examples required of each task will decrease as the number of tasks increases. This shows<br>that for suitably constructed hypothesis space families it is possible to share information<br>between tasks. This is discussed further after Theorem 4 below.<br>


162<br>
A MODEL OF INDUCTIVE BIAS LEARNING 



2.5 Choosing the Hypothesis Space Family H .<br>Theorem 2 only provides conditions under which er^ z (H )  and erQ (H )  are close, it does not guaran-<br>tee that erQ (H )  is actually small. This is governed by the choice of H . If H contains a hypothesis<br>space H  with a small value of erQ (H )  and the learner is able to find an H 2 H minimizing error on<br>the (n; m)  sample z  (i.e., minimizing er^ z (H ) ), then, for sufficiently large n  and m , Theorem 2 en-<br>sures that with high probability erQ (H )  will be small. However, a bad choice of H will mean there<br>is no hope of finding an H  with small error. In this sense the choice of H represents the  hyper-bias<br>of the learner.<br>Note that from a sample complexity point of view, the  optimal  hypothesis space family to choose<br>is one containing a single, minimal hypothesis space H that contains good solutions to all of the<br>problems in the environment (or at least a set of problems with high Q -probability), and no more.<br>For then there is no bias learning to do (because there is no choice to be made between hypothesis<br>spaces), the output of the bias learning algorithm is guaranteed to be a good hypothesis space for<br>the environment, and since the hypothesis space is minimal, learning any problem within the en-<br>vironment using H  will require the smallest possible number of examples. However, this scenario<br>is analagous to the trivial scenario in ordinary learning in which the learning algorithm contains a<br>single, optimal hypothesis for the problem being learnt. In that case there is no learning to be done,<br>just as there is no bias learning to be done if the correct hypothesis space is already known.<br>At the other extreme, if H contains a single hypothesis space H  consisting of all possible func-<br>tions from X ! Y then bias learning is impossible because the bias learner cannot produce a<br>restricted hypothesis space as output, and hence cannot produce a hypothesis space with improved<br>sample complexity requirements on as yet unseen tasks.<br>Focussing on these two extremes highlights the minimal requirements on H for successful bias<br>learning to occur: the hypothesis spaces H 2 H must be strictly smaller than the space of all<br>functions X ! Y  , but not so small or so “skewed” that none of them contain good solutions to a<br>large majority of the problems in the environment.<br>


It may seem that we have simply replaced the problem of selecting the right bias (i.e., selecting<br>the right hypothesis space H ) with the equally difficult problem of selecting the right hyper-bias (i.e.,<br>the right hypothesis space family H ). However, in many cases selecting the right hyper-bias is far<br>easier than selecting the right bias. For example, in Section 3 we will see how the feature selection<br>problem may be viewed as a bias selection problem. Selecting the right features can be extremely<br>difficult if one knows little about the environment, with intelligent trial-and-error typically the best<br>one can do. However, in a bias learning scenario, one only has to specify that a set of features should<br>exist, find a loosely parameterised set of features (for example neural networks), and then learn the<br>features by sampling from multiple related tasks.<br>
### **2.6 Learning Multiple Tasks** 

It may be that the learner is not interested in learning to learn, but just wants to learn a fixed set of n tasks from the environment (P ; Q) . As in the previous section, we assume the learner starts out with a hypothesis space family H , and also that it receives an (n; m) -sample z generated from the n distributions P1 ; : : : ; Pn<sup>.This time, however,the learneris simply lookingfor</sup> n hypotheses (h1 ; : : : ; hn ) , all contained in the same hypothesis space H , such that the average generalization error of the n hypotheses is minimal. Denoting (h1 ; : : : ; hn ) by h and writing P = (P1 ; : : : ; Pn ) , 

163 

this error is given by:<br>


1 n<br>erP (h) := n X erPi (hi ) (26)<br>i=1<br>1 n<br>= n X ZX �Y l (hi (x); y ) dPi (x; y );<br>i=1<br>and the empirical loss of h  on z  is<br>er^ z (h) := n1 Xn er^ zi (hi ) (27)<br>i=1<br>1 n 1 m<br>= n X m X l (hi (xij ); yij ):<br>i=1 j =1<br>As before, regardless of how the learner chooses (h1 ; : : : ; hn ) , if we can prove a uniform bound on<br>^<br>the probability of large deviation between erz (h) and erP (h) then any (h1 ; : : : ; hn ) that perform<br>well on the training sets z  will with high probability perform well on future examples of the same<br>tasks.<br>Theorem 4. Let P = (P1 ; : : : ; Pn )  be n  probability distributions on X � Y and let z  be an (n; m) -<br>sample generated by sampling m  times from X � Y according to each Pi . Let H = fH g  be any<br>permissible hypothesis space family. If the number of examples m  of each task satisfies<br>64 4C ( 16" ; H nl ) 16<br>m � max � n"2 log Æ ; "2 � (28)<br>then with probability at least 1 � Æ  (over the choice of z ), any h 2 H n  will satisfy<br>erP (h) � er^ z (h) + " (29)<br>(recall Definition 4 for the meaning of C ("; H nl ) ).<br>Proof. Omitted (follow the proof of the bound on m  in Theorem 2).<br>The bound on m in Theorem 4 is virtually identical to the bound on m in Theorem 2, and note<br>again that it depends inversely on the number of tasks n  (assuming that the first part of the “max”<br>expression is the dominate one). Whether this helps depends on the rate of growth of C ( 16" ; H nl )  as<br>a function of n . The following Lemma shows that this growth is always small enough to ensure that<br>we never do worse by learning multiple tasks (at least in terms of the upper bound on the number of<br>examples required per task).<br>Lemma 5. For any hypothesis space family H ,<br>C �"; H 1l � � C ( "; H nl ) � C �"; H 1l �n : (30)<br>


164<br>


A MODEL OF INDUCTIVE BIAS LEARNING<br>


2 . This behavior can be<br>


In Theorems 2, 3 and 4 the bounds on sample complexity all scale as 1="2 . This behavior can be<br>improved to 1="  if the empirical loss is always guaranteed to be zero (i.e., we are in the realizable<br>case). The same behavior results if we are interested in relative deviation between empirical and<br>true loss, rather than absolute deviation. Formal theorems along these lines are stated in Appendix<br>A.3.<br>
## **3. Feature Learning** 

The use of restricted feature sets is nearly ubiquitous as a method of encoding bias in many areas of machine learning and statistics, including classification, regression and density estimation. 



In this section we show how the problem of choosing a set of features for an environment of<br>related tasks can be recast as a bias learning problem. Explicit bounds on C ( H � ; ") and C ( H nl ; ")<br>are calculated for general feature classes in Section 3.2. These bounds are applied to the problem of<br>learning a neural network feature set in Section 3.3.<br>
### **3.1 The Feature Learning Model** 



Consider the following quote from Vapnik (1996):<br>


The classical approach to estimating multidimensional functional dependencies is<br>based on the following belief:<br>


Real-life problems are such that there exists a small number of “strong features,” simple<br>functions of which (say linear combinations) approximate well the unknown function.<br>Therefore, it is necessary to carefully choose a low-dimensional feature space and then<br>to use regular statistical techniques to construct an approximation.<br>
_Proof._ Let K denote the set of all functions (h1 ; : : : ; hn )l<sup>where each</sup> hi<sup>canbea memberof any</sup> hypothesis space H 2 H (recall Definition 1). Then H nl � K and so C ( "; H nl ) � C ("; K ) . By Lemma 29 in Appendix B, C ( "; K ) � C �"; H 1l �n and so the right hand inequality follows. For the first inequality, let P be any probability measure on X � Y and let P be the measure on (X � Y )n obtained by using P on the first copy of X � Y in the product, and ignoring all other elements of the product. Let N be an " -cover for ( H nl ; dP ) . Pick any hl 2 H 1l<sup>and</sup> let (g1 ; : : : ; gn )l 2 N be such that dP ((h; h; : : : ; h)l ; (g1 ; : : : ; gn )l ) � " . But by construction, dP ((h; h; : : : ; h)l ; (g1 ; : : : ; gn )l ) = dP (h; (g1 )l ) , which establishes the first inequality. By Lemma 5 log C �"; H 1l � � log C ("; H nl ) � n log C �"; H 1l � : (31) So keeping the accuracy parameters " and Æ fixed, and plugging (31) into (28), we see that the upper bound on the number of examples required of each task never _increases_ with the number of tasks, and at best decreases as O (1=n) . Although only an upper bound, this provides a strong hint that learning multiple related tasks should be advantageous on a “number of examples required per task” basis. In Section 3 it will be shown that for feature learning all types of behavior are possible, from no advantage at all to O (1=n) decrease. **2.7 Dependence on** " 

165 

In general a set of “strong features” may be viewed as a function f : X ! V mapping the input space X into some (typically lower) dimensional space V . Let F = ff g be a set of such feature maps (each f may be viewed as a set of features (f1 ; : : : ; fk ) if V = R k ). It is the f that must be “carefully chosen” in the above quote. In general, the “simple functions of the features” may be represented as a class of functions G mapping V to Y . If for each f 2 F we define the hypothesis space G Æ f := fg Æ f : g 2 G g , then we have the hypothesis space family H H := fG Æ f : f 2 F g: (32) Now the problem of “carefully choosing” the right features f is equivalent to the bias learning problem “find the right hypothesis space H 2 H ”. Hence, provided the learner is embedded within an environment of related tasks, and the capacities C ( H � ; ") and C ( H nl ; ") are finite, Theorem 2 tells us that the feature set f can be _learnt_ rather than carefully chosen. This represents an important simplification, as choosing a set of features is often the most difficult part of any machine learning problem. In Section 3.2 we give a theorem bounding C ( H � ; ") and C ( H nl ; ") for general feature classes. The theorem is specialized to neural network classes in Section 3.3. Note that we have forced the function class G to be the same for all feature maps f , although this is not necessary. Indeed variants of the results to follow can be obtained if G is allowed to vary with f . **3.2 Capacity Bounds for General Feature Classes** Notationally it is easier to view the feature maps f as mapping from X � Y to V � Y by (x; y ) 7! (f (x); y ) , and also to absorb the loss function l into the definition of G by viewing each g 2 G as a map from V � Y into [0; 1℄ via (v ; y ) 7! l (g (v ); y ) . Previously this latter function would have been denoted gl<sup>but in what follows we will drop the subscript</sup> l where this does not cause confusion. The class to which gl<sup>belongs will still be denoted by</sup> Gl<sup>.</sup> With the above definitions let Gl Æ F := fg Æ f : g 2 Gl ; f 2 F g . Define the capacity of Gl<sup>in</sup> the usual way, C ("; Gl ) := sup N ("; Gl ; dP ) P where the supremum is over all probability measures on V � Y , and dP (g ; g 0 ) := RV �Y jg (v ; y ) � g 0 (v ; y )j dP (v ; y ) . To define the capacity of F we first define a pseudo-metric d[P ;Gl ℄<sup>on</sup> F by “pulling back” the L1 metric on R through Gl<sup>as follows:</sup> d[P ;Gl ℄ (f ; f 0 ) := sup jg Æ f (x; y ) � g Æ f 0 (x; y )j dP (x; y ): (33) ZX �Y g 2Gl It is easily verified that d[P ;Gl ℄<sup>is a pseudo-metric.Note that for</sup> d[P ;Gl ℄<sup>to be well defined the supre-</sup> mum over Gl<sup>in the integrand must be measurable.This is guaranteed if the hypothesis space family</sup> H = fGl Æ f : f 2 F g is permissible (Lemma 32, part 4). Now define N ("; F ; d[P ;Gl ℄ ) to be the smallest " -cover of the pseudo-metric space (F ; d[P ;Gl ℄ ) and the " -capacity of F (with respect to Gl<sup>)</sup> as CGl ("; F ) := sup N ("; F ; d[P ;Gl ℄ ) P where the supremum is over all probability measures on X � Y . Now we can state the main theorem of this section. 

166 



A MODEL OF INDUCTIVE BIAS LEARNING<br>


For argument’s sake, assume the “simple functions” of the features (the class G of the previous<br>section) are squashed affine maps using the same sigmoid function � above (in keeping with the<br>“neural network” flavor of the features). Thus, each setting of the feature weights w generates a<br>hypothesis space:<br>


167<br>
**Theorem 6.** _Let_ H _be a hypothesis space family as in equation_ (32) _. Then for all_ "; "1 ; "2 > 0 _with_ " = "1 + "2<sup>_,_</sup> C ("; H nl ) � C ( "1 ; Gl )n CGl ("2 ; F ) (34) C ("; H � ) � CGl ( "; F ) (35) _Proof._ See Appendix B. **3.3 Learning Neural Network Features** In general, a set of features may be viewed as a map from the (typically high-dimensional) input space R d to a much smaller dimensional space R k ( k � d ). In this section we consider approximating such a feature map by a one-hidden-layer neural network with d input nodes and k output nodes (Figure 1). We denote the set of all such feature maps by f�w = (�w ;1 ; : : : ; �w ;k ) : w 2 D g where D is a bounded subset of R W ( W is the number of weights (parameters) in the first two layers). This set is the F of the previous section. Each feature �w ;i : R d ! [0; 1℄ , i = 1; : : : ; k is defined by l �w ;i (x) := � 0X vij hj (x) + vil +1 1 (36) j =1 A where hj (x) is the output of the j th node in the first hidden layer, (vi1 ; : : : ; vil +1 ) are the output node parameters for the i th feature and � is a “sigmoid” squashing function � : R ! [0; 1℄ . Each first layer hidden node hi : R d ! R , i = 1; : : : ; l , computes d hi (x) := � 0X uij xj + uid+1 1 (37) j =1 A where (ui1 ; : : : ; uid+1 ) are the hidden node’s parameters. We assume � is Lipschitz.<sup>5</sup> The weight vector for the entire feature map is thus w = (u11 ; : : : ; u1d+1 ; : : : ; ul 1 ; : : : ; ul d+1 ; v11 ; : : : ; v1l +1 ; : : : ; vk 1 ; : : : ; vk l +1 ) and the total number of feature parameters W = l (d + 1) + k (l + 1) . For argument’s sake, assume the “simple functions” of the features (the class G of the previous section) are squashed affine maps using the same sigmoid function � above (in keeping with the “neural network” flavor of the features). Thus, each setting of the feature weights w generates a hypothesis space: k 0 Hw := � X �i �w ;i + �k +1 : (�1 ; : : : ; �k +1 ) 2 D ; (38) ( i=1 ! ) where D 0 is a bounded subset of R k +1 . The set of all such hypothesis spaces, H := fHw : w 2 D g (39) 5. � is Lipschitz if there exists a constant K such that j� (x) � � (x0 )j � K jx � x0 j for all x; x0 2 R . 

Multiple Output Classes<br>n<br>k<br>Feature<br>l Map<br>d<br>


Input<br>
Figure 1: Neural network for feature learning. The feature map is implemented by the first two hidden layers. The n output nodes correspond to the n different tasks in the (n; m) - sample z . Each node in the network computes a squashed linear function of the nodes in the previous layer. is a hypothesis space family. The restrictions on the output layer weights (�1 ; : : : ; �k +1 ) and feature weights w , and the restriction to a Lipschitz squashing function are needed to obtain finite upper bounds on the covering numbers in Theorem 2. 

Finding a good set of features for the environment (P ; Q) is equivalent to finding a good hypothesis space Hw 2 H , which in turn means finding a good set of feature map parameters w . As in Theorem 2, the correct set of features may be learnt by finding a hypothesis space with small error on a sufficiently large (n; m) -sample z . Specializing to squared loss, in the present framework the empirical loss of Hw<sup>on</sup> z (equation (8)) is given by 2 ^ 1 n 1 m k erz (Hw ) = n X (�0 ;�1 ;:::inf;�k )2D 0 m X � X �l �w ;l (xij ) + �0 � yij (40) i=1 j =1 " l =1 ! # Since our sigmoid function � only has range [0; 1℄ , we also restrict the outputs Y to this range. 3.3.1 ALGORITHMS FOR FINDING A GOOD SET OF FEATURES Provided the squashing function � is differentiable, gradient descent (with a small variation on backpropagation to compute the derivatives) can be used to find feature weights w minimizing (40) (or at least a local minimum of (40)). The only extra difficulty over and above ordinary gradient descent is the appearance of “ inf ” in the definition of er^ z (Hw ) . The solution is to perform gradient descent over both the output parameters (�0 ; : : : ; �k ) for each node and the feature weights w . For more details see Baxter (1995b) and Baxter (1995a, chapter 4), where empirical results supporting the theoretical results presented here are also given. 

168 



A MODEL OF INDUCTIVE BIAS LEARNING<br>


3.3.2 SAMPLE COMPLEXITY BOUNDS FOR NEURAL-NETWORK FEATURE LEARNING<br>


The size of z  ensuring that the resulting features will be good for learning novel tasks from the same<br>environment is given by Theorem 2. All we have to do is compute the logarithm of the covering<br>numbers C ("; H nl )  and C ("; H � ) .<br>Theorem 7. Let H = �Hw : w 2 R W � be a hypothesis space family where each Hw is of the form<br>k<br>k<br>Hw := � X �i �w ;i (�) + �0 : (�1 ; : : : ; �k ) 2 R ;<br>( i=1 ! )<br>where �w = (�w ;1 ; : : : ; �w ;k )  is a neural network with W weights mapping from R d to R k . If the<br>feature weights w and the output weights �0 ; �1 ; : : : ; �k are bounded, the squashing function � is<br>Lipschitz, l  is squared loss, and the output space Y = [0; 1℄ (any bounded subset of R will do), then<br>there exist constants �; �0 (independent of "; W and k ) such that for all " > 0 ,<br>log C ("; H nl ) � 2 ( (k + 1)n + W ) log � (41)<br>"<br>0<br>log C ("; H � ) � 2W log � (42)<br>"<br>(recall that we have specialized to squared loss here).<br>Proof. See Appendix B.<br>Noting that our neural network hypothesis space family H is permissible, plugging (41) and (42)<br>into Theorem 2 gives the following theorem.<br>Theorem 8. Let H = fHw g be a hypothesis space family where each hypothesis space Hw is a<br>set of squashed linear maps composed with a neural network feature map, as above. Suppose the<br>number of features is k , and the total number of feature weights is W. Assume all feature weights and<br>output weights are bounded, and the squashing function � is Lipschitz. Let z  be an (n; m) -sample<br>generated from the environment (P ; Q) . If<br>1 1 1<br>n � O � "2 �W log " + log Æ �� ; (43)<br>and<br>1 W 1 1 1<br>m � O 2 k + 1 + log + log (44)<br>� " �� n � " n Æ ��<br>then with probability at least 1 � Æ  any Hw 2 H will satisfy<br>erQ (Hw ) � er^ z (Hw ) + ": (45)<br>


169<br>
BAXTER 

### 3.3.3 DISCUSSION 

1. Keeping the accuracy and confidence parameters " and Æ fixed, the upper bound on the number of examples required of each task behaves like O (k + W =n) . If the learner is simply learning n fixed tasks (rather than learning to learn), then the same upper bound also applies (recall Theorem 4). 

So when specialized to the traditional multiple-class, single task framework, Theorem 8 is consistent with the bounds already known. However, as we have already argued, problems such as face recognition are not really single-task, multiple-class problems. They are more appropriately viewed 

> 6. If each example can be classified with a “large margin” then naive parameter counting can be improved upon (Bartlett, 1998). 

2. Note that if we do away with the feature map altogether then W = 0 and the upper bound on m becomes O (k ) , independent of n (apart from the less important Æ term). So in terms of the upper bound, learning n tasks becomes just as hard as learning one task. At the other extreme, if we fix the output weights then effectively k = 0 and the number of examples required of each task decreases as O (W =n) . Thus a range of behavior in the number of examples required of each task is possible: from no improvement at all to an O (1=n) decrease as the number of tasks n increases (recall the discussion at the end of Section 2.6). 3. Once the feature map is learnt (which can be achieved using the techniques outlined in Baxter, 1995b; Baxter & Bartlett, 1998; Baxter, 1995a, chapter 4), only the output weights have to be estimated to learn a novel task. Again keeping the accuracy parameters fixed, this requires no more that O (k ) examples. Thus, as the number of tasks learnt increases, the upper bound on the number of examples required of each task decays to the minimum possible, O (k ) . 4. If the “small number of strong features” assumption is correct, then k will be small. However, typically we will have very little idea of what the features are, so to be confident that the neural network is capable of implementing a good feature set it will need to be very large, implying W � k . O (k + W =n) decreases most rapidly with increasing n when W � k , so at least in terms of the upper bound on the number of examples required per task, learning small feature sets is an ideal application for bias learning. However, the upper bound on the number of tasks does not fare so well as it scales as O (W ) . 3.3.4 COMPARISON WITH TRADITIONAL MULTIPLE-CLASS CLASSIFICATION A special case of this multi-task framework is one in which the marginal distribution on the input space PijX<sup>is the same for each task</sup> i = 1; : : : ; n , and all that varies between tasks is the conditional distribution over the output space Y . An example would be a multi-class problem such as face recognition, in which Y = f1; : : : ; ng where n is the number of faces to be recognized and the marginal distribution on X is simply the “natural” distribution over images of those faces. In that case, if for every example xij<sup>we have—in addition to the sample</sup> yij<sup>from the</sup> i th task’s conditional distribution on Y —samples from the remaining n � 1 conditional distributions on Y , then we can view the n training sets containing m examples each as one large training set for the multi-class problem with mn examples altogether. The bound on m in Theorem 8 states that mn should be O (nk + W ) , or proportional to the total number of parameters in the network, a result we would expect from<sup>6</sup> (Haussler, 1992). 

170 

as a (potentially infinite) collection of distinct binary classification problems. In that case, the goal of bias learning is not to find a single n -output network that can classify some subset of n faces well. It is to learn a set of features that can reliably be used as a fixed preprocessing for distinguishing any single face from other faces. This is the new thing provided by Theorem 8: it tells us that provided we have trained our n -output neural network on sufficiently many examples of _sufficiently many tasks_ , we can be confident that the common feature map learnt for those n tasks will be good for learning _any_ new, as yet unseen task, provided the new task is drawn from the same distribution that generated the training tasks. In addition, learning the new task only requires estimating the k output node parameters for that task, a vastly easier problem than estimating the parameters of the entire network, from both a sample and computational complexity perspective. Also, since we have high confidence that the learnt features will be good for learning novel tasks drawn from the same environment, those features are themselves a candidate for further study to learn more about the nature of the environment. The same claim could not be made if the features had been learnt on too small a set of tasks to guarantee generalization to novel tasks, for then it is likely that the features would implement idiosyncrasies specific to those tasks, rather than “invariances” that apply across all tasks. 

When viewed from a bias (or feature) learning perspective, rather than a traditional n -class classification perspective, the bound m on the number of examples required of each task takes on a somewhat different meaning. It tells us that provided n is large (i.e., we are collecting examples of a large number tasks), then we really only need to collect a few more examples than we would otherwise have to collect if the feature map was already known ( k + W =n examples vs. k examples). So it tells us that the burden imposed by feature learning can be made negligibly small, at least when viewed from the perspective of the sampling burden required of each task. 

### **3.4 Learning Multiple Tasks with Boolean Feature Maps** 

Ignoring the accuracy and confidence parameters " and Æ , Theorem 8 shows that the number of examples required of each task when learning n tasks with a common neural-network feature map is bounded above by O (k + W =n) , where k is the number of features and W is the number of adjustable parameters in the feature map. Since O (k ) examples are required to learn a single task once the true features are known, this shows that the upper bound on the number of examples required of each task decays (in order) to the minimum possible as the number of tasks n increases. This suggests that learning multiple tasks is advantageous, but to be truly convincing we need to = prove a lower bound of the same form. Proving lower bounds in a real-valued setting ( Y R ) is complicated by the fact that a single example can convey an infinite amount of information, so one typically has to make extra assumptions, such as that the targets y 2 Y are corrupted by a noise process. Rather than concern ourselves with such complications, in this section we restrict our attention to Boolean hypothesis space families (meaning each hypothesis h 2 H 1 maps to Y = f�1g and we measure error by discrete loss l (h(x); y ) = 1 if h(x) 6= y and l (h(x); y ) = 0 otherwise). We show that the sample complexity for learning n tasks with a Boolean hypothesis space family H is controlled by a “VC dimension” type parameter d H (n) (that is, we give nearly matching upper and lower bounds involving d H (n) ). We then derive bounds on d H (n) for the hypothesis space family considered in the previous section with the Lipschitz sigmoid function � replaced by a hard threshold (linear threshold networks). 

171 

As well as the bound on the number of examples required per task for good generalization across those tasks, Theorem 8 also shows that features performing well on O (W ) _tasks_ will generalize well to novel tasks, where W is the number of parameters in the feature map. Given that for many feature learning problems W is likely to be quite large (recall Note 4 in Section 3.3.3), it would be useful to know that O (W ) tasks are in fact _necessary_ without further restrictions on the environmental distributions Q generating the tasks. Unfortunately, we have not yet been able to show such a lower bound. There is some empirical evidence suggesting that in practice the upper bound on the number of tasks may be very weak. For example, in Baxter and Bartlett (1998) we reported experiments in which a set of neural network features learnt on a subset of only 400 Japanese characters turned out to be good enough for classifying some 2600 unseen characters, even though the features contained several hundred thousand parameters. Similar results may be found in Intrator and Edelman (1996) and in the experiments reported in Thrun (1996) and Thrun and Pratt (1997, chapter 8). While this gap between experiment and theory may be just another example of the looseness inherent in general bounds, it may also be that the analysis can be tightened. In particular, the bound on the number of tasks is insensitive to the size of the class of output functions (the class G in Section 3.1), which may be where the looseness has arisen. 

- 3.4.1 UPPER AND LOWER BOUNDS FOR LEARNING n TASKS WITH BOOLEAN HYPOTHESIS SPACE FAMILIES 

An important result in the theory of learning Boolean functions is Sauer’s Lemma (Sauer, 1972), of which we will also make use. 



We now generalize these concepts to learning n  tasks with a Boolean hypothesis space family.<br>
First we recall some concepts from the theory of Boolean function learning. Let H be a class of Boolean functions on X and x = (x1 ; : : : ; xm ) 2 X m . Hjx<sup>is the set of all binary vectors obtainable</sup> by applying functions in H to x : Hjx := f(h(x1 ); : : : ; h(xm )) : h 2 H g: Clearly jHjx j � 2m . If jHjx j = 2m we say H _shatters_ x . The _growth function_ of H is defined by �H (m) := xmax2X m ��Hjx �� : The _Vapnik-Chervonenkis dimension_ VCdim(H ) is the size of the largest set shattered by H : VCdim(H ) := maxfm : �H (m) = 2m g: 

**Lemma 9 (Sauer’s Lemma).** _For a Boolean function class_ H _with_ VCdim(H ) = d _,_ d m em d �H (m) � Xi=0 � i � � � d � ; _for all positive integers_ m _._ 

172 



A MODEL OF INDUCTIVE BIAS LEARNING<br>


Definition 5. Let H be a Boolean hypothesis space family. Denote the n � m matrices over the<br>input space X by X (n;m) . For each x 2 X (n;m) and H 2 H , define Hjx to be the set of (binary)<br>matrices,<br>8> h1 (x11 ) � � � h1 (x1m ) 9><br>2 3<br>Hjx := 6 ... ... ... 7 : h1 ; : : : ; hn 2 H :<br>< =<br>> ><br>:4 hn (xn1 ) � � � hn (xnm ) 5 ;<br>Define<br>H := H :<br>jx [ jx<br>H2 H<br>Now for each n > 0; m > 0 , define � H (n; m)  by<br>� H (n; m) := max �� H jx �� :<br>x2X (n;m)<br>Note that � H (n; m) � 2nm . If �� H jx �� = 2nm we say H shatters  the matrix x . For each n > 0  let<br>nm<br>d H (n) := maxfm : � H (n; m) = 2 g:<br>Define<br>1<br>d( H ) : = VCdim( H ) and<br>d( H ) : = max VCdim(H ):<br>H2 H<br>Lemma 10.<br>d ( H ) � d( H )<br>d ( H ) 1 d( H )<br>d H (n) � max ; d( H ) � + d ( H )<br>�� n � � 2 �� n � �<br>Proof. The first inequality is trivial from the definitions. To get the second term in the maximum<br>in the second inequality, choose an H 2 H with VCdim(H ) = d( H ) and construct a matrix<br>x 2 X (n;m) whose rows are of length d( H )  and are shattered by H . Then clearly H shatters x . For<br>the first term in the maximum take a sequence x = (x1 ; : : : ; xd ( H ) )  shattered by H 1  (the hypothesis<br>space consisting of the union over all hypothesis spaces from H ), and distribute its elements equally<br>among the rows of x  (throw away any leftovers). The set of matrices<br>8> h(x11 ) � � � h(x1m ) 9><br>2 3 1<br>6 ... ... ... 7 : h 2 H :<br>< =<br>> ><br>:4 h(xn1 ) � � � h(xnm ) 5 ;<br>where m = bd ( H )=n is a subset of H jx and has size 2nm .<br>Lemma 11.<br>nd H (n)<br>em<br>� H (n; m) �<br>� d H (n) �<br>173<br>
BAXTER 



Proof. Observe that for each n , � H (n; m) = �H (nm) where H is the collection of all Boolean<br>functions on sequences x1 ; : : : ; xnm obtained by first choosing n  functions h1 ; : : : ; hn from some<br>H 2 H , and then applying h1 to the first m  examples, h2 to the second m  examples and so on. By<br>the definition of d H (n) , VCdim(H ) = nd H (n) , hence the result follows from Lemma 9 applied to<br>H .<br>If one follows the proof of Theorem 4 (in particular the proof of Theorem 18 in Appendix<br>A) then it is clear that for all � > 0 , C ( H nl ; ") may be replaced by � H (n; 2m) in the Boolean<br>case. Making this replacement in Theorem 18, and using the choices of �; � from the discussion<br>following Theorem 26, we obtain the following bound on the probability of large deviation between<br>empirical and true performance in this Boolean setting.<br>Theorem 12. Let P = (P1 ; : : : ; Pn )  be n  probability distributions on X � f�1g and let z be an<br>(n; m) -sample generated by sampling m  times from X � f�1g  according to each Pi . Let H = fH g<br>be any permissible Boolean hypothesis space family. For all 0 < � � 1 ,<br>Pr f z : 9h 2 H n : erP (h) � er^ z (h) + "g � 4� H (n; 2m) exp (��2 nm=64): (46)<br>Corollary 13. Under the conditions of Theorem 12, if the number of examples m of each task<br>satisfies<br>88 22 1 4<br>m � 2 2d H (n) log + log (47)<br>" � " n Æ �<br>then with probability at least 1 � Æ  (over the choice of z ), any h 2 H n  will satisfy<br>erP (h) � er^ z (h) + " (48)<br>Proof. Applying Theorem 12, we require<br>4� H (n; 2m) exp (��2 nm=64) � Æ;<br>which is satisfied if<br>64 2em 1 4<br>m � 2 d H (n) log + log ; (49)<br>� � d H (n) n Æ �<br>where we have used Lemma 11. Now, for all a � 1 , if<br>1 1<br>m = 1 + a log 1 + a;<br>� e � � e �<br>then m � a log m . So setting a = 64d H (n)="2 , (49) is satisfied if<br>88 22 1 4<br>m � 2 2d H (n) log + log :<br>" � " n Æ �<br>


174<br>


A MODEL OF INDUCTIVE BIAS LEARNING<br>
Corollary 13 shows that any algorithm learning n tasks using the hypothesis space family H requires no more than 



1 1 1 1<br>m = O 2 d H (n) log + log (50)<br>� " � " n Æ ��<br>examples of each task to ensure that with high probability the average true error of any n  hypotheses<br>it selects from H n is within " of their average empirical error on the sample z . We now give a<br>theorem showing that if the learning algorithm is required to produce n  hypotheses whose average<br>true error is within "  of the  best possible error (achievable using H n ) for an arbitrary sequence of<br>distributions P1 ; : : : ; Pn , then within a log 1" factor the number of examples in equation (50) is also<br>necessary.<br>For any sequence P = (P1 ; : : : ; Pn ) of n probability distributions on X � f�1g , define<br>optP ( H n )  by<br>optP ( H n ) := hinf2 H n erP (h):<br>Theorem 14. Let H be a Boolean hypothesis space family such that H 1 contains at least two<br>functions. For each n = 1; 2; : : : ;  let An be any learning algorithm taking as input (n; m) -samples<br>z 2 (X � f�1g)(n;m) and producing as output n hypotheses h = (h1 ; : : : ; hn ) 2 H n . For all<br>0 < " < 1=64  and 0 < Æ < 1=64 , if<br>m < "12 � d 616H (n) + (1 � "2 ) n1 log � 8Æ (1 1� 2Æ ) ��<br>then there exist distributions P = (P1 ; : : : ; Pn ) such that with probability at least Æ (over the<br>random choice of z ),<br>erP (An (z)) > optP ( H n ) + "<br>Proof. See Appendix C<br>3.4.2 LINEAR THRESHOLD NETWORKS<br>Theorems 13 and 14 show that within constants and a log (1=") factor, the sample complexity of<br>learning n  tasks using the Boolean hypothesis space family H is controlled by the complexity pa-<br>rameter d H (n) . In this section we derive bounds on d H (n)  for hypothesis space families constructed<br>as thresholded linear combinations of Boolean feature maps. Specifically, we assume H is of the<br>form given by (39), (38), (37) and (36), where now the squashing function � is replaced with a hard<br>threshold:<br>1 if x � 0 ;<br>� (x) :=<br>(�1 otherwise ;<br>and we don’t restrict the range of the feature and output layer weights. Note that in this case the<br>proof of Theorem 8 does not carry through because the constants �; �0 in Theorem 7 depend on the<br>Lipschitz bound on � .<br>Theorem 15. Let H be a hypothesis space family of the form given in  (39) ,  (38) ,  (37)  and  (36) , with<br>a hard threshold sigmoid function � . Recall that the parameters d , l  and k  are the input dimension,<br>number of hidden nodes in the feature map and number of features (output nodes in the feature map)<br>
175 

respectively. Let W := l (d + 1) + k (l + 1) (the number of adjustable parameters in the feature<br>map). Then,<br>W<br>d H (n) � 2 + k + 1 log 2 ( 2e(k + l + 1)) :<br>� n �<br>Proof. Recall that for each w 2 R W  , �w : R d ! R k denotes the feature map with parameters w .<br>For each x 2 X (n;m) , let �w jx denote the matrix<br>2 �w (x11 ) � � � �w (x1m ) 3<br>6 ... ... ... 7 :<br>4 �w (xn1 ) � � � �w (xnm ) 5<br>Note that H jx is the set of all binary n � m  matrices obtainable by composing thresholded linear<br>functions with the elements of �w jx , with the restriction that the same function must be applied to<br>each element in a row (but the functions may differ between rows). With a slight abuse of notation,<br>define<br>�� (n; m) := max ����w jx : w 2 R W ��� :<br>x2X (n;m)<br>Fix x 2 X (n;m) . By Sauer’s Lemma, each node in the first hidden layer of the feature map computes<br>at most ( emn=(d + 1)) d+1 functions on the nm input vectors in x . Thus, there can be at most<br>l<br>(emn=(d + 1)) (d+1) distinct functions from the input to the output of the first hidden layer on<br>the nm  points in x . Fixing the first hidden layer parameters, each node in the second layer of the<br>l<br>feature map computes at most (emn=(l + 1)) +1  functions on the image of x  produced at the output<br>k (l +1)<br>of the first hidden layer. Thus the second hidden layer computes no more than ( emn=(l + 1))<br>functions on the output of the first hidden layer on the nm  points in x . So, in total,<br>l (d+1) k (l +1)<br>emn emn<br>�� (n; m) � :<br>� d + 1 � � l + 1 �<br>Now, for each possible matrix �w jx , the number of functions computable on each row of �w jx by a<br>k<br>thresholded linear combination of the output of the feature map is at most (em=(k + 1)) +1 . Hence,<br>the number of binary sign assignments obtainable by applying linear threshold functions to all the<br>n(k<br>rows is at most ( em=(k + 1)) +1) . Thus,<br>l (d+1) k (l +1) n(k +1)<br>emn emn emn<br>� H (n; m) � :<br>� d + 1 � � l + 1 � � n(k + 1) �<br>f (x) := x log x  is a convex function, hence for all a; b; > 0 ,<br>k a + l b + 1<br>f � (k f (a) + l f (b) + f ( ))<br>� k + l + 1 � k + l + 1<br>k a+l b+ k a l b<br>k + l + 1 1 1 1<br>) � :<br>� k a + l b + � � a � � b � � �<br>Substituting a = l + 1 , b = d + 1  and = n(k + 1)  shows that<br>W +n(k +1)<br>emn(k + l + 1)<br>� H (n; m) � : (51)<br>� W + n(k + 1) �<br>


176<br>


A MODEL OF INDUCTIVE BIAS LEARNING<br>


Hence, if<br>W emn(k + l + 1)<br>m > + k + 1 log 2 (52)<br>� n � � W + n(k + 1) �<br>then � H (n; m) < 2nm and so by definition d H (n) � m . For all a > 1 , observe that x > a log 2 x<br>if x = 2a log 2 2a . Setting x = emn(k + l + 1)=(W + n(k + 1))  and a = e(k + l + 1)  shows that<br>(52) is satisfied if m = 2(W =n + k + 1) log 2 (2e(k + l + 1)) .<br>Theorem 16. Let H be as in Theorem 15 with the following extra restrictions: d � 3 , l � k and<br>k � d . Then<br>1 W<br>d H (n) � + k + 1<br>2 �� 2n � �<br>Proof. We bound d( H )  and d( H )  and then apply Lemma 10. In the present setting H 1  contains all<br>three-layer linear-threshold networks with d  input nodes, l  hidden nodes in the first hidden layer, k<br>hidden nodes in the second hidden layer and one output node. From Theorem 13 in Bartlett (1993),<br>we have<br>VCdim( H 1 ) � dl + l (k � 1) + 1;<br>2<br>which under the restrictions stated above is greater than W =2 . Hence d( H ) � W =2 .<br>As k � d  and l � k we can choose a feature weight assignment so that the feature map is the<br>identity on k components of the input vector and insensitive to the setting of the reminaing d � k<br>components. Hence we can generate k + 1 points in X whose image under the feature map is<br>shattered by the linear threshold output node, and so d( H ) = k + 1 .<br>Combining Theorem 15 with Corrolary 13 shows that<br>1 W 1 1 1<br>m � O 2 + k + 1 log + log<br>� " �� n � " n Æ ��<br>examples of each task suffice when learning n  tasks using a linear threshold hypothesis space family,<br>while combining Theorem 16 with Theorem 14 shows that if<br>1 W 1 1<br>m � � 2 + k + 1 + log<br>� " �� n � n Æ ��<br>then any learning algorithm will fail on some set of n  tasks.<br>4. Conclusion<br>The problem of inductive bias is one that has broad significance in machine learning. In this paper<br>we have introduced a formal model of inductive bias learning that applies when the learner is able<br>to sample from multiple related tasks. We proved that provided certain covering numbers computed<br>from the set of all hypothesis spaces available to the bias learner are finite, any hypothesis space<br>that contains good solutions to sufficiently many training tasks is likely to contain good solutions to<br>novel tasks drawn from the same environment.<br>In the specific case of learning a set of features, we showed that the number of examples m<br>required of each task in an n -task training set obeys m = O (k + W =n) , where k is the number of<br>
log 2 x 

177 

features and W is a measure of the complexity of the feature class. We showed that this bound is essentially tight for Boolean feature maps constructed from linear threshold networks. In addition, we proved that the number of tasks required to ensure good performance from the features on novel tasks is no more than O (W ) . We also showed how a good set of features may be found by gradient descent. 

The model of this paper represents a first step towards a formal model of hierarchical approaches to learning. By modelling a learner’s uncertainty concerning its environment in probabilistic terms, we have shown how learning can occur simultaneously at both the base level—learn the tasks at hand—and at the meta-level—learn bias that can be transferred to novel tasks. From a technical perspective, it is the assumption that tasks are distributed probabilstically that allows the performance guarantees to be proved. From a practical perspective, there are many problem domains that can be viewed as probabilistically distributed sets of related tasks. For example, speech recognition may be decomposed along many different axes: words, speakers, accents, etc. Face recognition represents a potentially infinite domain of related tasks. Medical diagnosis and prognosis problems using the same pathology tests are yet another example. All of these domains should benefit from being tackled with a bias learning approach. 

Natural avenues for further enquiry include: 

- **Alternative constructions for** H **.** Although widely applicable, the specific example on feature learning via gradient descent represents just one possible way of generating and searching the hypothesis space family H . It would be interesting to investigate alternative methods, including decision tree approaches, approaches from Inductive Logic Programming (Khan et al., 1998), and whether more general learning techniques such as boosting can be applied in a bias learning setting. 

- � **Algorithms for automatically determining the hypothesis space family** H **.** In our model the structure of H is fixed _apriori_ and represents the _hyper-bias_ of the bias learner. It would be interesting to see to what extent this structure can also be learnt. 

- **Algorithms for automatically determining task relatedness.** In ordinary learning there is usually little doubt whether an individual _example_ belongs to the same learning task or not. The analogous question in bias learning is whether an individual learning task belongs to a given set of related tasks, which in contrast to ordinary learning, does not always have such a clear-cut answer. For most of the examples we have discussed here, such as speech and face recognition, the task-relatedness is not in question, but in other cases such as medical problems it is not so clear. Grouping too large a subset of tasks together as related tasks could clearly have a detrimental impact on bias-learning or multi-task learning, and there is emprical evidence to support this (Caruana, 1997). Thus, algorithms for automatically determining task-relatedness are a potentially useful avenue for further research. In this context, see Silver and Mercer (1996), Thrun and O’Sullivan (1996). Note that the question of task relatedness is clearly only meaningful _relative_ to a particular hypothesis space family H (for example, all possible collections of tasks are related if H contains every possible hypothesis space). 

- **Extended hierarchies.** For an extension of our two-level approach to arbitrarily deep hierarchies, see Langford (1999). An interesting further question is to what extent the hierarchy can be inferred from data. This is somewhat related to the question of automatic induction of structure in graphical models. 

178 

### **Acknowledgements** 

This work was supported at various times by an Australian Postgraduate Award, a Shell Australia Postgraduate Fellowship, U.K Engineering and Physical Sciences Research Council grants K70366 and K70373, and an Australian Postdoctoral Fellowship. Along the way, many people have contributed helpful comments and suggestions for improvement including Martin Anthony, Peter Bartlett, Rich Caruana, John Langford, Stuart Russell, John Shawe-Taylor, Sebastian Thrun and several anonymous referees. 

## **Appendix A. Uniform Convergence Results** 

Theorem 2 provides a bound (uniform over all H 2 H ) on the probability of large deviation between erQ (H ) and er^ z (H ) . To obtain a more general result, we follow Haussler (1992) and introduce the following parameterized class of metrics on R + : jx � y j d� [ x; y ℄ := ; x + y + � where � > 0 . Our main theorem will be a uniform bound on the probability of large values of ^ d� [erQ (H ); erz (H )℄ , rather than j erQ (H ) � erz (H )j . Theorem 2 will then follow as a corollary, as ^ will better bounds for the realizable case erz (H ) = 0 (Appendix A.3). **Lemma 17.** _The following three properties of_ d�<sup>_are easily established:_</sup> _1. For all_ r; s � 0 _,_ 0 � d� [ r; s℄ � 1 _2. For all_ 0 � r � s � t _,_ d� [r; s℄ � d� [r; t℄ _and_ d� [ s; t℄ � d� [r; t℄ _._ jr �sj jr �sj _3. For_ 0 � r; s � 1 _,_ � +2 � d� [ r; s℄ � � For ease of exposition we have up until now been dealing explicitly with hypothesis spaces H containing functions h : X ! Y , and then constructing loss functions hl<sup>mapping</sup> X � Y ! [0; 1℄ by hl (x; y ) := l (h(x); y ) for some loss function l : Y �Y ! [0; 1℄ . However, in general we can view hl<sup>just as a functionfrom an abstractset</sup> Z ( X � Y ) to [0; 1℄ and ignore its particular construction in terms of the loss function l . So for the remainder of this section, unless otherwise stated, all hypothesis spaces H will be sets of functions mapping Z to [0; 1℄ . It will also be considerably more convenient to transpose our notation for (n; m) -samples, writing the n training sets as columns instead of rows: z11 : : : z1n z = ... ... ... zm1 : : : zmn where each zij 2 Z . Recalling the definition of (X � Y )(n;m) (Equation 9 and prior discussion), with this transposition z lives in (X � Y ) (m;n) . The following definition now generalizes quantities like er^ z (H ) , erP (H ) and so on to this new setting. **Definition 6.** Let H1 ; : : : ; Hn<sup>be</sup> n sets of functions mapping Z into [0; 1℄ . For any h1 2 H1 ; : : : ; hn 2 Hn<sup>, let</sup> h1 � � � � � hn<sup>or simply</sup> h denote the map n ~ h(z ) = 1=n X hi (zi ) i=1 179 

~<br>for all z = (z1 ; : : : ; zn ) 2 Z n . Let H1 � � � � � Hn denote the set of all such functions. Given<br>~ ~<br>h 2 H1 � � � � � Hn and m  elements of (X � Y )n , (z1 ; : : : ; zm )  (or equivalently an element z  of<br>(X � Y )(m;n) by writing the ~zi as rows), define<br>^ 1 m ~<br>erz (h) := m X h(zi )<br>i=1<br>(recall equation (8)). Similarly, for any product probability measure P = P1 � � � � � Pn on<br>(X � Y )n , define<br>~ ~<br>erP (h) := h(z ) dP(z )<br>ZZ n<br>(recall equation (26)). For  any h; h0 : ( X � Y )n ! [0; 1℄ (not necessarily of the form h1 � � � � � hn ),<br>define<br>dP (h; h0 ) := jh(~z ) � h0 (~z )j dP(~z )<br>ZZ n<br>(recall equation (17)). For any class of functions H  mapping (X � Y )n  to [0; 1℄ , define<br>C ("; H ) := sup N ( "; H ; dP )<br>P<br>where the supremum is over all product probability measures on (X � Y )n  and N ( "; H ; dP )  is the<br>size of the smallest " -cover of H  under dP (recall Definition 4).<br>The following theorem is the main result from which the rest of the uniform convergence results<br>in this paper are derived.<br>Theorem 18. Let H � H1 � � � � � Hn be a permissible class of functions mapping (X � Y ) n  into<br>2 n<br>[0; 1℄ . Let z 2 (X � Y )(m;n) be generated by m � 2=(� � ) independent trials from (X � Y )<br>according to some product probability measure P = P1 � � � � � Pn . For all � > 0 , 0 < � < 1 ,<br>Pr z 2 (X � Y ) (m;n) : sup d� [er^ z (h); er P (h)℄ > �<br>� H �<br>2<br>� 4C (�� =8; H ) exp (�� � nm=8): (53)<br>The following immediate corollary will also be of use later.<br>Corollary 19. Under the same conditions as Theorem 18, if<br>8 4C � ��8 ; H � 2<br>m � max 2 log ; 2 ; (54)<br>� � n Æ � �<br>( )<br>then<br>Pr z 2 (X � Y )(m;n) : sup d� [ er^ z (h); er P (h)℄ > � � Æ (55)<br>� H �<br>


A.1 Proof of Theorem 18<br>


The proof is via a double symmetrization argument of the kind given in chapter 2 of Pollard (1984).<br>I have also borrowed some ideas from the proof of Theorem 3 in Haussler (1992).<br>
180 

### A.1.1 FIRST SYMMETRIZATION 



An extra piece of notation: for all z 2 (X � Y )(2m;n) , let z(1)  be the top half of z  and z(2)  be the<br>bottom half, viz:<br>z11 : : : z1n zm+1;1 : : : zm+1;n<br>z(1) = ... ... ... z(2) = ... ... ...<br>zm1 : : : zmn z2m;1 : : : z2m;n<br>The following lemma is the first “symmetrization trick.” We relate the probability of large deviation<br>between an empirical estimate of the loss and the true loss to the probability of large deviation<br>between two independent empirical estimates of the loss.<br>Lemma 20. Let H be a permissible set of functions from ( X � Y )n into [0; 1℄ and let P be a<br>2<br>probability measure on (X � Y )n . For all � > 0; 0 < � < 1  and m � �2 � ,<br>^<br>Pr z 2 (X � Y ) (m;n) : sup d� [erz (h); er P (h)℄ > �<br>� H �<br>^ ^ �<br>� 2 Pr z 2 Z (2m;n) : sup d� �erz(1) (h); er z(2) (h)� > : (56)<br>� H 2 �<br>Proof. Note first that permissibility of H guarantees the measurability of suprema over H<br>(Lemma 32 part 5). By the triangle inequality for d� , if d� �er^ z(1) (h); er P (h)� > � and<br>d� �er^ z(2) (h); er P (h)� < �=2 , then d� �er^ z(1) (h); er^ z(2) (h)� > �=2 . Thus,<br>Prnz 2 (X � Y )(2m;n) : 9h 2 H : d� �er^ z(1) (h); er^ z(2) (h)� > �2 o<br>� Prnz 2 ( X � Y ) (2m;n) : 9h 2 H : d� �er^ z(1) (h); er P (h)� > � and<br>(57)<br>^<br>d� �erz(2) (h); er P (h)� < �=2o :<br>By Chebyshev’s inequality, for any fixed h ,<br>Pr z 2 (X � Y )(m;n) : d� [er^ z (h); er P (h)℄ < �<br>n 2 o<br>^<br>� Pr z 2 (X � Y )(m;n) : jerz (h) � erP (h)j < �<br>� � 2 �<br>� 1 � erP (h)(1 �2 erP (h))<br>m� � =4<br>1<br>�<br>2<br>2<br>as m � 2=(� � )  and erP (h) � 1 . Substituting this last expression into the right hand side of (57)<br>gives the result.<br>


181<br>
BAXTER 

### A.1.2 SECOND SYMMETRIZATION 

The second symmetrization trick bounds the probability of large deviation between two empirical estimates of the loss (i.e. the right hand side of (56)) by computing the probability of large deviation when elements are randomly permuted between the first and second sample. The following definition introduces the appropriate permutation group for this purpose. 



Definition 7. For all integers m; n � 1 , let �(2m;n) denote the set of all permutations � of the<br>sequence of pairs of integers f(1; 1); : : : ; (1; n); : : : ; (2m; 1); : : : ; (2m; n)g  such that for all i , 1 �<br>i � m , either � (i; j ) = (m + i; j )  and � (m + i; j ) = (i; j )  or � (i; j ) = (i; j ) and � (m + i; j ) =<br>(m + i; j ) .<br>For any z 2 (X � Y )(2m;n) and any � 2 �(2m;n) , let<br>z� (1;1) : : : z� (1;n)<br>z� := ... ... ...<br>z� (2m;1) : : : z� (2m;n) :<br>Lemma 21. Let H = H1 � � � � � Hn be a permissible set of functions mapping (X � Y )n into<br>^ 1 M<br>[0; 1℄ (as in the statement of Theorem 18). Fix z 2 (X � Y ) (2m;n) and let H := ff ; : : : ; f g  be<br>an �� =8 -cover for (H ; dz ) , where dz (h; h0 ) := 21m P2i=1m jh(~zi ) � h0 (~zi )j  where the ~zi are the rows<br>of z . Then,<br>Pr � 2 �(2m;n) : sup d� �er^ z� (1) (h); er^ z� (2) (h)� > �<br>� H 2 �<br>M<br>� X Pr n� 2 �(2m;n) : d� �er^ z� (1) (f i ); er^ z� (2) (f i )� > �4 o ; (58)<br>i=1<br>where each � 2 �(2m;n) is chosen uniformly at random.<br>Proof. no suchFix h  for any � 2 �(2�m;n we are already done). ) and let h 2 H  be Choosesuch that f 2d�H�^er  such that ^ z� (1) (h);der^z (zh� ;(2)f )(h�)���>=8�= . Without loss 2 (if there is<br>of generality we can assume f is of the form f = f1 � � � � � fn . Now,<br>2 dz (h; f ) = P2i=1m ���Pnj =1 hj (zij ) � fj (zij )���<br>� � mn<br>= P2i=1m ���Pnj =1 hj (z� (i;j ) ) � fj (z� (i;j ) )���<br>� mn<br>���Pmi=1 Pnj =1 �hj (z� (i;j ) ) � fj (z� (i;j ) )� ���<br>� m n<br>� mn + Pi=1 Pj =1 �hj (z� (i;j ) ) + fj (z� (i;j ) )�<br>���P2i=mm+1 Pnj =1 �hj (z� (i;j ) ) � fj (z� (i;j ) )� ���<br>+<br>2m n<br>� mn + Pi=m+1 Pj =1 �hj (z� (i;j ) ) + fj (z� (i;j ) )�<br>^ ^ ^ ^<br>= d� �erz� (1) (h); er z� (1) (f )� + d� �erz� (2) (h); er z� (2) (f )� :<br>182<br>


A MODEL OF INDUCTIVE BIAS LEARNING<br>


Hence, by the triangle inequality for d� ,<br>2 dz (h; f )+ d� �er^ z� (1) (f ); er^ z� (2) (f )� � d� �er^ z� (1) (h); er^ z� (1) (f )�<br>�<br>^ ^ ^ ^<br>+ d� �erz� (2) (h); er z� (2) (f )� + d� �erz� (1) (f ); er z� (2) (f )� (59)<br>^ ^<br>� d� �erz� (1) (h); er z� (2) (h)� :<br>But �2 dz (h; f ) � �=4 by construction and d� �er^ z� (1) (h); er^ z� (2) (h)� > �=2 by assumption, so<br>(59) implies d� �er^ z� (1) (f ); er^ z� (2) (f )� > �=4 . Thus,<br>n� 2 �(2m;n) : 9h 2 H : d� �er^ z� (1) (h); er^ z� (2) (h)� > �2 o<br>� n� 2 �(2m;n) : 9f 2 H^ : d� �er^ z� (1) (f ); er^ z� (2) (f )� ) > �4 o ;<br>which gives (58).<br>Now we bound the probability of each term in the right hand side of (58).<br>n<br>Lemma 22. Let f : (X � Y ) ! [0; 1℄ be any function that can be written in the form f =<br>f1 � � � � � fn . For any z 2 (X � Y )(2m;n) ,<br>2<br>Pr n� 2 �(2m;n) : d� �er^ z� (1) (f ); er^ z� (2) (f )� > �4 o � 2 exp � �� 8� mn � ; (60)<br>where each � 2 �(2m;n) is chosen uniformly at random.<br>Proof. For any � 2 �(2m;n) ,<br>� m n � �<br>d� �er^ z� (1) (f ); er^ z� (2) (f )� = ��Pi=1 Pj =1 �fj (z2�m(i;j ) )n fj (z� (m+i;j ) )��� : (61)<br>� mn + Pi=1 Pj =1 fj (zij )<br>To simplify the notation denote fj (zij ) by �ij . For each pair ij , 1 � i � m , 1 � j � n , let<br>Yij be an independent random variable such that Yij = �ij � �m+i;j with probability 1=2 and<br>Yij = �m+i;j � �ij with probability 1=2 . From (61),<br>Pr n� 2 �(2m;n) : d� �er^ z� (1) (f ); er^ z� (2) (f )� > �4 o<br>8 � m n � 2m n 9<br>� � �<br>= Pr <� 2 �(2m;n) : ��X X �fj (z� (i;j ) ) � fj (z� (m+i;j ) )��� > 4 0� mn + X X �ij 1=<br>: �� i=1 j =1 �� i=1 j =1 A;<br>8� m n � 2m n 9<br>� � �<br>= Pr <�� X X Yij �� 4 0� mn + X X �ij 1=<br>:�� i=1 j =1 �� i=1 j =1 A;<br>For zero-mean independent random variables Y1 ; : : : ; Yk with bounded ranges ai � Yi � bi , Ho-<br>effding’s inequality (Devroye, Gy¨orfi, & Lugosi, 1996) is<br>� k � 2<br>Pr ( ����Xi=1 Yi ���� � � ) � 2 exp � Pki=1 (2b�i � ai )2 ! :<br>183<br>
BAXTER 



Noting that the range of each Yij is [�j�ij � �i+m;j )j; j�ij � �i+m;j )j℄ , we have<br>2<br>Pr 8<:������ Xi=1m Xj =1n Yij ������ > �4 0� mn + Xi2=1m Xj =1n �ij 1A9=; � 2 exp 0B� 32�2Ph�mimn=1 P+nj =1P(2i=1�mijP�nj�=1i+�m;jij i)2 1CA<br>Let � = P2i=1m Pnj =1 �ij . As 0 � �ij � 1 , Pmi=1 Pnj =1 (�ij � �m+ij )2 � � . Hence,<br>2<br>2 2m n<br>2 exp 0B� 32� Ph�mimn=1 P+nj =1P(i=1�ijP�j�=1i+�m;jij i)2 1C � 2 exp �� �2 (� mn32�+ � )2 � :<br>A<br>(� mn + � )2 =� is minimized by setting � = � mn  giving a value of 4� mn . Hence<br>2<br>Pr n� 2 �(2m;n) : d� �er^ z� (1) (f ); er^ z� (2) (f )� > �4 o � 2 exp � �� 8� mn � ;<br>as required.<br>A.1.3 PUTTING IT TOGETHER<br>For fixed z 2 ( X � Y ) (2m;n) , Lemmas 21 and 22 give:<br>Pr � 2 �(2m;n) : sup d� �er^ z� (1) (h); er^ z� (2) (h)� > �<br>� H 2 �<br>2<br>� � mn<br>� 2N (�� =8; H ; dz )) exp � :<br>� 8 �<br>Note that dz is simply dP where P = (P1 ; : : : ; Pn ) and each Pi is the empirical distribution that<br>puts point mass 1=m  on each zj i ; j = 1; : : : ; 2m  (recall Definition 3). Hence,<br>Pr � 2 �(2m;n) ; z 2 ( X � Y ) (2m;n) : sup d� �er^ z� (1) (h); er^ z� (2) (h)� > �<br>� H 2 �<br>2<br>� � mn<br>� 2C (�� =8; H ) exp � :<br>� 8 �<br>Now, for a random choice of z , each zij in z  is independently (but not identically) distributed and �<br>only ever swaps zij and zi+m;j (so that � swaps a zij drawn according to Pj with another component<br>drawn according to the same distribution). Thus we can integrate out with respect to the choice of<br>� and write<br>Pr z 2 (X � Y ) (2m;n) : sup d� �er^ z(1) (h); er^ z(2) (h)� > �<br>� H 2 �<br>2<br>� � mn<br>� 2C (�� =8; H ) exp � :<br>� 8 �<br>


Applying Lemma 20 to this expression gives Theorem 18.<br>


184<br>
A MODEL OF INDUCTIVE BIAS LEARNING 

### **A.2 Proof of Theorem 2** 



Another piece of notation is required for the proof. For any hypothesis space H  and any probability<br>measures P = (P1 ; : : : ; Pn )  on Z , let<br>er^ P (H ) := n1 Xn hinf2H erPi (h):<br>i=1<br>Note that we have used er^ P (H ) rather than erP (H ) to indicate that er^ P (H ) is another empirical<br>estimate of erQ (H ) .<br>With the (n; m) -sampling process, in addition to the sample z there is also generated a se-<br>quence of probability measures, P = ( P1 ; : : : ; Pn ) although these are not supplied to the learner.<br>This notion is used in the following Lemma, where Prf(z; P) 2 (X � Y ) (n;m) � P n : Ag  means<br>“the probability of generating a sequence of measures P  from the environment (P ; Q)  and then an<br>(n; m) -sample z  according to P  such that A holds”.<br>Lemma 23. If<br>Pr (z; P) 2 (X � Y )(n;m) � P n : sup d� [er^ z (H ); er^ P (H )℄ > � � Æ ; (62)<br>� H 2 � 2<br>and<br>Pr P 2 P n : sup d� [ er^ P (H ); er Q (H )℄ > � � Æ ; (63)<br>� H 2 � 2<br>then<br>Pr z 2 (X � Y )(n;m) : sup d� [er^ z (H ); erQ (H )℄ > � � Æ:<br>� H �<br>Proof. Follows directly from the triangle inequality for d� .<br>We treat the two inequalities in Lemma 23 separately.<br>A.2.1 INEQUALITY (62)<br>In the following Lemma we replace the supremum over H 2 H in inequality (62) with a supremum<br>over h 2 H n .<br>Lemma 24.<br>Pr (z; P) 2 (X � Y )(n;m) � P n : sup d� [ er^ z (H ); er^ P (H )℄ > �<br>� H �<br>� Pr (z; P) 2 (X � Y ) (n;m) � P n : sup d� [er^ z (h); er P (h)℄ > � (64)<br>H n<br>( l )<br>185<br>
BAXTER 



Proof. Suppose that (z; P) are such that sup H d� [er^ z (H ); er^ P (H )℄ > � . Let H satisfy this in-<br>equality. Suppose first that er^ z (H ) � er^ P (H ) . By the definition of er^ z (H ) , for all " > 0 there<br>exists h 2 H n := H � � � � � H  such that er^ z (h) < er^ z (H ) + " . Hence by property (3) of the d�<br>metric, for all " > 0 , there exists h 2 H n such that d� [ er^ z (h); er^ z (H )℄ < " . Pick an arbitrary h<br>satisfying this inequality. By definition, er^ P (H ) � erP (h) , and so er^ z (H ) � er^ P (H ) � erP (h) .<br>As d� [er^ z (H ); er^ P (H )℄ > � (by assumption), by the compatibility of d� with the ordering on the<br>^<br>reals, d� [erz (H ); er P (h)℄ > � = � + Æ , say. By the triangle inequality for d� ,<br>^ ^ ^ ^<br>d� [erz (h); er P (h)℄ + d� [erz (h); er z (H )℄ � d� [erz (H ); er P (h)℄ = � + Æ:<br>^<br>Thus d� [erz (h); er P (h)℄ > � + Æ � " and for any " > 0 an h satisfying this inequality can be<br>^<br>found. Choosing " = Æ  shows that there exists h 2 H n  such that d� [erz (h); er P (h)℄ > � .<br>If instead, er^ P (H ) < er^ z (H ) , then an identical argument can be run with the role of z and P<br>interchanged. Thus in both cases,<br>sup d� [er^ z (H ); er^ P (H )℄ > � ) 9h 2 H nl : d� [er^ z (h); er P (h)℄ > �;<br>H<br>which completes the proof of the Lemma.<br>By the nature of the (n; m)  sampling process,<br>Pr (z; P) 2 ( X � Y )(n;m) � P n sup : d� [ er^ z (h); er P (h)℄ > �<br>H n<br>( l )<br>= Pr z 2 (X � Y )(n;m) : sup d� [ er^ z (h); er P (h)℄ > � dQn (P): (65)<br>P 2PZ n ( H nl )<br>Now H nl � K � � � � � K  where K := fhl : h 2 H : H 2 H g  and H nl is permissible by the assumed<br>permissibility of H (Lemma 32, Appendix D). Hence H nl satisfies the conditions of Corollary 19<br>and so combining Lemma 24, Equation (65) and substituting �=2  for � and Æ =2  for Æ in Corollary<br>19 gives the following Lemma on the sample size required to ensure (62) holds.<br>Lemma 25. If<br>32 8C (�� =16; H nl ) 8<br>m � max 2 log ; 2<br>� � � n Æ � � �<br>then<br>Pr (z; P) 2 ( X � Y ) (n;m) � P n : sup d� [ er^ z (H ); er^ P (H )℄ > � � Æ :<br>� H 2 � 2<br>A.2.2 INEQUALITY (63)<br>Note that erP (H ) = n1 Pni=1 H � (Pi ) and erQ (H ) = E P �Q H � (P ) , i.e the expectation of H � (P )<br>where P is distributed according to Q . So to bound the left-hand-side of (63) we can apply Corollary<br>19 with n = 1 , m  replaced by n , H  replaced by H � , � and Æ  replaced by �=2  and Æ =2  respectively,<br>P replaced by Q  and Z replaced by P . Note that H � is permissible whenever H is (Lemma 32).<br>Thus, if<br>32 8C (�� =16; H � ) 8<br>n � max 2 log ; 2 (66)<br>� � � Æ � � �<br>186<br>


A MODEL OF INDUCTIVE BIAS LEARNING<br>


then inequality (63) is satisfied.<br>


Now, putting together Lemma 23, Lemma 25 and Equation 66, we have proved the following<br>more general version of Theorem 2.<br>


Theorem 26. Let H be a permissible hypothesis space family and let z  be an (n; m) -sample gen-<br>erated from the environment ( P ; Q) . For all 0 < �; Æ < 1  and � > 0 , if<br>32 8C (�� =16; H � ) 8<br>n � max 2 log ; 2<br>� � � Æ � � �<br>32 8C (�� =16; H nl ) 8<br>and m � max 2 log ; 2 ;<br>� � � n Æ � � �<br>then<br>Pr z 2 (X � Y )(n;m) : sup d� [er^ z (H ); er Q (H )℄ > � � Æ<br>� H �<br>To get Theorem 2, observe that erQ (H ) > er^ z (H ) + " ) d� [ er^ z (H ); er Q (H )℄ > "=(2 + � ) .<br>Setting � = "=(2 + � ) and maximizing �2 � gives � = 2 . Substituting � = "=4 and � = 2 into<br>Theorem 26 gives Theorem 2.<br>A.3 The Realizable Case<br>In Theorem 2 the sample complexity for both m and n scales as 1="2 . This can be improved to<br>1=" if instead of requiring er Q (H ) � er^ z (H ) + " , we require only that erQ (H ) � �er^ z (H ) + "<br>^<br>for some � > 1 . To see this, observe that erQ (H ) > erz (H )(1 + �)=(1 � �) + �� =(1 � �) )<br>^<br>d� [erz (H ); er Q (H )℄ > � , so setting �� =(1 � �) = "  in Theorem 26 and treating � as a constant<br>gives:<br>Corollary 27. Under the same conditions as Theorem 26, for all " > 0  and 0 < �; Æ < 1 , if<br>32 8C ((1 � �)"=16; H � ) 8<br>n � max � �(1 � �)" log Æ ; �(1 � �)" �<br>32 8C ((1 � �)"=16; H nl ) 8<br>and m � max � �(1 � �)"n log Æ ; �(1 � �)" � ;<br>then<br>Pr �z 2 ( X � Y ) (n;m) : supH erQ (H ) � 11 +� �� er^ z (H ) + "� � Æ:<br>^<br>These bounds are particularly useful if we know that erz (H ) = 0 , for then we can set � = 1=2<br>�<br>(which maximizes �(1 �) ).<br>Appendix B. Proof of Theorem 6<br>Recalling Definition 6, for H of the form given in (32), H nl can be written<br>H nl = fg1 Æ f � � � � � gn Æ f : g1 ; : : : ; gn 2 Gl and f 2 F g :<br>To f�: ( write X � HY nl)n as ! a ( composition V � Y )n  by of two function classes note that if for each f : X ! V we define<br>�<br>f (x1 ; y1 ; : : : ; xn ; yn ) := (f (x1 ); y1 ; : : : ; f (xn ); yn )<br>187<br>
BAXTER 



then g1 Æ f � � � � � gn Æ f = g1 � � � � � gn Æ f� . Thus, setting Gln := Gl � � � � � Gl and F := ff�: f 2<br>F g ,<br>H nl = Gln Æ F : (67)<br>The following two Lemmas will enable us to bound C ("; H nl ) .<br>Lemma 28. Let H : X � Y ! [0; 1℄ be of the form H = Gl Æ F where X � Y �!F V � Y �G!l<br>[0; 1℄ . For all "1 ; "2 > 0 ,<br>C ("1 + "2 ; H ) � CGl ("1 ; F ) C ("2 ; Gl ):<br>Proof. Fix a measure P on X � Y and let F be a minimum size "1 -cover for (F ; d[P ;Gl ℄ ) . By<br>definition jF j � CGl ("1 ; F ) . For each f 2 F let Pf be the measure on V � Y defined by Pf (S ) =<br>�1 �1<br>P (f (S )) for any set S in the � -algebra on V � Y ( f is measurable so f (S ) is measurable).<br>Let Gf be a minimum size "2 -cover for (Gl ; dPf ) . By definition again, jGf j � C ("2 ; Gl ) . Let<br>N := fg Æ f : f 2 F and g 2 Gf g . Note that jN j � CGl ("1 ; F )C ("2 ; Gl ) so the Lemma will be<br>proved if N can be shown to be an "1 + "2 -cover for (H ; dP ) . So, given any g Æ f 2 H choose<br>f 0 2 F such that d[P ;Gl ℄ (f ; f 0 ) � "1 and g 0 2 Gf 0 such that dPf 0 (g ; g 0 ) � "2 . Now,<br>0 0 0 0 0 0<br>dP (g Æ f ; g Æ f ) � dP (g Æ f ; g Æ f ) + dP (g Æ f ; g Æ f )<br>� d[P ;Gl ℄ (f ; f 0 ) + dPf 0 (g ; g 0 )<br>� "1 + "2 :<br>where the first line follows from the triangle inequality for dP and the second line follows from<br>the facts: dP (g Æ f 0 ; g 0 Æ f 0 ) = dPf (g ; g 0 ) and dP (g Æ f ; g Æ f 0 ) � d[P ;Gl ℄ (f ; f 0 ) . Thus N is an<br>"1 + "2 -cover for (H ; dP )  and so the result follows.<br>Recalling the definition of H1 � � � � � Hn (Definition 6), we have the following Lemma.<br>Lemma 29.<br>n<br>C ("; H1 � � � � � Hn ) � Y C ("; Hi )<br>i=1<br>Proof. Fix a product probability measure P = P1 � � � � � Pn on (X � Y )n . Let N1 ; : : : ; Nn be<br>" -covers of (H1 ; dP1 ) : : : ; (Hn ; dPn ) . and let N = N1 � � � � � Nn . Given h = h1 � � � � � hn 2<br>H1 � � � � � Hn , choose g1 � � � � � gn 2 N such that dPi (hi ; gi ) � "  for each i = 1; : : : ; n . Now,<br>� n n �<br>1 � �<br>dP (h1 � � � � � hn ; g1 � � � � � gn ) = n ZZ n �� X hi (zi ) � X gi (zi )�� dP(z1 ; : : : ; zn )<br>� i=1 i=1 �<br>1 n<br>� n X dPi (hi ; gi )<br>i=1<br>� ":<br>Thus N is an " -cover for H1 � � � � � Hn and as jN j = Qni=1 jNi j  the result follows.<br>


188<br>


A MODEL OF INDUCTIVE BIAS LEARNING<br>


B.1 Bounding C ( "; H nl )<br>From Lemma 28,<br>n n<br>C �"1 + "2 ; Gl Æ F � � C ( "1 ; Gl ) CGl n �"2 ; F � (68)<br>and from Lemma 29,<br>n n<br>C ( "1 ; Gl ) � C ("1 ; Gl ) : (69)<br>Using similar techniques to those used to prove Lemmas 28 and 29, CGl n ("; F ) can be shown to<br>satisfy<br>CGl n ("2 ; F ) � CGl ( "2 ; F ) : (70)<br>Equations (67), (68), (69) and (70) together imply inequality (34).<br>B.2 Bounding C ("; H � )<br>We wish to prove that C ("; H � ) � CGl ( "; F ) when H is a hypothesis space family of the form<br>H = fGl Æ f : f 2 F g . Note that each H � 2 H � corresponds to some Gl Æ f , and that<br>�<br>H (P ) = inf erP (g Æ f ):<br>g 2Gl<br>Any probability measure Q  on P induces a probability measure QX �Y on X � Y  , defined by<br>QX �Y (S ) = P (S ) dQ(P )<br>ZP<br>for any S in the � -algebra on X � Y  . Note also that if h; h0 are bounded, positive functions on an<br>arbitrary set A , then<br>� �<br>�� inf h(a) � inf h0 (a)�� � sup �� h(a) � h0 (a)�� : (71)<br>�a2A a2A � a2A<br>Let Q  be any probability measure on the space P of probability measures on X � Y  . Let H1� ; H2�<br>be two elements of H � with corresponding hypothesis spaces Gl Æ f1 ; Gl Æ f2 . Then,<br>� �<br>dQ (H1� ; H2� ) = �� inf erP (g Æ f1 ) � inf erP (g Æ f2 )�� dQ(P )<br>ZP � g 2Gl g 2Gl �<br>� sup jerP (g Æ f1 ) � erP (g Æ f2 )j dQ(P ) (by (71) above)<br>ZP g 2Gl<br>� sup jg Æ f1 (x; y ) � g Æ f2 (x; y )j dP (x; y ) dQ(P )<br>ZP ZX �Y g 2Gl<br>= d[QX �Y ;Gl ℄ (f1 ; f2 ):<br>The measurability of supGl g Æ f is guaranteed by the permissibility of H (Lemma 32 part 4, Ap-<br>pendix D). From dQ (H1� ; H2� ) � d[QX �Y ;Gl ℄ (f1 ; f2 )  we have,<br>N ("; H � ; dQ ) � N �"; F ; d[QX �Y ;Gl ℄ � ; (72)<br>


which gives inequality (35).<br>


189<br>
BAXTER 



B.3 Proof of Theorem 7<br>
In order to prove the bounds in Theorem 7 we have to apply Theorem 6 to the neural network hypothesis space family of equation (39). In this case the structure is 



Substituting these two expressions into (75) and (76) and applying Theorem 6 yields Theorem<br>7.<br>


190<br>
R d �!F R k �!G [0; 1℄ k where G = f(x1 ; : : : ; xk ) 7! � Pi=1 �i xi + �0 : (�0 ; �1 ; : : : ; �k ) 2 U g for some bounded � � subset U of R k +1 and some Lipschitz squashing function � . The feature class F : R d ! R k is the set of all one hidden layer neural networks with d inputs, l hidden nodes, k outputs, � as the squashing function and weights w 2 T where T is a bounded subset of R W . The Lipschitz restriction on � and the bounded restrictions on the weights ensure that F and G are Lipschitz classes. Hence there exists b < 1 such that for all f 2 F and x; x0 2 R d , kf (x) � f (x0 )k < bkx � x0 k and for all g 2 G and x; x0 2 R k , jg (x) � g (x0 )j < bkx � x0 k where k � k is the L1<sup>norm</sup> in each case. The loss function is squared loss. Now, gl (x; y ) = l (g (x); y ) = (g (x) � y )2 , hence for all g ; g 0 2 G and all probability measures k P on R � [0; 1℄ (recall that we assumed the output space Y was [0; 1℄ ), dP (gl ; gl0 ) = ��(g (v ) � y )2 � (g 0 (v ) � y )2 �� dP (v ; y ) ZRk �[0;1℄ � 2 ��g (v ) � g 0 (v )�� dPRk (v ); (73) ZRk where PRk is the marginal distribution on R k derived from P . Similarly, for all f ; f 0 2 F and d probability measures P on R � [0; 1℄ , d[P ;Gl ℄ (f ; f 0 ) � 2b kf (x) � f 0 (x)k dPRd (x): (74) ZRd Define C �"; G ; L1 � := sup N �"; G ; L1 (P )� ; P where the supremum is over all probability measures on (the Borel subsets of) R k , and N �"; G ; L1 (P )� is the size of the smallest " -cover of G under the L1 (P ) metric. Similarly set, C �"; F ; L1 � := sup N �"; F ; L1 (P )� ; P where now the supremum is over all probability measures on R d . Equations (73) and (74) imply C ("; Gl ) � C " ; G ; L1 (75) � 2 � CGl ( "; F ) � C " ; F ; L1 (76) � 2b � Applying Theorem 11 from Haussler (1992), we find C " ; Gl ; L1 � 2eb 2k +2 � 2 � � " � C " ; F ; L1 � 2eb2 2W : � 2b � � " � 

## **Appendix C. Proof of Theorem 14** 



This proof follows a similar argument to the one presented in Anthony and Bartlett (1999) for<br>ordinary Boolean function learning.<br>


First we need a technical Lemma.<br>


Lemma 30. Let � be a random variable uniformly distributed on f1=2 + � =2; 1=2 � � =2g , with<br>0 < � < 1 . Let �1 ; : : : ; �m be i.i.d. f1; �1g -valued random variables with Pr(�i = 1) = � for all<br>i . For any function f mapping f1; �1gn ! f1=2 + � =2; 1=2 � � =2g ,<br>1 � m� 2<br>Pr f �1 ; : : : ; �m : f (�1 ; : : : ; �m ) 6= �g > 1 � q1 � e 1�� 2 :<br>4<br>" #<br>Proof. Let N (� )  denote the number of occurences of +1  in the random sequence � = (�1 ; : : : ; �m ) .<br>The function f can be viewed as a decision rule, i.e. based on the observations � , f tries to guess<br>�<br>whether the probability of +1  is 1=2 + � =2  or 1=2 � =2 . The optimal decision rule is the Bayes<br>estimator: f (�1 ; : : : ; �m ) = 1=2 + � =2  if N (� ) � m=2 , and f (�1 ; : : : ; �m ) = 1=2 � � =2  otherwise.<br>Hence,<br>Pr ( f (� ) 6= �) � 21 Pr �N (� ) � m2 ����� = 21 � �2 �<br>+ 12 Pr �N (� ) < m2 ����� = 12 + �2 �<br>> 21 Pr �N (� ) � m2 ����� = 21 � �2 �<br>�<br>which is half the probability that a binomial (m; 1=2 � =2)2) random variable is at least m=22 . By<br>Slud’s inequality (Slud, 1977),<br>1 m� 2<br>Pr (f (� ) 6= �) > 2 Pr Z � s 1 � � 2 !<br>where Z is normal (0; 1) . Tate’s inequality (Tate, 1953) states that for all x � 0 ,<br>Pr ( Z � x) � 1 1 � p1 � e�x2 :<br>2 h i<br>Combining the last two inequalities completes the proof.<br>Let x 2 X (n;m) be shattered by H , with m = d H (n) . For each row i  in x  let Pi be the set of<br>all 2d distributions P on X � f�1g such that P (x; 1) = P (x; 0) = 0 if x  is not contained in the<br>i th row of x , and for each j = 1; : : : ; d H (n) , P (xij ; 1) = (1 � � )=(2d H (n)) and P (xij ; �1) =<br>(1 � � )=(2d H (n)) . Let P := P1 � � � � � Pn .<br>Note that for P = (P1 ; : : : ; Pn ) 2 P , the optimal error optP ( H n )  is achieved by any sequence<br>h� = (h�1 ; : : : ; h�n )  such that h�i (xij ) = 1  if and only if Pi (xij ; 1) = (1 + � )=(2d H (n)) , and H n<br>always contains such a sequence because H shatters x . The optimal error is then<br>opt P ( H n ) = erP (h� ) = n1 Xi=1n Pi fh�i (x) 6= y g = n1 Xi=1n dXjH=1(n) 21d H�(�n) = 1 �2 � ;<br>191<br>


=2)2) random variable is at least m=22 . By<br>


BAXTER<br>


and for any h = (h1 ; : : : ; hn ) 2 H n ,<br>erP (h) = optP ( H n ) + � jf (i; j ) : hi (xij ) 6= h�i (xij )g j : (77)<br>nd H (n)<br>For any (n; m) -sample z , let each element mij in the array<br>m11 � � � m1d H (n)<br>m(z) := ... ... ...<br>mn1 � � � mnd H (n)<br>equal the number of occurrences of xij in z .<br>Now, if we select P = (P1 ; : : : ; Pn ) uniformly at random from P , and generate an (n; m) -<br>sample z  using P , then for h = An (z)  (the output of the learning algorithm) we have:<br>� �<br>E ( jf (i; j ) : hi (xij ) 6= hi (xij )g j) = X P (m)E ( jf (i; j ) : hi (xij ) 6= hi (xij )g j j m)<br>m<br>n d H (n)<br>�<br>= X P (m) X X P ( h(xij ) 6= h (xij )jmij )<br>m i=1 j =1<br>where P (m)  is the probability of generating a configuration m  of the xij under the (n; m) -sampling<br>process and the sum is over all possible configurations. From Lemma 30,<br>P (h(xij ) 6= h� (xij )jmij ) > 1 21 � r1 � e� m1�ij��22 3 ;<br>4 4 5<br>hence<br>E � nd H1(n) jf (i; j ) : hi (xij ) 6= h�i (xij )g j� > nd H1(n) Xm P (m) Xi=1n dXjH=1(n) 41 241 � r1 � e� m1�ij��22 35<br>� 1 1 � q1 � e� d H (nm�)(12�� 2 ) (78)<br>4<br>" #<br>by Jensen’s inequality. Since for any [0; 1℄ -valued random variable Z , Pr(Z > x) � E Z � x , (78)<br>implies:<br>Pr 1 jf(i; j ) : hi (xij ) 6= h�i (xij )gj > � � > (1 � � )�<br>� nd H (n) �<br>where<br>� := 1 1 � q1 � e� d H (nm�)(12�� 2 ) (79)<br>4<br>" #<br>and � 2 [0; 1℄ . Plugging this into (77) shows that<br>Pr f (P; z) : erP (An (z)) > optP ( H n ) + � �� g > (1 � � )�:<br>


and for any<br>


192<br>


A MODEL OF INDUCTIVE BIAS LEARNING<br>


Since the inequality holds over the random choice of P , it must also hold for some specific choice<br>of P . Hence for any learning algorithm Ann there is some sequence of distributions P  such that<br>


of P . Hence for any learning algorithm Ann there is some sequence of distributions P  such that<br>Pr fz : erP (An (z)) > optP ( H n ) + � �� g > (1 � � )�:<br>Setting<br>(1 � � )� � Æ; and � �� � "; (80)<br>ensures<br>Pr f z : erP (An (z)) > opt P ( H n ) + "g > Æ: (81)<br>Assuming equality in (80), we get<br>Æ " 1 � �<br>� = 1 � � ; � = Æ � :<br>Solving (79) for m , and substituting the above expressions for � and � shows that (81) is satisfied<br>provided<br>m � d H (n) � Æ" �2 � 1 �� � �2 � 1 log 8Æ (1(1�����)2 2Æ ) (82)<br>" #<br>Setting � = 1 � aÆ for some a > 4 ( a > 4 since � < 1=4 and � = Æ =(1 � � ) ), and assuming<br>"; Æ � 1=(k a)  for some k > 2 , (82) becomes<br>m � d Ha(2n) �1 � k2 � log 8(aa�2 2) : (83)<br>Subject to the constraint a > 4 , the right hand side of (83) is approximately maximized at a =<br>8:7966 , at which point the value exceeds d H (n)(1 � 2=k )=(220"2 ) . Thus, for all k � 1 , if "; Æ �<br>1=9k  and<br>d H (n) �1 � k2 �<br>m � 220"2 ; (84)<br>then<br>Pr f z : erP (An (z)) > opt P ( H n ) + "g > Æ:<br>To obtain the Æ -dependence in Theorem 14 observe that by assumption H 1  contains at least two<br>functions h1 ; h2 , hence there exists an x 2 X  such that h1 (x) 6= h2 (x) . Let P � be two distributions<br>concentrated on (x; 1) and (x; �1) such that P � (x; h1 (x)) = (1 � ")=2 and P � (x; h2 (x)) =<br>(1 � ")=2 . Let P+ := P + � � � � � P +  and P� := P � � � � � � P � be the product distributions on<br>(X � f�1g) n generated by P � , and h1 := (h1 ; : : : ; h1 ); h2 := (h2 ; : : : ; h2 ) . Note that h1 and h2<br>are both in H n . If P  is one of P�� and the learning algorithm An chooses the wrong hypothesis h ,<br>then<br>


193<br>
P�� and the learning algorithm erP (h) � optP ( H n ) = ": 

Now, if we choose P uniformly at random from fP+ ; P� g and generate an (n; m) -sample z ac-<br>cording to P , Lemma 30 shows that<br>1 nm"2<br>Prf(P; z) : erP (An (z)) � optP ( H n ) + "g > 1 � q1 � e 1�"2 ;<br>4<br>" #<br>which is at least Æ  if<br>1 � "2 1 1<br>m < "2 n log 8Æ (1 � 2Æ ) (85)<br>provided 0 < Æ < 1=4 . Combining the two constraints on m : (84) (with k = 7 ) and (85), and using<br>1<br>maxfx1 ; x2 g � 2 (x1 + x2 )  finishes the proof.<br>Appendix D. Measurability<br>In order for Theorems 2 and 18 to hold in full generality we had to impose a constraint called<br>“permissibility” on the hypothesis space family H . Permissibility was introduced by Pollard (1984)<br>for ordinary hypothesis classes H . His definition is very similar to Dudley’s “image admissible<br>Suslin” (Dudley, 1984). We will be extending this definition to cover hypothesis space families.<br>Throughout this section we assume all functions h map from (the complete separable metric<br>space) Z into [0; 1℄ . Let B (T )  denote the Borel � -algebra of any topological space T . As in Section<br>2.2, we view P , the set of all probability measures on Z , as a topological space by equipping it<br>with the topology of weak convergence. B (P )  is then the � -algebra generated by this topology. The<br>following two definitions are taken (with minor modifications) from Pollard (1984).<br>Definition 8. A set H  of [0; 1℄ -valued functions on Z  is  indexed  by the set T if there exists a function<br>f : Z � T ! [0; 1℄ such that<br>H = f f ( � ; t) : t 2 T g :<br>Definition 9. The set H  is  permissible  if it can be indexed by a set T such that<br>1. T is an analytic subset of a Polish 7 space T , and<br>2. the function f : Z � T ! [0; 1℄ indexing H by T is measurable with respect to the product<br>� -algebra B (Z ) � B (T ) .<br>An analytic subset T of a Polish space T is simply the continuous image of a Borel subset X<br>of another Polish space X . The analytic subsets of a Polish space include the Borel sets. They<br>are important because projections of analytic sets are analytic, and can be measured in a complete<br>measure space whereas projections of Borel sets are not necessarily Borel, and hence cannot be<br>measured with a Borel measure. For more details see Dudley (1989), section 13.2.<br>Lemma 31. H11 � � � � � Hn : (X � Y )n ! [0; 1℄ is permissible if H1 ; : : : ; Hn are all permissible.<br>


Lemma 31. H11<br>Proof. Omitted.<br>


We now define permissibility of hypothesis space families.<br>


7. A topological space is called  Polish  if it is metrizable such that it is a complete separable metric space.<br>
194 

**Definition 10.** _A hypothesis space family_ H = fH g _is_ permissible _if there exist sets_ S _and_ T _that are analytic subsets of Polish spaces_ S _and_ T _respectively, and a function_ f : Z � T � S ! [0; 1℄ _, measurable with respect to_ S � B (T ) � B (S ) _, such that_ H = �ff ( � ; t; s) : t 2 T g : s 2 S �: Let (X ; �; �) be a measure space and T be an analytic subset of a Polish space. Let A(X ) denote the analytic subsets of X . The following three facts about analytic sets are taken from Pollard (1984), appendix C. (a) If (X ; �; �) is complete then A(X ) � � . (b) A(X � T ) contains the product � -algebra � � B (T ) . (c) For any set Y in A(X � T ) , the projection �X Y of Y onto X is in A(X ) . Recall Definition 2 for the definition of H � . In the following Lemma we assume that (Z ; B (Z )) has been completed with respect to any probability measure P , and also that (P ; B (P )) is complete with respect to the environmental measure Q . **Lemma 32.** _For any permissible hypothesis space family_ H _, 1._ H nl<sup>_is permissible._</sup> _2._ fh 2 H : H 2 H g _is permissible. 3._ H _is permissible for all_ H 2 H _. 4._ supH<sup>_and_</sup> inf H<sup>_are measurable for all_</sup> H 2 H _. 5._ H � _is measurable for all_ H 2 H _. 6._ H � _is permissible. Proof._ As we have absorbed the loss function into the hypotheses h , H nl<sup>issimplythesetofall</sup> n -fold products H � � � � � H such that H 2 H . Thus (1) follows from Lemma 31. (2) and (3) are immediate from the definitions. As H is permissible for all H 2 H , (4) can be proved by an identical argument to that used in the “Measurable Suprema” section of Pollard (1984), appendix C. For (5), note that for any Borel-measurable h : Z ! [0; 1℄ , the function h : P ! [0; 1℄ defined by h(P ) := RZ h(z ) dP (z ) is Borel measurable Kechris (1995, chapter 17). Now, permissibility of H automatically implies permissibility of H := fh : h 2 H g , and H � = inf H<sup>so</sup> H � is measurable by (4). Now let H be indexed by f : Z � T � S ! [0; 1℄ in the appropriate way. To prove (6), define g : P � T � S ! [0; 1℄ by g (P ; t; s) := RZ f (z ; t; s) dP (z ) . By Fubini’s theorem g is a B (P ) � B (T ) � B (S ) -measurable function. Let G : P � S ! [0; 1℄ be defined by G(P ; s) := inf t2T g (P ; t; s) . G indexes H � in the appropriate way for H � to be permissible, provided it can be shown that G is B (P ) � B (S ) -measurable. This is where analyticity becomes important. Let g� := f(P ; t; s) : g (P ; t; s) > �g . By property (b) of analytic sets, A (P � T � S ) contains g�<sup>.</sup> The set G� := f(P ; s) : G(P ; s) > �g is the projection of g�<sup>onto</sup> P � S , which by property (c) is also analytic. As (P ; B (P ); Q) is assumed complete, G�<sup>is measurable, by property (a).Thus</sup> G is a measurable function and the permissibility of H � follows. 

195 

## **References** 

- Abu-Mostafa, Y. (1993). A method for learning from hints. In Hanson, S. J., Cowan, J. D., & Giles, C. L. (Eds.), _Advances in Neural Information Processing Systems 5_ , pp. 73–80 San Mateo, CA. Morgan Kaufmann. 

- Anthony, M., & Bartlett, P. L. (1999). _Neural Network Learning: Theoretical Foundations_ . Cambridge University Press, Cambridge, UK. 

- Bartlett, P. L. (1993). Lower bounds on the VC-dimension of multi-layer threshold networks. In _Proccedings of the Sixth ACM Conference on Computational Learning Theory_ , pp. 44–150 New York. ACM Press. Summary appeared in Neural Computation, 5, no. 3. 

- Bartlett, P. L. (1998). The sample complexity of pattern classification with neural networks: the size of the weights is more important than the size of the network. _IEEE Transactions on Information Theory_ , _44_ (2), 525–536. 

- Baxter, J. (1995a). _Learning Internal Representations_ . Ph.D. thesis, Department of Mathematics and Statistics, The Flinders University of South Australia. Copy available from http://wwwsyseng.anu.edu.au/ � jon/papers/thesis.ps.gz. 

- Baxter, J. (1995b). Learning internal representations. In _Proceedings of the Eighth International Conference on Computational Learning Theory_ , pp. 311–320. ACM Press. Copy available from http://wwwsyseng.anu.edu.au/ � jon/papers/colt95.ps.gz. 

- Baxter, J. (1997a). A Bayesian/information theoretic model of learning to learn via multiple task sampling. _Machine Learning_ , _28_ , 7–40. 

- Baxter, J. (1997b). The canonical distortion measure for vector quantization and function approximation. In _Proceedings of the Fourteenth International Conference on Machine Learning_ , pp. 39–47. Morgan Kaufmann. 

- Baxter, J., & Bartlett, P. L. (1998). The canonical distortion measure in feature space and 1-NN classification. In _Advances in Neural Information Processing Systems 10_ , pp. 245–251. MIT Press. 

- Berger, J. O. (1985). _Statistical Decision Theory and Bayesian Analysis_ . Springer-Verlag, New York. 

- Blumer, A., Ehrenfeucht, A., Haussler, D., & Warmuth, M. K. (1989). Learnability and the vapnikchervonenkis dimension. _Journal of the ACM_ , _36_ , 929–965. 

- Caruana, R. (1997). Multitask learning. _Machine Learning_ , _28_ , 41–70. 

- Devroye, L., Gy¨orfi, L., & Lugosi, G. (1996). _A Probabilistic Theory of Pattern Recognition_ . Springer, New York. 

- Dudley, R. M. (1984). _A Course on Empirical Processes_ , Vol. 1097 of _Lecture Notes in Mathematics_ , pp. 2–142. Springer-Verlag. 

- Dudley, R. M. (1989). _Real Analysis and Probability_ . Wadsworth & Brooks/Cole, California. 

196 

- Gelman, A., Carlin, J. B., Stern, H. S., & Rubim, D. B. (Eds.). (1995). _Bayesian Data Analysis_ . Chapman and Hall. 

- Good, I. J. (1980). Some history of the hierarchical Bayesian methodology. In Bernardo, J. M., Groot, M. H. D., Lindley, D. V., & Smith, A. F. M. (Eds.), _Bayesian Statistics II_ . University Press, Valencia. 

- Haussler, D. (1992). Decision theoretic generalizations of the pac model for neural net and other learning applications. _Information and Computation_ , _100_ , 78–150. 

- Heskes, T. (1998). Solving a huge number of similar tasks: a combination of multi-task learning and a hierarchical Bayesian approach. In Shavlik, J. (Ed.), _Proceedings of the 15th International Conference on Machine Learning (ICML ’98)_ , pp. 233–241. Morgan Kaufmann. 

- Intrator, N., & Edelman, S. (1996). How to make a low-dimensional representation suitable for diverse tasks. _Connection Science_ , _8_ . 

- Kechris, A. S. (1995). _Classical Descriptive Set Theory_ . Springer-Verlag, New York. 

- Khan, K., Muggleton, S., & Parson, R. (1998). Repeat learning using predicate invention. In Page, C. D. (Ed.), _Proceedings of the 8th International Workshop on Inductive Logic Programming (ILP-98)_ , LNAI 1446, pp. 65–174. Springer-Verlag. 

- Langford, J. C. (1999). Staged learning. Tech. rep., CMU, School of Computer Science. http://www.cs.cmu.edu/ � jcl/research/ltol/staged latest.ps. 

- Mitchell, T. M. (1991). The need for biases in learning generalisations. In Dietterich, T. G., & Shavlik, J. (Eds.), _Readings in Machine Learning_ . Morgan Kaufmann. 

- Parthasarathy, K. R. (1967). _Probabiliity Measures on Metric Spaces_ . Academic Press, London. 

- Pollard, D. (1984). _Convergence of Stochastic Processes_ . Springer-Verlag, New York. 

- Pratt, L. Y. (1992). Discriminability-based transfer between neural networks. In Hanson, S. J., Cowan, J. D., & Giles, C. L. (Eds.), _Advances in Neural Information Processing Systems 5_ , pp. 204–211. Morgan Kaufmann. 

- Rendell, L., Seshu, R., & Tcheng, D. (1987). Layered concept learning and dynamically-variable bias management. In _Proceedings of the Tenth International Joint Conference on Artificial Intelligence (IJCAI ’87)_ , pp. 308–314. IJCAI , Inc. 

- Ring, M. B. (1995). _Continual Learning in Reinforcement Environments_ . R. Oldenbourg Verlag. 

- Russell, S. (1989). _The Use of Knowledge in Analogy and Induction_ . Morgan Kaufmann. 

- Sauer, N. (1972). On the density of families of sets. _Journal of Combinatorial Theory A_ , _13_ , 145–168. 

- Sharkey, N. E., & Sharkey, A. J. C. (1993). Adaptive generalisation and the transfer of knowledge. _Artificial Intelligence Review_ , _7_ , 313–328. 

197 

- Silver, D. L., & Mercer, R. E. (1996). The parallel transfer of task knowledge using dynamic learning rates based on a measure of relatedness. _Connection Science_ , _8_ , 277–294. 

- Singh, S. (1992). Transfer of learning by composing solutions of elemental sequential tasks. _Machine Learning_ , _8_ , 323–339. 

- Slud, E. (1977). Distribution inequalities for the binomial law. _Annals of Probability_ , _4_ , 404–412. 

- Suddarth, S. C., & Holden, A. D. C. (1991). Symolic-neural systems and the use of hints in developing complex systems. _International Journal of Man-Machine Studies_ , _35_ , 291–311. 

- Suddarth, S. C., & Kergosien, Y. L. (1990). Rule-injection hints as a means of improving network performance and learning time. In _Proceedings of the EURASIP Workshop on Neural Networks_ Portugal. EURASIP. 

- Sutton, R. (1992). Adapting bias by gradient descent: An incremental version of delta-bar-delta. In _Proceedings of the Tenth National Conference on Artificial Intelligence_ , pp. 171–176. MIT Press. 

- Tate, R. F. (1953). On a double inequality of the normal distribution. _Annals of Mathematical Statistics_ , _24_ , 132–134. 

- Thrun, S. (1996). Is learning the n-th thing any easier than learning the first?. In _Advances in Neural Information Processing Systems 8_ , pp. 640–646. MIT Press. 

- Thrun, S., & Mitchell, T. M. (1995). Learning one more thing. In _Proceedings of the International Joint Conference on Artificial Intelligence_ , pp. 1217–1223. Morgan Kaufmann. 

- Thrun, S., & O’Sullivan, J. (1996). Discovering structure in multiple learning tasks: The TC algorithm. In Saitta, L. (Ed.), _Proceedings of the 13th International Conference on Machine Learning (ICML ’96)_ , pp. 489–497. Morgen Kaufmann. 

- Thrun, S., & Pratt, L. (Eds.). (1997). _Learning to Learn_ . Kluwer Academic. 

- Thrun, S., & Schwartz, A. (1995). Finding structure in reinforcement learning. In Tesauro, G., Touretzky, D., & Leen, T. (Eds.), _Advances in Neural Information Processing Systems_ , Vol. 7, pp. 385–392. MIT Press. 

- Utgoff, P. E. (1986). Shift of bias for inductive concept learning. In _Machine Learning: An Artificial Intelligence Approach_ , pp. 107–147. Morgan Kaufmann. 

- Valiant, L. G. (1984). A theory of the learnable. _Comm. ACM_ , _27_ , 1134–1142. 

- Vapnik, V. N. (1982). _Estimation of Dependences Based on Empirical Data_ . Springer-Verlag, New York. 

- Vapnik, V. N. (1996). _The Nature of Statistical Learning Theory_ . Springer Verlag, New York. 

198 

