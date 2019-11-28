---
title: Neural Networks in Artificial Intelligence
draft: false

---

Issues in Neural Network Modeling
=================================

Introduction
------------

The past five years or so have seen a substantial amount of work being
done in the area of neural network modeling. This research attempts to
build model neural networks that solve significant psychological
problems, such as natural language understanding, visual processing,
etc.

A neural network is a computational model that is a directed graph
composed of nodes (sometimes referred to as units or neurons) and
connections between the nodes. With each node is associated a number,
referred to as the node's activation. Similarly, with each connection in
the network, a number is also associated, called its weight. These are
(very roughly) based on the firing rate of a biological neuron and the
strength of a synapse (connection between two neurons) in the brain.
There are usually some special nodes with their activations externally
set, called the input nodes; there may be, in addition, some nodes that
are distinguished as output nodes.

Each node's activation level, or activation for short, is based on the
activations of the nodes that have connections directed at it, and the
weights on those connections. A rule that updates the activations is
typically called the update rule. Typically, all the activations would
be updated simultaneously. Thus a neural network is a parallel model.
Because of the lack of general availability of parallel computers,
neural networks are typically simulated on conventional serial
computers. Learning in a neural network typically occurs by adjustment
of the weights, via a learning rule. The network is typically trained to
either complete an input pattern, classify an input pattern, or compute
a function of its input. At the beginning of learning, with the weights
all "wrong", the network performs badly at one of these tasks: at the
end, with the weights adjusted, one hopes that it will perform well.
Typically the update or learning rules do not change only the weights.
After learning, the weights are usually not changed further, unless
something new must be learned. Many network connection schemes, update
rules, and learning rules have been invented: these are covered in gory
detail in Chapter [2](#learn_methods){reference-type="ref"
reference="learn_methods"}.

Neural network models, like Al itself, date back to the 1950s. At the
beginning of the 1980s, many researchers, discouraged at the speed of
progress in traditional, symbolic Al, turned back to neural network
models. They felt that this line of research had been unjustly hurt by
the publication, in 1969, of Minsky and Papert's book Perceptrons, which
pointed up the limitations of a particular kind of neural model, the
two-layer perceptron. In 1985, a special issue of Cognitive Science was
devoted to the subject of "connectionism", which was a new name for the
field of neural network modeling, and which emphasized the idea that it
was the topology of the connections in a network that was critical to
its behavior. In this book, I will use the phrases "neural network
model," "connectionist model," and "connectionist network"
interchangeably.

Another reason for the renewed popularity of connectionist models was
the fact that parallel computers began to become available.
Connectionist models are one important variety of parallel computational
models. In 1986, a two-volume set of books entitled "Parallel
Distributed Processing: Explorations in the Microstructure of Cognition"
was published which reported research done by a group of cognitive
scientists at the University of California at San Diego, and led by
David Rumelhart and Jay McClelland [@Rumelhart1986AProcessing]. The most
important result in those volumes, although they contained many
important ideas, was the report of the discovery of the error
back-propagation algorithm for learning the weights in an associative
network (see Chapter 2). Much of the connectionist literature since then
has consisted of studies of what can be done with networks that are
trained using this algorithm.

Rumelhart, McClelland, and co-workers advocated a particular type of
network model, a distributed model, in which a typical concept, such as
a word, would be represented by a pattern of activation across a set of
nodes. This is in contrast with local representation, in which a
particular concept is represented by the activation of a single node. We
discuss the relative advantages of local versus distributed
representations in section [1.4](#reptypes){reference-type="ref"
reference="reptypes"}.

The Statistical Nature of Connectionist Models
----------------------------------------------

A critical fact about neural networks is that they are statistical
associative models. A typical network model has a set of input patterns
and a set of output patterns. The role of the network is to perform a
function that associates each input pattern with an output pattern. A
learning algorithm, such as back-propagation, uses the statistical
properties of a set of input/output pairs, called the training set, to
generalize, that is, generate outputs from novel inputs. Without the
ability to generalize, neural network models would be like look-up
tables, which are not very interesting. For a formal discussion of
generalization in learning, see Valiant (1984).

It is important to recognize the difference between statistical and
rule-based inference. Statistical inference allows for exceptions and
randomness in the association between two variables, whereas rules tend
to be rigid. In a neural network model, the history of the system---that
is, what training it has experienced---determines the system's response
to a new stimulus. Often, rule-based systems are non-adaptive, that is,
they do not respond to observed changes in the stimulus environment,
although they can be made to be adaptive. Rule-based systems can be made
to handle exceptions as well, at the expense of making the rules more
complex.

Thus neural networks derive their inspiration from two distinct yet
related fields: associationist psychology and neuroscience.
Associationist psychology has a long history: behaviorism is one form of
it, but the idea that human memory works associatively dates back at
least to classical times. Neuroscience and associationist psychology
have an uneasy alliance that comes from the simple observation that
neurons synapse with one another, therefore the firing rates of such
neurons are associated. If the brain is simply a web of such
associations, perhaps the mind is as well.

As Touretzky [@Touretzky1988ConnectionismAttachment] points out, usually
a connectionist system either classifies its input or performs some
other function on it. In either case, the function computed tends to be
a continuous one, with relatively similar outputs being assigned to
similar inputs. There may be a certain number of discontinuities
intrinsic to a classification task, since the outputs in such a task are
a discrete set of symbols representing the sets in which the inputs are
classified. Some type of measurement of similarity between patterns is
critical for a statistical process such as a neural network. In
computing its output for a given input, a connectionist model computes a
type of correlation between its input and the set of stored weights
associated with a given node in the layer above the input. Typically
this correlation is a dot (scalar) product; if $x_i$ and $w_i$ are the
$i$th components of the input and weight vectors respectively, then the
dot product is given by $$\sum_{i} x_i w_i$$

Yet there are many methods of forming statistical correlations between
patterns. A review may be found in Kohonen
[@Kohonen1988Self-OrganizationMemory]; a summary of that review follows.

Probably the best known measure of the distance between two vectors $x$
and $y$ is the Euclidean distance, given by
$$\sqrt(\sum_{i} (x_i-y_i)^2))$$ This generalizes to the Minkowski
metric, given by $$(\sum_{i} (x_i-y_i)^n)^\frac{1}{n}$$ which, when
$n$=l, is sometimes referred to as the "Manhattan" distance (because in
order to get between two points in Manhattan (New York City) one must
move along a grid). When $n$=2, it is the Euclidean distance.

If what matters is not the magnitude of the vectors that are being
compared, but their relative orientation $\theta$, this is given by
$$\cos\theta=\frac{x*y}{|x||y|}$$

The formula $\cos\theta=1$ implies x and y are parallel (and thus as
similar as possible); $\cos\theta$ implies $x$ and $y$ are orthogonal.

In fuzzy logic [@Zadeh1973OutlineProcesses], two scalars similar is
given by $$e(x,y) = max(min(x,y ),min(1-x,l-y))$$ where $x$ and $y$ are
drawn from the interval between 0 and 1 inclusive. In this formula, $x$
and $y$ are variables that represent the degree of truth of two
propositions, and $e$ represents the degree to which they are
equivalent. These differences, when they are taken between vector
components, can be combined using the Minkowski metric above with some
value of $n$ (most typically $n-1$ or 2).

Kohonen points out that this method is simpler to compute than a dot
product, since the maximum and minimum functions are simpler to compute
than the product function, and that, in forming the distance, this
method counts weak signal data more than the dot product does. All of
the similarity measures mentioned so far deal with real-valued input
vectors.

For discrete-valued vectors, one similarity metric is the Hamming
distance. This is simply the number of vector components in which two
vectors $x$ and $y$ differ. Each element of each vector being compared
is drawn from a finite set of symbols. In the case of binary vectors,
the Hamming distance is given by the sum of the exclusive-or function
(xor) across all components: $$\sum_{i} xor(x_i,y_i)$$. The exclusive-or
function takes two binary arguments and is one if exactly one of the
arguments is one, and zero otherwise. (This differs from the ordinary or
function in that the or function is one if either of its arguments are
one, that is, one of them or both.)

Although most of the computations carried out in connectionist models
involve the dot product of weights with input (often then composed with
a function, such as a sigmoid function), it is helpful to keep in mind
that this is only one of the many similarity functions that could be
used.

The Relevance of the Brain
--------------------------

Research in neural networks stems from the idea that simulating, on a
computer, the way that the brain processes information may prove useful
in understanding thought processes. Neural network research dates from
the 1940s. In 1943, McCulloch and Pitts published their classic paper "A
Logical Calculus of the Ideas Immanent in Nervous Activity"
[@McCulloch1943AActivity]. McCulloch and Pitts's neurons were simple
logic gates, the and, or, and not gates familiar to logic designers.
McCulloch and Pitts proved that a computer built out of these "formal
neurons" was Turing-equivalent, that is, equivalent to the most powerful
class of computing devices known, which are known as Turing machines.

As Cowan and Sharp [@Cowan1988NeuralIntelligence] and many others have
pointed out, actual neurons are much more complex than simple logic
gates, and "their complexities can be accurately simulated only by
intricate computer chips". These complexities have been elucidated only
in recent years, since McCulloch and Pitts's work. Most neuron models
that have been developed for neural network research are less complex
than actual neurons. Neural network researchers argue that since their
models are Turing-equivalent, they can simulate any computation at all,
so there is no need to use more complex neuron models. The brain
consists of approximately $10^{11}$ to $10^{12}$ neurons connected in a
complex fashion. Neural network and brain researchers believe that the
way that the neurons are connected to one another is critical to
understanding the behavior of the brain as an information-processing
system.

Distributed vs. Local Connectionism {#reptypes}
-----------------------------------

The brain, like the rest of the body, is built of unreliable components.
Neurons can wither and die. How is it possible that the brain continues
to function fairly reliably over a long period, despite this? This was a
question that interested John von Neumann, a mathematician who was one
of the founders of computer science
[@VonNeumann1956ProbabilisticComponents]. He devised a neural network
that utilized redundant neurons, using a "voting" protocol. In such a
net, a set of neurons "vote" on whether or not another neuron should
fire. If a majority of the neurons that input to the neuron in question
fire, i.e. vote "yes", then the outputting neuron will fire (and
possibly input into some other neurons). Thus if an outputting neuron
starts out with a strong majority, the failure of some of the neurons
inputting to it to fire will not affect its outputting, and thus
performance will not be degraded. Redundancy has been the cornerstone of
much work in reliable systems. Randell and his co-workers
[@Randell1978ReliabilityDesign] review work in this area.

Much of the debate among connectionists concerns the degree to which
information should be localized in a single neuron
[@Barlow1972SinglePsychology], or distributed across many neurons. A
distributed memory is the latter kind, one in which a symbol---for
example, an ordinary word---is represented by a pattern of activation.
While studies of brain function have proved that different
activities---for example, speech, vision, or motor control---are
localized in various specialized parts of the brain, many people believe
that single neurons do not represent high-level pieces of information.
The argument goes as follows: if one had a neuron that represented, for
example, one's grandmother, then there would be people running around
who were perfectly normal except for their inability to recognize their
grandmother (i.e. whose grandmother neurons had died). Since there are
no such people (as far as we know), therefore there cannot be any
grandmother neurons (or yellow Volkswagen neurons, etc.), or there must
be many redundant copies of a grandmother neuron. Human memory appears
to suffer from "uniform degradation"; instead of individual memories
becoming lost, performance on recall of all memories becomes worse and
worse (with age, injury and/or disease).

Local representations are also subject to a combinatorial explosion, in
which a node is needed for every concept. This is because there are an
infinite number of concepts, because of the natural combinatorial nature
of language, and each one would require an individual node. For example,
in a local representation, one would have to have a "tall blond man"
node, whereas in a distributed representation one would have nodes for
"tall", "blond", and "man", all of which would be activated
simultaneously to represent "tall blond man". These individual nodes are
often referred to as microfeatures to emphasize the fact that individual
concepts can be decomposed into them. Microfeatures are often chosen on
the basis of a researcher's feeling that the chosen concepts are somehow
basic to a complete semantics of the concepts being represented, but
very few researchers have chosen their microfeatures in a principled
way. For limits on how this might be done, see works on semantics such
as those by Barwise and Perry [@Barwise1983SituationsAttitudes] and
Jackendoff [@Jackendoff1983SemanticsCognition].

Research done at the University of Rochester emphasizes the use of local
representations in connectionist models [@Feldman1986NeuralKnowledge].
Feldman, Ballard, and their colleagues and students at Rochester have
applied these types of models to problems in knowledge representation,
vision, and language comprehension. For examples, see the reviews of the
work of Shastri in Chapter 4, Sabbah in Chapter 6, and Fanty im Chapter
7. The main advantage of local representations is they are relatively
easy to understand and implement. The main disadvantage is the
combinatorial explosion in the needed number of nodes. In distributed
representations, information is represented redundantly, but a single
neuron may participate in the representation of several pieces of
information.

This idea is not new in psychology: for instance, Pribram
[@Pribram1971LanguagesBrain] proposed a "holographic" theory of memory,
in which every memory was stored as a hologram. A small piece of a
hologram may be used to reconstruct the entire image (at lower
resolution), thus, the hologram contains redundancy. In the
connectionist formulation, a distributed memory is typically viewed as a
pattern of activation over a set of interconnected nodes in a neural
network. The relationship between local and distributed representations
is analogous to that between unary and higher radix number systems. In a
unary system, the number of symbols needed to represent a set of $n$
numbers is proportional to $n$, whereas in a binary or higher radix
system it is proportional to $\log_{r}n$, where $r$ is the radix.

The radix of the system is, in a distributed model, equivalent to the
number of states a given node can have. Normally there are two,
activated or deactivated, although nodes may have anywhere from two to
an infinite number of states. One problem with a distributed memory is
*crosstalk*. Consider three nodes A, B, and C, and three concepts 1, 2,
and 3. If concept 1 is represented by AB (i.e., nodes A and B
activated), concept 2 by BC, and concept 3 by AC, then if any two of the
concepts are activated, the third one will be as well, even if it is not
there. This is crosstalk; the concept erroneously evoked is called a
ghost.

One particular technique for constructing a distributed memory is known
as coarse-coding. Coarse-coding is best understood as a way to represent
a digitized image. Suppose there are two layers of neuron units, each of
which is a two-dimensional array, so that the input layer represents an
ordinary digitized image array, stored in the activations in the units.
Suppose one output unit is associated with each of a square array of
units in the input, say a 3 by 3 or a 4 by 4 array, then it responds
when any of the units in that small array are active. The array of units
that the output unit responds to is called its receptive field.
Typically, in a coarse-coded memory, output units have overlapping
receptive fields.

Figure
[\[fig\_cc\_symbol\_memory\]](#fig_cc_symbol_memory){reference-type="ref"
reference="fig_cc_symbol_memory"} shows a coarse-coded memory with 16
input units (shown as white circles), 9 output units (shown as grey
circles), and receptive fields of 4 units per output unit. The receptive
fields overlap so that the left two units of one output unit's field
compose the right two units of the receptive field of the output unit to
the right of it, and similarly for the output units above and below each
other.

![A Course-Coded Symbol Memory](coarsecoded.PNG)

[\[fig\_cc\_symbol\_memory\]]{#fig_cc_symbol_memory
label="fig_cc_symbol_memory"}

Coarse coding is a distributed memory technique because a single unit's
activation in the input units corresponds to a pattern of activation in
the output units. For instance, if unit (2,2) of the local input pattern
is activated, then units (1,1), (1,2), (2,1), and (2,2) of the
coarse-coded output units would be activated. Coarse-coding is a useful
method of reducing the number of units that is required to represent
some stimulus. However, since a coarse-coded image reduces resolution,
local detail is often lost. Distributed models in general require less
units than local models; the more distributed the model, the fewer
units.

Rosenfeld and Touretzky [@Rosenfeld1988AMemories] review techniques for
coarse-coded memory representation. They use the phrases coarse-coded
memory and distributed memory interchangeably. They define a
coarse-coded symbol memory (CCSM) as: (1) a set of N units with
binary-valued activations, (2) a set of $\alpha$ symbols and (3) a
mapping of each of the symbols onto a bit pattern in the units. That is,
a symbol is---as in most definitions of distributed memory---represented
by a pattern of activation across a set of binary units. Each unit has a
receptive field consisting of all the symbols in each unit for which it
is activated. A ghost is a pattern in the memory that corresponds to a
symbol that was not intended to be stored; it is the result of
crosstalk.

The failure rate of a CCSM is the rate at which ghosts emerge. They
define $P_{ghost}$ as the probability that a ghost will emerge, given
that the CCSM has stored a certain number of items $k$. They note that a
local representation is one in which $k=N=a$ and $P_{ghost}=0$, that is,
one in which each symbol has one unit assigned to it. Aside from the
terminology, there is no difference between a CCSM and a binary function
on a finite set of symbols.

One CCSM that they explore is the random receptors model, in which each
unit is assigned to each symbol with a probability $s$. They show that
the probability of a ghost is minimized when $s=(l/(k+l))$, where $k$ is
the number of items stored. They also show that, for this model, this
implies that the number of symbols a in the alphabet can be related to
$P_{ghost}$, the number of units $N$, and the number of symbols stored
$k$ (if $k$ is large) by a function
$$a(N,k,Pghost)=P_{ghost} e^{O-368(N/k)}$$ They note that this implies
that the capacity $k$ can be increased linearly by linearly increasing
the number of units $N$, because of the term $N/k$ in the above formula.
It also shows that the probability of representing a ghost symbol can be
reduced by reducing the number of symbols to be represented, or
increasing the number of units, which makes intuitive sense.

Rosenfeld and Touretzky go on to analyze the effects of dividing up a
CCSM into two or more CCSMs. They conclude that, unless you have some
information about which pairs of symbols will be stored at once, it does
not pay to split the system. They compare the coarse coding scheme used
by Touretzky and Hinton in DUCS (see section 3.2 for a complete
description) with the random receptors method. The DUCS memory uses
randomness, but in a more structured way than purely random receptors.
They note that the random receptors method gives a lower probability of
getting a ghost for a given number of system units and patterns stored,
but the DUCS system typically only needs slightly more units to achieve
the same result. They conclude that DUCS needs more units than a random
receptor model since there is more redundancy in representing a given
symbol. In the random receptor model, the number of units in a receptive
field of a unit is dependent on the number of symbols that are stored,
in DUCS it is fixed. This leads to sub-optimal performance.

They consider another distributed representation, the wickelphone
representation used by Rumelhart and McClelland [@Rumelhart1986OnVerbs]
(see section 7.12) in their model of the learning of the past tense
forms of English verbs, in which the phonemes of words are represented
by the phoneme name and its left and right context phonemes (since a
sound is different depending on its context). The name and context
phonemes are each represented by a microfeature vector of 11 bits
representing various phonetic features, such as voicing, place of
articulation, etc. A wickelphone therefore requires 33 bits of
information to describe it. They represent the wickelphones as
wickelfeatures, in which one wickelfeature node represents a conjunction
of three phonemes. A single wickelfeature node always is compatible with
several or many actual wickelphones; for instance, it could be
compatible with the letters "t","d", and "k". Their main point in
reviewing the wickelphone representation in the context of coarse coding
is to illustrate that microfeature-based distributed representations and
coarse-coding are not necessarily incompatible, but can complement one
another.

Designing a distributed memory is equivalent to assigning receptive
fields to the units in the system. They note that a stochastic method
for generating receptive fields can cause unacceptable variation in the
size of the fields, and affect the performance of the system. They have
developed a method for generating optimal fixed sized receptive fields,
the bounded overlap method, but this requires an exponential search,
which is too expensive in the typical case of a few thousand units
[@Rosenfeld1988AMemories]. They suggest a practical method for designing
a distributed memory whereby each unit in the CCSM has its receptive
field set to a consecutive set of $F$ symbols, chosen cyclically and
repeatedly, from the symbols in the symbol set in some order. The
receptive fields are then shuffled randomly for some time, by exchanging
a symbol between two receptive fields, those of units $x$ and $y$. They
exchange a symbol that is in the receptive field of $y$ but not $x$ for
one that is in the receptive field of x but not y. They note that the
pattern size (the size of a pattern representing a symbol) is $L=NF/a$,
where $a$ is the number of symbols, and the expected overlap $OE$
between two patterns ia $$OE = \frac{L(F-l)}{(a-l)}$$

We want to minimize the degree to which the overlap C(i,j) between the
patterns for the ith and jth symbols differs from the expected overlap,
as measured by the variance
$$V=\frac{1}{2}\sum_{i \neq j} [C(i,j)-O_E]$$

The receptive fields are shuffled until the variance ceases to be
reduced significantly.

Distributed Models: A Critique
------------------------------

Oden [@Oden1988WhyHope] gives a useful discussion of distributed models.
One of his main points is that a model that looks distributed is always
local from another point of view. For instance, in the model of schemata
presented by Rumelhart et al.[@Rumelhart1986SchemataModels], a bedroom
is represented by a series of nodes representing the objects in it. The
model is distributed with respect to the bedroom, but local with respect
to the objects in it. Similarly, in the case of coarse-coding, when a
stimulus is encoded in terms of the responses of detectors that have
overlapping areas of sensitivity, the array of responses is distributed
with respect to the stimuli, but is local with respect to the detectors.

Oden points out that distributivity is not a characteristic
predominantly of connectionist models; even the ordinary binary encoding
of a number is distributed, since each bit can participate in the
representation of many numbers. Distributivity, according to Oden, is a
question of degree and perspective. Oden tackles the common claim of
connectionists that the nodes in a network can be non-symbolic, that is,
that they can have no semantic interpretation, especially in a
distributed network. This claim is a resulting of misconstruing a
complex, hard-to-make interpretation as no interpretation. Oden gives
the rotation of a coordinate system as an example. A pair of coordinates
that could be readily interpreted in terms of one system could not be as
readily interpreted in terms of another, rotated, system. Yet all the
original information is preserved.

To give another example: if I take a digitized picture of Lenin and run
it through some invertible transformation so that the picture becomes
unrecognizable, does this make the result non-symbolic of Lenin? I think
not: all the information necessary to construct the picture of Lenin is
still there (since the transformation is invertible).

The third property of connectionist models that Oden discusses is their
continuousness. He notes that all models must deal with continuous data,
since the real world is continuous, but models differ in the degree to
which they transform these data to continuous or binary variables. Some
connectionist models have neurons that fire "on" when their input
exceeds a threshold, and "off"otherwise. These are discrete models; the
non-linear sigmoid response curve and the linear response curve of other
connectionist models are continuous models (see section 2.3).

Oden notes that connectionist models provide for "the use of best-fit
pattern matching, which in turn allows for the kinds of categorization,
content-addressability, and automatic generalization that (they) are
known for." Traditional cognitive models make discrete classifications
based on necessary criteria or cutoffs; they do not allow for there to
be degrees of fit to a set of possible concepts, as there is in a
connectionist model, where a concept node may be more or less activated,
or in fuzzy set theory, in which an element may have a degree of
membership in a set that is any value intermediate between 0 and 1. An
example of how a connectionist model may closely correspond to a
particular symbolic, cognitive approach-----the fuzzy propositional
approach-----is given in the following section.

Connectionist Models and the Fuzzy Propositional Approach
---------------------------------------------------------

Oden [@Oden1988FuzzyModels] has worked out a relationship between his
earlier work in symbolic cognitive science, in what he termed the fuzzy
propositional approach (based on fuzzy logic
[@Zadeh1973OutlineProcesses]), and connectionist models. In the fuzzy
propositional model, each proposition $A$ is assigned a truth value
$t(A)$ between 0 and 1. One form of this model uses a multiplicative law
for conjunction:

$$t(A \land B)=t(A)t(B)$$

The following rule is used for disjunction:

$$t(A \lor B)=(l-t(A))(l-t(B))$$

The result of amplifying a truth value $t(very(A))$ is given by
$t(A)^v$, where $v$ is some constant. These formulas are borrowed from
probability theory, although it is different to say that $A$ is partly
true than it is to say that $A$ is has a certain probability of being
true.

Starting with some experiments that Oden and Rueckl
[@Oden1986TakingWords] did on stimuli that vary continuously on two
dimensions, Oden derives some response formulas. The exact stimuli were
"eat" and "lot" (see Figure
[\[fig\_eat\_lot\]](#fig_eat_lot){reference-type="ref"
reference="fig_eat_lot"}).

![Stimuli simular to those used in experiments](eat_lot.PNG)

[\[fig\_eat\_lot\]]{#fig_eat_lot label="fig_eat_lot"}

The difference between them is based on a continuous variation of the
height of the loop in the "e" or the "l" (in their hand-written
versions) and the depth of the dip in the line that connects the "a" or
the "o" with the "t". If t is the truth value of the first letter being
"l" and "d" is the truth value of the second letter being "a", this
gives the formula for the truth value of "eat" relative to "lot" to be

$$\frac{(1-t)d}{(1-t)d+t(1-d)}$$

To map his scheme on to a connectionist network, he postulates that
there are nodes whose activation corresponds to "short," "tall," "deep,"
and "flat," where the first two of these refer to the loop of the e or
l, and the second two correspond to the dip between the a or o and the
t. Thus, "short" and "deep" would be positively connected to the "eat"
node, and "tall" and "fat" would be negatively connected to the "eat"
node. The "lot" node would have opposite connections. Such a network
would perform similarly to his fuzzy-logic-based system.

Oden makes the case that a symbolic system, such as the fuzzy
propositional model, can be used to understand connectionist systems,
which otherwise would be collections of nodes that operate as if by
magic. He argues that the symbolic and the neuronal levels are
complementary, not in conflict. It is not sufficient to design a system
that learns a task by adjustment of weights; one must be able to give a
semantic interpretation of the system.

Philosophical Issues
--------------------

Connectionism has stimulated vigorous discussion of its usefulness as a
research strategy for cognitive science. To its detractors, it is a new
form of associationism, and the debate on its merits simply a rehash of
the late 1950s debate between behaviorists and cognitivists, which the
cognitivists basically won, in that cognitivism became the prevailing
school of thought. Of course, connectionism is explicitly
representational, while behaviorism was anti-representational.

The best known philosophical arguments for connectionism, and against
it, are, respectively, those of Smolensky
[@Smolensky1988OnConnectionism], and Fodor and Pylyshyn
[@Fodor1988ConnectionismAnalysis]. Smolensky was a contributor to the
PDP books [@Rumelhart1986AProcessing]. Fodor and Pylyshyn are well known
proponents of the symbol-manipulation approach to AI; for a sample of
the earlier philosophical work, see Pylyshyn
[@Pylyshyn1984ComputationScience] and Fodor
[@Fodor1981Representations:Science]. The book edited by Graubard
[@Graubard1988TheFoundations] contains several articles concerned with
the philosophical controversy around connectionism.

Smolensky's "Proper Treatment" of Connectionism
-----------------------------------------------

Smolensky [@Smolensky1988OnConnectionism], in his controversial article
\"On the Proper Treatment of Connectionism\" attempts to characterize
the limitations and advantages of the connectionist approach, and to
reconcile it to the traditional approach to AI. He characterizes
connectionism as the \"sub-symbolic\" approach, as opposed to
traditional AI as embodied in the Physical Symbol System Hypothesis
(PSSH) of Newell [@Newell1980PhysicalSystems]. He interprets the PSSH as
stating that the "subsymbols" in the connectionist paradigm are
constituents of the symbols used by traditional AI. The two paradigms
also have different levels on which they operate; the symbolic approach
uses what Smolensky calls the conceptual level, and the connectionist
approach uses what is called the sub-conceptual level.

Smolensky says that natural language has provided the major theoretical
focus of the symbolic paradigm. Cultural knowledge about specific
domains is typically embodied in language, and linguistic symbol lists,
in the form of rules, are used in conjunction with some type of logic to
create simulations of human action. The machine acts as a rule
interpreter, which is a model of conscious rule application. In addition
to this rule interpreter, Smolensky posits the existence of a second,
unconscious processor, which acts on knowledge drawn from individual
experience to perform tasks such as intuitive expert game playing, motor
coordination, that is, almost all skilled action. He calls this
processor the intuitive processor.

He considers the following possible assertions, all of which he will
reject: that the intuitive processor deals with "inguistically
formalized rules" which are applied sequentially, that the program of
the intuitive processor is itself symbolic, and that these programs are
similar to those of the conscious rule interpreter. He rejects these
assertions, as most connectionists do, because---as yet---models of
human performance that are based on them lead to too much brittleness
and inflexibility, because the amounts of knowledge that would have to
be embodied to make them workable is too large, and because they lead to
few insights about how the brain works. (Fodor and Pylyshyn argue that
there is no particular reason why they should lead to such insights; see
the next section.)

After rejecting a symbolic approach to the intuitive processor Smolensky
considers the opposite extreme; that the intuitive processor uses the
same architecture as the brain does. The trouble with this hypothesis is
that we don't know what the brain's architecture is. Instead of this
hypothesis, Smolensky advances the hypothesis that the intuitive
processor has a connectionist architecture. The version of connectionism
that Smolensky advocates is what he calls the "connectionist dynamical
system hypothesis." This hypothesis views a connectionist system as a
parallel computer containing many processors, with each of which is
associated a number (or possibly, a set of numbers). Thus the state of
the system can be described by a vector, a state vector of activations.
The system has an equation describing how the state vector evolves in
time, which Smolensky calls the activation evolution equation. The state
of the connections in the system (that is, the weights on them), can
also be described by an equation, which Smolensky calls the connection
evolution equation. Thus a connectionist system is, for Smolensky, a
dynamical system such asthat found in physics. Typically these are
governed by differential equations.

Next, Smolensky considers the meaning of the activations in a
connectionist system. Each of the activations does not constitute an
entire symbol; rather a symbol (e.g., word) is represented as a
distributed pattern of activation across the system, each unit is
subsymbolic and participates in the pattern for many symbols. He takes
this to a (possibly new) extreme, by hypothesizing that the subsymbolic
behavior of a connectionist system is not explicable in terms of the
conceptual level. This rejects the idea (see Fodor and Pylyshyn, next
section) that connectionist models are implementations of symbolic
processes. Smolensky believes that if connectionist modeling is an
implementation theory, the connectionist research program is defeated.
The mere fact that connectionist networks and Von Neumann machines
(conventional serial computers) can simulate one another does not reduce
connectionism to an implementation theory, because a Von Neumann
simulation of a connectionist machine does not manipulate the kinds of
linguistic-level symbols used in a typical rule-based system.

Smolensky reviews three methodologies for choosing features at the
sub-conceptual level. This first is the borrowing of these features from
previous symbolic models, such as was done in Rumelhart and McClelland's
[@Rumelhart1986OnVerbs] model of the formation of the past tense, where
phonetic features were used (see section 7.12) The second is the
learning of the relevant features in hidden units using learning
procedures such as back-propagation. The third method is to choose
features in such a way so as to tune a system so that it matches human
performance.

One technique that does this, according to Smolensky, is
multi-dimensional scaling [@Shepard1962TheII], which looks at the raw
corpus of data and extracts vectors which can be used to represent the
stimuli. Smolensky points out that, if we want to look to the, brain for
guidelines on how to derive features at the sub-conceptual level, we
lack information. We have more information in vision than in any other
domain---in a domain such as language processing the information given
is virtually nil. Thus Smolensky points out (the protestations of some
other connectionists to the contrary), that the semantics of the
features at the sub-conceptual level are, at this point, more closely
related to the semantics of concepts at the conceptual level than they
are to the activations of neurons in the brain. Moreover, the actual
activity of the brain is much more complex than is reflected in most
current connectionist models. Yet, clearly, because of the rough
correspondence between connectionist models' architecture and the brain,
connectionist models are likely to operate under similar principles to
the brain.

Smolensky concludes that connectionist models are at a level
intermediate between that of symbolic models and the brain, and should
not be seen as biological models themselves. He argues that a reduction
will someday have to be made from successful connectionist models to
neural circuits. The expression of neural circuits directly in models is
made difficult by insufficient knowledge of the dynamic behavior of the
brain, according to Smolensky. The degree to which a connectionist model
can be approximated by a symbolic model depends on whether or not the
process being modeled is one of conscious rule application or of
intuition. Conscious rule application processes modeled on the
connectionist level can be described "with reasonable precision" on the
conceptual level, but intuitive processes can be described only roughly
on the conceptual level. This is because the symbolic level relies
strongly on its own implementation language for its functioning.

On the other hand, connectionist models can serve as an implementation
language for conventional symbolic processes. However, the details of
how this is done are not completely specified, although Touretzky and
Hinton [@Touretzky1985SymbolsArchitecture]
[@Touretzky1987RepresentingNetwork] have done valuable work in this
area. Smolensky divides knowledge that is useful in interpreting stimuli
into two sets: P-knowledge (parallel knowledge) and S-knowledge
(sequential knowledge). P-knowledge can be used in parallel; e.g., a
listener attempting to understand a sentence can simultaneously use
syntactic, phonetic, morphological and semantic knowledge. On the other
hand, S-knowledge cannot be used in parallel; a player of a game has to
execute a single rule before s/he can contemplate the execution of a
second rule. According to Smolensky, P-knowledge is much more context
dependent than S-knowledge, because it is necessary to know which
aspects of the P-knowledge can operate in conjunction with one another.

Next, Smolensky attempts to characterize what it is about a model that
makes it cognitive, and how connectionist models can be cognitive. He
defines a cognitive system as one that maintains a large set of goal
conditions under a variety of environmental conditions. A thermostat is
not cognitive because it does not maintain a large set of environmental
condition. Thus complexity is, for Smolensky, the acid test as to
whether a model is cognitive. One important task that cognitive models
undertake is what Smolensky calls the "prediction-goal," that is, to
predict missing features of the environment from the features that are
present in the stimuli. Closely related to this goal is the
"prediction-from-examples" goal, that is, to use previous examples to
continuously improve performance on the prediction problem.

Smolensky answers the argument of Fodor and Pylyshyn
[@Fodor1988ConnectionismAnalysis] (see the next section) that
connectionist models are limited in their usefulness, since mental
states have constituent structure, like that represented in a parse
tree, and connectionist models don't. Clearly, this applies to localist
connectionist models, which are subject to combinatorial explosion in
the number of nodes, since there must be a node for every combination of
concepts (e.g., the tall-blond-man-with-one-black-shoe node), but
Smolensky, who advocates distributed representations, thinks that they
are less vulnerable to this kind of criticism. He considers the idea of
Pylyshyn [@Pylyshyn1984ComputationScience] that in a distributed
connectionist system the representation of "coffee" is equal to the
representation of "cup with coffee" minus the representation of "cup."
If the representation for "cup with coffee" consists of units
representing features like "solid container," "handle," "brown liquid,"
"curved liquid," etc., then when we remove all of the features of "cup"
from this representation, we are left with a representation of coffee,
but in the context of cup, that is, we have a representation of coffee
that has the coffee in the shape that is contained by the cup. Other
representations of coffee could be made in other contexts; for
Smolensky, there is no context-free representation of coffee.

The difference between local symbolic representations of a word and
distributed connectionist representations is that, with the former type
of representations, the context is established by the connections that
it makes with other local symbols (as in a semantic network); with the
latter type, the context is contained in the pattern of activation
itself. This disturbs people who believe in context-free symbols.

Smolensky argues that connectionist models should be intrinsically
continuous, that is, have activations that are real-valued rather than
discrete, in order to escape from the brittleness and inflexibility of
conventional symbolic models. This allows for the integration of
multiple constraints with different weight, which all-or none rule-based
systems cannot give you.

Smolensky believes that even the many connectionist models which use
binary data in their computations need not be fundamentally discrete,
but can all be mapped onto models that use real-valued variables. He
argues, therefore, that little is gained by the use of discrete models,
and their common use is based on the fact that digital computers are
basically discrete, and so researchers have a tendency to think in
discrete terms. But the brain is a highly parallel analog computer,
which deals with real values, and analog computers can be built that
embody connectionist models: for example, Smolensky cites Anderson
[@Anderson1986LearningSystems] and Cohen [@Cohen1986DesignProcessing].

Smolensky reformulates the idea that multiple "soft" constraints can
simultaneously contribute to a solution to a problem, such as mapping an
input to its correct output, in terms of what he calls the "Best Fit
Principle." This principle says that the system as a whole arrives at a
solution which is statistically the "best fit" to the input, as
specified by the various constraints known to the system. This is
expressed mathematically in terms of the harmony function H, which is
maximized by the machine. Harmony theory
[@Smolensky1986InformationTheory] gives the theoretical underpinnings of
the harmony function and the dynamic behavior of networks with respect
to it.

Smolensky gives an example of how a network can exhibit behavior
normally thought of as rule-governed: his system that does qualitative
physics. In this system, the knowledge of Ohm's law is embodied in the
configuration of parts of the system representing current, voltage, and
resistance. Each of these computes its value in parallel, with hundreds
of microdecisions (activation changes). Macrodecisions are the result of
many microdecisions. If a system changes, as the result of receiving
input, to a state in which Ohm's law is satisfied, this is a
macrodecision, even though underlying behavior is following Ohm's law as
such. The relationship is like that between quantum mechanics and
Newtonian mechanics; examined at a gross level a physical system seems
Newtonian, but underneath it is really obeying quantum mechanics.

Harmony theory is illustrated by the work on schemata of Rumelhart and
his co-workers [@Rumelhart1986ParallelModels] (see section 4.2). This
model simulates the look-up of schemata on the basis of some triggering
information. Basically, what it does is perform a search in harmony
space. Although no schemata are explicitly stored---just correlations
between rooms and objects---the schemata are emergent phenomena. When
primed on some of a room's contents---say an oven and a cabinet---the
system performs a search in harmony space, which leads to a peak which
corresponds to the complete set of the room's descriptors. Yet the
schemata are not directly present; they are higher-level descriptors of
rooms than are explicitly represented.

Connectionism: A New Form of Associationism?
--------------------------------------------

Fodor and Pylyshyn [@Fodor1988ConnectionismAnalysis] advance a detailed
critique of connectionism, and a defense of the classical view (as they
term it) of cognition as rule-governed manipulation of symbols, as in
LISP-based AI or production systems. Basically, they view connectionism
as a more sophisticated form of associationism, masquerading in new
clothes. Associationism, in one of its forms, behaviorism, reigned as
the supreme psychological theory through the 1950s, until it was
displaced by classical cognitivism. For a history of cognitivism and its
victory over behaviorism, see Gardner [@Gardner1985TheRevolution]. Fodor
and Pylyshyn feel that many of the same arguments used against the
associationism of the 1950s---one of the most famous of which was
Chomsky's review [@Chomsky1959ReviewBehavior] of Skinner's Verbal
Behavior---can be used against the associationism of the 1980s,
connectionism.

This is not to say that connectionism and behaviorism can be equated;
connectionism, for one thing, is representational, believing that
structures in the brain and mind represent objects and states of affairs
in the world. Behaviorism (at least in the forms advanced by Skinner in
his famous arguments against "mentalism," which is what he called
theories that use mental representations) is not representational, it is
what Fodor and Pylyshyn call "eliminativist." This means that
behaviorism does not think representations are important, but rather
that behavior is, and, in fact, all talk about representations is
unscientific. Fodor and Pylyshyn think that all the distinctions made in
the connectionist literature between symbolic and sub-symbolic
representations miss the main point: that all nodes in a connectionist
network are symbolic. (A representation is the same thing as a symbol,
in their opinion.) A single node is a symbol; so is a pattern of
activation across nodes, much as a bit is a symbol (if it is causally
attached to something in the world), and so is a bit vector.

Connectionist representations are typically collections of activated
nodes representing microfeatures (or rather, features, since what is
typically used in the connectionist literature as a "microfeature" is
something like "human," which is hardly an elementary concept). Fodor
and Pylyshyn's main objection to this form of representation is that it
is unstructured; that is, it exhibits little of the compositionality of
representation that classical representations exhibit. For instance,
they posit that if the sentence "John loves Mary" is represented by
three activated nodes (+johnsubject +loves +mary-object), and we receive
the additional information that "Bill hates Sally," so that we now have
the vector (+john-subject +bill-subject +loves +hates +mary-object
+sallyobject); we now have crosstalk producing the additional sentences
"John loves Sally," "Bill hates Mary," "John hates Mary," "Bill loves
Mary," "John hates Sally" and "Bill loves Sally." One way to avoid this
is to have one node representing the entire complex concept of each
sentence; the problem is there are so many possible sentences that no
physically realizable brain could possibly contain enough nodes. The
other way would be to allow the representations to have an internal
tree-like structure such as (((John subject) loves (Mary object)) ((Bill
subject) hates (Sally object)). The problem with doing this, according
to Fodor and Pylyshyn, is that no one, to their knowledge, has shown how
to do this in a connectionist architecture. Touretzky
[@Touretzky1989TowardsManipulation] discusses issues involved in doing
compositional semantics in connectionist networks, with reference to the
problem of attaching prepositional phrases to noun phrases in sentences.

The key to this problem is that each link between nodes in a neural
network represents a causal relationship between the nodes. Semantic
networks, which can be more general than neural networks, allow for
labeled links; the links can signify "causes," "is contained in,"
"precedes," etc. A connectionist network that Fodor and Pylyshyn suggest
that draws inferences from a node standing for both A and B to ones
standing for A and B is shown in Figure
[\[fig\_AB\]](#fig_AB){reference-type="ref" reference="fig_AB"}.
Although the nodes are labelled to make it clear what they signify,
these labels are not part of the connectionist representation. Fodor and
Pylyshyn point out that the difference between this network and the
classical implementation of this inference is that, in the classical
case, the symbol string "A&B" contains as a part the strings A and B,
whereas this is clearly not true of the top node in the diagram. The
causality shown in Figure [\[fig\_AB\]](#fig_AB){reference-type="ref"
reference="fig_AB"} is insufficient to account for the compositionality,
at least in this simple form.

![A connectionist network that represents the assertion that A and B
implies A and B individually](AB.PNG)

[\[fig\_AB\]]{#fig_AB label="fig_AB"}

Fodor and Pylyshyn point out, additionally, that connectionist models
learn that concepts are statistically related. So, for instance, a
connectionist model learns that A follows from A and B because the two
statements are statistically related in the environment. It does not
observe the structure of A and B in order to infer A, which is obviously
one of the salient features of A and B. Two basic properties of language
augur well for the classical theory and badly for connectionist theory,
according to Fodor and Pylyshyn. These are two well-known properties of
language, its productivity and its systematicity. Any natural language
consists of, for all practical purposes, an infinite number of
well-formed sentences; this is referred to as the language's
productivity. They quote a remark of Rumelhart and McClelland
[@Rumelhart1986OnVerbs] in which they say that recursive center-embedded
sentences---such as "The dog the man walked barked"---are hard to
process, and that this is evidence that recursive capabilities are not
central to cognitive capability. Fodor and Pylyshyn dispute this point,
citing examples in which recursive embeddings are both easily understood
and natural. In order to handle recursive embeddings, you need a
classical (Von Neumann machine) architecture---one might presume---or a
connectionist implementation thereof. So if you think that recursive
productive features are critical to language, one might be tempted to
choose a classical theory, according to Fodor and Pylyshyn.

Many sentences of the same structure occur in any given language, such
as "the tree is in the park" and "the man is in the office"; this is the
systematicity of language. This is what has caused linguists to posit
the existence of syntactic categories. These categories necessitate the
creation of linguistic structures that go beyond the simple lists of
features that connectionist structures imply.

Fodor and Pylyshyn take some time to consider some of the issues that
have made connectionism so appealing to so many in cognitive science.
First of all, the issue of parallelism: the idea is often advanced that
since the brain has many neurons active at once, and no apparent central
control, any plausible cognitive architecture would have to be
"massively" parallel. Of course, on the lowest level, a standard Von
Neumann computer is highly parallel, since electrical signals are active
throughout both CPU and memory, but on a higher level of abstraction it
is not parallel. Fodor and Pylyshyn point out that not only
connectionist models that are parallel; classical symbolic programming
languages can be parallel, for instance Hewett's
[@Hewett1977ViewingMessages] ACTORS system and Hillis's
[@Hillis1985TheMachine] parallel version of LISP for his Connection
Machine.

Designing algorithms for networks of traditional processors and
shared-memory multiprocessors has become a fertile area of computer
science research. Fodor and Pylyshyn point out that not only
connectionist representations can be distributed. A traditional memory
register can be distributed; all that is necessary is for it to divide
its contents and spread them out across the machine's memory. If it is
desired, a transformation like a Fourier transform can be applied to the
data so that, if part of the transform is lost (after it is split into
chunks and distributed across the memory), the original signal can still
be constructed, albeit with a loss of quality (the sort of "uniform
degradation" that connectionists often talk about). They point out that
an array can be functionally local, but distributed across the entire
machine (using a hash table, for instance); representations are
damage-resistant unless they are physically localized in memory and
non-redundant.

Another point about connectionist models that is often raised is their
ability to deal with representations and stimuli that are continuous
rather than all-or-none. Fodor and Pylyshyn point out that there are
lots of classical models that use continuous variables; for instance the
use of Bayesian probabilistic inference techniques in expert systems or
systems that use fuzzy logic [@Zadeh1973OutlineProcesses]. Statistical
properties of stimuli can emerge from the interaction of many smaller
deterministic processes. (For instance, to oversimplify, the
classification of a bird can emerge from a set of features, some of
which are necessary; the classification process may depend on how many
non-necessary features are present, their strength, or a function
therof.)

It is often maintained that connectionist architectures implicitly model
rule-governed behavior whereas classical architectures make the rules
explicit (and presumably implicit rules are better). Fodor and Pylyshyn
claim that classical architectures can be rule implicit, when functions
are wired in to the machine. Connectionist architectures can be rule
explicit, as well,- but only by implementing a classical recursive
machine. They add that if explicit rule learning turns out to be an
important part of psychological theory, then connectionist systems are
in trouble.

A major argument that is advanced for connectionist architectures is
that they are biologically plausible, that is, that they are consistent
with known facts about the brain. Fodor and Pylyshyn briefly summarize
these facts as follows: connection patterns are important to how the
brain processes information; memory is distributed, not localized;
neurons work with synaptic thresholds; a neuron is more likely to fire,
given a certain amount of input at its synapse, if it has recently
fired. Fodor and Pylyshyn claim that none of these facts strongly
constrain what type of architecture the mind must have. Low-level
structure---and this is a point that Fodor has emphasized in many of his
books---does not necessarily reflect high-level structure; the theory of
atoms, doesn't look anything like biology, for instance. There is no
reason that the brain could not implement a recursive classical
architecture. Moreover, Fodor and Pylyshyn lament that the naive
implementation of these \"brain-like\" models has created a revival of
associationist psychology.

Fodor and Pylyshyn do not want to dismiss connectionism totally, they
just want to reduce it to the level of a theory of how specific
psychological algorithms are implemented. Such "properties" of computers
such as the idea that their memory is permanent, that they use
exhaustive search, that they are logical and don't make mistakes, are
not properties of an algorithm but of its underlying implementation in a
digital computer; algorithms can be devised that "forget," that perform
faulty reasoning, make errors in retrieval, etc.; basically any process
that can be theorized about can be embodied in a conventional computer
(this is the Church-Turing thesis, the widely accepted statement that
Turing machines are universal, that is, that they can compute anything
we can conceive of computing).

According to David Marr [@Marr1978RepresentingInformation], every
process can be understood on two levels: the level of formal
specification (theory), and the level of implementation. The problem
that Fodor and Pylyshyn have with connectionist models is that their
authors offer them as theories of cognition, not as implementations of
higher-level formal processes. I would add, in defense of
connectionists, that studying lower-level neuronal implementations may
produce interesting new insights about higher-level processes, much as
physics has informed theory formation in chemistry. Of course, there is
no reason that neural theories need constrain psychological theories
much at all, if the brain turns out to be a general purpose machine,
like a Von Neumann computer.

Fodor and Pylyshyn consider four possible routes that connectionism
could take from here, which are: (1) to maintain its present course; (2)
to admit structured mental representations but retain an associationist
account of their processing; (3) to reduce connectionism (and
neuroscience) to the status of an implementation theory; or (4) to
accept connectionist accounts of a certain subject of cognitive
processes---notably the forming of statistical inferences from sets of
stimuli---but do not accept connectionist accounts of such phenomena as
linguistic productivity and regularity. It is hard to say what route
connectionism will take; much depends on the success of the current
research program.

Learning and Relaxation {#learn_methods}
=======================

Introduction
------------

Most work in neural networks involves learning. The goal of most neural
network models is to learn relationships between stimuli. There are at
least three ways that learning models can be classified. The first way
concerns the nature of what is learned. A learning model can be a
*hetero-associator* or an *auto-associator*. A hetero-associator is a
network that computes a function between a set of inputs and a set of
outputs. An auto-associator is a network that completes an incomplete
input pattern. These two types of models are not different in principle,
because a hetero-associator can always be reduced to an auto-associator
by concatenating an input pattern and its associated output pattern, to
make an input pattern of the auto-associator. Thus the performance of
the hetero-associator can be then achieved in the auto-associator, by
simply presenting, as the partial input of the auto-associator, the
input pattern for the heter-oassociator, and having the machine complete
the pattern to produce what was the output pattern of the
hetero-associator.

Auto-associative memory is extremely useful for the organism, and it is
the way that the human memory system seems to work. Everyone has the
almost constant experience of having a memory evoked by a particular cue
that formed part of the memory: for instance, seeing a hat similar to
one one's father used to wear reminds you of him wearing that hat.
Content-addressable memories are also useful for database applications,
because one typically wants to look up some record in a database based
on some part of it (which is called the key.) Traditional approaches
have involved building indices on each item of a record for which one
wants to look for, each key. Organizing a memory in such a way so that
content-addressability is an automatic feature would be desirable; it
seems that this is what the brain has done.

The second way that learning models can be classified applies only to
hetero-associators. These can be classified based on what they compute.
Typically they compute either a general function of their input, in
which there are about as many outputs as inputs, or a classification,
where a large set of input patterns is mapped onto a relatively small
set of output patterns, which represent sets into which the input
patterns are classified.

Yet another way that learning models can be classified is in terms of
the amount of guidance that the learning process receives from an
outside agent, typically referred to as the *teacher*. *Unsupervised
learning* occurs without a teacher; such a learning algorithm---the
various kinds of competitive learning discussed in sections 2.15 through
2.18 are examples---that learns to classify the input into sets without
being told anything. It does this clustering solely on the basis of the
intrinsic statistical properties of the set of inputs.

*Supervised learning* adjusts weights on the basis of the difference
between the values of output units, given an input pattern, and the
desired pattern, given by the teacher. Error backpropagation (see
section 2.12) is a supervised learning procedure.

The third type of learning has a long history in psychology:
*reinforcement learning*. This is a type of supervised learning in which
very little information is given the algorithm; typically one bit, which
signifies if the output that the network provided to the given stimulus
is good or bad. Many people feel this type of learning (which must of
necessity proceed more slowly than supervised learning, since less
information is given) is more psychologically valid, since people are
not normally provided with a complete example of the desired behavior,
especially in situations in which they are not explicitly taught (such
as in child language learning).

While most work in connectionism is limited to learning of weights, many
other things can also be learned, such as the topology of the network,
the activation functions, even the learning rules themselves. Several
authors have explored the possible of learning where in a net to place
nodes and connections (see, for example [@Ash1989DynamicNetworks]
[@Dieterich1988Knowledge-intensiveLearning]
[@Honavar1989PerceptualModels] [@Honavar1989ExperimentalNetworks]).

Relaxation is the process whereby the unit activations (not the weights)
change over time until they evolve to a state in which activations are
no longer changing, and thus the network can be said to have
\"relaxed\", i.e. fallen into a state of little activity. Relaxation
differs from learning in that only activations change; in learning, the
weights change. Some network paradigms, notably feed-forward networks,
which will be discussed later on in this chapter, require only one
update per unit in order to reach their final state. Other types of
networks, such as the Boltzmann machine (also discussed later in this
chapter), require many updates, and thus undergo relaxation. Relaxation
is especially applicable to constraint satisfaction problems such as
vertex-labeling in line drawings [@Waltz1975GeneratingShadows] and line
and edge detection and enhancement [@Zucker1977AnEnhancement]. Mackworth
[@Mackworth1977ConsistancyRelations] supplied a useful discussion of
methods for the satisfaction of multiple constraints. For other
discussions of relaxation, see [@Hummel1983OnProcesses] and
[@Geman1984StochasticImages].

The following sections give an overview of the main neural network
designs and learning methods. We start with a review of the different
types of model neurons devised by Feldman and Ballard
[@Feldman1982ConnectionistProperties], which can be used to construct
connectionist models. We then discuss two of the earliest devices, the
Adeline and the Perceptron, which are relatively simple and limited in
their computational power. In both of these models the
error---difference between the desired and actual output---is used as a
corrective to bring the performance of the model closer to that desired.
Thus these two models employ supervised learning.

We then proceed to another simple associative network that uses Hebbian
learning, that of Anderson. This is a network that uses matrix
multiplication to compute associations between input and output vectors,
and which uses as correctives to its matrix values correlations
(products) of single components of these vectors. Thus, following Hebb
[@Hebb1949TheBehavior], connections between components that are
simultaneously active are strengthened. This is a simple form of
associative learning.

We then move on to another type of associative learning, that of Kohonen
[@Kohonen1988Self-OrganizationMemory]. In the auto-associative version
of Kohonen's work, he views an input vector as a corrupted version of
its true value. He uses a mathematical technique, the Gram-Schmidt
process, to compute the stored vector that is closest to the noisy
input.

These first four models are linear associators, since, in each, their
output is a linear combination of their input. The remainder of the
models discussed are non-linear, i.e. their outputs are non-linear
functions of their inputs. A non-linear model can compute a much greater
variety of functions than a linear model, although it is surprising how
much linear models can handle.

The first non-linear model we discuss is that of Hopfield. In this
model, neurons reset themselves randomly and asynchronously if their
weighted inputs exceed a threshold. Hopfield's network is
autoassociative. By setting the weights in his network in a particular
fashion, Hopfield is able to show that the state of his network always
converges to a stable state. Any such network, Hopfield shows, has a set
of stable limit points, which can be used to store memories. Thus the
network functions as an auto-associator, because each input---that is,
initial state of the network---leads to a stable state that corresponds
to a stored memory. Moreover, unlike a linear associator, the stored
patterns need not form a linearly independent set.

We then discuss the more complex neuron model offered by Hopfield and
Tank, in which a neuron's behavior is modeled by a differential
equation. We discuss the application of this neuron model to problems in
optimization, in particular, the traveling salesman problem.

Hopfield used his initial network to store memories that correspond to
local minima in a function that, borrowing from thermodynamics, he
defines as the energy function of the network. If the goal of the model
is not to store a set of memories, but to find a global minimum in the
energy function that corresponds to an optimal solution of some
constraint-satisfaction problem, then an extension of the Hopfield
network, called the Boltzmann machine, is used. The Boltzmann machine is
basically a stochastic version of the Hopfield network. The state of a
given node in the network is based both on how much input it is
receiving as well as a parameter called the temperature. The higher the
temperature, the more randomness there is in the system. This type of
network lends itself well to finding global minima. We also discuss a
learning rule to make a Boltzmann machine reflect the state of the
environment.

Next, in our discussion of network paradigms, we turn to the best-known
neural network algorithm, the error back-propagation algorithm. This is
an extension of the Perceptron to systems with one or more layers of
hidden units between the input and the output. In this algorithm, the
difference between the desired output and the actual output is used to
adjust the connections first between the output layer and the hidden
layer right below it. This error is then propagated backward in the
network to layers below the top hidden layer and ultimately used to
adjust connections between the input units and units above them.
Rumelhart and his co-workers show that this algorithm converges to a
local minimum in the error---that is, in the difference between the
desired and actual outputs.

We discuss a variety of problems to which Rumelhart and his co-authors
applied the back-propagation algorithm. These include the computation of
the exclusive-or function, which the two-layer Perceptron was unable to
handle.

We then turn to a variety of unsupervised learning methods. The first
three examples we consider are versions of competitive learning, in
which various digits or sets of units compete to recognize features in
the input. These algorithms resemble Darwinian natural selection. We
then consider a class of algorithms that, while not explicitly neural
network algorithms, also take their inspiration from evolution: genetic
algorithms. These are algorithms that model populations of organisms as
populations of bit strings, in which each bit string is itself a
solution to a problem. Both competitive learning and genetic algorithms
"evolve" solutions to problems.

The final type of algorithms that we consider are reinforcement
algorithms. The three main ones that we discuss all were developed at
the University of Massachusetts, in Andrew Barto's group. The first of
these algorithms is called the associative reward/penalty $(A_{R-P})$
algorithm, which is used to classify a set of inputs. The system learns
the classification of inputs based on a reinforcement signal. We then
discuss the work of Sutton on what he calls temporal difference methods,
which are useful in predicting events that are in time series.

We then discuss the work of Anderson, one of Barto's students. Anderson
applied both back-propagation, and the AR.p algorithm, to a set of
problems. He also adapted the work of Sutton to use in the problem of
learning heuristics for problem solving for problems normally associated
with heuristic search. This type of problem solving has not been
attacked by many other researchers using the connectionist paradigm.
Chapter two concludes with discussions of various extensions of the
error back-propagation algorithm, of attempts that have been made to
model sequential phenomena in neural networks, of a method for
compressing information using back-propagation, and of an attempt to
embody recursive structures in a neural network.
