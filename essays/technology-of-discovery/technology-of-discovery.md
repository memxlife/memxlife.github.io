# The Technology of Discovery: How Intelligence Learns the World

Every era confronts the same fundamental problem: **the world is vastly more complex than any observer’s ability to perceive, represent, and understand it.**

Reality does not arrive already organized into variables, equations, categories, or datasets. Nature does not tell us which measurements matter, which apparent patterns are accidental, which relationships are causal, or which abstractions will remain valid beyond the situations we have already observed.

Those structures must be discovered.

Newton could observe only a narrow slice of the physical world. Darwin had no access to DNA. Early statisticians worked with datasets that would now fit comfortably in a spreadsheet. The pioneers of machine learning operated with computers far less powerful than a modern phone. Even today’s largest artificial-intelligence systems encounter only partial representations of reality: text, images, audio, video, sensor readings, and the feedback generated through limited interactions.

Yet knowledge progresses.

How?

One way to understand this progression is not as a succession of disconnected movements—rationalism, experimental science, probability, statistics, machine learning, deep learning, and large language models—but as the evolution of a single process:

> **How does a limited intelligence discover useful structure in a world it can observe only partially?**

Seen from this perspective, two historical threads run in parallel.

The first is the history of **epistemology**: our evolving understanding of how knowledge should be acquired, justified, tested, and revised.

The second is the history of **domain science and technology**: our expanding understanding of particular parts of reality, together with the instruments and systems we build from that understanding.

These threads continually interact. But neither develops in an intellectual vacuum. At any historical moment, what can be discovered is constrained by three practical capabilities:

1. how much of the world can be observed and recorded;
2. how much computation can be performed over those observations; and
3. what mathematical and algorithmic methods exist for converting observations into knowledge.

The history of discovery is therefore also a history of changing constraints.

From Descartes and Newton to Bayes, classical statistics, Breiman’s “two cultures,” deep learning, and large language models, the central problem has remained surprisingly stable. What has changed is the division of labor among **human prior knowledge, empirical data, and computation**.

That changing division of labor is the deeper story behind the evolution of modern learning.

---

## 1. Reality, Observation, and Representation

Suppose there is some enormously complicated process $F$ generating the phenomena we observe.

We never encounter $F$ in its entirety. Instead, an observational process gives us fragments:

$$
D = O(F),
$$

where $O$ represents the limitations of our instruments, experiments, sampling procedures, language, and attention.

We then construct a representation $M$ from those observations:

$$
D \rightarrow M.
$$

The resulting model may be a physical law, a causal diagram, a statistical distribution, a neural network, or an informal theory expressed in natural language. Whatever its form, the model is not reality itself. It is a compressed representation designed to preserve certain structures that matter for some purpose.

This gives us three distinct levels:

$$
\text{Reality}
\rightarrow
\text{Observation}
\rightarrow
\text{Representation}.
$$

A fourth level is computation: the procedure through which a representation is constructed, fitted, tested, or used.

These levels are frequently confused.

A statistical equation is a representation, not reality. A neural-network architecture is both a representational framework and a computational design. A training objective determines what information the learning process is rewarded for preserving. A dataset is not merely a neutral sample of the world; it is the output of an observational and selection process.

The familiar statement that “all models are wrong, but some are useful” points to this distinction. A useful model does not reproduce the full structure of reality. It captures regularities that are sufficiently dominant, stable, and relevant for a particular task.

A Newtonian model ignores atomic structure while accurately predicting many macroscopic motions. A medical risk model may ignore most of a patient’s biology while still improving a clinical decision. A language model does not reproduce the full causal process that created human civilization, but it can nevertheless learn powerful regularities from humanity’s linguistic traces.

The critical question is therefore not whether a model captures everything. No model does.

The question is:

> **Which structure does the model preserve, under what conditions, and for what purpose?**

This question connects philosophy, science, statistics, and machine learning.

---

## 2. Two Parallel Histories

### Epistemology: How Should We Know?

The first thread is epistemology.

Epistemology asks how an intelligent observer should form reliable beliefs about a world that is only partially accessible.

What should count as evidence?

Should knowledge begin with reason or experience?

How should hypotheses be compared?

How should uncertainty be represented?

When should a belief be revised or abandoned?

Descartes is associated with systematic doubt, rational analysis, and the decomposition of difficult problems into simpler components. The empirical tradition emphasized observation and experiment. Galileo’s work exemplified the productive combination of mathematical reasoning with controlled observation. Kepler extracted mathematical regularities from astronomical measurements. Newton synthesized mathematical methods, empirical results, and physical concepts into a compact theoretical system.

It would be historically simplistic to say that Cartesian reasoning directly produced Newtonian mechanics. Newton emerged from the interaction of several traditions: rational analysis, empirical observation, mathematical innovation, and the accumulated results of earlier scientists.

But together these developments established a powerful epistemological pattern:

$$
\text{observe}
\rightarrow
\text{abstract}
\rightarrow
\text{deduce}
\rightarrow
\text{predict}
\rightarrow
\text{test}.
$$

Later developments in probability and statistics added a crucial element: uncertainty.

Bayesian reasoning, in its modern epistemological interpretation, offers a general picture of knowledge as iterative updating:

$$
\text{prior belief}
+
\text{new evidence}
\rightarrow
\text{posterior belief}.
$$

The posterior then becomes the prior for the next round of learning.

This creates a continuing cycle:

$$
\text{existing knowledge}
\rightarrow
\text{observation}
\rightarrow
\text{update}
\rightarrow
\text{prediction or action}
\rightarrow
\text{new observation}.
$$

This is more than a statistical formula. It is a general architecture for learning.

### Domain Science and Technology: What Structure Exists?

The second thread is domain science and technology.

Physics asks what structures govern matter, motion, energy, space, and time.

Chemistry studies molecular composition and transformation.

Biology investigates inheritance, evolution, metabolism, development, and ecosystems.

Economics studies production, exchange, incentives, institutions, and collective behavior.

Engineering converts discovered regularities into controllable systems.

Computer science studies information, algorithms, and computation.

Machine learning asks how useful representations and behaviors can be acquired from data and experience.

Each domain develops its own concepts, instruments, theories, and experimental practices. But domain knowledge does not progress independently of epistemology.

New ways of reasoning enable new discoveries. New discoveries expose weaknesses in existing methodologies. Technologies built from scientific knowledge create better observational instruments. Better instruments reveal phenomena that existing theories cannot explain.

The two threads therefore form a feedback loop:

$$
\text{epistemic methodology}
\rightarrow
\text{scientific discovery}
\rightarrow
\text{technology}
\rightarrow
\text{better observation}
\rightarrow
\text{new epistemic problems}.
$$

The telescope did not merely produce more astronomical facts; it changed what could count as astronomical evidence. The microscope did the same for biology. Statistical sampling changed how populations could be studied. Digital sensors and the internet changed the scale at which human activity could be observed.

Technology does not merely apply knowledge. It transforms the conditions under which knowledge can be produced.

---

## 3. Early Science: The Human Mind as the Learning Algorithm

In the early development of modern science, observations were scarce, computation was largely manual, and mathematical tools were limited.

Under those conditions, human intelligence had to perform most of the work of structural discovery.

Consider Newtonian mechanics.

The achievement was not simply solving equations after the relevant variables had been provided. The deeper achievement was identifying the right abstractions:

- mass;
- position;
- velocity;
- acceleration;
- force;
- momentum;
- time.

These are not raw sensory inputs. They are conceptual representations through which observations become mathematically intelligible.

Once the right representations had been discovered, a vast range of physical behavior could be compressed into a small number of equations.

In contemporary machine-learning language, early scientists had to perform several tasks largely inside their own minds:

- representation learning;
- feature selection;
- hypothesis generation;
- model selection;
- causal abstraction;
- error analysis.

The resulting theories had to be compact enough to express symbolically, manipulate by hand, and test with limited observations.

This constraint favored low-dimensional, interpretable models. Scientific elegance was not only an aesthetic preference. It was also an adaptation to severe limitations in data and computation.

The success of Newtonian mechanics can therefore be viewed as an extraordinary act of human compression. A complicated range of terrestrial and celestial phenomena was represented through a compact mathematical structure.

Calculus itself was part of the capability breakthrough. It was not merely a description of nature; it was a new algorithmic and representational technology that made certain forms of reasoning possible.

This illustrates a general principle:

> **What can be discovered depends not only on what is true, but also on what the available representational and computational tools make thinkable.**

When data and computation are scarce, strong human priors must carry more of the burden.

---

## 4. From Deterministic Laws to Statistical Knowledge

The success of classical mechanics encouraged the hope that nature might be understandable through compact deterministic laws.

But many important systems did not yield so easily.

Measurements contained noise. Biological populations varied. Economic outcomes depended on large numbers of interacting agents. Social behavior was shaped by unobserved variables. Even deterministic systems could become practically unpredictable when their initial conditions were incompletely known.

Probability and statistics introduced a new technology of knowledge.

Instead of claiming complete access to the mechanism, one could separate what was modeled from what remained unresolved:

$$
Y = f_{\theta}(X) + \epsilon.
$$

Here, $f_{\theta}$ represents the structure the model attempts to capture. The term $\epsilon$ represents what remains outside that structure.

But $\epsilon$ can have several meanings:

- measurement error;
- omitted variables;
- unresolved interactions;
- sampling variation;
- intrinsic stochasticity;
- limitations in the model;
- limitations in the observer.

Statistics therefore does not necessarily imply that reality itself is fundamentally random. Often, probability represents an observer’s incomplete access to a system.

This was an important epistemological shift.

A model no longer had to say:

> “This equation completely describes reality.”

It could instead say:

> “Given the information available, this is a disciplined representation of the structure we understand and the uncertainty we do not.”

Bayesian inference made this relationship particularly explicit:

$$
P(H \mid D)
=
\frac{P(D \mid H)P(H)}{P(D)}.
$$

The prior $P(H)$ expresses what is believed before the new evidence. The likelihood $P(D \mid H)$ describes how expected the observations would be under a hypothesis. The posterior $P(H \mid D)$ represents the updated state of belief.

Modern machine learning is not automatically Bayesian in a technical sense. Most neural networks are trained through optimization rather than full Bayesian inference. Nevertheless, the broader epistemological pattern remains relevant:

$$
\text{prior structure}
+
\text{observations}
+
\text{an updating procedure}
\rightarrow
\text{a revised model}.
$$

That pattern links statistical reasoning to the larger history of discovery.

---

## 5. The Capability Frontier of Discovery

Why do different historical periods favor different methodologies?

The answer is not simply that later thinkers are more intelligent or philosophically sophisticated than earlier ones. A major part of the answer lies in changing capabilities.

At any point in history, the feasible methods of learning are constrained by three interdependent resources.

### 5.1 Observation and Data

What can be observed, at what resolution, at what cost, and at what scale?

Telescopes expanded the observable universe.

Microscopes opened the cellular world.

Precision clocks enabled new measurements of motion and navigation.

Laboratory instruments transformed chemistry and physics.

Clinical trials changed medicine.

DNA sequencing transformed biology.

Satellites transformed climate science and geography.

Digital systems transformed language, behavior, commerce, and communication into machine-readable records.

A larger dataset does not automatically produce better knowledge. Data may be biased, incomplete, noisy, or disconnected from the causal question being asked. But without adequate observation, many structures remain empirically inaccessible.

### 5.2 Computation

How many candidate explanations can be evaluated, and how complex can they be?

When computation is performed by hand, a model with a few parameters is not merely elegant; it is practical.

Electronic computers expanded the size of feasible statistical calculations. Parallel computing expanded it further. Graphics processors and specialized accelerators made the training of large neural networks economically possible. Distributed systems enabled models and datasets that could not fit on a single machine.

Computation changes the amount of search that can be delegated to a machine.

With little compute, humans must narrow the hypothesis space before calculation begins.

With abundant compute, a learning algorithm can explore much larger spaces and discover structures that humans did not explicitly enumerate.

### 5.3 Mathematics and Algorithms

More data and hardware are not enough. Algorithms determine what can be extracted from them.

Calculus created a language for continuous change.

Probability theory formalized uncertainty.

Linear algebra made high-dimensional representation and computation possible.

Numerical analysis enabled approximate solutions when symbolic ones were unavailable.

Optimization provided methods for fitting complex models.

Backpropagation made multilayer neural networks trainable.

Convolution encoded spatial regularities into neural architectures.

Attention enabled flexible, content-dependent interaction across sequences.

An algorithmic improvement can reduce the amount of computation or data required to achieve a given result. In that sense, algorithms multiply the value of the other resources.

Together, these capabilities define a moving frontier:

$$
\boxed{
\text{Observation and Data}
\quad+\quad
\text{Computation}
\quad+\quad
\text{Mathematics and Algorithms}
}
$$

The combination is what matters. Data without algorithms may remain uninterpretable. Algorithms without data may have nothing from which to learn. Computation without suitable representations may perform an enormous amount of useless search.

As the capability frontier moves, the practical methodology of discovery moves with it.

---

## 6. Breiman’s Two Cultures as a Historical Transition

Leo Breiman’s 2001 paper, *Statistical Modeling: The Two Cultures*, described a division in statistical practice.

The first culture begins with an explicit probabilistic model of how data are generated.

For example:

$$
Y
=
\beta_0
+
\beta_1X_1
+
\cdots
+
\beta_pX_p
+
\epsilon.
$$

The researcher chooses the variables, functional form, probability distribution, and relevant assumptions. Data are then used to estimate the unknown parameters and quantify uncertainty.

The second culture treats the data-generating mechanism as substantially unknown:

$$
X
\rightarrow
\boxed{\text{unknown mechanism}}
\rightarrow
Y.
$$

Rather than specifying the mechanism in advance, the researcher trains an algorithm to approximate the mapping from $X$ to $Y$ and evaluates how well it generalizes to new data.

This is commonly presented as a conflict between statistics and machine learning.

But the deeper difference is not that one culture uses structure while the other ignores it.

Both use structure.

The difference is **where the structure is placed and how it is justified**.

In traditional statistical modeling, humans specify much of the structure explicitly before estimation.

In algorithmic modeling, humans specify a learning framework—a hypothesis class, architecture, objective, optimization procedure, and evaluation protocol—and allow data and computation to determine more of the detailed structure.

Breiman’s two cultures can therefore be interpreted as two different allocations of cognitive labor:

$$
\text{human-specified structure}
\quad \longleftrightarrow \quad
\text{machine-discovered structure}.
$$

This also explains why the cultures emphasize different forms of validation.

A traditional statistical analysis often asks whether inference is valid under a stated set of assumptions.

An algorithmic approach often asks whether the learned system performs well on unseen data.

Neither criterion is sufficient by itself.

A statistically elegant model can be badly misspecified.

A highly predictive algorithm can exploit unstable correlations, fail under distribution shift, or provide little causal understanding.

The two cultures therefore have complementary strengths and complementary failure modes.

The more productive question is not which culture should eliminate the other. It is:

> **How much structure should humans specify, how much should machines discover, and what evidence should justify the resulting claims?**

---

## 7. Machine Learning Does Not Remove Structure

Machine learning is sometimes described as allowing “the data to speak for themselves.”

Data do not speak for themselves.

Every learning system contains inductive biases: assumptions that make some explanations easier to learn than others.

A convolutional neural network encodes locality and repeated spatial structure.

A graph neural network encodes relations among connected entities.

An equivariant model preserves specified transformations.

A transformer allows information to be routed dynamically among positions through attention.

An autoregressive language model decomposes a sequence distribution into a chain of next-token predictions.

Regularization favors some solutions over others.

A loss function defines which errors matter.

A dataset determines which parts of reality the system encounters.

A training curriculum determines the order and distribution of experience.

Even the choice of what to predict imposes a view of what information should be retained.

Machine learning therefore does not eliminate prior knowledge. It often transforms prior knowledge from a compact, human-readable equation into a distributed set of architectural and procedural choices.

This is why it is useful to distinguish between **explicit structure** and **implicit structure**.

A linear model states its structural assumptions directly.

A deep neural network may express weaker but broader assumptions through its architecture, optimization, and data. Its detailed representation is then learned rather than written down.

The meaningful contrast is not:

$$
\text{structure}
\quad \text{versus} \quad
\text{no structure}.
$$

It is:

$$
\text{structure specified in advance}
\quad \text{versus} \quad
\text{structure discovered through learning}.
$$

Most successful systems combine both.

---

## 8. Deep Learning and the Automation of Representation

Traditional machine learning frequently depended on handcrafted features.

Humans decided how images, speech, text, or behavior should be represented, and a learning algorithm operated over those representations.

Deep learning changed this division of labor.

A multilayer network can learn intermediate representations jointly with the final task. Instead of specifying every relevant feature, humans define an architecture and objective through which useful features can emerge.

This is a major conceptual shift.

Early scientists had to discover the relevant variables themselves. They had to invent concepts such as force, gene, field, molecule, utility, and temperature before those concepts could support formal models.

Deep learning begins to automate part of this representational work.

An image model can learn visual features rather than relying entirely on manually designed detectors.

A speech model can learn acoustic representations.

A language model can learn distributed representations of syntax, semantics, style, and recurring reasoning patterns without receiving an explicit rulebook for language.

The machine is no longer only estimating parameters within a representation designed by humans.

It is participating in the construction of the representation itself.

This does not mean that the model discovers the uniquely correct structure of reality. Learned representations are shaped by the objective, architecture, data, and environment. They preserve what is useful for the training process.

But the scale of representational discovery is new.

Human beings design the conditions for learning; the machine performs a large portion of the detailed compression.

---

## 9. Large Language Models and the Scaling Regime

Large language models push this strategy further.

Human beings do not manually encode millions of rules covering grammar, rhetoric, programming, historical associations, scientific discourse, and conversational behavior.

Instead, developers provide several ingredients:

- a general architecture;
- a prediction objective;
- a large corpus of data;
- an optimization process;
- substantial computation;
- post-training signals that shape behavior.

The detailed representational structure is acquired through training.

Language is particularly powerful because it contains compressed traces of human knowledge and activity. Scientific papers, software, debates, explanations, stories, records, and instructions all leave linguistic evidence.

An LLM trained on text does not directly observe the world in the way an experimental scientist does. It observes a cultural and informational projection of the world. Nevertheless, that projection contains a remarkable amount of structure.

The resulting system represents an extreme redistribution of cognitive labor.

At the Newtonian end of the spectrum, humans discover the abstractions and write a compact model. Computation plays a limited role.

At the large-model end, humans design the learning process, while optimization over data constructs billions of parameters that collectively encode a vast learned representation.

The historical shift can be summarized as a changing mixture:

$$
\text{Human prior knowledge}
+
\text{Data}
+
\text{Computation}.
$$

When data and computation are scarce, human prior knowledge must do more of the work.

When data and computation become abundant, weaker or more general prior structure can be combined with large-scale search.

Scaling is therefore not merely “making models larger.” It is a strategy for shifting more structural discovery from manual specification to computation.

But scaling also produces a new constraint.

It is expensive.

---

## 10. The Return of Structure

The early success of large-scale learning can create the impression that progress requires only more data, more parameters, and more computation.

But every resource encounters limits.

High-quality data are finite.

Training consumes capital and energy.

Inference introduces latency and operating cost.

Some tasks provide weak or ambiguous feedback.

Static datasets may not contain the experiences needed for new capabilities.

More computation can compensate for weak structure, but the compensation may be inefficient.

The frontier therefore turns again toward a familiar question:

> **How can a system learn more from less?**

This is the problem of efficiency, and efficiency is closely related to structure.

Useful prior structure narrows the search space.

A physically informed model need not rediscover every conservation law from examples.

A retrieval system need not store every changing fact inside model parameters.

A tool-using model need not approximate arithmetic or database lookup internally when an external system can perform the operation reliably.

A verifier supplies information about which candidate solutions satisfy a constraint.

A memory system preserves useful experience across episodes.

A curriculum determines which experiences should appear and in what order.

An agent can select actions that generate informative feedback rather than passively accepting a fixed dataset.

Not every efficiency improvement is an inductive bias. Better hardware, numerical precision, compilers, distributed systems, and optimization techniques can reduce cost without introducing new domain assumptions.

But many major gains come from adding structure to the learning process.

A useful informal principle is:

> **The less valid prior structure we provide, the more we generally pay in data, computation, and search. The more valid structure we possess, the more efficiently we can learn.**

The word “valid” matters. A correct inductive bias improves efficiency. A mistaken one can prevent the learner from discovering the truth.

Structure is therefore both an advantage and a risk.

It helps learning by excluding possibilities. But reality may lie among the possibilities excluded.

The central design problem is not to maximize prior structure. It is to use enough structure to learn efficiently while preserving enough flexibility to be corrected by evidence.

---

## 11. Discovery as a Spiral

The history of knowledge is not a straight-line movement from human-designed models to structureless machine learning.

It is better understood as a spiral:

$$
\text{observe}
\rightarrow
\text{discover structure}
\rightarrow
\text{encode structure}
\rightarrow
\text{predict or act}
\rightarrow
\text{observe the result}
\rightarrow
\text{revise the structure}.
$$

Once discovered, structure becomes prior knowledge for the next stage.

Newtonian mechanics becomes a prior for later physics.

Chemical knowledge constrains biological investigation.

Biological knowledge informs medicine.

Scientific theories guide the design of instruments and experiments.

A pretrained model provides representations for later tasks.

Human feedback becomes data for post-training.

Model-generated solutions can become training material when they can be independently verified.

Tools extend the range of operations a model can perform.

Agents can interact with environments to obtain new evidence.

Each cycle begins with more accumulated structure than the previous one. But each cycle also encounters new anomalies, scales, and domains that the existing structure cannot explain.

Knowledge therefore advances through alternating phases:

1. **expansion**, in which richer data and more flexible models reveal patterns that existing theories miss; and
2. **compression**, in which discovered regularities are distilled into reusable concepts, architectures, laws, or procedures.

Flexible learning systems are powerful instruments of expansion.

Explicit theories and inductive biases are powerful instruments of compression.

Progress requires both.

---

## 12. From Static Models to Active Learning Systems

This history also suggests a possible next transition.

The dominant abstraction of classical statistics is the fitted model.

The dominant abstraction of machine learning is the trained predictor.

The dominant abstraction of deep learning is a learned representation combined with a predictor or generator.

But the increasingly important unit may be the **learning system**.

A learning system may include:

- a foundation model;
- short- and long-term memory;
- external retrieval;
- computational tools;
- planning;
- verification;
- interaction with an environment;
- mechanisms for selecting new observations;
- procedures for revising behavior.

Its operation is no longer merely:

$$
X \rightarrow \hat{Y}.
$$

It is closer to:

$$
\text{observe}
\rightarrow
\text{model}
\rightarrow
\text{reason}
\rightarrow
\text{act}
\rightarrow
\text{receive feedback}
\rightarrow
\text{update}.
$$

At this point, the connection to epistemology becomes explicit.

An active learning system must confront questions that philosophers and scientists have asked for centuries:

What should I believe?

How uncertain should I be?

What evidence would change my belief?

Which experiment would distinguish among competing explanations?

What observation should I seek next?

When is a prediction reliable?

When has the environment changed?

Which parts of my current representation should be preserved, and which should be revised?

These are no longer only philosophical questions. They are becoming engineering requirements.

A system that generates plausible answers without checking reality remains an incomplete learning system. A system that can act, test, verify, and revise begins to approximate the full epistemic loop.

This may be the deeper meaning of agentic AI and model self-improvement. The goal is not simply to produce a larger static model. It is to build systems capable of organizing their own interaction with evidence.

However, genuine self-improvement requires more than generating additional text. The system needs feedback that is connected to reality. Otherwise, it risks amplifying its own errors.

The essential problem is therefore not self-generation but **epistemically grounded self-correction**.

---

## 13. Prediction, Explanation, and Understanding

Breiman’s two cultures also raise a deeper question: Is accurate prediction equivalent to understanding?

Not necessarily.

A system can predict well by exploiting regularities without identifying the causal mechanism that produced them. Such a system may fail when the environment changes or when an intervention alters the relationships it learned.

Conversely, an interpretable theoretical model may capture an important causal mechanism while remaining less accurate for short-term prediction because it omits many secondary factors.

Prediction and explanation are therefore related but distinct achievements.

Prediction asks:

> Given these conditions, what is likely to happen?

Explanation asks:

> What structure produced this outcome, and how would the outcome change under intervention?

A mature science usually needs both.

Prediction tests whether a representation captures stable regularities.

Explanation supports transfer, intervention, and conceptual compression.

Machine learning expanded our predictive reach. The next challenge is to use flexible models not only to approximate outcomes but also to help identify stable abstractions, causal mechanisms, and experiments that can distinguish among competing accounts.

The most valuable future systems may not replace scientific theories with opaque predictors. They may help scientists move more rapidly through the entire loop:

$$
\text{pattern}
\rightarrow
\text{hypothesis}
\rightarrow
\text{experiment}
\rightarrow
\text{evidence}
\rightarrow
\text{revised theory}.
$$

In that role, artificial intelligence becomes part of the technology of discovery rather than merely a technology of prediction.

---

## 14. A Unified View

We can now place Descartes, Newton, Bayes, classical statistics, Breiman, deep learning, and large language models within a single conceptual history without claiming that they were all doing the same thing.

They were not.

But they occupy different stages in humanity’s evolving effort to discover structure under constraint.

The epistemological thread asks:

> **How should knowledge be acquired, justified, and revised?**

The domain science and technology thread asks:

> **What structures exist in this part of the world, and how can they be understood or used?**

Their development is enabled by the capability frontier:

$$
\boxed{
\text{Observation and Data}
\quad+\quad
\text{Computation}
\quad+\quad
\text{Mathematics and Algorithms}
}
$$

And the practical methodology of each era reflects a particular allocation of cognitive labor:

$$
\boxed{
\text{Human Priors}
\quad+\quad
\text{Empirical Evidence}
\quad+\quad
\text{Computational Search}
}
$$

Early science relied heavily on human abstraction because data and computation were scarce.

Statistics formalized reasoning under uncertainty and partial knowledge.

Machine learning delegated more search to algorithms.

Deep learning delegated part of representation discovery to optimization.

Large language models expanded the scale of this approach by combining general architectures, massive datasets, and extraordinary computation.

Now, as scaling becomes costly, progress increasingly depends on better structure: better objectives, better data, better algorithms, better memory, better tools, better feedback, and better ways to connect learning systems to reality.

The movement is not from models to model-free learning.

It is from one arrangement of structure, data, and computation to another.

---

## Conclusion: The Technology of Knowing

The history from Newton to large language models is not only the history of scientific theories or increasingly powerful machines.

It is the history of the **technology of knowing**.

Human beings have always faced a world too complicated to observe completely and too large to represent in full. Progress has depended on finding compressions: concepts, laws, probabilities, models, representations, and algorithms that preserve the structures relevant to prediction and action.

As our capabilities have expanded, the balance of discovery has shifted.

When observation and computation were severely limited, human intelligence had to perform most of the abstraction.

When data and computation became abundant, machines could search larger spaces and learn representations that humans could not specify manually.

As brute-force scaling encounters new constraints, discovered structure becomes valuable again—not as a retreat from data-driven learning, but as the means by which learning becomes more efficient, reliable, and transferable.

This produces a recurring cycle:

$$
\text{use existing knowledge}
\rightarrow
\text{collect evidence}
\rightarrow
\text{discover new structure}
\rightarrow
\text{encode that structure}
\rightarrow
\text{learn more efficiently}.
$$

The cycle has no final stage.

Every model reveals some regularities and hides others. Every new instrument opens part of reality while imposing its own limitations. Every method of learning solves one bottleneck and exposes another.

Breiman’s “two cultures” therefore represent one important moment inside a much longer history. The enduring issue is not whether humans should specify models or machines should learn them.

It is how intelligence—human, artificial, or combined—should divide the work of discovery among prior knowledge, observation, and computation.

That is the central challenge of science.

It is also the central challenge of artificial intelligence.

And beneath both lies the same timeless question:

> **Given limited observations, limited resources, and an immensely complex world, how can an intelligent system discover the structures that matter—and use them to understand what comes next?**

## Selected References

- Bayes, Thomas. “An Essay towards Solving a Problem in the Doctrine of Chances.” 1763.
- Box, George E. P. “Science and Statistics.” *Journal of the American Statistical Association*, 1976.
- Breiman, Leo. “Statistical Modeling: The Two Cultures.” *Statistical Science*, 2001.
- Descartes, René. *Discourse on the Method*. 1637.
- Newton, Isaac. *Philosophiæ Naturalis Principia Mathematica*. 1687.
- Vaswani, Ashish, et al. “Attention Is All You Need.” 2017.
