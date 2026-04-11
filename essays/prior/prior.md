# Prior Discovery: Iterative Falsification in Search of World Structure

In much of modern engineering and machine learning, science is quietly treated as a data-first activity. We gather data, fit models, optimize metrics, and improve performance. This paradigm is extraordinarily effective when the task is already well defined. If the problem is known, the objective is fixed, and success can be measured by a benchmark, then data-driven iteration can be fast, practical, and powerful.

But the moment the problem shifts from the known to the unknown, this paradigm begins to reveal its limits. It is one thing to optimize within an established frame. It is another to ask what the frame itself should be. It is one thing to fit patterns in observations. It is another to understand why those observations arise in the first place. And it is precisely at this boundary—where we are no longer merely solving problems, but trying to discover what the world is really doing—that a different view of science becomes necessary.

If we step back and re-examine the nature of scientific inquiry, a neglected fact comes into view: **scientific discovery does not truly begin with data. It begins with structure.** Experience is not the starting point, but the constraint. Data is not the source, but the evidence. A model is not merely a curve-fitting device, but a hypothesis about the generative organization of the world.

This essay argues for a view of science better suited to the exploration of the unknown: a structure-centered, falsification-driven, causally organized paradigm of discovery. In this view, the goal of science is not to accumulate observations until a rule emerges automatically. It is to formulate structural hypotheses about how the world works, to derive their consequences, and to use experience to progressively refine, pressure-test, and sometimes overturn them.

## The World Comes Before Experience

We often say that we “learn laws from data.” The phrase sounds natural, but logically it is misleading. Data consists of observations, and observations are outcomes. They do not carry within themselves the mechanism that generated them. We may observe countless trajectories of moving objects, but trajectories do not, by themselves, yield Newtonian mechanics. We may record complex streams of visual input, but pixels do not directly reveal the constraints of the physical world.

A more serious starting point is to acknowledge that the world already possesses structure before we observe it. That structure is reflected in generative mechanisms, in lawful constraints, in the boundaries between what can happen and what cannot. These structures are not produced by experience. They are what make experience possible in the first place.

When an object slides across a table, its motion is not arbitrary. It is shaped by contact constraints, friction, inertia, geometry, and other factors that jointly define a hidden generative rule. What we observe is only the visible trace of that rule. The trajectory is the effect. The structure is the cause.

Seen in this light, the task of science is not to extract patterns from data as if the patterns were already self-explanatory. The task is to propose hypotheses about the underlying structures that generate the data, and then to use experience to move closer to those structures.

## Theory Is Not a Summary of Experience, but a Guess About Structure

Once we accept that structure precedes observation, our understanding of theory must change as well. A theory is no longer an inductive summary of experience. It becomes a compressed conjecture about the world’s generative organization.

To put forward a model is to say: perhaps the world works like this. That claim does not arise mechanically from data. It may be informed by prior knowledge, physical intuition, analogy, or creative insight, but it is still a conjecture. As many scientists have noted, the birth of a theory is not itself governed by a strict logical procedure. There is no general algorithm for discovery.

But once a theory has been proposed, everything changes. It enters the realm of testing, and that is where scientific method truly begins.

From the theory, we derive consequences. These consequences may take the form of experimental predictions, system behaviors, statistical regularities, or dynamical trajectories. We then compare these derived consequences with the world. If the world disagrees, the theory must be revised or abandoned. If the world agrees, the theory is not thereby proven true; it has merely survived another round of scrutiny.

Scientific progress, then, is not fundamentally the accumulation of confirmations. It is the progressive elimination of structurally inadequate hypotheses.

## The Real Role of Experience: Not to Generate Knowledge, but to Filter Hypotheses

Under this view, the role of experience is transformed. Experience is not the origin of knowledge. It is a filtering mechanism, a constraint on the space of possible explanations.

Every experiment, every measurement, every observation cuts into the hypothesis space. It does not tell us directly what the right answer is. Instead, it tells us which answers cannot be right. As more observations accumulate, the hypothesis space contracts, and the remaining structural possibilities become sharper and more realistic.

More importantly, the scientific value of experience often lies not in success, but in failure. When a model breaks down under an important condition, what is exposed is not just an error term. What is exposed is a structural inadequacy. Perhaps a crucial variable was omitted. Perhaps independence was wrongly assumed. Perhaps a constraint that seemed negligible is in fact fundamental.

This is why the old phrase *the devil is in the details* is so apt. The deepest information is often not found in superficial agreement, but in the pattern of mismatch. A failed prediction is not merely a setback. It is often the most informative event in the entire research cycle.

## From Physical Modeling to Mathematical Modeling to Computational Realization

If the purpose of scientific discovery is to identify structure, then modeling must follow a disciplined causal chain. That chain can be expressed simply:

**physical modeling → mathematical modeling → computational realization**

Physical modeling comes first. At this stage, we propose how the world may be organized. What are the relevant variables? What entities interact? What constraints, conservation laws, symmetries, or invariants govern the system? This stage determines the substance of the problem.

Mathematical modeling is the formalization of these physical assumptions. It turns intuitive structure into analyzable form. Once the structure is mathematically expressed, we can reason with it, derive implications from it, compare alternatives, and identify where ambiguities remain. If a physical hypothesis cannot be translated into a clear mathematical structure, that usually means the original intuition is still too vague.

Computational realization is the stage at which the mathematical model is turned into an executable system. But its purpose is often misunderstood. It is not merely there to “make the solution work.” Its deeper purpose is to construct a testable environment—one in which the consequences of our assumptions can be made concrete and exposed to pressure.

By running the system, we can observe how the model behaves under varying conditions, where it fails, what details matter, and which structural commitments survive contact with reality.

The crucial point is that these three levels must remain causally aligned. The computational system should not be designed arbitrarily, detached from physical assumptions. Nor should the mathematical model be treated as a formal convenience with no real structural commitment. If the chain breaks, the process degenerates into a pile of engineering tricks rather than a genuine inquiry into the organization of the world.

## Computation Is Not for Solving the Problem; It Is for Exposing the Problem

In engineering culture, computation is often treated as the end goal. The model trains. The metric improves. The system runs. The benchmark is beaten. These are treated as success.

But under a structural view of scientific inquiry, computation plays a deeper and more demanding role.

Computation is a magnifier. It takes the assumptions hidden in our models and pushes them into concrete operational settings where their strengths and weaknesses become visible. A model may appear successful on simple data simply because the task is too small or too forgiving. Once the environment becomes richer, its structural weaknesses can no longer hide.

For this reason, the output of computation should not be interpreted merely as an answer. It should be interpreted as feedback about where our understanding is still wrong or incomplete. Success tells us that the current structure remains viable within a certain range. Failure tells us where refinement is needed. Both outcomes matter. In fact, the latter is often more valuable.

## Why the World Is Learnable: Structure Determines Data Efficiency

If the world were completely arbitrary, then no finite amount of data could support meaningful learning of its governing principles. The reason science works at all is that the world itself has certain favorable properties.

One is low-dimensionality. The raw observational space may be extremely high-dimensional, as in images, videos, or multimodal sensor streams. But the true generative degrees of freedom that govern the system are often far fewer. This allows complex phenomena to be compressed into more tractable structure.

Another is local smoothness and local convexity. Within bounded operating regimes, the response of a system to perturbation is often continuous, stable, and predictable. Small changes do not typically produce arbitrary discontinuities. This means that local observations can reveal more global regularities.

A third is structured interaction. Variables in real systems rarely interact in a fully unstructured, all-to-all way. Their relations are often sparse, layered, or modular. That structure makes decomposition possible, and decomposition reduces difficulty.

These properties help explain why high data efficiency is possible in some scientific domains. It is not because we have found a magical algorithm. It is because the world itself is structurally organized in ways that finite observation can exploit.

## Structure Discovery in Embodied Intelligence

This perspective becomes especially important in embodied intelligence. Vision, touch, proprioception, and action are not separate worlds; they are different projections of the same physical reality. If we simply learn mappings between these signals, we may achieve task performance, but we do not necessarily understand the environment.

The deeper challenge is to uncover the physical rules that generate these signals. A robot that pushes an object, changes its viewpoint, or applies force is not merely collecting more data. It is conducting an experiment. It is probing the hidden structure of the environment to infer dynamics, constraints, affordances, and stability.

In this setting, learning is no longer just an input-output mapping problem. It becomes a process of designing actions that reveal structure. The agent shifts from passive observer to active investigator. It is no longer just a data processor. It becomes a structure discoverer.

## Science as an Endless Iterative Process

Beneath all of this lies an important epistemic commitment: our knowledge is almost certainly incomplete, and often substantially wrong. Our first models are crude. They capture fragments of structure, but never the whole. They are missing details, and the details matter.

Science does not advance by accumulating final truths. It advances by repeatedly proposing, deducing, testing, failing, refining, and trying again. Each iteration produces a model that is slightly sharper, slightly more detailed, slightly more causally faithful than the last. The process does not culminate in perfect certainty. It converges, at best, toward models that have not yet been overturned at the current level of resolution.

To understand the world, then, is not to possess an ultimate truth. It is to construct a structural model that, for now, explains the relevant phenomena, survives severe tests, and remains open to revision.

## Conclusion

If one had to summarize this view of science in a single sentence, it would be this:

**Science does not construct laws from data. It advances by continually proposing and testing structural hypotheses, gradually approaching the generative rules of the world that exist prior to experience yet become visible through it.**

In this process, physical modeling sets the direction, mathematical modeling provides the language, computational realization serves as the testing ground, and empirical feedback drives revision. Together, they form a discovery loop centered on structure and propelled by falsification.

This is not the easiest path. It demands more than optimization, more than curve fitting, and more than benchmark progress. But when the goal is not just to solve a known problem, but to enter the unknown, it is the most reliable path we have.
