# Small Semantics, Large Design Spaces

**Can tensor regularity separate optimization freedom from verification
complexity?**<br>
**Status:** Exploratory research proposal; the central hypothesis has not been
tested<br>
**Date:** 2026-07-26

## Abstract

Agentic AI can search a large software–hardware design space, but it cannot
guarantee that every lowering, schedule, or hardware choice is correct. A
formal checker can provide that guarantee only when the system has a precise
and manageable model.

Tensor computing may have the right structure. In a bounded tensor region,
operations, shapes, dependencies, and data movement are often known before
execution. Tiling, layout, placement, buffering, routing, scheduling, and the
GPU microarchitecture itself still create many possible implementations. The
key possibility is that these are mostly choices **between designs**, not
unpredictable choices **inside one running design**.

This proposal has two questions:

1. **Why might tensor computing have a large optimization space but a small
   verification-relevant runtime space for each selected design?**
2. **Can one frozen staged-resource contract compose workload, compiler,
   runtime, architecture, and implementation obligations without hidden
   target-specific semantics or unacceptable loss of good designs?**

The model treats microarchitecture, compiler, and runtime as resource
decisions committed at manufacture time, compile time, and runtime.

The first goal is not to build a complete language or an end-to-end LLM
system. The first question tests whether the project is tractable. The second
constructs the cross-layer model if that opportunity is real. An AI agent then
searches the model; it is not the source of its correctness.

## 1. The core problem

End-to-end AI infrastructure design contains many coupled choices:

$$
\begin{aligned}
\text{tensor program}
&\rightarrow \text{numerical form}
\rightarrow \text{tiling and fusion}\\
&\rightarrow \text{layout and placement}
\rightarrow \text{schedule and communication}\\
&\rightarrow \text{GPU microarchitecture}
\rightarrow \text{RTL and physical implementation}.
\end{aligned}
$$

A choice at one level changes the legal and efficient choices below it. For
example, tile size changes data reuse, storage demand, parallelism, and
communication. An AI agent may search these interactions well, but its answer
is still a statistical proposal. A mistake found after RTL integration or
tape-out is expensive.

Software and hardware are coupled. For a chosen machine, software selects
transformations, schedules, mappings, and resource allocations. But the
machine is also a design variable: compute arrays, pipelines, memories, ports,
links, and supported operations can be changed to serve the workload. The
search target is therefore a matched software–hardware pair, not software on a
permanently fixed GPU.

The naive response is to model the whole system and verify it. That usually
fails because verification must cover every runtime alternative: branch
outcomes, arbitration decisions, queue states, aliases, cache states,
speculation, interrupts, and event interleavings. Even a small controller can
have an enormous reachable state space.

The real question is therefore not whether tensor workloads are large. They
are. It is whether their **performance freedom** can be placed mainly at
design time, without creating the same amount of **runtime control
uncertainty**.

The two research questions depend on each other. If design freedom necessarily
becomes runtime uncertainty, end-to-end verification will not scale. If the
layer-specific priors cannot be composed, a local proof will not give an
end-to-end guarantee.

## 2. Question I: Why optimization and verification may separate

### 2.1 Design alternatives are not runtime alternatives

Freeze a finite workload suite \(\mathcal T\), architecture and schedule
bounds, a technology backend, and a physical budget \(B\). One manufactured
hardware design \(H\) is shared across the suite; compile-time choices \(c_t\)
and runtime policies \(\omega_t\) may vary by task. Define:

$$
\mathcal Z_B(\mathcal T)
=
\left\{
(H,\{c_t,\omega_t\}_{t\in\mathcal T})
\;\middle|\;
\operatorname{Accepted}_B
\left(H,\{c_t,\omega_t\}_{t\in\mathcal T}\right)
\right\}/{\equiv}.
$$

\(\operatorname{Accepted}_B\) requires a well-formed bounded architecture,
accepted software and hardware certificates, successful generation, and the
declared physical budget under the named backend. The equivalence
\(\equiv\) removes registered aliases, symmetries, and idle-time differences.
The resulting set is finite. Its design richness is:

$$
D_B(\mathcal T)
=
\log_2\left|\mathcal Z_B(\mathcal T)\right|.
$$

After one \(H\) is manufactured and \(c_t\) is compiled, the unchosen hardware
and compiler alternatives are not runtime states. Only decisions made by
\(\omega_t\) branch during execution.

For one selected design \(z\), let \(T_z\) be its runtime transition system.
Let \(\alpha_\varphi\) be a preregistered abstraction proved to preserve the
property \(\varphi\). Over every admitted input \(x\) and environment trace
\(\xi\), define:

$$
V_\varphi(z)
=
\left|
\alpha_\varphi
\left(
\bigcup_{x,\xi}
\operatorname{Reach}(T_z;x,\xi)
\right)
\right|.
$$

The abstraction cannot be chosen after seeing the result or collapse states
without a preservation proof. Raw state count is still not a complete measure
of verification difficulty, so checker time, peak memory, and certificate size
are measured separately. \(D_B(\mathcal T)\) and \(V_\varphi(z)\) describe two
different sources of complexity:

$$
\boxed{
\text{many candidate designs}
\;\not\Rightarrow\;
\text{many runtime alternatives in each design}
}
$$

An optimizer or agent explores \(\mathcal Z_B(\mathcal T)\). A verifier checks
one chosen \(z\). Search and counting cost are reported separately from the
cost of checking that candidate.

### 2.2 Why static tensor dataflow may create this separation

Suppose \(n\) independent tiles may be assigned to \(p\) compute units. There
are already:

$$
p^n
$$

possible mappings, before considering tile shapes, layouts, routes, buffering,
or schedules. This is a large optimization space.

After one mapping and schedule are selected, however, the unit, route, and
start time of each tile can be fixed. For the closed, fixed-latency static
subclass \(z_{\mathrm{static}}\), the strong property to prove is:

$$
\begin{aligned}
\forall x,\;
\forall e\in\operatorname{Exec}(z_{\mathrm{static}},x),\;
\forall\tau:\qquad
\alpha_{\mathrm{ctl}}(e_\tau)=F_{z_{\mathrm{static}}}(\tau).
\end{aligned}
$$

This quantifies over all executions, including any internal nondeterminism.
If it holds, changing \(x\) changes tensor values but not the projected control
trace, whose branching degree is one. Dynamic environments and runtime
policies are evaluated with \(V_\varphi\) instead of this stronger equation.

This is the proposed source of the separation:

- data dependencies are resolved before execution;
- performance choices are encoded as design parameters;
- tensor values are handled symbolically by functional proofs; and
- runtime control does not rediscover the schedule.

This claim needs proof, not intuition. Tensor memories still contain an
enormous number of possible values. We must prove that erasing those payloads
does not change future control behavior. Variable latency, data-dependent
routing, arbitration, sparse execution, faults, and dynamic batching may
break the property.

The distinction is therefore not “CPU versus accelerator” or even “control
flow versus dataflow.” It is:

$$
\boxed{
\text{dependencies fixed at design time}
\quad\text{versus}\quad
\text{dependencies discovered at runtime}
}
$$

## 3. What computing history teaches us

Earlier models solved different problems by preserving different facts.

- **Lovelace** separated an operation, its operands, and the mechanism that
  performs it. This made implementation a distinct object of study [R1].
- **Turing** made mechanical computation mathematically precise and independent
  of one machine. Universality answered what can be computed, not what is
  physically efficient [R2].
- **Shannon** connected Boolean meaning to relay construction and
  simplification. This joined semantics, synthesis, and physical economy
  [R3].
- **Von Neumann and his collaborators** faced limited equipment, storage, and
  reliability. They treated architecture as a hierarchy and made time–equipment
  tradeoffs explicit [R4, R5].
- **General-purpose ISAs and dynamic ILP machines** preserved one software
  contract across varied programs while recovering parallelism at runtime.
  This flexibility placed more performance behavior in mutable machine state
  [R6, R7].
- **Systolic and tensor systems** exploited regular dependencies by exposing
  spatial mapping, communication, and schedules [R8, R9].

The lesson is methodological. A useful computation model is not the most
general one. It preserves the structure needed for its target problem and
removes distinctions that do not matter. Our task is to discover whether
tensor workloads admit such a boundary; we should not assume it in advance.

Modern tensor compilers expose schedules and mappings [R10, R11]. Verified
tensor compilers prove selected transformations [R12]. Timing-aware hardware
languages check resource and cycle contracts [R13]. Formal accelerator
interfaces connect software-visible behavior to implementations [R14].
Architecture tools search mappings and estimate cost [R15]. The open question
is not whether static schedules are easier to analyze. Synchronous dataflow
and resource-aware hardware languages already exploit fixed dependencies
[R16, R17]. This proposal instead asks whether one frozen contract can jointly
measure design richness, representation loss, and per-design verification cost
as decisions move among manufacture, compile, and runtime, while carrying
replayable evidence across the full chain.

## 4. Question II: Can one contract compose the cross-layer priors?

A DSL is not the starting point. The order should be:

$$
\text{physical prior}
\rightarrow
\text{mathematical model}
\rightarrow
\text{checked contract}
\rightarrow
\text{DSL}.
$$

### 4.1 Priors from each layer

A **prior** here means known structure that introduces design variables,
constraints, or both. It need not be a literal law of physics.

The **workload prior** includes typed tile operations, bounded iteration
domains, producer–consumer dependencies, tensor shapes, and whether values can
change control.

The **system-software prior** includes legal graph transformations, numerical
formats, tiling and fusion rules, memory allocation, synchronization, runtime
services, and the mapping and scheduling choices exposed by the compiler.

The **architecture prior** includes:

- finite compute units, storage, ports, and links;
- spatial topology and legal routes;
- locality and data-movement cost;
- pipeline latency, issue rate, and synchronization;
- resource sharing, capacity, and conflict rules; and
- the operations and data types implemented directly.

The **physical-implementation prior** includes wire delay, clocking, area,
power, energy, reliability, fabrication technology, and layout rules. Some of
these can be checked by implementation tools; others require calibrated models
or measurement. They must not be silently treated as theorems.

These priors enter through three commitment stages:

1. **Manufacture time:** microarchitecture design creates the resource fabric
   — compute units, memories, pipelines, ports, and links. These resources are
   fixed after fabrication.
2. **Compile time:** the compiler transforms the dataflow graph and chooses
   mostly static tiling, placement, storage, routing, and scheduling on that
   fabric.
3. **Runtime:** the runtime allocates what remains dynamic, such as requests,
   queues, batches, arbitration, and load balance.

Hardware design is therefore resource **construction**; compiler and runtime
design are two stages of resource **allocation**. They can be modeled together
because they act on the same operations, capacities, dependencies, and costs.
Every decision \(q\) should state when it becomes fixed:

$$
\kappa(q)
\in
\left\{
\text{manufacture},
\text{compile},
\text{runtime}
\right\}.
$$

The policy \(\omega\) may itself be selected before execution. Its individual
allocation actions are runtime decisions. The label \(\kappa(q)\) applies to
the value of a decision, not merely to the code that computes it.

Moving a decision later may improve adaptation to input or load, but it also
adds mutable control state. Moving it earlier may simplify verification, but
it can lose performance when runtime conditions vary. The best commitment
stage is itself a co-design choice.

At each lowering step, a prior introduces choices and then constrains them.
Across all choices, the design tree branches. Along one chosen lowering path,
the represented implementation set narrows:

$$
\gamma_{i+1}(x_{i+1})
\subseteq
\gamma_i(x_i).
$$

The certificate for that step must prove that the narrowing preserves the
upper-level meaning. This is the formal role of an intermediate
representation: it is a contract for the new information introduced at that
layer, not merely a new syntax.

Let
\(z=(H(h),\{c_t,\omega_t\}_{t\in\mathcal T})\)
be one accepted joint design from Section 2, and let \(I_h\) be the artifact
produced by the named backend \(\beta\). End-to-end optimization is simply:

$$
z^*
\in
\arg\min_{z\in\mathcal Z_B(\mathcal T)}
J(\mathcal T,z).
$$

Here \(J\) is a fixed scalar objective. If weights or budgets are not
justified, the study reports the Pareto frontier for performance, power, and
area instead; there is then no single “best PPA” point.
Physical feasibility inside \(\mathcal Z_B(\mathcal T)\) must be labeled as
measured or predicted. A prediction is not a formal physical guarantee.

Thus hardware contributes both constraints and optimization choices. A wider
array may expose more parallel schedules but require more area and bandwidth.
A larger buffer may reduce traffic but increase access energy or cycle time.
A new interconnect may remove one conflict and create another.

### 4.2 A joint design contract

Microarchitecture synthesis must use a fixed, typed architecture grammar:

$$
h\in\Theta_A,
\qquad
H(h)=\Gamma_A(h).
$$

The agent may choose component counts, capacities, topology, and compositions
allowed by \(\Gamma_A\). It may not invent a new transition rule or change the
checker for each candidate. Extending the grammar creates a new model version
with new proof obligations.

The first model needs these objects:

$$
\begin{aligned}
P&=(O,E,\mathrm{Ty}),
&&\text{the typed tensor operations and dependencies},\\
H(h)&=(R_h,C_h,L_h,\Delta_h),
&&\text{the machine resources, capacities, links, and timing rules},\\
c&=(\theta,\mu,\lambda,s,\rho),
&&\text{the compile-time tiles, mapping, storage, schedule, and routes},\\
\omega&=\text{the bounded runtime allocation policy},\\
I_h&=\Gamma_{I,\beta}(H(h)),
&&\text{the lower-level implementation of the chosen hardware}.
\end{aligned}
$$

Here \(h\) contains microarchitecture choices such as array dimensions,
pipeline structure, buffer capacities, bank counts, ports, and network
topology. \(P\) states what must be computed, \(H(h)\) states what the selected
machine can do, \(c\) fixes the static software realization, \(\omega\) governs
the remaining runtime choices, and \(I_h\) is an RTL or other executable
realization. In the strictly static case, \(\omega\) has only one legal action
at every step.

Concurrency, sharing, timing, and conflict are not informal annotations. They
become constraints. If action \(a\) starts at \(s_a\), ends at \(e_a\), and
uses \(u(a,r,\tau)\) units of resource \(r\), then:

$$
\forall r,\tau:\qquad
\sum_{a:\,s_a\le \tau<e_a}
u(a,r,\tau-s_a)
\le C_h(r).
$$

For every dependency \((a,b)\in E\):

$$
s_b\ge e_a.
$$

Storage lifetimes, address overlap, routes, and link use receive similar
finite constraints. One checker connects the software realization to tensor
meaning:

$$
\begin{aligned}
&\operatorname{Check}_{\mathrm{SW}}(P,H,c,\omega,\pi_S)
=\operatorname{accept}\\
&\qquad\Longrightarrow
\operatorname{Obs}(H,c,\omega)=\operatorname{Obs}(P)
\;\land\;
\operatorname{Safe}(H,c,\omega).
\end{aligned}
$$

Another checker connects the generated hardware to its architectural
contract:

$$
\begin{aligned}
&\operatorname{Check}_{\mathrm{HW}}(I,H,\pi_H)
=\operatorname{accept}\\
&\qquad\Longrightarrow
\operatorname{Impl}(I)\sqsubseteq H.
\end{aligned}
$$

The refinement relation must preserve every fact used by the software proof,
including timing, backpressure, faults, and synchronization. Under that
compatibility condition, the checks compose:

$$
\begin{aligned}
&\operatorname{Check}_{\mathrm{SW}}(P,H,c,\omega,\pi_S)
=\operatorname{accept}
\;\land\;
\\[-2pt]
&\operatorname{Check}_{\mathrm{HW}}(I,H,\pi_H)
=\operatorname{accept}
\;\land\;
\operatorname{Compatible}(I,H,c,\omega)\\
&\qquad\Longrightarrow
\operatorname{Obs}
\left(
\operatorname{Exec}(I,c,\omega)
\right)
=
\operatorname{Obs}(P)
\;\land\;
\operatorname{Safe}(I,c,\omega).
\end{aligned}
$$

If the second proof is not yet available, conformance between \(I\) and \(H\)
is an open obligation, not an implicit fact. Such a candidate may support an
architecture-level pilot, but it does not count as end-to-end accepted.
Requiring this proof is how synthesizing a new GPU microarchitecture becomes
part of the same checked hierarchy.

### 4.3 Small and complete for a declared scope

“Complete” should mean **closed for the stated guarantee**: every fact that
can change functional behavior or checked resource legality is derived from
the model or listed as an external assumption. It does not mean a complete
theory of all hardware.

“Minimal” should also be relative to a workload and a goal. Report model
complexity as the tuple:

$$
\mathbf C(K)
=
\left(
\#\text{primitives},
\#\text{rules},
\#\text{assumptions},
\text{trusted bytes}
\right).
$$

Let \(r_{\mathrm{rep}}(K)\) measure performance lost against a richer
independent design space. We seek models that are Pareto-minimal in
\(\mathbf C(K)\), subject to:

$$
\operatorname{Sound}(K),
\qquad
r_{\mathrm{rep}}(K)\le\varepsilon,
\qquad
\operatorname{VerifyCost}(K)\le B_{\mathrm v}.
$$

Thus the smallest model is not the one with the fewest words. It is the
simplest model that remains sound, verifiable, and rich enough to contain good
designs. The earlier four-primitive kernel—zero, copy, transfer, and tile
multiply–accumulate—is one candidate for this experiment, not yet a justified
answer.

## 5. The first experiment

We should begin with a structure experiment, not an arbitrary end-to-end GPU.

### 5.1 Test the separation

Use one bounded tiled matrix-multiplication family and one parameterized GPU
template. Construct a static baseline in which operation durations, mappings,
routes, and schedules are fixed by the candidate. Then vary three axes
independently:

1. **Manufacture-time choices:** vary array dimensions, pipeline depth,
   local-memory size and banking, port counts, interconnect topology, and
   supported tile operations.
2. **Compile-time choices:** add tile shapes, layouts, placements, buffering,
   routes, and static schedules.
3. **Runtime choices:** add variable latency, arbitration, bounded queues,
   dynamic batching, and data-dependent routing one feature at a time.

The first two axes enlarge the static joint design space. The third tests when
a performance mechanism turns design-time freedom into runtime state. Where
possible, the same allocation decision should be implemented once statically
and once dynamically. This directly measures the cost and benefit of moving
its commitment stage.

Two matched controls isolate the claimed separation:

- add unchosen architecture or compiler alternatives while holding the checked
  candidate \(z\) fixed; this should increase counting or search cost, not
  \(V_\varphi(z)\) or the cost of checking \(z\);
- hold the number of available designs fixed while adding runtime state and
  branching inside \(z\); this should change \(V_\varphi(z)\) and checking
  cost, not design richness.

For each point, measure:

- the number of canonical designs and canonical near-optimal designs;
- design counting and search time;
- projected reachable control states and runtime branching;
- checker or model-checker time, peak memory, and certificate size;
- best attainable cost relative to a richer independent model;
- RTL or simulator conformance to the selected hardware contract;
- measured performance, power, area, timing, and energy for generated hardware
  points; and
- whether two equal-shape inputs can produce different control traces.

For a positive objective \(J\), representation loss is:

$$
r_{\mathrm{rep}}(K)
=
\frac{J_K^*-J_U^*}{J_U^*},
$$

where \(J_K^*\) is the best design in the small model and \(J_U^*\) is the best
design in a richer independent model.

The experiment supports the hypothesis only if increasing design-time freedom
creates many meaningful and near-optimal designs without a corresponding
increase in per-design runtime branching or verification cost.

### 5.2 Test the cross-layer contract

For every selected design \((H,c,\omega)\), follow one receipt through:

$$
P
\longrightarrow
(H,c,\omega)
\longrightarrow
I.
$$

The receipt must show which facts were derived, selected, imposed, measured,
or assumed. The software certificate must replay against \(P\) and \(H\). The
hardware certificate must replay against \(H\), or the missing conformance
proof must remain visibly open. Physical cost must come from a named,
versioned backend.

This test asks whether the priors really compose. It fails if a lower layer
changes upper-level meaning, if a correctness-relevant constraint disappears
between layers, or if an empirical physical claim is presented as a formal
guarantee.

It weakens or falsifies the hypothesis if:

- good performance requires substantial runtime choice;
- verification cost grows with unchosen design alternatives;
- tensor payloads affect future control after they are projected away;
- the small model has large representation loss;
- competitive microarchitectures cannot be expressed or generated; or
- target-specific semantics must be hidden inside machine tables or unchecked
  assumptions.

Thresholds should be set after a pilot and frozen before the confirmatory
study. No numerical success claim is made here.

## 6. The role of Agentic AI

Agentic AI is useful only after the model defines a sound search space:

$$
\boxed{
\text{fallible high-capacity proposer}
\;+\;
\text{small deterministic verifier}
}
$$

The agent proposes \(H\), \(c\), \(\omega\), and possibly \(I\). It does not
define their meaning, change the rules, or approve its own result. The software
checker validates \(c\) and \(\omega\) against \(P\) and \(H\); the hardware
checker validates \(I\) against \(H\). Search quality and representation
quality must be measured separately. A poor agent may fail to find a good
design in a good space. A restrictive model may contain no good design,
regardless of the agent.

Formal legality and physical prediction must also remain separate:

$$
\text{proved correctness under }H
\neq
\text{accuracy of }H\text{ for a physical chip}.
$$

Latency, energy, area, and physical timing require calibration against
simulation, synthesis, FPGA measurements, or silicon.

## 7. Claim boundary and next step

This proposal does not claim that all tensor computing is static, that CPUs
cannot be verified, or that four primitives are sufficient. It targets bounded
compute-heavy regions whose control and communication may be fixed before
execution. Dynamic batching, sparse expert routing, KV-cache management,
collectives, faults, and variable-latency memory may need a different or
larger model.

The immediate next step is to build the smallest finite experiment that plots
four quantities separately:

$$
\begin{array}{c}
\text{joint software–hardware design richness},\\
\text{verification-relevant runtime state and branching},\\
\text{per-design verification cost},\\
\text{measured physical quality}.
\end{array}
$$

Only if those curves separate should we invest in a full DSL, proof stack,
general GPU generator, or agent benchmark. The current research state is
therefore deliberately incomplete: the central contribution is a falsifiable
question about the structure of tensor computation and its physical machines.
Per-design verification may still grow with the size of the selected hardware.
The claim is independence from unchosen alternatives, not constant checking
cost for arbitrarily large machines.

## References

[R1] Ada Lovelace, “Notes” in *Sketch of the Analytical Engine Invented
by Charles Babbage*, 1843.
[Text](https://psychclassics.yorku.ca/Lovelace/lovelace.htm)

[R2] Alan M. Turing, “On Computable Numbers, with an Application to the
Entscheidungsproblem,” 1936–1937.
[DOI](https://doi.org/10.1112/plms/s2-42.1.230)

[R3] Claude E. Shannon, “A Symbolic Analysis of Relay and Switching
Circuits,” 1938.
[PDF](https://tubes.mit.edu/6S917/_static/2025/resources/shannon38.pdf)

[R4] Arthur W. Burks, Herman H. Goldstine, and John von Neumann,
*Preliminary Discussion of the Logical Design of an Electronic Computing
Instrument*, 1946/1947.
[IAS archive](https://albert.ias.edu/entities/publication/49465635-0aa6-4299-921b-98c70c876e51)

[R5] John von Neumann, “The General and Logical Theory of Automata,”
Hixon Symposium lecture, 1948.
[PDF](https://www.vordenker.de/ggphilosophy/jvn_the-general-and-logical-theory-of-automata.pdf)

[R6] Gene M. Amdahl, Gerrit A. Blaauw, and Frederick P. Brooks Jr.,
“Architecture of the IBM System/360,” 1964.
[DOI](https://doi.org/10.1147/rd.82.0087)

[R7] Robert M. Tomasulo, “An Efficient Algorithm for Exploiting Multiple
Arithmetic Units,” 1967.
[DOI](https://doi.org/10.1147/rd.111.0025)

[R8] H. T. Kung, “Why Systolic Architectures?” 1982.
[DOI](https://doi.org/10.1109/MC.1982.1653825)

[R9] Norman P. Jouppi et al., “In-Datacenter Performance Analysis of a
Tensor Processing Unit,” 2017.
[Google Research](https://research.google/pubs/in-datacenter-performance-analysis-of-a-tensor-processing-unit/)

[R10] Jonathan Ragan-Kelley et al., “Halide: A Language and Compiler for
Optimizing Parallelism, Locality, and Recomputation,” 2013.
[DOI](https://doi.org/10.1145/2491956.2462176)

[R11] Siyuan Feng et al., “TensorIR: An Abstraction for Automatic
Tensorized Program Optimization,” 2023.
[DOI](https://doi.org/10.1145/3575693.3576933)

[R12] Amanda Liu et al., “Verified Tensor-Program Optimization via
High-Level Scheduling Rewrites,” 2022.
[DOI](https://doi.org/10.1145/3498717)

[R13] Rachit Nigam, Pedro Henrique Azevedo de Amorim, and Adrian
Sampson, “Modular Hardware Design with Timeline Types,” 2023.
[DOI](https://doi.org/10.1145/3591234)

[R14] Bo-Yuan Huang et al., “Application-Level Validation of Accelerator
Designs Using a Formal Software/Hardware Interface,” 2024.
[DOI](https://doi.org/10.1145/3639051)

[R15] Angshuman Parashar et al., “Timeloop: A Systematic Approach to DNN
Accelerator Evaluation,” 2019.
[NVIDIA Research](https://research.nvidia.com/publication/2019-03_timeloop-systematic-approach-dnn-accelerator-evaluation)

[R16] Edward A. Lee and David G. Messerschmitt, “Synchronous Data Flow,”
1987.
[Paper page](https://ptolemy.berkeley.edu/publications/papers/87/synchdataflow/)

[R17] Rachit Nigam et al., “Predictable Accelerator Design with
Time-Sensitive Affine Types,” 2020.
[arXiv](https://arxiv.org/abs/2004.04852)

<script type="text/javascript" async
  src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/MathJax.js?config=TeX-MML-AM_CHTML">
</script>
