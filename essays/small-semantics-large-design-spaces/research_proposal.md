# Small Semantics, Large Design Spaces

**A Verifiable Tile-Compute Model for Agentic AI Software–Hardware Co-Design**<br>
**Status:** Research proposal; no experimental results are claimed<br>
**Date:** 2026-07-26

## Abstract

Many costly regions of AI workloads are built from regular tensor tiles. Their
iteration domains, data dependencies, and data movement can often be made
explicit before execution. This creates an opportunity that is weaker for
irregular general-purpose code: performance choices may be moved from hidden
runtime speculation into an explicit design-time description.

This proposal tests one central hypothesis:

> **A fixed four-primitive tile-compute kernel can generate many verified
> software–hardware designs while retaining designs near the optimum of a
> richer independent space. An AI agent may search this space, but it cannot
> change its meaning or approve its own output.**

The kernel has four value-producing primitives: zero, copy, transfer, and tile
multiply–accumulate. Typed rules govern storage, dependence, placement,
routing, resources, and time. Tile sizes, layouts, schedules, mappings, and
machine parameters still create many possible designs. Every accepted design
is interpreted by the same fixed semantics and carries a replayable
certificate.

This changes the role of Agentic AI. The agent is not the source of
correctness. It is an untrusted search procedure over a large, structured
space. A deterministic checker verifies functional refinement, event order,
finite resources, storage, routes, and completion. The model's representation
regret and the agent's search regret are measured separately. Latency, energy,
and area remain empirical predictions and are validated separately from formal
correctness.

The first study, **Model Zero**, covers bounded tiled matrix multiplication,
exact signed-integer arithmetic, discrete time, static schedules, and two
abstract machine families: a cooperative tensor engine and a systolic array.
It tests six claims: four primitive forms remain frozen across the two
families; the extracted checker is sound; the kernel induces many distinct
legal designs; its best designs are near the independently computed reference
optimum; a fixed agent can find good checked designs under a fixed budget; and
one frozen cost model predicts held-out backend measurements within registered
limits.

The formal theorem ends at timed architectural actions, not RTL or silicon.
Model Zero is a test of the idea on exact tiled computation, not a model of all
LLM execution. Dynamic batching, KV-cache policies, sparse routing, sampling,
collectives, failures, and physical implementation are later extensions.

## 1. The problem

### 1.1 Tile-dominant computation creates a special opportunity

An AI workload is often shown as a graph of tensor operations. Its execution
depends on decisions across the full stack:

$$
\begin{aligned}
\text{tensor computation}
&\rightarrow \text{numerical choices}
\rightarrow \text{graph rewrites}
\rightarrow \text{tiling} \\
&\rightarrow \text{layout and placement}
\rightarrow \text{parallel schedule}
\rightarrow \text{communication} \\
&\rightarrow \text{machine resources}
\rightarrow \text{architectural actions}
\rightarrow \text{implementation}.
\end{aligned}
$$

These decisions interact. A tile size changes reuse, storage, traffic,
parallelism, and legal accumulation order. A data layout changes both locality
and bank conflicts. A hardware primitive can save time but consume more area.
A communication schedule can reduce traffic yet create a cyclic wait.
Optimizing each layer in isolation can therefore reject the best system design
or accept an inconsistent one.

Lowering does more than translate syntax. Each step chooses facts that were
previously open, such as a tile shape, precision, placement, route, or start
time. These choices narrow the design space. A sound lowering must state the
new choice, define its meaning, and show that it still implements the contract
above.

The physical prior is that many compute-heavy tensor regions have bounded,
repeated iteration spaces and regular producer–consumer relations. Their main
performance decisions can therefore be stated before execution. This does not
describe every part of an LLM system. It is the hypothesis to test for a
declared tile-dominant domain.

### 1.2 The distinction is explicit versus hidden performance control

General-purpose processors must perform well across programs with branches,
pointer-dependent memory access, interrupts, and other data-dependent
behavior. High-performance implementations often discover useful instruction
parallelism and locality at runtime through mechanisms such as out-of-order
issue, speculation, caches, prefetching, and arbitration. These mechanisms are
not wrong or unverifiable. They place important performance decisions inside a
large dynamic machine state.

Tile-oriented execution offers another contract. Dependencies, storage
lifetimes, communication, resource use, and start times can often be selected
by a compiler or design tool and exposed directly. Performance can then come
from a statically chosen composition rather than from an
implementation-internal runtime choice.

The boundary is not CPU versus accelerator. An ISA may expose tile operations;
Intel AMX is one concrete example [R55]. Accelerators may also contain dynamic
control. The relevant comparison is:

$$
\boxed{
\text{general-purpose ISA contract that abstracts over microarchitecture}
\quad\text{versus}\quad
\text{explicit timed tile-resource contract}
}
$$

The proposal asks whether the second contract is sufficient for a useful AI
workload domain.

### 1.3 Few primitives can induce a large design space

Let \(K\) be a fixed semantic kernel. Its primitive and proof rules should be
inspectable and mechanized. Few primitive forms need not imply few designs:

$$
\left|\operatorname{Prim}(K_0)\right|=4,
\qquad
\left|\mathcal Z_K(t)\right|
\text{ can be large}.
$$

For one task \(t\), the space may contain the product of choices for tiling,
fusion, layout, placement, buffering, pipelining, routing, and machine
parameters:

$$
\mathcal Z_K(t)
\subseteq
\mathcal T_t
\times\mathcal F_t
\times\mathcal L_t
\times\mathcal P_t
\times\mathcal B_t
\times\mathcal S_t
\times\mathcal R_t
\times\mathcal H_t.
$$

The scientific opportunity is not to make optimization easy. It is to make
every design choice explicit and give every accepted combination one fixed
meaning. A four-primitive grammar can still generate a combinatorial space.

### 1.4 The agent searches; the model and checker constrain it

Language-model agents are good at proposing and revising designs across
different notations and tools. Their output is still produced by statistical
inference. A confident answer is not evidence that all reductions are complete,
all values are available before use, or all shared resources are legal.

Compiler errors, simulations, and tests are useful, but they answer limited
questions. A finite test checks only the tested inputs and executions.
Simulation checks the behavior encoded by the simulator, which may omit a
constraint or share the same mistake as the design. Formal acceptance requires
a defined specification and a checker whose acceptance rule can itself be
proved sound.

The proposed architecture is:

$$
\boxed{
\text{untrusted high-capacity proposer}
+
\text{fixed four-primitive semantic kernel}
+
\text{deterministic checker}
}
$$

The agent searches the space induced by the kernel. The checker decides whether
a candidate belongs to the verified subset. The agent may choose parameters
and compositions, but it may not invent a new primitive, change an operation's
meaning, or waive a failed obligation.

### 1.5 Correctness and physical prediction are different claims

For a declared abstract machine, a checker can prove facts such as:

- every output element receives exactly the required reduction terms;
- an action reads only available data;
- overlapping actions do not exceed a resource capacity;
- a buffer is not overwritten while its value is live; and
- every action in a finite static schedule issues and completes.

These are deductive claims: if the model assumptions hold, the conclusion
follows.

Physical latency, energy, and area are different. They depend on details such
as synthesis, placement, routing, clocking, voltage, memory behavior, and
measurement noise. Their validity must be tested on held-out artifacts. A
formal checker may use an admitted upper timing bound as an assumption, but it
must report that assumption rather than present it as a physical theorem.

This separation is central:

$$
\text{formal correctness under a model}
\neq
\text{empirical accuracy of the model}.
$$

An error found after architecture freeze, RTL integration, physical design, or
tape-out has a long feedback loop. Formal checking does not replace simulation,
timing analysis, physical signoff, or silicon measurement. It rejects defined
error classes before those stages.

## 2. Scientific question and claims

### 2.1 Central scientific question

> **Can a fixed four-primitive, machine-checkable kernel generate many legal
> designs for bounded tile programs while retaining near-optimal designs from
> a richer independent space? Can an untrusted AI agent search that space
> effectively?**

“Small” is shorthand for the frozen four-primitive value kernel, not a claim
that the full theory or trusted code is absolutely minimal. “Complete” is also
scoped: every fact that affects the stated correctness and exact-cost claims
must be derived from the model or listed as an external assumption. It does not
mean that the kernel represents every legal machine.

This question has six parts:

1. **Kernel economy:** Can one four-primitive semantic kernel remain unchanged
   for a held-out machine family, with every rule, assumption, and trusted byte
   reported?
2. **Soundness:** Does checker acceptance imply functional, temporal, storage,
   route, resource, and bounded-completion validity?
3. **Richness:** Does the kernel induce many distinct legal designs, not
   merely many equivalent syntax trees?
4. **Optimization sufficiency:** Is the best induced design close to the best
   design in a richer independent reference space?
5. **Search:** Can a fixed AI agent find a checked design close to the kernel's
   own optimum under a fixed budget?
6. **Physical prediction:** Do costs derived from checked traces predict
   held-out backend measurements within fixed limits?

Each part requires different evidence. A size count cannot prove soundness. A
soundness theorem cannot show that the space contains good designs. A strong
agent cannot repair a weak representation. A formal trace cannot prove a
physical latency estimate.

Let \(K\) be the kernel, \(t\) a task, and \(B\) a finite expression and
schedule bound. The verified semantic design space is:

$$
\mathcal M_K(t,B)
=
\left\{
\operatorname{Can}(e)
\;\middle|\;
e\in\operatorname{Expr}(K,t,B),
\;
\exists\pi:
\operatorname{Check}_K(e,\pi)=\operatorname{accept}
\right\}.
$$

\(\operatorname{Can}\) maps equivalent encodings to one semantic design record.
It preserves every distinction used for correctness or exact cost, including
placement, route, schedule, and resource counts.

An independent reference language \(U\) defines a richer space:

$$
\mathcal R(t)
=
\left\{
\operatorname{Can}(u)
\;\middle|\;
u\in U(t),
\;
\operatorname{Valid}_U(u)
\right\}.
$$

The reference implementation must not call \(K\), its elaborator, or its
checker. The first relation to establish is:

$$
\mathcal M_K(t,B)\subseteq\mathcal R(t).
$$

The proposal does not require equality. A useful restricted kernel may exclude many
legal but unnecessary designs.

Every task scored by \(\mathrm{H}_4\) or \(\mathrm{H}_5\) must satisfy:

$$
\mathcal M_K(t,B)\neq\varnothing,
\qquad
\mathcal R(t)\neq\varnothing.
$$

An empty space fails the corresponding claim; its optimum is not silently
defined.

For a registered positive cost \(J_t\), define:

$$
J_U^*(t)
=
\min_{c\in\mathcal R(t)}J_t(c),
\qquad
J_K^*(t,B)
=
\min_{c\in\mathcal M_K(t,B)}J_t(c).
$$

The kernel's normalized representation regret is:

$$
r_K(t,B)
=
\frac{J_K^*(t,B)-J_U^*(t)}
{J_U^*(t)}.
$$

The kernel is optimization-sufficient at tolerance \(\varepsilon\) when
\(r_K(t,B)\le\varepsilon\) for every held-out task.

For an agent-produced checked design \(\widehat c_t\), the total gap separates
exactly:

$$
\underbrace{
J_t(\widehat c_t)-J_U^*(t)
}_{\text{total gap}}
=
\underbrace{
J_t(\widehat c_t)-J_K^*(t,B)
}_{\text{agent search regret}}
+
\underbrace{
J_K^*(t,B)-J_U^*(t)
}_{\text{representation regret}}.
$$

This decomposition prevents two wrong conclusions. Poor agent search must not
be blamed on the model, and a restrictive model must not be excused by a
capable agent.

For an expression \(g\), write
\(J_t(g):=J_t(\operatorname{Can}(g))\).
The constrained search problem remains:

$$
\begin{aligned}
\min_{z\in\operatorname{Expr}(K,t,B)}\quad & J_t(z) \\
\text{subject to}\quad
& \exists\pi:
\operatorname{Check}_K(z,\pi)=\operatorname{accept}.
\end{aligned}
$$

For one objective, \(J_t\) is fixed before search. For several costs, the study
will either use a declared scalar objective

$$
J_t(z)=w^{\mathsf T}K_{\mathrm{cost}}(z),\qquad w\ge 0,
$$

or report the Pareto set. The choice of \(w\), the hard constraints, and all
cost-model versions are part of the task specification. A Pareto design is one
for which no other legal design is at least as good on every declared cost and
strictly better on one.

### 2.2 Precise terms

**Frozen benchmark.** A manifest \(\mathcal B_0\) fixes the input domain,
machine schemas, parameter ranges, time and size bounds, observation relation,
and two separate languages. \(U_0\) is an independent finite reference
language used to define the design universe. \(G_0\) is the proposed contract
grammar implementing kernel \(K_0\). The induced benchmark domain is
\(D_0=D(\mathcal B_0)\). Every kernel-economy, richness, regret, and soundness
claim applies only to this frozen benchmark.

**Small kernel.** Kernel size is the registered tuple:

$$
C(K)
=
\left(
\left|\operatorname{Prim}(K)\right|,
\left|\operatorname{Rule}(K)\right|,
\left|\operatorname{Assumption}(K)\right|,
\operatorname{TCBBytes}(K)
\right).
$$

The primitive, proof-rule, external-assumption, and trusted-code counting
methods are fixed before the held-out machine is encoded. The semantic-kernel
size and deployed trusted-code size are reported separately.
Model Zero fixes four primitive forms. It does not claim that \(C(K_0)\) is
absolutely minimal or infer total simplicity from the primitive count. The
machine-checked result in \(\mathrm{H}_2\) tests whether the resulting rules
and checker are in fact mechanizable.

Machine-description complexity is reported separately:

$$
C_H(H)
=
\left(
|R_H|,
|\operatorname{Routes}_H|,
|\operatorname{TableRows}_H|,
|\operatorname{DemandForms}_H|
\right).
$$

A manifest may supply capacities, routes, durations, resource-demand tables,
and exact counts for existing primitives. It may not add value semantics,
predicates, or executable code. This prevents target semantics from being
hidden in an unreported “data” interface.

**Semantic richness.** Raw syntax does not count. The number of distinct legal
designs must also ignore pure schedule slack. For richness only,
\(\operatorname{RichCan}(c)\) keeps the task, machine, action/value graphs,
operations, shapes, storage, placement, routes, and pairwise action
order-or-overlap relation. It removes identifier aliases, exact start-cycle
offsets, and cost values. Thus uniform delays and inserted idle time do not
create new classes.

Define the full and near-optimal richness:

$$
\begin{aligned}
S_K^{\mathrm{all}}(t,B)
&=
\log_2
\left|
\left\{
\operatorname{RichCan}(c)
\mid
c\in\mathcal M_K(t,B)
\right\}
\right|,\\
S_K^{(\eta)}(t,B)
&=
\log_2
\left|
\left\{
\operatorname{RichCan}(c)
\;\middle|\;
c\in\mathcal M_K(t,B),
\;
J_t(c)\le(1+\eta)J_K^*(t,B)
\right\}
\right|.
\end{aligned}
$$

The confirmatory study uses \(\eta=0.10\). A large full space with only one
good design does not pass.

For reference, the alias-free accepted-space size remains:

$$
S_K^{\mathrm{accepted}}(t,B)
=
\log_2
\left|
\mathcal M_K(t,B)
\right|.
$$

**Optimization sufficiency.** The kernel need not represent every reference
design. It is sufficient only relative to a task distribution, objective,
bound, and regret threshold. This is weaker and more useful than universal
coverage.

**External behavior.** Model Zero observes termination status and output tensor
values. Its evaluator is total: every run returns either
\(\operatorname{Normal}(o)\) or a named \(\operatorname{Fault}(f)\). A valid
candidate returns \(\operatorname{Normal}(o)\) for every registered input. A
task may also declare completion time as an external contract. Internal events,
occupancy, and traffic remain proof, cost, or diagnostic data; they are not
automatically part of functional equivalence.

**Soundness.** If the checker accepts a candidate, that candidate satisfies the
formal validity judgment:

$$
\operatorname{Check}_{\mathcal B_0,H}(z,\pi)=\operatorname{accept}
\Rightarrow
\operatorname{Valid}_{\mathcal B_0,H}(z).
$$

This implication must be proved from the semantics. Passing tests or detecting
mutants does not prove it.

**Decision completeness.** The reverse direction—every semantically valid
candidate has a certificate the checker accepts—is a separate, optional
theorem. A conservative checker may reject legal designs; those false
rejections must be measured. Because \(\mathcal M_K\) counts accepted designs,
its representation regret includes both grammar restriction and conservative
checker rejection unless decision completeness is proved. The two losses are
reported separately when possible.

**Verification closure.** Every behavior that can affect the scoped correctness
judgment must be either derived from a declared primitive or named as an
external assumption. This is not a complete theory of hardware.

**Agent search regret.** Agent quality is measured against \(J_K^*(t,B)\), the
best design available inside the fixed kernel. It is not measured only against
another agent or prompt.

**Refinement and composition.** Vertical refinement connects two representation
levels. Horizontal composition connects components at one level. They require
different rules.

### 2.3 Falsifiable claims

Section 5.1 states the six claims, their primary evidence, exact pass rules,
and the narrow conclusion each result would allow.

## 3. What prior work teaches us

The survey is organized by the problem each abstraction was built to solve.
The point is not that early researchers had one complete hardware theory. They
did not. The point is that they developed a method: define a small object,
state what it means, show how objects compose, and keep the connection to a
lower physical level explicit.

### 3.1 Historical foundations

| Work | Problem | Response | Lesson for this proposal |
|---|---|---|---|
| Babbage and Lovelace [R1] | A calculating machine was tied to particular calculations. | Separate operations, their operands, and the mechanism that carries them out. | A system description should separate meaning from one implementation. |
| Turing [R2] | “Mechanical procedure” lacked a precise mathematical definition. | Define a small machine and a universal interpreter. | A simple model can make a broad question exact, but computability alone says nothing about physical efficiency. |
| Shannon [R3] | Relay circuits were hard to analyze and simplify systematically. | Map switching circuits to Boolean expressions and synthesize circuits from those expressions. | A useful theory connects meaning, construction, and a physical cost. |
| McCulloch and Pitts [R4] | Isolated truth functions did not explain time-dependent network behavior. | Model networks of simple units with delayed activity. | Equal outputs do not imply equal schedules or implementations. |
| Burks, Goldstine, and von Neumann [R5] | Early computers faced severe equipment, storage, and reliability limits. | Treat hardwired operations and synthesized sequences as explicit time–equipment choices. | The choice of physical primitives is part of optimization. |
| Von Neumann [R6] | Complex automata could not be understood as one undivided object. | Study elementary units, compose them through explicit contracts, and justify each contract at a lower level. | A white box can be hierarchical: a component is abstract locally but still carries a refinement duty. |
| Wilkes and Stringer [R7] | Irregular control logic was hard to design and change. | Implement control through a stored microprogram. | The software–hardware boundary can move through another interpretation layer. |

This history gives a constructive pattern:

$$
\boxed{
\text{primitives}
+
\text{composition rules}
+
\text{interpretation rules}
+
\text{physical assumptions}
}
$$

For this project, a model is white-box at a chosen level when every accepted
behavior and legality judgment follows from declared primitives, and each
primitive either has a checked refinement below or is named as an external
assumption.

### 3.2 Two ways to obtain performance

The System/360 work separated programmer-visible architecture from machine
organization [R53]. The RISC-V specification likewise avoids requiring an
in-order, out-of-order, or microcoded implementation [R54]. This separation
lets one instruction contract support many microarchitectures.

The System/360 Model 91 used dynamic register tagging and a common data bus to
find independent operations at runtime without requiring specially scheduled
code [R49]. This solved an important general-purpose problem: useful parallelism
could be recovered even when the instruction stream did not state it directly.
Later processors added more speculation, prediction, caching, and dynamic
scheduling. These mechanisms preserve a stable instruction-level contract
while hiding many performance choices below it.

Tile-oriented machines explore a different point. Systolic architectures map
regular dataflow onto arrays with local, rhythmic communication [R50]. The
first production TPU paired a large matrix unit with software-managed memory
and reported a deterministic execution model, contrasting it with
time-varying CPU and GPU mechanisms [R51]. Hennessy and Patterson later framed
domain-specific architectures as a major path for architecture improvement
[R52].

These works do not prove this proposal's hypothesis. They identify its physical
prior: regular workload structure may let a contract expose choices that a
conventional ISA often leaves to lower software or implementation layers. The
relevant object is the execution contract, not the device label. A CPU with
explicit tile operations could implement the proposed contract, while an
accelerator with hidden arbitration might not.

### 3.3 Tensor programs and verified lowering

Halide separates an algorithm from its performance schedule [R10]. TVM,
TensorIR, and Ansor expose and search tensor schedules for heterogeneous
targets [R11, R26, R12]. StableHLO defines a portable tensor-operation contract
[R14]. Exo and Exo 2 build safe user-defined schedules from checked
transformations [R32, R31]. These systems show that schedule choices should be
explicit rather than hidden in generated code.

Verified tensor scheduling uses machine-checked, semantics-preserving rewrites
[R40]. The later ATL compiler proves lowering from a functional tensor
language to imperative loop nests [R41]. CompCert and Alive2 are broader
precedents for verified compilation and translation validation [R28, R29].
The gap targeted here is how such correctness evidence connects to cycle-level
resource binding for a selected machine configuration.

### 3.4 Time, resources, and safe hardware composition

Synchronous dataflow makes rates explicit so schedules and buffer bounds can be
analyzed [R8]. Timed automata add quantitative time to transition systems [R9].
Dahlia uses a resource-aware type system to reject unpredictable HLS programs
[R33]. Kôika and Kami show that small formal hardware languages can support
deterministic semantics and deductive refinement [R34, R35].

Filament encodes and enforces cycle-level timing and structural interface
constraints [R42]. The 2024 Parafil v1 preprint lifts Filament's timing-safety
checks to all instances of parameterized generators [R43]. Piezo makes its
static-control fragment a semantic refinement of its dynamic fragment in Calyx
[R36, R44]. Their reported guarantees begin at hardware-control or generator
interfaces, not at a proved tensor and numerical specification.

### 3.5 Formal software–hardware interfaces

The Instruction-Level Abstraction, or ILA, is a hierarchical, software-visible
accelerator contract that supports ILA-to-ILA and ILA-to-FSM equivalence
[R38]. 3LA uses it as an ISA-like software–hardware interface, combines it with
flexible compiler matching, and generates instruction-level simulators for
end-to-end application validation [R39].

MLIR provides infrastructure for multiple intermediate representations [R13],
but infrastructure alone does not prove that a cross-dialect lowering is
correct. mlir-tv applies SMT-based translation validation to selected MLIR
tensor and lowering transformations [R27]. The First-Class Verification
Dialects framework provides semantics-supporting dialects and
dialect-independent verification tools [R46]. The 2024 K-CIRCT preprint gives
executable layered semantics for a substantial part of CIRCT hardware IR
[R45].

These systems make a cross-layer certificate chain more plausible. They also
narrow the novelty claim: this proposal must contribute a precise connection
among existing kinds of contracts, not merely another formal IR.

### 3.6 Architecture search and physical cost

Timeloop and MAESTRO represent mappings and estimate reuse, traffic,
performance, and hardware cost [R16, R17]. Aladdin evaluates accelerators before
RTL using calibrated models [R15]. Accelergy composes action counts with
component energy estimates [R37].

Their key lesson is that exact structural counts and empirical parameters
should be separate. A legal trace can determine operation count, bytes moved,
live storage, and abstract occupancy. It cannot prove that a physical
implementation has a given clock rate, energy, or area.

### 3.7 AI agents for software and hardware design

SWE-agent shows that a purpose-built interface can change a fixed agent's
success rate [R18]. ChipNeMo, VeriGen, VerilogEval, RTLLM, AutoChip,
GPT4AIGChip, and Chip-Chat show useful forms of domain adaptation, RTL
generation, tool feedback, accelerator generation, and conversational hardware
design [R19–R25].

Two 2026 preprints move closer to formal and end-to-end evaluation. FormalRTL
reports a multi-agent RTL pipeline with equivalence checking against executable
software reference models [R47]. HSCO-Bench benchmarks agents on kernel
identification, accelerator-rich system generation, and software mapping
[R48]. Neither reports a replayable refinement certificate from tensor
semantics through timed resource constraints.

### 3.8 The remaining gap

Verified tensor systems can prove scheduling and lowering steps. Timing-aware
hardware languages can safely compose resource-constrained modules and
parameterized generators. ILA and 3LA connect applications to formally
specified accelerators. Formal MLIR and CIRCT work gives semantics to important
compiler and hardware representations. Agent systems can generate RTL, use
formal checks, and attempt end-to-end co-design.

Across the systems surveyed here, the proposal tests one combined claim not
reported end to end:

> Can one frozen four-primitive tile kernel induce a large space of distinct checked
> software–hardware designs, retain near-optimal designs from a richer
> independent space, and give an AI agent enough structure to search that
> space effectively?

The replayable tensor-to-action certificate is the mechanism that makes every
accepted point verifiable. Design-space richness and near-optimality are
separate empirical claims. This proposal does not claim a new scheduling
language, hardware language, cost model, or agent architecture.

## 4. Proposed model

### 4.1 Model Zero is finite by construction

Model Zero is described by a frozen manifest:

$$
\mathcal B_0=
\left(
\mathcal S_0,
V_0,
T_{\max},
N_{\max},
\mathcal H_{\mathrm{coop}},
\mathcal H_{\mathrm{systolic}},
U_0,
K_0,
G_0,
\mathcal E_0
\right).
$$

\(\mathcal E_0\) freezes the canonical schema, objective, size-accounting rule,
richness threshold, regret tolerances, and inclusion test.

The matrix shapes are:

$$
\mathcal S_0=
\left\{
(M,N,K)\mid M,N,K\in\{4,8,16\}
\right\},
$$

and the input values are:

$$
V_0=\{-2,-1,0,1,2\}.
$$

For a program with shape \((M,N,K)\), the registered input domain is:

$$
\mathcal I_{M,N,K}
=
V_0^{M\times K}
\times
V_0^{K\times N}.
$$

Write \(\mathcal I(P)=\mathcal I_{M,N,K}\) when \(P\) has that shape.

The core experiment uses exact signed-integer arithmetic. A 16-bit accumulator
cannot overflow in this domain because:

$$
\left|
\sum_{k=0}^{K-1}A_{mk}B_{kn}
\right|
\le 16\cdot 2\cdot 2=64.
$$

\(T_{\max}\) bounds schedule time and \(N_{\max}\) bounds the action graph.
Every identifier, operation, shape, placement, address, start time, and route
comes from a finite manifest set. Routes are simple paths with a fixed maximum
length. Certificates also have a fixed size bound. Thus a finite grammar cannot
silently generate an unbounded candidate set.

\(U_0\) defines candidates directly as finite tables of matrix-index products,
dependencies, value locations, routes, start times, and per-cycle machine use.
It is split into:

$$
U_0=
U_0^{\mathrm{legal}}
\uplus
U_0^{\mathrm{fault}}.
$$

Only \(U_0^{\mathrm{legal}}\) is used to define the reference design space and
its optimum.
\(U_0^{\mathrm{fault}}\) is used for rejection and diagnostic tests. The
reference language does not call the proposed kernel, grammar, elaborator, or
checker. \(K_0\) is the fixed semantic kernel, and \(G_0\) is its typed surface
grammar. Two independent implementations define \(U_0\). For enumerated tasks,
their counts and canonical record hashes must agree. For larger exact searches,
one implementation's candidate and optimality certificate must be checked by
the other.

A legal reference record \(u\in U_0^{\mathrm{legal}}\) and model expression
\(g\in G_0\) are functionally equivalent when they produce equal outputs for
every registered input:

$$
u\equiv_{\mathrm{fun}} g
\quad\Longleftrightarrow\quad
\operatorname{Valid}_{U_0}(u)
\land
\operatorname{Valid}_{G_0}(g)
\land
\operatorname{shape}(u)=\operatorname{shape}(g)
\land
\left(
\forall x\in\operatorname{Inputs}(u;V_0),\;
\operatorname{Run}_{U_0}(u,x)
=
\operatorname{Run}_{G_0}(g,x)
\right).
$$

Functional equivalence is not implementation identity. Richness and
optimization must preserve the choices that distinguish two implementations.
Both languages therefore normalize into one neutral record:

$$
\operatorname{Can}(c)
=
\left(
\operatorname{task},
\operatorname{machine},
\operatorname{valueGraph},
\operatorname{actionGraph},
\operatorname{opShape},
\lambda_A,
\lambda_V,
\rho,
s,
E,
K_{\mathrm{exact}},
\operatorname{outputNF}
\right).
$$

\(\lambda_A\) records action placement, \(\lambda_V\) records memory locations
and ranges, \(\rho\) records routes, \(s\) records exact start times, and \(E\)
records elaborated durations and resource demands. These fields derive tiling,
layout, buffering, dataflow, and timing rather than naming them informally.
\(\operatorname{outputNF}\) is the exact matrix-index product normal form.
Identifier names and explicitly registered graph or machine symmetries are
normalized away. No other difference is discarded.

The canonicalizers must satisfy:

$$
\operatorname{Can}(c_1)=\operatorname{Can}(c_2)
\quad\Longleftrightarrow\quad
c_1\equiv_{\mathrm{design}}c_2,
$$

where \(\equiv_{\mathrm{design}}\) means equality of every field above modulo
only the registered aliases. They must also preserve validity and every
objective computed from the exact structural vector:

$$
\operatorname{Valid}(c)
\Rightarrow
\operatorname{Valid}_{\mathrm{Can}}
\left(
\operatorname{Can}(c)
\right),
\qquad
J_t(c)
=
J_t
\left(
\operatorname{Can}(c)
\right).
$$

These properties are proved for the \(U_0\) and \(G_0\) normalizers. Each
kernel-induced design carries a matching reference record and certificate
\(\pi_{\mathrm{eq}}\). The equivalence checker recomputes both neutral
records:

$$
\operatorname{EqCheck}(u,g,\pi_{\mathrm{eq}})
=
\operatorname{accept}
\quad\Longleftrightarrow\quad
\operatorname{Valid}_{U_0}(u)
\land
\operatorname{Valid}_{G_0}(g)
\land
\operatorname{Can}_{U_0}(u)
=
\operatorname{Can}_{G_0}(g).
$$

Random or exhaustive input testing is used to find mistakes, but it is not
accepted as the equivalence certificate.

Thus the required relation is inclusion:

$$
\mathcal M_{K_0}(t,B)
\subseteq
\mathcal R_0(t).
$$

Equality is neither expected nor required. Every held-out task family must
contain a task with a strictly richer reference space. The experiment then
asks how many useful designs the smaller kernel induces and how much optimum
quality, if any, it loses.

The two machine sets are finite lists of legal configurations. Every machine
record gives named resources, typed capacities, locations, routes, operation
support, exact integer durations, and finite per-cycle resource demands. All
values in the manifest are frozen and hashed before evaluation.

The two target families are:

- a cooperative tensor engine with local memory, banks, ports, transfer
  engines, tile-compute units, and explicit events; and
- a systolic array with processing elements, local registers, neighbor links,
  injection resources, and drain resources.

One target is used while developing the core. The core is then frozen before
the second target is encoded. This tests whether the checker is genuinely
target-independent. The second target is a one-shot holdout. If it exposes a
missing primitive or checker rule, the confirmatory test fails. Any revised
core must be tested on a new untouched target family; the failed holdout cannot
be reused as final evidence.

Model Zero excludes floating-point approximation, dynamic arbitration,
unbounded queues, dynamic allocation, failures, clock-domain crossing, RTL
equivalence, physical timing closure, and manufacturing effects. Each is a
possible later extension, not an implicit part of the first claim.

### 4.2 Representation levels

Model Zero has four levels:

$$
\begin{aligned}
L_0 &: \text{mathematical matrix-multiplication contract},\\
L_1 &: \text{tiled, single-assignment value graph},\\
L_2 &: \text{placement, routes, and static start times},\\
L_3 &: \text{timed architectural actions expanded from the machine contract}.
\end{aligned}
$$

RTL is outside the first refinement theorem. A backend may emit RTL, but that
artifact carries an open proof obligation until a separate
\(L_3\rightarrow\mathrm{RTL}\) validation step exists.

Each lowering returns a lower representation and a certificate:

$$
P_i
\xrightarrow{\;\ell_i\;}
\left(
P_{i+1},\pi_i,\mathcal A_i
\right).
$$

\(\pi_i\) is evidence that the new representation refines the old one.
\(\mathcal A_i\) records external conformance assumptions, such as the claim
that a physical target implements the abstract machine table. Every deductive
proof obligation must close before acceptance; an unresolved proof obligation
cannot be relabeled as an assumption.

### 4.3 Finite action semantics

For a selected machine \(H\), the proof model is:

$$
M_0(\mathcal B_0,H)=
\left(
Q,A,R_H,\longrightarrow_H,\operatorname{Run},
\operatorname{Valid}_{\mathcal B_0,H}
\right),
$$

where \(Q\) is the set of finite configurations, \(A\) is a finite action
grammar, \(R_H\) is the machine's set of named resources,
\(\longrightarrow_H\) defines issue and completion, and
\(\operatorname{Run}\) returns either a normal external observation or a named
fault. It is defined for every finite candidate and input.

A complete candidate is:

$$
z=
\left(
P,H,G,\lambda_A,\lambda_V,\rho,s
\right).
$$

\(P\) is the source tensor contract and
\(H\in\mathcal H_{\mathrm{coop}}\cup\mathcal H_{\mathrm{systolic}}\).
\(G\) is a typed action graph with at most \(N_{\max}\) actions.
\(\lambda_A\) maps actions to legal machine locations. \(\lambda_V\) maps each
logical value to a memory resource and a half-open address range:

$$
\lambda_V(v)=
\left(
m_v,[lo_v,hi_v)
\right),
\qquad
hi_v-lo_v=\operatorname{size}(v).
$$

\(\rho\) maps transfers to bounded routes, and
\(s:A_G\rightarrow\{0,\ldots,T_{\max}-1\}\) gives start times. Non-transfer
actions use the empty route.

Logical values are single-assignment. Physical reuse is modeled by mapping
different logical values to the same storage at non-overlapping live times. An
action contains:

$$
a=
\left(
\operatorname{id},
\operatorname{op},
\operatorname{inputs},
\operatorname{output},
\operatorname{pred},
\operatorname{shape}
\right).
$$

The initial primitive set is:

$$
\operatorname{Prim}(K_0)
=
\left\{
\operatorname{zero},
\operatorname{copy},
\operatorname{transfer},
\operatorname{tile\_mac}
\right\}.
$$

Changing this set creates a new kernel version and requires a fresh holdout.
The candidate chooses legal operations, placement, routes, and start times. It
cannot choose its own duration or resource demand. All operations use signed
16-bit values in Model Zero. Action evaluation is total. It returns
\(\operatorname{Normal}(v)\) when every result fits in signed 16-bit form and
\(\operatorname{Fault}(\mathrm{overflow})\) otherwise. Unsupported types and
shapes return named faults. For explicit finite index sets \(I,J,K\), the
normal value rules are:

$$
\begin{aligned}
\llbracket\operatorname{zero}_{I,J}\rrbracket()_{ij}
&=0,\\
\llbracket\operatorname{copy}_{I,J}\rrbracket(X)_{ij}
&=X_{ij},\\
\llbracket\operatorname{transfer}_{I,J,\ell_1,\ell_2}\rrbracket(X)_{ij}
&=X_{ij},\\
\llbracket\operatorname{mac}_{I,J,K}\rrbracket(A,B,C)_{ij}
&=C_{ij}+\sum_{k\in K}A_{ik}B_{kj}.
\end{aligned}
$$

Copy preserves a value within one location. Transfer preserves the value across
two locations and must occupy every resource on its declared route.

The machine contract expands an action through one fixed table lookup and one
fixed route-fold rule. Expansion is a total option-valued function:

$$
\operatorname{Expand}
\left(
H,a,\lambda_A(a),\rho(a)
\right)
\in
\operatorname{Option}
\left(
\operatorname{ElaboratedAction}
\right),
$$

with:

$$
\operatorname{Expand}
\left(
H,a,\lambda_A(a),\rho(a)
\right)
=
H.\operatorname{table}
\left[
\operatorname{op}(a),
\operatorname{shape}(a),
\lambda_A(a),
\operatorname{routeClass}(\rho(a))
\right]
=
\operatorname{Some}
\left(
d_a,u_a,b_a
\right)
$$

where \(d_a\) is the exact abstract duration, \(u_a(r,\delta)\) is a finite
resource-demand table, and \(b_a\) contains exact structural counts such as
bytes transferred. Every action must satisfy
\(1\le d_a\le T_{\max}\). A new target may add resource records and table
rows, but it may not add executable expansion code. Route classes are finite
manifest labels; the fixed route fold adds the declared demand of each
traversed link. A missing row or illegal route returns
\(\operatorname{None}\).

The whole candidate elaborates only when every action succeeds:

$$
\operatorname{Elab}_H(z)
=
\operatorname{Some}(E)
\quad\Longleftrightarrow\quad
\forall a\in A_G,\;
\exists d_a,u_a,b_a:
\operatorname{Expand}
\left(H,a,\lambda_A(a),\rho(a)\right)
=
\operatorname{Some}(d_a,u_a,b_a)
\land
E(a)=(d_a,u_a,b_a).
$$

All later uses of \(d_a\), \(u_a\), and \(b_a\) are bound by such an
elaboration \(E\). If elaboration fails, the total evaluator returns a named
fault and the checker rejects the candidate.

A configuration is:

$$
q=
\left(
t,\operatorname{Val},\operatorname{Started},
\operatorname{Completed},\operatorname{Events}
\right).
$$

An action issues at \(s_a\) only when its inputs and predecessor events are
available and the static resource checks pass. Logical values are immutable.
Their physical storage remains live until every consuming action completes.
The action's output and completion event become visible at:

$$
e_a=s_a+d_a.
$$

Each cycle has two ordered phases. First, actions with \(e_a=t\) apply their
registered value rule and publish outputs and completion events. Second,
actions with \(s_a=t\) issue. Thus a consumer may start in the cycle when its
producer completes, but it cannot see a partial write.

An input is available at cycle zero. Every other value is available when its
unique producer completes:

$$
\operatorname{avail}(v)=
\begin{cases}
0, & v\text{ is a program input},\\
e_{\operatorname{prod}(v)}, & \text{otherwise}.
\end{cases}
$$

Every input must be available when the action starts:

$$
\forall v\in\operatorname{inputs}(a):
\quad
\operatorname{avail}(v)\le s_a.
$$

Every declared predecessor must also be complete:

$$
\forall b\in\operatorname{pred}(a):
\quad
e_b\le s_a.
$$

Resource use is legal exactly when:

$$
\forall r\in R_H,\;
\forall t\in\{0,\ldots,T_{\max}-1\}:
\quad
\sum_{\substack{a\\s_a\le t<e_a}}
u_a(r,t-s_a)
\le \operatorname{CapToken}_H(r).
$$

Compute units, ports, banks, links, pipeline issue slots, and transfer engines
are distinct named resources. \(\operatorname{CapToken}_H(r)\) gives capacity
in the resource's declared unit, such as issue slots or port transactions.
Memory bytes use the separate capacity
\(\operatorname{CapBytes}_H(m)\). An exclusive resource has capacity one.
Routes are legal only when every hop is present in the machine graph, and
their expanded demand includes every occupied link.

Let \(T_{\mathrm{obs}}=T_{\max}+1\) be the final observation point. The last
required time for a value is:

$$
\operatorname{last}(v)
=
\max
\left(
\{e_a\mid v\in\operatorname{inputs}(a)\}
\cup
\{T_{\mathrm{obs}}\mid v\text{ is a final output}\}
\cup
\{\operatorname{avail}(v)\}
\right).
$$

Its live interval is
\(I_v=[\operatorname{avail}(v),\operatorname{last}(v))\). If
\(\lambda_V(v)=(m_v,[lo_v,hi_v))\), then:

$$
\forall m,\;
\forall t\in\{0,\ldots,T_{\max}\}:
\quad
\sum_{\substack{v:m_v=m\\t\in I_v}}
\operatorname{size}(v)
\le \operatorname{CapBytes}_H(m).
$$

For every value,
\(0\le lo_v<hi_v\le\operatorname{CapBytes}_H(m_v)\).
Overlapping live values in the same memory must have disjoint address ranges.
The evaluator is total on all bounded syntax. It returns the first named fault
under a fixed order when elaboration, typing, arithmetic, timing, routing,
storage, or resource checks fail. Otherwise, it executes the two cycle phases
above and returns \(\operatorname{Normal}(o)\). Define:

$$
\operatorname{AllRunsNormal}_{\mathcal B_0,H}(z)
\quad\Longleftrightarrow\quad
\forall x\in\mathcal I(P),\;
\exists o:
\operatorname{Run}_{\mathcal B_0,H}(z,x)
=
\operatorname{Normal}(o).
$$

The semantic validity judgment binds a successful elaboration before using any
duration or demand:

$$
\begin{aligned}
\operatorname{Valid}_{\mathcal B_0,H}(z)
\equiv{}&
\exists E:\;
\operatorname{Elab}_H(z)=\operatorname{Some}(E)\\
&\land
\operatorname{Bounded}(z)
\land \operatorname{WellTyped}(G)
\land \operatorname{SingleAssignment}(G)\\
&\land \operatorname{Acyclic}(G)
\land \operatorname{ReductionComplete}(G)\\
&\land \operatorname{ValueRefines}(G,P,E)
\land \operatorname{TemporalLegal}(z,E)
\land \operatorname{ResourceLegal}(z,E)\\
&\land \operatorname{StorageLegal}(z,E)
\land \operatorname{RouteLegal}(z,E)
\land \operatorname{Complete}(z,E)\\
&\land \operatorname{AllRunsNormal}_{\mathcal B_0,H}(z).
\end{aligned}
$$

\(\operatorname{ReductionComplete}\) requires every output element to contain
each required reduction index exactly once and no extra term.
Successful elaboration excludes unsupported operations, shapes, locations, and
routes. \(\operatorname{AllRunsNormal}\) excludes arithmetic faults.
\(\operatorname{Complete}\) requires every registered action to issue and
finish by \(T_{\max}\). Well-formed graphs require every produced value to be
consumed or declared as an output, so \(\operatorname{last}(v)\) is defined.
The certificate is evidence for this judgment; it is not part of the
candidate's meaning.

There are no hidden stall or arbitration rules in Model Zero. A proposed
static trace either satisfies the finite constraints or is rejected.
Completion by \(T_{\max}\) is a bounded safety property, not a theorem about
general liveness.

### 4.4 Behavior, refinement, and composition

For input \(x=(A,B)\), the \(L_0\) behavior is:

$$
\llbracket P_{M,N,K}\rrbracket(A,B)_{mn}
=
\sum_{k=0}^{K-1}A_{mk}B_{kn}.
$$

The external observation is termination status and the output matrix.
Resource occupancy, internal events, traffic, and routes remain safety,
diagnostic, or cost data.

Model Zero is deterministic. Every level in one refinement chain uses the
source domain \(\mathcal I(P_0)\).
\(\operatorname{Safe}_i(P_i)\) is the conjunction of all
well-formedness, value, time, route, storage, and resource rules required at
level \(i\), including normal termination for every input and all inherited
rules; rules that do not yet apply are true.
Let
\(\alpha_{ji}:\mathcal O_j\rightarrow\mathcal O_i\) hide events below level
\(i\). The total function
\(\operatorname{Run}_i(P_i,x)\) returns either
\(\operatorname{Normal}(o_i)\) or a named fault. For any \(j>i\), exact
refinement is:

$$
\begin{aligned}
P_j\sqsubseteq_{ji} P_i
\quad\Longleftrightarrow\quad
&\operatorname{Safe}_j(P_j)
\land\operatorname{Safe}_i(P_i)\\
&\land
\forall x\in\mathcal I(P_0):
\exists o_j\in\mathcal O_j,\;
o_i\in\mathcal O_i:\\
&
\operatorname{Run}_j(P_j,x)=\operatorname{Normal}(o_j)\\
&\land
\operatorname{Run}_i(P_i,x)=\operatorname{Normal}(o_i)\\
&\land
\alpha_{ji}
(o_j)=o_i.
\end{aligned}
$$

The certificate checker must satisfy:

$$
\operatorname{Check}_i(P_i,P_{i+1},\pi_i)
=\operatorname{accept}
\Rightarrow
P_{i+1}\sqsubseteq_{i+1,i} P_i.
$$

For three levels, vertical refinement composes when the projections compose:

$$
\begin{aligned}
P_2\sqsubseteq_{21}P_1
\land P_1\sqsubseteq_{10}P_0
\land
\alpha_{10}\circ\alpha_{21}=\alpha_{20}
\;\Rightarrow\;
P_2\sqsubseteq_{20}P_0.
\end{aligned}
$$

A later floating-point profile must replace equality with a defined error
relation and prove how error bounds compose.

Horizontal composition uses a different theorem. A component summary is:

$$
\operatorname{Summary}(Q)=
\left(
I_Q,O_Q,W_Q,D_Q,S_Q
\right),
$$

where \(I_Q\) and \(O_Q\) are interface values, \(W_Q\) is the set of written
values, \(D_Q(r,t)\) is per-cycle resource demand, and \(S_Q(m,t)\) is storage
occupancy. A source component \(P\) also declares
\(\operatorname{Rely}(P,p)\), the values accepted at input port \(p\), and
\(\operatorname{Guarantee}(P,p)\), the values it may produce at output port
\(p\). Each component refinement carries an interface projection \(\beta_k\)
from \(Q_k\) ports and values to \(P_k\) ports and values. Let \(\omega_P\)
wire source ports and \(\omega_Q\) wire their lower-level counterparts. For
Model Zero:

$$
\begin{aligned}
\operatorname{Compat}_H
\left(
Q_1,Q_2;P_1,P_2,\beta_1,\beta_2,\omega_Q,\omega_P
\right)
\equiv{}&
 \operatorname{InterfaceCompatible}
 \left(Q_1,Q_2,P_1,P_2,\omega_Q,\omega_P\right)\\
&\land
\operatorname{WireSound}
\left(\omega_Q,\omega_P,\beta_1,\beta_2\right)\\
&\land
\operatorname{ContractCompatible}
\left(P_1,P_2,\omega_P\right)\\
&\land W_{Q_1}\cap W_{Q_2}=\varnothing\\
&\land \operatorname{Acyclic}(G_1\cup_{\omega_Q} G_2)\\
&\land
\operatorname{CombinedTemporalLegal}_H(Q_1,Q_2,\omega_Q)\\
&\land
\operatorname{CombinedRouteLegal}_H(Q_1,Q_2,\omega_Q)\\
&\land
\forall r,t:
D_{Q_1}(r,t)+D_{Q_2}(r,t)
\le\operatorname{CapToken}_H(r)\\
&\land
\operatorname{CombinedStorageLegal}_H(Q_1,Q_2,\omega_Q).
\end{aligned}
$$

For every source wire from output port \(p\) to input port \(q\),
\(\operatorname{ContractCompatible}\) requires:

$$
\operatorname{Guarantee}(P_{\mathrm{src}},p)
\subseteq
\operatorname{Rely}(P_{\mathrm{dst}},q).
$$

\(\operatorname{InterfaceCompatible}\) requires connected values to have equal
types and shapes. \(\operatorname{WireSound}\) uses \(\beta_1,\beta_2\) to
require every lower-level wire to project to its declared source wire and
forbids an undeclared cross-wire.
\(\operatorname{CombinedTemporalLegal}\) requires every cross-component
consumer to start after its producer completes. These premises make both the
source and lower compositions well-defined. Let \(\beta_{\parallel}\) be the
interface projection induced by \(\beta_1,\beta_2,\omega_Q,\omega_P\) after
connected internal ports are hidden. The horizontal theorem to prove is:

$$
\begin{aligned}
Q_1\sqsubseteq_{\beta_1} P_1
\land
Q_2\sqsubseteq_{\beta_2} P_2
\land
\operatorname{Compat}_H
\left(
Q_1,Q_2;P_1,P_2,\beta_1,\beta_2,\omega_Q,\omega_P
\right)
\;\Rightarrow\;
Q_1\parallel_{\omega_Q} Q_2
\sqsubseteq_{\beta_{\parallel}}
P_1\parallel_{\omega_P} P_2.
\end{aligned}
$$

This theorem covers disjoint writers and static resources only. Shared-state
protocols, dynamic arbitration, and blocking communication are outside Model
Zero.

Lowering introduces a constraint, but a constraint is not itself semantics.
Let \(\gamma_i(P_i)\) be the concrete designs represented at level \(i\). A
valid lowering narrows the set:

$$
\gamma_{i+1}(P_{i+1})
\subseteq
\gamma_i(P_i),
$$

and the certificate proves that the narrowing preserves the upper contract.
This makes the role of a hardware-specific prior precise:

1. a choice narrows the search space;
2. semantics defines what the choice means; and
3. a certificate shows that the choice remains correct.

### 4.5 Exact counts and empirical cost

For an accepted candidate:

$$
\tau=
\operatorname{Trace}_H(z)
=
\left\{
\left(
a,s_a,e_a,u_a,b_a,\lambda_A(a),\rho(a)
\right)
\mid a\in A_G
\right\}.
$$

Fixed folds over this trace determine exact structural quantities:

$$
K_{\mathrm{exact}}(\tau)
=
\left(
\#\mathrm{operations},
\mathrm{bytes},
\mathrm{link\ uses},
\mathrm{peak\ storage},
\mathrm{resource\ cycles},
\mathrm{abstract\ makespan}
\right).
$$

Each metric has a declared unit: operations, bytes, link-cycles,
resource-token cycles, storage bytes, or abstract cycles. “Exact” means exact
relative to the manifest's action granularity and tables, not exact physical
behavior. The count checker has its own proof goal:

$$
\operatorname{CountCheck}_H(\tau,k)=\operatorname{accept}
\Rightarrow
k=K_{\mathrm{exact},H}(\tau).
$$

An implementation prediction is separate:

$$
\widehat K_\theta(\tau,H_{\mathrm{physical}})
=
f_\theta
\left(
K_{\mathrm{exact}}(\tau),
H_{\mathrm{physical}}
\right).
$$

\(\theta\) is calibrated from simulation, synthesis, FPGA measurement, or
silicon measurement. Each source is reported separately.

Model Zero legality uses only the exact abstract duration
\(d_a=d_H(a)\in\mathbb N\) stored in the frozen manifest. Duration intervals
and robust scheduling are later extensions. A fitted duration
\(\widehat d_a\) may guide optimization, but it cannot change the formal
legality result.

Every parameter is labeled as:

- **derived:** proved from an earlier representation;
- **selected:** chosen by the optimizer;
- **imposed:** required by the target;
- **measured:** obtained from an artifact;
- **assumed:** not yet validated; or
- **learned:** predicted by a statistical model.

The final receipt preserves these labels.

### 4.6 Workflow and trust boundary

Proof-carrying code established the useful pattern of an untrusted producer
supplying evidence to a small checker [R30]. Model Zero applies that pattern to
co-design:

$$
\text{specify}
\rightarrow
\text{propose}
\rightarrow
\text{check}
\rightarrow
\text{diagnose}
\rightarrow
\text{revise}
\rightarrow
\text{evaluate cost}.
$$

The proof trusted base contains the formal definitions and the proof
assistant's kernel. The deployed semantic checker and exact-count folds will be
extracted from those proved definitions. A separately handwritten checker does
not satisfy \(\mathrm{H}_2\).

The executable trusted base also contains the canonical parser, certificate
decoder, runtime, and the extraction compiler unless a separate refinement
proof removes one of them. Receipts include the normalized syntax and its hash
so parsing can be audited.

The agent, search heuristic, solver whose result can be replayed, reference
interpreter, learned cost model, and artifact backend are untrusted. If a
solver result cannot be replayed, that solver is named as part of the trusted
base. The reference interpreter is an independent test reference; it is not a
substitute for the checker theorem.

Acceptance is always relative to the source specification and machine
manifest. It does not prove that an undocumented physical target matches the
manifest. The abstract validity theorem is unconditional for the supplied
manifest. Any claim that a physical target conforms to that manifest remains a
separate, explicit assumption.

Each receipt hashes the specification, manifest, candidate, and checker. It
records the decision, closed obligations, external assumptions, exact counts,
and cost-model provenance. A rejection includes a failed predicate and a
counterexample when available; an acceptance has no open deductive obligation.
A separate freeze receipt records \(C(K_0)\), the kernel and checker hashes,
the registered limits, and every file permitted to vary when a new machine
manifest is added.

## 5. Research method

### 5.1 Evidence required for each claim

| Claim | Required evidence | Primary endpoint | Pass rule | Allowed conclusion |
|---|---|---|---|---|
| \(\mathrm{H}_1\): four-primitive frozen-kernel portability | Registered kernel and manifest complexity; one held-out machine family | \(C(K_0)\), \(C_H(H)\); semantic or checker-code changes after freeze | The four primitive forms and manifest schema remain fixed; all complexity is reported; the held-out machine adds records but no value semantics, predicates, or executable code | One four-primitive kernel transfers across two Model Zero machine families |
| \(\mathrm{H}_2\): sound composition | Machine-checked theorems, extracted checker, and end-to-end certificates | Proof status and explicit premises | Every proof goal closes; each external premise is checked or stated | Soundness conditional on the stated premises |
| \(\mathrm{H}_3\): useful semantic richness | Independent canonical enumeration or a checkable counting certificate | Full-space richness \(S_{K_0}^{\mathrm{all}}\); \(10\%\)-near-optimal richness \(S_{K_0}^{(0.10)}\) | At least \(90\%\) of held-out tasks have \(2^{20}\) distinct designs and \(2^{10}\) designs within \(10\%\) of the kernel optimum; aliases and pure schedule slack count once | The four-primitive kernel induces a large space and at least \(2^{10}\) distinct near-optimal designs on that task fraction |
| \(\mathrm{H}_4\): optimization sufficiency | Independently certified kernel and reference optima under one exact objective | Maximum held-out representation regret, \(\max_t r_{K_0}(t,B)\) | Every kernel design maps into the reference space, every held-out family contains a reference-only design, and worst-case representation regret is at most \(5\%\) | The kernel loses at most \(5\%\) of attainable objective quality in the frozen study |
| \(\mathrm{H}_5\): agent search | Controlled runs scored against the exact kernel optimum | Fraction of runs returning a checked design with agent search regret at most \(0.10\) | The lower 95% family-clustered confidence bound on that fraction is at least \(0.80\) at budget \(B^\ast\) | The fixed agent usually finds a design within \(10\%\) of the kernel optimum |
| \(\mathrm{H}_6\): cost validity | Whole design families held out from one frozen backend | Exact-count mismatch; backend-completion rate; conditional held-out latency median absolute percentage error | Zero count mismatches; the lower 95% confidence bound on completion is at least \(0.95\); the upper 95% confidence bound on conditional median error is at most \(0.10\) | Completion and latency accuracy for that backend and sampled domain |

No experiment may replace the evidence type in this table. In particular,
mutation testing cannot replace a soundness proof, and a theorem cannot replace
physical measurement.

### 5.2 Common protocol

Before the test set is exposed, the study will freeze:

- the manifest, kernel, grammar, task generator, objectives, and canonical
  design schema;
- the kernel-size accounting rules, expression bounds, richness threshold, regret
  tolerances, and invalid-output rule;
- the formal semantics, checker, and independent reference implementation;
- the named hand-designed, compiler, and solver baselines, their versions, and
  a reproducible task-eligibility rule;
- development, validation, and test families;
- mutation operators and legal boundary cases;
- prompts, model and API versions, tool and token budgets, and randomization;
- solver, compiler, simulator, synthesis, and proof-tool versions;
- timeout, retry, crash, and missing-run rules; and
- primary endpoints, power analysis, confidence intervals, and rules for
  interpreting multiple secondary tests.

Task families, not only parameter combinations, are held out. Every run keeps
raw inputs, outputs, trajectories, logs, artifact hashes, and failure status.
Compilation, synthesis, timeout, and measurement failures remain in the
results.

The study keeps three kinds of exhaustive evidence separate:

1. **Symbolic input proof:** the value-refinement theorem quantifies over every
   matrix whose elements are in \(V_0\). It does not enumerate
   \(5^{M K+K N}\) input pairs.
2. **Exact design search:** \(U_0^{\mathrm{legal}}\) and the kernel-induced
   space are exhaustively enumerated, or independently checked certificates
   prove canonical class counts, inclusion, and both optima for each registered
   confirmatory task. If any required computation does not finish, the related
   claim is marked insufficient.
3. **Full micro-test:** a smaller manifest uses
   \((M,N,K)=(1,1,2)\),
   \(V_\mu=\{-1,0,1\}\),
   \(N_{\max}=8\), and \(T_{\max}=8\), with one tiny configuration from each
   machine family. All \(81\) input pairs and all bounded reference and kernel
   candidates are enumerated.

### 5.3 Kernel size, semantic richness, and optimization sufficiency

The two implementations of \(U_0\) define legal and faulty design classes
without calling the proposed kernel or checker. The kernel, size-accounting
rule, canonicalizer, and checker are frozen after development on one target
family. The second target is then encoded once using only manifest records.

The kernel-economy gate is:

$$
\operatorname{Prim}(K_0)
=
\left\{
\operatorname{zero},
\operatorname{copy},
\operatorname{transfer},
\operatorname{tile\_mac}
\right\}
\land
\Delta_{\mathrm{semantics}}=0
\land
\Delta_{\mathrm{checker}}=0
\land
\Delta_{\mathrm{manifest\ schema}}=0
$$

after the held-out machine is exposed. \(C(K_0)\) is reported, not hidden
behind the four-primitive count. The soundness proof and measured checker cost
determine whether this kernel is actually manageable; no absolute minimality
is claimed.

The study reports:

- primitive, proof-rule, parser, elaborator, checker, and total trusted-code
  size;
- resource, route, table-row, and demand-form counts for each machine
  manifest;
- each post-freeze code change, which fails \(\mathrm{H}_1\);
- the number and logarithm of canonical legal designs by task;
- inclusion failures and a strict-reference witness;
- kernel and reference optima and their representation regret;
- checking time and memory against state-space size; and
- the effect of removing each primitive on semantic-space size and optimum
  quality.

Every kernel-induced class must map into the independent reference:

$$
\operatorname{Included}(K_0,U_0,t,B)
\quad\Longleftrightarrow\quad
\forall g\in\operatorname{Expr}(K_0,t,B),\;
\left(
\exists\pi_g:
\operatorname{Check}_{K_0}(g,\pi_g)=\operatorname{accept}
\right)
\Rightarrow
\exists u\in U_0^{\mathrm{legal}}(t),
\pi_{\mathrm{eq}}:
\operatorname{EqCheck}(u,g,\pi_{\mathrm{eq}})
=
\operatorname{accept}.
$$

The reference must also contain at least one design outside the kernel-induced
space in every held-out task family \(\mathcal F\):

$$
\forall\mathcal F\in\operatorname{Families}(\mathcal T_{\mathrm{test}}),\;
\exists t\in\mathcal F:
\mathcal M_{K_0}(t,B)
\subsetneq
\mathcal R_0(t).
$$

This prevents a circular result in which the “richer” universe is merely
another encoding of the kernel. Strictness requires exhaustive non-membership
or a proved invariant of \(K_0\) that a legal reference witness violates. A
search timeout is insufficient.

For each task, exact enumeration or a checkable model-counting certificate
computes \(S_{K_0}^{\mathrm{all}}\) and
\(S_{K_0}^{(0.10)}\) from Section 2.2. The richness canonicalizer keeps
meaningful graph, mapping, route, storage, and order-or-overlap differences but
removes identifier aliases, global time shifts, and pure idle slack.

The \(20\)-bit full-space threshold means at least \(2^{20}\) distinct
implementation structures. The \(10\)-bit near-optimal threshold requires at
least \(2^{10}\) such structures within \(10\%\) of the kernel optimum. These
are registered engineering tests, not universal definitions of “large.”
Per-axis value counts and pairwise interaction counts are reported as
diagnostics, not used as an undefined pass rule.

The same independent objective \(J_t\) is applied to both spaces. The
confirmatory \(\mathrm{H}_4\) and \(\mathrm{H}_5\) objective uses only exact
trace quantities. \(\mathrm{H}_6\)'s median prediction error does not justify a
claim about optimizer-selected physical optima. Such a claim would require a
separate uniform-error or selection-regret experiment.
Before held-out tasks are exposed, the study freezes a baseline registry
\(\mathcal B_{\mathrm{base}}\): named hand designs, compiler versions, solver
configurations, and a deterministic rule that says which baseline applies to
which task. Every admitted valid baseline must normalize into
\(\mathcal R_0(t)\), and:

$$
J_U^*(t)
\le
\min_{b\in\mathcal B_{\mathrm{base}}(t)}
J_t(b).
$$

A missing admitted baseline or a violated inequality makes the reference
evidence insufficient. No AI agent participates in this test.
The representation test computes:

$$
r_{K_0}(t,B)
=
\frac{
J_{K_0}^*(t,B)-J_U^*(t)
}{
J_U^*(t)
}.
$$

\(\mathrm{H}_4\) uses one frozen scalar objective. It does not establish
Pareto-frontier coverage; that would require a separate componentwise
\(\varepsilon\)-dominance test.

Primitive ablation is diagnostic, not a claim of absolute minimality. A
primitive is useful for this benchmark when removing it either prevents a
target encoding or increases the best attainable cost by more than a
registered tolerance:

$$
\Delta_p(t)
=
\frac{
J_{K_0\setminus\{p\}}^*(t,B)-J_{K_0}^*(t,B)
}{
J_{K_0}^*(t,B)
}.
$$

### 5.4 Formal proof and checker validation

The primary formal result is:

$$
\operatorname{Check}_{\mathcal B_0,H}(z,\pi)=\operatorname{accept}
\Rightarrow
\operatorname{Valid}_{\mathcal B_0,H}(z).
$$

A proof assistant will check this theorem and the stated vertical and
horizontal composition rules. The deployed core checker is extracted from
this proved definition. If the reverse implication is claimed, it requires a
separate proof and a certificate-existence argument.

The implementation is also tested against an independently written interpreter
and enumerator. The registered fault classes include:

1. a missing reduction tile;
2. a duplicated product;
3. use before transfer completion;
4. overwrite of a live value;
5. buffer overflow;
6. port or bank overuse;
7. pipeline issue-rate violation;
8. illegal layout or route;
9. arithmetic overflow;
10. a cyclic event graph;
11. overlapping use of an exclusive resource; and
12. an unregistered machine or cost assumption.

Mutants start from independently validated legal designs. Equivalent mutants
are removed. Legal near-boundary cases measure false rejection. The initial
target is at least 100 independent base designs per fault class. If several
mutants share a base design, the analysis treats that base as one cluster.
False-acceptance and false-rejection counts and 95% cluster-aware confidence
bounds are reported for each fault class, not only as one pooled number.
Held-out mutation families and independently authored faults reduce
co-development bias. These tests measure suite sensitivity; they do not prove
soundness beyond the theorem.

### 5.5 Cost-model validation

The first confirmatory study uses one frozen implementation backend. Other
simulators, FPGA measurements, or physical devices are separate studies, not
pooled evidence.

After a pilot fixes variance and sample size, the study will hold out whole
shape, dataflow, array, or memory families. A family is a generation template,
not a single parameter setting. The confirmatory set will contain at least 20
disjoint held-out families and about 100 held-out configurations within a
larger set of at least 300 attempted configurations. The family is the
resampling unit. Synthesis, simulation, timing, and measurement failures remain
in the completion analysis.

The proposed model is compared with:

- a constant predictor;
- operation-count and byte-count linear models;
- a roofline-style model;
- an applicable analytical tool such as Timeloop, MAESTRO, or Accelergy; and
- cycle-level simulation when available.

An independently written trace counter checks every quantity called exact.
For measured latency \(y_j>0\) and prediction \(\widehat y_j\), define:

$$
\operatorname{APE}_j
=
\frac{\left|\widehat y_j-y_j\right|}{y_j},
\qquad
\operatorname{MdAPE}
=
\operatorname{median}_j\operatorname{APE}_j.
$$

The confirmatory test has two gates. First, it requires zero exact-count
mismatches and a lower one-sided 95% family-clustered bootstrap confidence
bound of at least \(0.95\) for backend completion. Second, among completed
artifacts, it requires an upper one-sided 95% family-clustered bootstrap bound
of at most \(0.10\) on \(\operatorname{MdAPE}\). Accuracy is reported as
conditional on completion, and it cannot pass if the completion gate fails.
A failed attempt is counted as a completion failure; it is not assigned an
invented prediction error.

The 90th-percentile error, rank correlation, top-\(k\) selection regret, and
prediction-interval coverage and width are secondary diagnostics. Energy and
area require separate registered endpoints. The 10% latency bound is an
engineering target for one frozen backend, not a universal constant.

### 5.6 Controlled agent study

Every condition is scored against the same independently defined task and
exact kernel optimum. Conditions \(\mathrm{C}_1\) through \(\mathrm{C}_3\)
produce the same typed Model Zero artifact. A fixed adapter maps
\(\mathrm{C}_0\) output into that artifact. Thus every primary condition is
scored inside the same accepted canonical space
\(\mathcal M_{K_0}(t,B)\). The conditions are:

- \(\mathrm{C}_0\): a conventional native format with compiler, simulator, and
  test feedback; the fixed adapter emits a \(G_0\) candidate;
- \(\mathrm{C}_1\): the typed model with the same conventional feedback but no
  semantic checker feedback;
- \(\mathrm{C}_2\): model representation with binary accept or reject;
- \(\mathrm{C}_3\): model representation with structured diagnostics.

An adapter failure or an output outside \(\mathcal M_{K_0}(t,B)\) in
\(\mathrm{C}_0\) counts as an invalid design. The adapter, output grammars,
tools, and exact information initially shown in each condition are frozen
before testing. Direct RTL generation is retained only as a descriptive
real-world baseline because it changes the output language, abstraction level,
and search space at once; it is not part of the causal contrast. Random search,
a conventional optimizer, and the exact kernel optimum provide lower,
practical, and upper search references.

The study fixes the agent version, prompt information, context policy,
temperature, task order, and one matched budget at a time. Budget-response
curves report tokens, tool calls, wall time, and monetary cost. Tasks are held
out by family, and each run starts with an independent context.

Let \(\widehat g_t\) be the expression returned by the agent. If the checker
accepts it, define its canonical design
\(\widehat c_t=\operatorname{Can}(\widehat g_t)\) and agent search regret:

$$
r_A(t)
=
\frac{
J_t(\widehat c_t)-J_{K_0}^*(t,B)
}{
J_{K_0}^*(t,B)
}.
$$

An invalid or missing \(\widehat g_t\) has \(r_A(t)=+\infty\). At the primary
budget \(B^\ast\), define:

$$
Y_{10}
=
\mathbf 1
\left[
\left(
\exists\pi:
\operatorname{Check}_{K_0}(\widehat g_t,\pi)=\operatorname{accept}
\right)
\land
r_A(t)\le 0.10
\right].
$$

\(\mathrm{H}_5\) passes only if the lower 95% family-clustered confidence bound
on \(\Pr(Y_{10}=1\mid\mathrm C_3)\) is at least \(0.80\). This endpoint combines
correctness and quality. A reliable agent that returns poor designs does not
pass.

A paired bootstrap samples whole task families, then tasks within each family.
A pilot-based power simulation fixes the final sample size before the test set
is opened. The full-interface contrast
\(\Pr(Y_{10}=1\mid\mathrm C_3)-\Pr(Y_{10}=1\mid\mathrm C_0)\) tests whether the
kernel and checker help relative to conventional feedback. The contrast
\(\mathrm C_3-\mathrm C_2\) isolates structured diagnostics;
\(\mathrm C_2-\mathrm C_1\) isolates binary checking; and
\(\mathrm C_1-\mathrm C_0\) compares the typed representation with the native
format. These are secondary mechanism tests.

The study reports representation regret and agent search regret separately.
If \(\mathrm{H}_4\) fails, a low \(r_A\) shows only that the agent searched a
weak space well. \(\mathrm{H}_4\) and \(\mathrm{H}_5\) therefore remain claims
about the frozen exact surrogate. \(\mathrm{H}_6\) evaluates prediction
accuracy but does not turn either result into a physical-optimality claim.

### 5.7 From the proof pilot to LLM-relevant motifs

Model Zero is a proof pilot for exact tiled matrix multiplication. Passing it
does not establish that the kernel is sufficient for an LLM system.

Before making a broader tile-dominant LLM claim, a new versioned kernel must
repeat the six gates on three bounded motifs:

1. a fused matrix–activation–matrix block, including materialized and fused
   intermediates;
2. bounded causal prefill attention, including materialized and streamed
   schedules; and
3. bounded decode attention with explicit KV-cache append, layout, prefetch,
   and reduction choices.

Activation, exponential, reciprocal, rounding, exceptional values, and
accumulation order must have fixed semantics. Searching over numerical
approximations is a later experiment.

Further extensions add sparse expert routing, dynamic batching and arbitration,
bounded queues, multi-device collectives, shared-link contention, and failures.
Each new primitive creates a new kernel version with a new semantics, proof
obligation, independent reference extension, and untouched holdout. A failed
holdout cannot be repaired and reused as confirmatory evidence.

## 6. Risks, research gates, and claim boundary

### 6.1 Main risks

| Failure | Conclusion |
|---|---|
| The held-out target needs new semantics or checker code | The frozen-kernel claim fails; version the kernel and use a new holdout. |
| The slack-normalized space is small or has too few near-optimal designs | The richness claim fails even if the checker is sound. |
| A kernel design has no valid reference match | The inclusion or independent-reference implementation is wrong. |
| Representation regret exceeds \(0.05\) | The kernel is too restrictive for the registered objective. |
| Local proofs do not compose | Strengthen the interface or require an explicit whole-system check. |
| Checking is too slow | Report the usable bounded subset; finiteness alone does not imply practical speed. |
| Held-out cost prediction fails | Keep exact counts; reject or revise the empirical model. |
| The agent has high search regret | Keep the kernel and checker if useful with enumerators or solvers; reject the agent-search claim. |

### 6.2 Research gates

The project advances only after each gate:

1. define the finite task domain, independent reference, canonical schema, and
   exact objective;
2. build and prove the kernel on one development machine;
3. freeze the kernel, checker, size-accounting rule, richness threshold, and
   regret tolerances;
4. encode the held-out machine without semantic or checker changes;
5. prove inclusion, count canonical designs, and certify both optima;
6. validate one implementation-cost backend before reporting empirical physical
   predictions;
7. run the powered agent study; and
8. repeat the process on new untouched transformer-motif holdouts before making
   an LLM-level claim.

### 6.3 What success would and would not establish

Success would establish, for Model Zero:

- one reported four-primitive kernel frozen across two abstract machine
  families, with no target-specific semantic or checker code;
- a sound checker for finite scheduled tensor actions;
- a replayable certificate chain from matrix meaning to timed architectural
  actions;
- at least \(2^{20}\) slack-normalized designs, including at least \(2^{10}\)
  designs within \(10\%\) of the kernel optimum, on at least \(90\%\) of
  held-out tasks;
- at most \(5\%\) representation regret under one exact registered objective;
- the registered probability that one fixed agent finds a checked design
  within \(10\%\) of the kernel optimum; and
- held-out accuracy bounds for one versioned cost model and backend.

It would not establish:

- an absolutely minimal or complete theory of hardware;
- that all LLM computation is regular or covered by Model Zero;
- a fundamental CPU–accelerator distinction;
- correctness of generated RTL, gates, layout, or silicon;
- physical timing, energy, or area without measurement;
- general liveness, failures, unbounded workloads, or dynamic distributed
  execution;
- model-training convergence or application-level ML accuracy; or
- global optimality outside an independently solved finite design universe.

The current research state is incomplete. No theorem, checker, benchmark,
physical measurement, or agent result has yet been produced.

### 6.4 Conclusion

The proposal tests whether tile regularity changes a basic design tradeoff.
The four primitive forms and trusted checker remain fixed, while composition
and parameters create a large optimization space. Every accepted point remains
verifiable because it is built from the same rules.

The agent supplies search capacity, not correctness. The decisive tests are
whether the kernel retains near-optimal designs and whether the agent can find
them. If either fails, the two regret terms show whether the problem is the
representation or the search procedure. If Model Zero succeeds, the same gates
must be repeated on bounded transformer motifs before the claim is extended to
LLM infrastructure.

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

[R4] Warren S. McCulloch and Walter Pitts, “A Logical Calculus of the
Ideas Immanent in Nervous Activity,” 1943.
[DOI](https://doi.org/10.1007/BF02478259)

[R5] Arthur W. Burks, Herman H. Goldstine, and John von Neumann,
*Preliminary Discussion of the Logical Design of an Electronic Computing
Instrument*, 1946/1947.
[IAS archive](https://albert.ias.edu/entities/publication/49465635-0aa6-4299-921b-98c70c876e51)

[R6] John von Neumann, “The General and Logical Theory of Automata,”
Hixon Symposium lecture, 1948.
[PDF](https://www.vordenker.de/ggphilosophy/jvn_the-general-and-logical-theory-of-automata.pdf)

[R7] M. V. Wilkes and J. B. Stringer, “Micro-programming and the Design
of the Control Circuits in an Electronic Digital Computer,” 1953.
[DOI](https://doi.org/10.1017/S0305004100028322)

[R8] Edward A. Lee and David G. Messerschmitt, “Synchronous Data Flow,”
1987.
[Paper page](https://ptolemy.berkeley.edu/publications/papers/87/synchdataflow/)

[R9] Rajeev Alur and David L. Dill, “A Theory of Timed Automata,” 1994.
[PDF](https://www.cis.upenn.edu/~alur/TCS94.pdf)

[R10] Jonathan Ragan-Kelley et al., “Halide: A Language and Compiler for
Optimizing Parallelism, Locality, and Recomputation in Image Processing
Pipelines,” 2013.
[DOI](https://doi.org/10.1145/2491956.2462176)

[R11] Tianqi Chen et al., “TVM: An Automated End-to-End Optimizing
Compiler for Deep Learning,” 2018.
[USENIX](https://www.usenix.org/conference/osdi18/presentation/chen)

[R12] Lianmin Zheng et al., “Ansor: Generating High-Performance Tensor
Programs for Deep Learning,” 2020.
[USENIX](https://www.usenix.org/conference/osdi20/presentation/zheng)

[R13] Chris Lattner et al., “MLIR: Scaling Compiler Infrastructure for
Domain Specific Computation,” 2021.
[Google Research](https://research.google/pubs/mlir-scaling-compiler-infrastructure-for-domain-specific-computation/)

[R14] OpenXLA, “StableHLO Specification,” accessed 2026-07-26.
[Specification](https://openxla.org/stablehlo/spec)

[R15] Yakun Sophia Shao et al., “Aladdin: A Pre-RTL,
Power-Performance Accelerator Simulator Enabling Large Design Space
Exploration of Customized Architectures,” 2014.
[Paper page](https://vlsiarch.eecs.harvard.edu/publications/aladdin-pre-rtl-power-performance-accelerator-simulator-enabling-large-design)

[R16] Angshuman Parashar et al., “Timeloop: A Systematic Approach to
DNN Accelerator Evaluation,” 2019.
[NVIDIA Research](https://research.nvidia.com/publication/2019-03_timeloop-systematic-approach-dnn-accelerator-evaluation)

[R17] Hyoukjun Kwon et al., “Understanding Reuse, Performance, and
Hardware Cost of DNN Dataflows: A Data-Centric Approach,” 2019.
[NVIDIA Research](https://research.nvidia.com/publication/2019-10_understanding-reuse-performance-and-hardware-cost-dnn-dataflows-data-centric)

[R18] John Yang et al., “SWE-agent: Agent-Computer Interfaces Enable
Automated Software Engineering,” 2024.
[arXiv](https://arxiv.org/abs/2405.15793)

[R19] Mingjie Liu et al., “ChipNeMo: Domain-Adapted LLMs for Chip
Design,” 2023.
[arXiv](https://arxiv.org/abs/2311.00176)

[R20] Shailja Thakur et al., “VeriGen: A Large Language Model for
Verilog Code Generation,” 2023.
[arXiv](https://arxiv.org/abs/2308.00708)

[R21] Mingjie Liu et al., “VerilogEval: Evaluating Large Language
Models for Verilog Code Generation,” 2023.
[Repository](https://github.com/NVlabs/verilog-eval)

[R22] Yao Lu et al., “RTLLM: An Open-Source Benchmark for Design RTL
Generation with Large Language Model,” 2023.
[arXiv](https://arxiv.org/abs/2308.05345)

[R23] Shailja Thakur et al., “AutoChip: Automating HDL Generation Using
LLM Feedback,” 2023.
[arXiv](https://arxiv.org/abs/2311.04887)

[R24] Yonggan Fu et al., “GPT4AIGChip: Towards Next-Generation AI
Accelerator Design Automation via Large Language Models,” 2023.
[arXiv](https://arxiv.org/abs/2309.10730)

[R25] Sean A. Blocklove et al., “Chip-Chat: Challenges and
Opportunities in Conversational Hardware Design,” 2023.
[PDF](https://verificationacademy.com/forums/uploads/short-url/hv4m31iwKciCxQsGiKuhwRHYW3G.pdf)

[R26] Siyuan Feng et al., “TensorIR: An Abstraction for Automatic
Tensorized Program Optimization,” 2023.
[DOI](https://doi.org/10.1145/3575693.3576933)

[R27] Seongwon Bang et al., “SMT-Based Translation Validation for
Machine Learning Compiler,” 2022.
[DOI](https://doi.org/10.1007/978-3-031-13188-2_19)

[R28] Xavier Leroy, “A Formally Verified Compiler Back-end,” 2009.
[arXiv](https://arxiv.org/abs/0902.2137)

[R29] Nuno P. Lopes et al., “Alive2: Bounded Translation Validation for
LLVM,” 2021.
[Paper page](https://web.ist.utl.pt/nuno.lopes/pubs.php?id=alive2-pldi21)

[R30] George C. Necula, “Proof-Carrying Code,” 1997.
[DOI](https://doi.org/10.1145/263699.263712)

[R31] Yuka Ikarashi et al., “Exo 2: Growing a Scheduling Language,”
ASPLOS, 2025.
[DOI](https://doi.org/10.1145/3669940.3707218)

[R32] Yuka Ikarashi et al., “Exocompilation for Productive Programming
of Hardware Accelerators,” 2022.
[PDF](https://people.csail.mit.edu/yuka/pdf/exo_pldi2022_full.pdf)

[R33] Rachit Nigam et al., “Predictable Accelerator Design with
Time-Sensitive Affine Types,” 2020.
[arXiv](https://arxiv.org/abs/2004.04852)

[R34] Thomas Bourgeat et al., “The Essence of Bluespec: A Core Language
for Rule-Based Hardware Design,” 2020.
[Paper page](https://adam.chlipala.net/papers/KoikaPLDI20/)

[R35] MIT CSAIL, “Kami: A Modular Deductive Hardware Verification
Platform.”
[Project page](https://www.csail.mit.edu/research/kami-modular-deductive-hardware-verification-platform)

[R36] Rachit Nigam et al., “Calyx: A Compiler Infrastructure for
Accelerator Generators,” 2021.
[arXiv](https://arxiv.org/abs/2102.09713)

[R37] Yannan Nellie Wu et al., “Accelergy: An Architecture-Level Energy
Estimation Methodology for Accelerator Designs,” 2019.
[PDF](https://accelergy.mit.edu/paper.pdf)

[R38] Bo-Yuan Huang et al., “Instruction-Level Abstraction: A Uniform
Specification for System-on-Chip Verification,” 2019.
[DOI](https://doi.org/10.1145/3282444)

[R39] Bo-Yuan Huang et al., “Application-Level Validation of
Accelerator Designs Using a Formal Software/Hardware Interface,” 2024.
[DOI](https://doi.org/10.1145/3639051)

[R40] Amanda Liu et al., “Verified Tensor-Program Optimization via
High-Level Scheduling Rewrites,” 2022.
[DOI](https://doi.org/10.1145/3498717)

[R41] Amanda Liu et al., “A Verified Compiler for a Functional Tensor
Language,” 2024.
[DOI](https://doi.org/10.1145/3656390)

[R42] Rachit Nigam, Pedro Henrique Azevedo de Amorim, and Adrian
Sampson, “Modular Hardware Design with Timeline Types,” 2023.
[DOI](https://doi.org/10.1145/3591234)

[R43] Rachit Nigam, Ethan Gabizon, Edmund Lam, and Adrian Sampson,
“Correct and Compositional Hardware Generators,” 2024 preprint,
arXiv:2401.02570v1.
[arXiv v1](https://arxiv.org/abs/2401.02570v1)

[R44] Caleb Kim et al., “Unifying Static and Dynamic Intermediate
Languages for Accelerator Generators,” 2024.
[DOI](https://doi.org/10.1145/3689790)

[R45] Jianhong Zhao, Jinhui Kang, and Yongwang Zhao, “K-CIRCT: A
Layered, Composable, and Executable Formal Semantics for CIRCT Hardware
IRs,” 2024 preprint.
[arXiv](https://arxiv.org/abs/2404.18756)

[R46] Mathieu Fehr et al., “First-Class Verification Dialects for
MLIR,” 2025.
[DOI](https://doi.org/10.1145/3729309)

[R47] Kezhi Li et al., “FormalRTL: Verified RTL Synthesis at Scale,”
2026 preprint.
[arXiv](https://arxiv.org/abs/2603.08738)

[R48] Pei-Huan Tsai et al., “HSCO-Bench: An Agent-Driven End-to-End
Hardware-Software Co-design Benchmark for Systems-on-Chip,” 2026
preprint.
[arXiv](https://arxiv.org/abs/2605.19399)

[R49] Robert M. Tomasulo, “An Efficient Algorithm for Exploiting
Multiple Arithmetic Units,” 1967.
[DOI](https://doi.org/10.1147/rd.111.0025)

[R50] H. T. Kung, “Why Systolic Architectures?” 1982.
[DOI](https://doi.org/10.1109/MC.1982.1653825)

[R51] Norman P. Jouppi et al., “In-Datacenter Performance Analysis of a
Tensor Processing Unit,” 2017.
[Google Research](https://research.google/pubs/in-datacenter-performance-analysis-of-a-tensor-processing-unit/)

[R52] John L. Hennessy and David A. Patterson, “A New Golden Age for
Computer Architecture,” 2019.
[DOI](https://doi.org/10.1145/3282307)

[R53] Gene M. Amdahl, Gerrit A. Blaauw, and Frederick P. Brooks Jr.,
“Architecture of the IBM System/360,” 1964.
[DOI](https://doi.org/10.1147/rd.82.0087)

[R54] RISC-V International, “The RISC-V Instruction Set Manual, Volume
I: Unprivileged Architecture,” 20250508 edition, accessed 2026-07-26.
[Specification](https://docs.riscv.org/reference/isa/v20250508/unpriv/intro.html)

[R55] Intel, “Advanced Matrix Extensions Intrinsics,” accessed
2026-07-26.
[Reference](https://www.intel.com/content/www/us/en/developer/articles/code-sample/advanced-matrix-extensions-intrinsics-functions.html)

<script type="text/javascript" async
  src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/MathJax.js?config=TeX-MML-AM_CHTML">
</script>
