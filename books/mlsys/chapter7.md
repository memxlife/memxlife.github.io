# Chapter: Deep Learning Compilers — From Generality to Specialization, and Back to Learned Search

## 1. Why Deep Learning Needs a Compiler at All

A deep learning model is written in the language of meaning. We talk about attention, feed-forward networks, layer normalization, convolution, activation functions, and loss functions. Hardware, however, does not execute meaning. Hardware executes instructions. It moves bytes between memory levels, schedules warps, issues tensor-core instructions, resolves dependencies, and stalls when data does not arrive on time.

A deep learning compiler exists because these two worlds are far apart.

At the model level, we express *what* computation we want. At the hardware level, we must decide *how* to realize that computation efficiently on a concrete machine. The distance between those two levels is not a small syntactic gap. It is a large optimization problem. The compiler is the system that bridges that gap.

For deep learning, this problem became important only after neural networks grew large enough and useful enough that raw execution efficiency started to matter in production. Once deep networks moved from research prototypes to real systems, the problem was no longer merely functional correctness. It was no longer enough that a model ran. The model had to run fast, fit in memory, use the hardware well, and scale across real deployment environments.

This chapter develops a structured understanding of that problem. The key claim is simple:

> A deep learning compiler is a software system for automating a large, structured optimization problem over the mapping from high-level tensor computation to hardware execution.

That mapping has changed over time. The nature of the workload changed. The nature of the hardware changed. The dominant bottlenecks changed. As a result, the dominant compiler designs also changed. Over roughly the last decade, one can read the history of deep learning compilers through four landmarks:

1. **The pre-LLM compiler era**, represented most clearly by **TVM**, where the core problem was *many models × many hardware backends*.
2. **The first LLM-era simplification**, represented by **Triton**, where the problem collapsed toward *Transformer workloads on NVIDIA GPUs*.
3. **The return of exposed control**, represented by **TileLang**, where automation alone was no longer sufficient for increasingly specialized kernels.
4. **The new learned-search era**, represented here by **CUDA-L2**, where large language models plus reinforcement learning begin to automate kernel optimization again, but at a different level.

The deeper point is that each of these systems is not just a new tool. Each is an answer to a different version of the same underlying question:

> Given a tensor workload and a target machine, what structure should be exposed, what structure should be hidden, and who should search the optimization space: the human, the compiler, or a learned agent?

---

## 2. The Optimization Problem Before the Compiler

Before discussing compilers, we must first clarify *what exactly is being optimized*.

When we studied GPU programming, the central picture was already visible. A tensor computation can be viewed as an **iteration space**: the set of all loop-index tuples that define the work. For matrix multiplication,

$$
C[i,j] = \sum_k A[i,k]B[k,j],
$$

a natural implementation is

```cpp
for (int i = 0; i < M; ++i) {
  for (int j = 0; j < N; ++j) {
    float acc = 0.0f;
    for (int k = 0; k < K; ++k) {
      acc += A[i][k] * B[k][j];
    }
    C[i][j] = acc;
  }
}
```

The iteration space is the set

$$
\{(i,j,k) \mid 0 \le i < M,\; 0 \le j < N,\; 0 \le k < K\}.
$$

A GPU kernel is, in essence, a mapping from this iteration space to hardware execution and the memory hierarchy.

Two goals dominate this mapping.

### 2.1 Parallelism

The first goal is to break a large computation into many smaller computation-intensive subtasks that can run concurrently on many processing units. On a GPU, those units are arranged hierarchically: grids, blocks, warps, and threads. Good parallelization increases compute utilization. Bad parallelization leaves arithmetic units idle.

### 2.2 Locality

The second goal is to reduce the cost of data movement. Arithmetic and memory are physically separate. Registers are fast but tiny. Shared memory and L1 are also fast but limited. HBM is large but far slower and much farther away. A high-performance kernel must keep reusable data as high in the hierarchy as possible, for as long as possible.

This is why so many seemingly different kernel tricks all reduce to the same underlying idea: do more useful computation per byte moved.

### 2.3 The Canonical Optimization Knobs

The optimization techniques introduced in the lecture are not isolated tricks. They are the standard degrees of freedom of tensor kernel optimization.

#### Tiling

Tiling partitions the iteration space into smaller blocks. In GEMM, one may tile the output space into, say, \(128 \times 128\) output tiles, and tile the reduction dimension by a chosen block size. Tiling improves locality because a tile can fit in shared memory or registers, and it improves parallelism because different tiles can be assigned to different blocks or warps.

#### Memory placement

Memory placement decides where intermediate data lives: global memory, shared memory, registers, or sometimes specialized on-chip storage. For matrix multiplication, tiles of \(A\) and \(B\) are typically loaded into shared memory so they can be reused by many threads within a block.

#### Loop permutation

Loop permutation changes the order of iteration. The mathematical computation is unchanged, but the reuse pattern changes dramatically. In matrix multiplication, changing the order from \(i\)-\(j\)-\(k\) to \(k\)-\(i\)-\(j\) changes which values are reused in inner loops and thus changes memory locality.

#### Fusion and fission

Fusion combines multiple operators or loops into one larger computation. Fission splits a large fused computation into smaller pieces. Fusion often reduces memory traffic because intermediate results no longer have to be materialized to global memory. Fission may improve parallelism, reduce register pressure, or simplify scheduling.

#### Unrolling

Loop unrolling expands multiple loop iterations into straight-line code. This reduces loop-control overhead and can increase instruction-level parallelism, but it also increases code size and register pressure.

#### Thread/block mapping

This determines which piece of the iteration space is computed by which block, warp, or thread. This choice strongly affects coalescing, reuse, occupancy, and synchronization patterns.

#### Synchronization

Synchronization coordinates shared work across threads. It is necessary for correctness whenever data is shared, but it also introduces overhead. Excessive synchronization can destroy throughput; insufficient synchronization breaks correctness.

---

## 3. GPU Kernel Optimization as a Structured Discrete Search Space

The lecture’s most important abstraction is that all these choices together form a **structured discrete search space**.

A kernel is not optimized by a single decision. It is optimized by a tuple of decisions:

$$
\theta = (\theta_{\text{tile}},\theta_{\text{mem}},\theta_{\text{perm}},\theta_{\text{fusion}},\theta_{\text{unroll}},\theta_{\text{map}},\theta_{\text{sync}}).
$$

Each component is usually discrete. Tile sizes are selected from a finite set. Loop orders are permutations. Unroll factors are integers. Thread mappings are categorical choices. Synchronization patterns are discrete design choices. Real kernels typically involve many such variables, often well beyond ten and sometimes beyond twenty.

The difficulty is not only that the space is large. The difficulty is that the variables are **strongly coupled**.

- A larger tile may improve data reuse but overflow shared memory.
- More unrolling may improve instruction-level parallelism but blow up register pressure.
- A different loop order may improve locality but break vectorization.
- More aggressive fusion may reduce HBM traffic but lower occupancy.
- Better thread mapping may improve coalescing but introduce warp divergence elsewhere.

In other words, performance is not a sum of independent choices. It is an interaction surface.

This is why, in the worst case, the problem is often described as combinatorial and intractable. In practice, compilers therefore use heuristics, pruning, integer programming, mixed-integer programming, cost models, rule systems, autotuning, or learned search.

This is also why the compiler stack exists. One giant optimization problem is too difficult to handle at once. The system must be decomposed into layers.

---

## 4. Why the Compiler Stack Exists

A deep learning compiler stack exists because one computation needs many representations.

At the top, the computation is meaningful to humans. At the bottom, it is meaningful to hardware. The representations in the middle preserve different kinds of structure. Each representation exists because some optimizations require one kind of visibility while other optimizations require another.

The essential layers are:

1. **Model layer** — functional neural network structure.
2. **Graph IR** — a graph of operators and data dependencies.
3. **Tensor IR** — loop-and-memory aware tensor programs.
4. **Hardware IR** — device-specific execution representation.
5. **Runtime** — dynamic launch, memory, and coordination decisions.

This stack is not bureaucracy. It is decomposition.

A model says *what function* is being computed. A graph IR says *which operators depend on which*. A tensor IR says *what the loop structure and memory accesses look like*. A hardware IR says *how the work is mapped to the machine*. Runtime says *how the compiled plan is actually launched and coordinated right now*.

Performance comes from preserving and exploiting the right structure at each layer.

---

## 5. The Historical Shift: Before and After the LLM Era

The history of deep learning compilers is easier to understand if we divide it into two broad regimes.

### 5.1 The pre-LLM era: diversity everywhere

From roughly 2017 to 2020, deep learning was characterized by diversity. Models included CNNs, RNNs, LSTMs, GANs, hybrid architectures, and early attention models. Operators were numerous and varied. At the same time, hardware was fragmented: NVIDIA GPUs, Google TPUs, ARM CPUs, FPGAs, mobile accelerators, Huawei Ascend, Graphcore IPUs, and more.

The core problem of this era was therefore:

> How do we write a model once and run it efficiently everywhere?

This is the world in which systems like **TVM** made the most sense. The goal was generality. The compiler had to reason about many models and many backends.

### 5.2 The LLM era: convergence, then pressure

After 2020, the landscape changed. Workloads converged around the Transformer. Hardware converged around NVIDIA GPUs and the CUDA ecosystem. This massively simplified the compiler problem in one sense: it was no longer necessary to optimize everything for everyone.

But the simplification came with new pressure. Transformer training and inference became so expensive that every percentage point of efficiency translated directly into infrastructure cost. A model developer could no longer ignore low-level performance. This is the world in which **Triton** made sense: a high-productivity GPU kernel DSL for Python users.

Then the problem changed again. Transformer-derived models and inference kernels became increasingly specialized: FlashAttention variants, grouped-query attention, MLA, quantized kernels, fused epilogues, and hardware-specific pipeline tricks. At this point, a black-box compiler abstraction was no longer enough. This is the world in which **TileLang** emerged.

Finally, as the exposed optimization space became too large for humans again, a new answer appeared: instead of handing more knobs to humans, use **LLMs plus search and reinforcement learning** to explore the discrete space. This is the world of **CUDA-L2** and the broader LLM-for-AI-infra direction.

---

## 6. TVM: The Pre-LLM Deep Learning Compiler

### 6.1 The motivation behind TVM

TVM is the canonical deep learning compiler of the pre-LLM era. Its motivation was clear and historically important.

Frameworks such as TensorFlow, Caffe, and early PyTorch depended heavily on manually optimized kernels written for specific hardware. This made the support matrix explode: many operators multiplied by many backends. Supporting all combinations manually was expensive, brittle, and slow to evolve.

TVM’s answer was to build a unified compiler stack that could take high-level models, lower them through intermediate representations, and automatically generate efficient code for diverse hardware backends. The foundational TVM paper framed this as an end-to-end optimization problem for deep learning models, not merely a local code-generation problem.

### 6.2 The TVM stack: graph, tensor, codegen, runtime

At a high level, TVM separates the compilation problem into several layers.

- At the **graph level**, TVM reasons about whole-model structure: which operators exist, how they are connected, what can be fused, which constants can be folded, and how memory can be planned.
- At the **tensor-program level**, TVM reasons about the actual implementation of an operator: loops, memory scopes, tiling, parallelization, vectorization, tensorization, and caching.
- At the **backend level**, TVM lowers the scheduled tensor program into target-specific code such as LLVM IR, CUDA, ROCm, or other backends.
- At **runtime**, TVM handles module loading, memory, launching, and coordination.

Historically, TVM’s high-level graph IR was **Relay**; more recent TVM work increasingly emphasizes **Relax** for graph-level abstraction and dynamic-shape support. On the tensor side, TVM evolved from **Tensor Expression (TE)** toward **TensorIR (TIR)** as a more explicit and transformation-friendly representation. The current TVM documentation describes Relax as the graph abstraction for ML models and TensorIR as the tensor program abstraction for loops, buffers, and hardware-related scheduling structure. 

### 6.3 Compute–schedule separation: TVM’s core intellectual move

The central idea of TVM is **compute–schedule separation**.

The compute definition describes the mathematical logic of the operator. The schedule describes how that computation is executed.

For matrix multiplication, a TE-style definition looks like this conceptually:

```python
k = te.reduce_axis((0, K), name="k")
C = te.compute(
    (M, N),
    lambda i, j: te.sum(A[i, k] * B[k, j], axis=k)
)
```

This defines *what* should be computed. It says nothing yet about tiling, loop order, memory scope, vectorization, or threading.

Then the schedule says *how*:

```python
s = te.create_schedule(C.op)
i, j = s[C].op.axis
io, ii = s[C].split(i, factor=32)
jo, ji = s[C].split(j, factor=32)
s[C].reorder(io, jo, ii, ji)
s[C].parallel(io)
s[C].vectorize(ji)
```

This tiny example expresses several deep ideas at once.

- `split` performs tiling.
- `reorder` changes loop order to improve locality and vectorization feasibility.
- `parallel` exposes parallel work.
- `vectorize` maps the innermost contiguous iteration to SIMD or vectorized execution.

TVM’s TensorIR documentation explains this general style as a sequence of transformations over tensor programs. A TensorIR program contains buffers, loop nests, and computation blocks; scheduling is the process of invoking transformation primitives such as loop reordering, partitioning, and tensorization to move the program toward an efficient implementation. 

This separation is one of the most important ideas in modern ML compilers. It decouples correctness from performance. A model or operator can remain mathematically stable while many schedules are explored automatically.

### 6.4 Graph-level optimization in TVM

At the graph level, TVM can perform hardware-agnostic transformations before detailed scheduling begins. TVM’s pass infrastructure and graph-level IR support optimizations such as operator fusion, constant folding, dead code elimination, layout alteration, and memory planning. 

#### Operator fusion

Operator fusion is among the most important graph-level optimizations. Consider

$$
y = \operatorname{GELU}(XW + b).
$$

Without fusion, `MatMul`, `Add`, and `GELU` are separate operators. Intermediate tensors are written out and read back. With fusion, these operations may be collapsed into one larger kernel so the intermediate value remains in registers or cache rather than being materialized in HBM.

The lecture’s Conv2D → BiasAdd → ReLU → BatchNorm example makes the same point. Convolution is compute-heavy. BiasAdd, ReLU, and inference-time BatchNorm are mostly elementwise and memory-bound. Fusing them reduces memory traffic and kernel-launch overhead.

The key conditions for legal and profitable fusion are:

1. One-directional data dependency.
2. Computation that can be composed into one loop nest.
3. Semantic equivalence.
4. Compatible access patterns so the same intermediate can stay on-chip.

### 6.5 Tensor-level optimization in TVM

Once we move into TensorIR, the program becomes loop-and-memory aware. This is where TVM’s schedule space resembles structured CUDA optimization.

The TensorIR documentation emphasizes that tensor programs explicitly represent buffers, loops, and blocks, and that the abstraction is designed to expose hardware acceleration options such as threading, memory access, and specialized instructions. 

Typical tensor-level schedule primitives include:

- loop splitting / tiling
- loop reordering
- parallelization / binding
- vectorization
- caching (`cache_read`, `cache_write`)
- tensorization
- software pipelining

This is why TensorIR is best understood as a structured symbolic encoding of the CUDA optimization space. The underlying degrees of freedom are similar, but the representation is designed for systematic transformation rather than handwritten code.

### 6.6 AutoTVM and MetaSchedule

Manual scheduling does not scale. The search space grows too quickly. TVM’s answer was **AutoTVM**, and later **MetaSchedule**.

#### AutoTVM

AutoTVM uses human-defined schedule templates plus machine-learning-guided search. The basic loop is:

1. Define the schedule space through parameterized templates.
2. Generate candidate schedules.
3. Compile and run them on the real device.
4. Measure latency.
5. Train or update a cost model to predict performance.
6. Use the cost model to guide further exploration.

This is the classic autotuning loop: measurement, prediction, search, feedback.

#### MetaSchedule

MetaSchedule generalizes this idea. The 2021 TVM RFC describes Meta Schedule as a probabilistic scheduling DSL on TensorIR that unifies AutoTVM and AutoScheduler-style approaches, directly operating over schedule primitives in TIR and making the automation infrastructure customizable at every layer. It is explicitly described there as TVM’s third-generation automatic scheduling system. 

The key conceptual shift is that MetaSchedule searches over **IR transformation space**, not merely over human-authored templates. This makes it more extensible: when new scheduling primitives such as tensorization or loop partitioning appear, the search system can incorporate them without requiring handcrafted templates for every operator.

### 6.7 Why TVM mattered

TVM mattered because it changed the mental model of ML systems. Instead of treating optimized kernels as fixed artifacts hand-authored for each framework and each backend, TVM treated optimization as a programmable and partially learnable compilation process.

Its strengths were exactly what the pre-LLM era needed:

- graph-level reasoning over many models,
- tensor-level scheduling over many operators,
- many backends,
- and automated tuning to survive the explosion of combinations.

Its weakness, which became more visible later, was that this broad generality comes with complexity. The stack becomes deep. Cross-layer effects become hard to reason about. Debugging becomes difficult because an optimization at one layer may hurt another.

---

## 7. Triton: The First Compiler of the LLM Era

### 7.1 Why Triton emerged

Triton belongs to a different world than TVM.

By the early LLM era, the problem had simplified. The dominant workload had converged around Transformer-style operators, and the dominant platform had converged around NVIDIA GPUs. In that regime, broad hardware generality was no longer the first priority. What mattered was enabling model developers—who mostly worked in Python—to write custom GPU kernels without becoming full-time CUDA experts.

Triton’s design goal is therefore not “optimize everything everywhere.” Its design goal is closer to:

> Give Python users a way to write fast custom GPU kernels with near-CUDA performance while hiding most low-level machinery.

The Triton documentation presents exactly this positioning. Its tutorials emphasize short, high-performance kernels for matrix multiplication, fused softmax, fused attention, grouped GEMM, and persistent matmul, all written in a Pythonic DSL that is customizable yet much more concise than raw CUDA. 

### 7.2 Triton’s program model

The core mental model of Triton is that the user writes **block-level programs**.

A Triton kernel is decorated with `@triton.jit`. The programmer writes code as if one program instance operates on one block or tile of data. Under the hood, Triton maps those program instances onto the underlying CUDA execution hierarchy.

The lecture slide comparison captured the abstraction well:

- CUDA exposes global/shared/register memory directly; Triton largely automates memory management.
- CUDA exposes blocks, warps, and threads explicitly; Triton makes the block/program the primary user-visible unit.
- CUDA requires manual tensor-core use and vectorization; Triton automates much of this.

### 7.3 `@triton.jit`, `program_id`, `arange`, and `mask`

The most important building blocks of Triton are visible even in a simple vector-add example.

- `@triton.jit` marks the function for JIT compilation.
- `tl.program_id(axis)` returns the program instance ID along a launch dimension.
- `tl.arange` creates vectorized offsets inside a block.
- `tl.load(..., mask=...)` and `tl.store(..., mask=...)` provide masked memory access for boundary handling.

Conceptually, the flow is:

1. Triton analyzes the Python kernel.
2. It specializes it to runtime constants such as block size.
3. It lowers through LLVM / MLIR / PTX-related machinery.
4. It caches the compiled binary.
5. It launches the kernel.

The lecture slides describe this explicitly: analyze the Python function, generate LLVM/PTX code, cache the binary, and launch on the GPU. The same slides emphasize dynamic specialization and caching. These are not incidental features; they are central to why Triton feels usable while still producing fast kernels. 

### 7.4 A simple way to read Triton code

A Triton kernel typically looks like this, conceptually:

```python
@triton.jit
def kernel(x_ptr, y_ptr, z_ptr, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    z = x + y
    tl.store(z_ptr + offsets, z, mask=mask)
```

Read this as follows:

- one **program instance** handles one **block** of `BLOCK` elements,
- `tl.arange` creates the vector of lane offsets inside that block,
- `mask` protects the boundary,
- and the code operates on vectors, not explicitly on CUDA threads.

This is the crucial abstraction shift. Triton hides thread-level parallelism inside vector-style block computation.

### 7.5 Triton for GEMM and attention

Triton’s matrix-multiplication tutorial states the main idea very clearly: a blocked GEMM algorithm is implemented so that each doubly nested block iteration is handled by a dedicated Triton program instance, with explicit pointer arithmetic for block loads and automatic performance tuning through configuration parameters like block sizes and warp counts. The documentation also highlights program reordering for improved L2 hit rate. 

The fused-softmax tutorial shows another key Triton strength: bandwidth-bound operators benefit from fusion because data can be loaded once, processed on-chip, and written back once, instead of materializing multiple intermediate tensors in HBM. 

The fused-attention tutorial and persistent-matmul tutorial show the next level: Triton is expressive enough to implement state-of-the-art attention and more advanced matmul variants, while still exposing autotuning over parameters such as block size, number of warps, and number of stages. 

### 7.6 Why Triton worked so well in the first LLM era

The lecture makes a subtle but important point: Triton is not magic. It worked especially well because the problem had temporarily simplified.

If the dominant workload family is Transformer-like, and the dominant target is NVIDIA CUDA GPUs, then the compiler can be highly specialized. Under those conditions, it is possible to expose a small, pleasant programming interface while still delivering strong performance.

This is why Triton could often deliver something like 80–90% of well-written CUDA performance with vastly lower development complexity for key operators such as GEMM and FlashAttention.

### 7.7 The limitations of Triton

Triton’s strength is its abstraction. Triton’s weakness is also its abstraction.

As soon as kernels become more irregular or more heavily customized, black-box scheduling begins to hit a ceiling. The user cannot directly control many of the details that become important in difficult kernels:

- exact shared-memory layout,
- swizzling patterns,
- warp specialization,
- fine-grained thread binding,
- explicit multi-stage pipelines,
- certain complex fused kernels.

Triton’s elegance comes from collapsing the optimization space. When the workload family broadens again, this collapse can become a limitation.

---

## 8. TileLang: When the Knobs Return

### 8.1 Why TileLang appeared

TileLang emerges exactly at the point where Triton’s simplicity stops being enough.

As LLM kernels became more specialized and as GPUs themselves became more complex, model developers and systems researchers increasingly needed more explicit control. The challenge was no longer merely to write “a fast custom kernel.” The challenge was to write very specific kernels whose performance depended on subtle choices in layout, pipelining, thread binding, tensor-core mapping, and shape-dependent scheduling.

TileLang’s answer is not to abandon abstraction, but to shift abstraction one level down:

> Keep tiles as the main conceptual unit, but expose more of the scheduling and memory machinery around them.

### 8.2 Tile as a first-class object

TileLang’s documentation explicitly says that a tile is the heart of its programming model: a tile is a shaped portion of data that can be owned and manipulated by a warp, a thread block, or another equivalent parallel unit. In `T.Kernel`, the execution context defines block indices and thread count, and explicit user-facing intrinsics place tile buffers in physical memory spaces such as shared memory. 

This is a decisive difference from Triton. In Triton, the user largely writes block-level logic and leaves many scheduling details to the compiler. In TileLang, the tile remains the unit of reasoning, but the user is allowed to explicitly shape more of the schedule around it.

### 8.3 What TileLang exposes

TileLang is built on top of TVM-style infrastructure and explicitly decouples dataflow from scheduling space. The TileLang paper and documentation emphasize that the scheduling space includes thread binding, layout, tensorization, and pipeline control. 

Concretely, TileLang provides user-visible primitives such as:

- `T.Kernel(...)` for kernel launch context,
- `T.alloc_shared(...)` for shared-memory allocation,
- `T.alloc_fragment(...)` for register fragments,
- `T.copy(...)` and `T.async_copy(...)` for explicit data movement across memory scopes,
- `T.Pipelined(...)` for expressing software pipelining along a loop dimension,
- `T.Parallel(...)` for parallel copying or loop parallelism,
- `T.gemm(...)` and related compute primitives,
- and layout / swizzle utilities for memory layout control.

The TileLang documentation groups the DSL around exactly these categories: data movement, compute primitives, control helpers, diagnostics, and advanced operations such as memory barriers and warp-group operations. 

### 8.4 The three levels of control

The TileLang matrix-multiplication tutorial describes multiple levels of control. Level 1 is relatively comfortable and high level. Level 2 exposes thread blocks, shared memory, fragments, pipelining, and tile-level intrinsics. Level 3 approaches hand-written CUDA/HIP-style explicitness and is intended for experts who need control over thread-level behavior or even inline PTX-like details. 

This is important conceptually. TileLang is not simply “Triton with more knobs.” It is attempting to span a wider range of programmability-performance trade-offs.

### 8.5 Memory layout: the key example

The lecture correctly emphasizes that one of the most important things TileLang exposes is **memory layout**.

In difficult kernels, the exact arrangement of tiles in shared memory or register fragments matters enormously. Bank conflicts, alignment, tensor-core fragment layout, and coalescing all depend on these details.

TileLang therefore allows the user to control:

- tile arrangement in shared memory,
- swizzle transforms to avoid bank conflicts,
- padding and alignment,
- fragment layout for tensor-core-friendly access.

The TileLang documentation includes dedicated layout and swizzle utilities, including helper layouts for GEMM fragments and linear layouts. It also documents data-movement primitives that explicitly stage data between Global, Shared, and Fragment scopes. 

### 8.6 Pipelining and warp specialization

TileLang also exposes pipeline control much more directly than Triton.

The matmul tutorial explicitly highlights `T.Pipelined(...)` to express software pipelining along the K dimension, and the broader transformation documentation includes warp-specialized producer-consumer passes that rewrite eligible pipelined loops into producer and consumer branches with explicit barrier synchronization. 

This matters because many high-performance kernels rely on overlapping data movement and compute through carefully structured pipelines. Triton exposes some of this indirectly through parameters such as `num_stages`; TileLang gives the user more direct influence over pipeline depth, buffering, and even warp-specialization structure.

### 8.7 Why TileLang is powerful—and why it is painful

TileLang is powerful because it reopens a part of the optimization space that Triton collapsed.

That is exactly why it can push beyond Triton on difficult kernels such as:

- complex attention variants,
- quantized kernels,
- kernels with unusual memory layouts,
- cross-hardware scheduling scenarios,
- fused kernels that demand explicit control over on-chip staging.

But that is also why it is painful. The developer must now reason about memory layout, thread binding, pipelining, and tensorization explicitly. The usability cost is real.

This explains the strong opinion in the lecture. From a human-factors standpoint, handing these low-level knobs back to model developers is deeply undesirable. The whole point of the compiler was to automate such search. TileLang is a useful engineering response to a real problem, but it is, in a deeper sense, a sign that automation was not yet good enough.

---

## 9. CUDA-L2: Using Learned Search to Beat Human Heuristics

### 9.1 Why CUDA-L2 matters conceptually

CUDA-L2 is not just another kernel optimizer. It represents a different idea:

> Instead of giving more optimization knobs back to humans, train a system to search the large discrete kernel design space more effectively than humans do.

The CUDA-L2 paper focuses on half-precision GEMM on A100. It combines LLMs and reinforcement learning to optimize HGEMM kernels across 1,000 \((M,N,K)\) configurations and reports beating strong baselines including cuBLAS and cuBLASLt on average in its evaluation setting. 

This matters for two reasons.

First, GEMM is the canonical workload in LLMs. Second, cuBLAS and cuBLASLt are already the result of years of expert engineering. Beating them is strong evidence that the search space still contains high-quality solutions that humans either do not consider or do not consider systematically enough.

### 9.2 The system ingredients

According to the paper, CUDA-L2 extends CUDA-L1 with several important ingredients:

1. continued pretraining on more diverse CUDA code,
2. multi-stage RL training moving from broader kernel optimization toward matmul specialization,
3. richer Nsight Compute profiling metrics in the context,
4. retrieval-augmented context to inject architectural or optimization knowledge not already present in the foundation model. 

The reward is execution speed. Candidate kernels are compiled, run, and measured on real hardware. The search is therefore grounded in actual execution behavior, not only in static reasoning.

### 9.3 Why learned search can beat human experts

The lecture’s explanation is exactly right: humans do not search the whole space. Humans search with rules.

A human expert usually carries strong priors such as:

- tile sizes should divide the tensor dimensions,
- padding is undesirable overhead,
- prefetch depth will probably be small,
- a certain swizzle is “the standard one,”
- double buffering is usually good—or usually not worth it,
- a temporary staging layout is probably necessary.

These rules are not irrational. They are compressed expertise. But they also prune the search space before the search begins.

CUDA-L2 can therefore win not because it has mystical insight, but because it is willing to try options that a human would reject early.

### 9.4 Concrete cases discovered by CUDA-L2

The lecture slides summarize six cases that make this point vividly.

#### Case 1: padding to unlock a better tile

Humans often insist that tile sizes should divide matrix dimensions cleanly. CUDA-L2 found configurations where padding slightly larger dimensions unlocked a much better tile size. In the slide example, a tile size of 160 did not divide 8192 and required only about 1.6% padding overhead, yet yielded much better performance than more “natural” tile sizes like 128 or 256. The paper reports exactly this phenomenon as a key example. 

The lesson is subtle: the local cost of padding may be much smaller than the global gain in utilization and reduced register pressure.

#### Case 2: deep prefetch, not just K+1

Standard kernels often prefetch one tile ahead in the K dimension. CUDA-L2 discovered that deeper lookahead such as K+4 could hide latency better on A100, where async copy latency is high enough that shallow prefetch is not always sufficient. 

This is a perfect example of architecture-sensitive optimization. A human may not explore deep prefetch aggressively because the interaction between latency hiding, buffering cost, and shape dependence is hard to reason about manually.

#### Case 3: direct 128-bit register → shared writes

The lecture explains this case especially well. A human often introduces a temporary staging tensor between registers and shared memory to repack or reorder data before the shared-memory write. CUDA-L2 found cases where the intermediate staging step was unnecessary: when shape, alignment, and ordering were already favorable, it used direct wide 128-bit stores from registers to shared memory. This reduced memory instructions and simplified the epilogue pipeline. 

This illustrates an important theme: human designs often include conservative staging steps “just in case.” Learned search can discover when those steps are actually redundant.

#### Case 4: double-buffered register fragments, but only when worthwhile

Double buffering is a classic technique: while one buffer feeds MMA compute, another prefetches the next tile. CUDA-L2 learned both *how* to use double buffering and *when not to use it*. The slide explicitly notes that it avoids the optimization when register pressure would overflow. 

This is exactly the kind of tradeoff humans understand conceptually but find hard to optimize precisely across many shapes: the gain from latency hiding versus the loss from reduced occupancy due to register pressure.

#### Case 5: staggered A/B prefetch scheduling

Rather than always issuing prefetch-A and prefetch-B together before compute, CUDA-L2 found cases where staggering them around compute improved instruction-level parallelism and smoothed memory traffic. 

Again, this is the sort of low-level scheduling pattern whose payoff depends on precise instruction mix and warp scheduling behavior.

#### Case 6: shape-specific CTA swizzling

CUDA-L2 also tuned CTA block swizzling differently for small, medium, and large GEMMs to improve L2 reuse. Small GEMMs often used no swizzle, medium ones small-stride swizzles, and large ones large-stride variants. 

This is a beautiful example of why learned search is attractive: there may be no single “best” swizzle policy. The best choice depends on shape and cache behavior.

### 9.5 What CUDA-L2 teaches us

CUDA-L2 teaches three important lessons.

First, the kernel optimization space remains underexplored even in mature domains like GEMM.

Second, human kernel engineering is limited not only by knowledge but also by search strategy. Humans search with strong priors and limited patience. Learned systems can search more broadly and more systematically.

Third, the right long-term answer to the complexity exposed by systems like TileLang may not be to keep burdening humans with more control. The better answer may be to let machines search those controls.

---

## 10. The Broader Emerging Direction: LLM for AI Infrastructure

CUDA-L2 is part of a broader movement. The lecture’s final slide correctly groups recent work into several clusters:

- specialized post-trained kernel models,
- agentic iterative search systems,
- hardware-aware and profiling-aware optimization,
- evolutionary or population-based search,
- cross-hardware expansion,
- and benchmarking / evaluation infrastructure.

The shared idea across these works is that AI infrastructure itself is becoming a target for AI-driven automation. The compiler is no longer only a handwritten optimization system. Increasingly, it becomes an environment in which a learned agent proposes, measures, revises, and accumulates optimization strategies.

This is a historically important shift. TVM used machine learning to guide autotuning. CUDA-L2 and related systems go further: they increasingly treat low-level optimization itself as a reasoning-and-search problem that LLM-based systems can participate in.

In that sense, the phrase from the lecture is exactly right:

> Use magic to defeat magic.

More soberly stated, use machine learning to optimize the execution of machine learning.

---

## 11. Putting the Four Systems Together

We can now compare the four landmarks more cleanly.

### TVM

TVM is the most representative **compiler-centric** system. It assumes the problem is broad, heterogeneous, and multi-layered. It therefore builds a rich IR stack, separates compute from schedule, and uses graph optimization plus tensor-level autotuning to cover a wide space.

### Triton

Triton is a **programmer-centric but compiler-assisted** system for a narrower regime. It assumes the workload/hardware pair is sufficiently constrained that a much simpler DSL can expose the right block-level abstraction and still perform well.

### TileLang

TileLang is a **control-restoring** system. It keeps a tile abstraction but re-exposes memory layout, pipelining, thread binding, and tensorization because the simplified abstraction of Triton is no longer sufficient for highly customized kernels.

### CUDA-L2

CUDA-L2 is a **learned-search** system. It says: the optimization space is too large and shape-dependent for humans to search reliably, so let an LLM+RL system search it directly using execution-time reward and profiling signals.

These are not merely four tools. They are four different answers to the question of where automation should live.

---

## 12. The Real Takeaway

The deepest lesson of this chapter is not any single compiler mechanism. It is the following.

A computation does not have one representation. It must have many.

At the model layer, we preserve semantics and design intent.
At the graph layer, we preserve global dependency structure.
At the tensor layer, we preserve loops, buffers, and scheduleable structure.
At the hardware layer, we preserve execution and resource constraints.
At runtime, we preserve dynamic coordination and launch behavior.

Different optimizations become possible only when the right structure is visible. That is why the stack exists.

The second deep lesson is that the kernel optimization problem is fundamentally a structured discrete search problem. That is why manual CUDA optimization is hard. That is why compiler stacks are layered. That is why compute–schedule separation matters. That is why Triton could temporarily simplify the problem. That is why TileLang had to re-expose more controls. And that is why LLM+RL systems such as CUDA-L2 are now so compelling.

The final lesson is historical.

Deep learning compilers have evolved not because compiler researchers arbitrarily changed taste, but because the shape of the world changed.

- When the world was heterogeneous, we needed broad compiler abstractions.
- When the world converged, we needed high-productivity specialized DSLs.
- When specialization created new complexity, we needed finer-grained control.
- When finer-grained control became inhuman to manage, we turned back to automation through learned search.

This cycle is not over.

If anything, the next generation of ML systems will likely push even harder toward a synthesis:

> keep the compiler’s structured, layered view of the problem, but let learned agents search the difficult parts of the design space automatically.

That is not only a technical direction. It is also a practical one. The optimization space is already too large for the human brain. The systems that win will be the systems that preserve structure, expose the right abstractions, and automate the search where human reasoning becomes brittle.

---
## 13. Latest Development 

Since CUDA-L2 (Dec. 2025), this area has expanded quickly, and the work now falls into a few fairly clear clusters rather than one single line of progress. The most useful way to read the space is: specialized post-trained models, agentic search systems, hardware/profiling-aware search, evolutionary search, and new benchmarks/evaluation infrastructure. A recent survey of the area organizes it in almost exactly this way and is a good umbrella reference.  

The first cluster is specialized LLMs trained for kernel generation itself. Before CUDA-L2, the key precursors were AutoTriton for Triton generation, which combined SFT with RL and execution-aware rewards, and CUDA-L1, which used contrastive RL on CUDA optimization and reported strong speedups on KernelBench. CUDA-L2 then pushed this line further for HGEMM specifically, coupling LLM generation with RL and execution-speed reward, and reported beating cuBLAS, cuBLASLt-heuristic, and cuBLASLt-autotuned baselines on A100 in its setting. After that, 2026 work has focused on making RL training more stable and less gameable: Dr. Kernel introduced KernelGYM, multi-turn RL, profiling-based rewards, and TRLOO to reduce bias in turn-level credit assignment; CUDA Agent scaled agentic RL with synthetic data, a skill-augmented environment, and stable training tricks; and DRTriton pushed large-scale synthetic-data RL for Triton kernels. The broad trend here is that the community no longer treats kernel coding as generic codegen; it is becoming its own post-training domain with custom environments, rewards, and data pipelines.  

The second cluster is agentic iterative search without heavy model training. A lot of systems now treat kernel optimization as a closed-loop program synthesis problem: generate candidate code, compile, test correctness, profile, mutate, and keep the winners. Early 2025 examples include CUDA-LLM, which used an iterative refinement loop over CUDA kernels, and NVIDIA’s DeepSeek-R1 + inference-time scaling experiment, which framed attention-kernel generation as a search problem at inference time. Then the 2026 systems became much more systematic: AutoKernel runs an autonomous optimization loop over Triton or CUDA kernels for PyTorch bottlenecks with a strong correctness harness; StitchCUDA uses planner/coder/verifier agents for end-to-end CUDA generation; and KernelFalcon uses deep agents, hierarchical decomposition, and persistent memory, reporting 100% correctness on KernelBench in Meta’s blog post. The key shift is from “LLM writes one kernel” to “LLM orchestrates a long-horizon search process.”  

The third cluster is hardware-aware and profiling-aware search. This line tries to close the gap between generic code reasoning and what human performance engineers actually do, which is reason from bottlenecks, counters, and hardware structure. SwizzlePerf is a strong example: it feeds architecture details, access patterns, profiling logs, and historical reflections into the LLM, and reported recovering expert-level swizzling decisions for GEMM while improving L2 hit rate and speed on a kernel suite. PRAGMA integrates fine-grained profiling directly into the reasoning loop and reports gains over profiling-free baselines. KernelBand casts kernel optimization as a hierarchical multi-armed bandit problem, using profiling to guide exploration/exploitation and reduce wasted trials. A very recent paper, Improving Efficiency of GPU Kernel Optimization Agents using a Domain-Specific Language and Speed-of-Light Guidance, pushes the same idea further: the abstraction the agent works in matters, and the system should stop searching once it is near hardware limits. This whole branch is moving toward bottleneck-aware search, not blind mutation.  

The fourth cluster is evolutionary and population-based search, which is arguably the fastest-growing subfield after CUDA-L2. The motivating idea is that kernel optimization landscapes are irregular, non-monotonic, and full of local optima, so maintaining a population of diverse candidates is better than doing single-chain local refinement. GPU Kernel Scientist used an evolutionary workflow for AMD MI300/HIP kernels. cuPilot introduced strategy-coordinated multi-agent evolution, using “strategy” as an intermediate semantic representation plus roofline-guided prompts. K-Search added a co-evolving world model so planning and code instantiation are partially decoupled; it reports strong gains on complex FlashInfer kernels like GQA, MLA, and MoE. KernelFoundry uses MAP-Elites quality-diversity search, meta-prompt evolution, and hardware-aware parameter tuning. AVO makes the variation operators themselves agentic and reports gains over cuDNN and FlashAttention-4 on Blackwell attention kernels. Kernel-Smith combines an evolutionary agent with a post-training recipe derived from evolutionary traces, and reports state-of-the-art overall KernelBench performance plus transfer to MetaX’s MACA backend and real integrations such as SGLang and LMDeploy. The common pattern is that search is no longer just “sample more completions”; it is becoming structured population dynamics over kernel programs.  

A fifth, very important cluster is heterogeneous-hardware and non-CUDA expansion. One of the big questions after CUDA-L2 was whether this field would remain NVIDIA/CUDA-centric. The answer now looks like no. KernelEvolve targets heterogeneous production settings across NVIDIA GPUs, AMD GPUs, Meta accelerators, and CPUs, and explicitly spans multiple programming abstractions including Triton and CuTe. KernelCraft provides a benchmark for agentic close-to-metal kernel generation on emerging hardware with new ISAs. CuTeGen pushes LLM-driven generation into the CuTe/CUTLASS-style design space. Earlier QiMeng systems also matter here: QiMeng-GEMM, QiMeng-TensorOp, and QiMeng-Attention all used LLM-guided search to generate hardware-aware operators across targets, and they were among the first works to make “LLM + structured operator optimization” concrete rather than just speculative.  

Another major development is better benchmarking and evaluation discipline. KernelBench became the central public benchmark for asking whether models can generate fast and correct kernels from PyTorch workloads. More recent work has started to attack weaknesses in evaluation: KernelCraft extends toward emerging hardware and close-to-metal ISAs; several papers stress robust correctness harnesses, multi-turn logging, and resistance to reward hacking; and the survey explicitly highlights evaluation robustness and generalization as one of the field’s main open issues. This matters because many reported wins depend heavily on whether you compare against eager PyTorch, torch.compile, cuBLAS heuristics, or fully autotuned vendor libraries, and whether you allow long search budgets.  

In summary, CUDA-L2 showed that LLM+RL can beat even heavily optimized matrix kernels in narrow settings; the next wave generalized that insight into full agentic systems that combine planning, profiling, evolutionary search, and sometimes RL post-training. The strongest recent systems are converging on a common recipe: domain-specific abstraction, rigorous correctness checking, hardware feedback, structured search over populations or multi-turn trajectories, and memory of past successful edits. The main remaining bottlenecks are sample efficiency, reliable reward design, fair evaluation against strong baselines, and cross-hardware generalization.

---

## 14. A Compact Mental Model for Students

When you study any deep learning compiler from now on, ask five questions.

### 1. What problem regime is it designed for?

Is it solving many-model/many-backend generality, or a narrow high-value workload on one dominant platform?

### 2. What representation does it operate on?

Graph IR, tensor IR, schedule IR, hardware IR, runtime—or some mixture?

### 3. What structure is visible at that level?

Global dependencies? Loop nests? Memory scopes? Thread hierarchy? Pipeline shape?

### 4. What optimization knobs are exposed, and to whom?

To the human programmer, to a schedule language, to a cost model, or to a learned agent?

### 5. What is the search strategy?

Handwritten heuristics? Templates? Cost-model-guided search? Evolution? Reinforcement learning? LLM agents?

If you can answer those five questions, you usually understand the essence of the system.

---

## 15. Selected References for Further Reading

### Core compiler systems

- Apache TVM documentation: architecture, Relax, TensorIR, and scheduling.
- TVM OSDI paper: *TVM: An Automated End-to-End Optimizing Compiler for Deep Learning*.
- MetaSchedule RFC / TensorIR auto-scheduling materials.

### Triton

- Triton documentation and tutorials: vector add, fused softmax, GEMM, fused attention, grouped GEMM, persistent matmul.
- OpenAI Triton project materials.

### TileLang

- *TileLang: A Composable Tiled Programming Model for AI Systems*.
- TileLang official documentation, especially language basics, instructions, layout/swizzle, and matmul tutorials.

### CUDA-L2 and learned kernel optimization

- *CUDA-L2: Surpassing cuBLAS Performance for Matrix Multiplication through Reinforcement Learning*.
- KernelBench and the broader LLM-for-AI-infra literature.

---

## 16. Final Summary

Deep learning compilers exist because high-level tensor computations must be mapped onto low-level hardware execution, and that mapping is a large structured optimization problem. The optimization knobs—tiling, memory placement, loop order, fusion, unrolling, thread mapping, and synchronization—form a discrete, coupled search space that is too large to solve manually at scale. The compiler stack exists because different layers preserve different structures, and different optimizations require different visibility.

TVM represents the broad, pre-LLM answer: a general compiler stack with graph IR, tensor scheduling, and autotuning across many backends. Triton represents the first LLM-era answer: a Pythonic, block-level DSL specialized enough to make custom GPU kernels practical for model developers. TileLang represents the return of exposed control once Triton’s black-box automation hits a ceiling on increasingly specialized kernels. CUDA-L2 represents the next step: letting learned systems search low-level kernel design spaces more broadly and systematically than humans do. Together, these systems tell one coherent story: the future of AI systems lies not in any single compiler abstraction, but in a layered, structured synthesis of representation, optimization, and increasingly automated search.

<script type="text/javascript" async
  src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/MathJax.js?config=TeX-MML-AM_CHTML">
</script>
