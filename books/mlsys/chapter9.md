# **Chapter 9: Distributed Training with Data Parallelism**

Modern large language models (LLMs) have reached unprecedented scale, with parameter counts ranging from billions to trillions. While these models have achieved remarkable performance, their sheer size introduces fundamental challenges in computation, memory, and system design. A single machine can no longer accommodate the storage and compute requirements of such models. As a result, **distributed training** has become a necessity rather than an optimization. This chapter provides a systematic overview of distributed training for LLMs. We begin with a quantitative analysis of model scale, followed by an examination of compute and memory requirements. We then introduce core data-parallel training paradigms, including parameter server and all-reduce architectures, and explain collective communication techniques such as All-Reduce,  and discuss memory optimization techniques.



---

## **9.1 Quantitative Foundations of Large Models**

### **9.1.1 Parameter Scaling in Transformers**

A parameter in a neural network is simply a scalar value within a weight matrix that is learned during training. In Transformer architectures, the total number of parameters grows rapidly due to three key components:

* **Embedding Layer**:
  The parameter count is:
  $$
  N_{\text{embed}} = n_{\text{vocab}} \times d_{\text{model}}
  $$
  

  For example, a vocabulary of 50,000 and embedding dimension of 12,000 yields hundreds of millions of parameters.
  
* **Multi-Head Attention (MHA)**:
  Each head requires projection matrices for Query, Key, and Value:
  $$
  W_Q, W_K, W_V \in \mathbb{R}^{d_{\text{model}} \times d_{\text{head}}}
  $$
  
* **Layer Stacking**:
  With ( n_{\text{layers}} ) layers, total parameters scale linearly with depth.

* **Feed-Forward Network (MLP)**:

$$
N_{\text{MLP}} = 2 \cdot d_{\text{ffn}} \cdot d_{\text{model}} + d_{\text{ffn}} + d_{\text{model}}
$$

​	Using common design rules:

​	(1) \( d_{\text{ffn}} = 4 d_{\text{model}} \)

​	(2) \( d_{\text{head}} \cdot n_{\text{head}} = d_{\text{model}} \)

* **The Total Parameter Count**:

$$
N \approx n_{\text{layers}} \cdot (12 d_{\text{model}}^2 + 5 d_{\text{model}})
$$

​	This compact formula explains how models scale to hundreds of billions of parameters.



### **9.1.2 Training FLOPs**

The total computational cost is approximated as:

$$
\text{FLOPs} \approx 6 \times N \times D
$$

where:

* \( N \) : number of parameters
* \( D \) : number of training tokens

The factor 6 accounts for:

* Forward pass: ~2 FLOPs per parameter-token
* Backward pass: ~4 FLOPs

For a 175B parameter model trained on 300B tokens:

$$
\text{FLOPs} \approx 3.14 \times 10^{23}
$$



### **9.1.3 Hardware Provisioning**

Despite high theoretical GPU performance, actual utilization is limited by:

* communication overhead
* memory bandwidth
* synchronization delays

Typical **Model FLOPs Utilization (MFU)** is around 50%. Achieving a target training time (e.g., 30 days) may require thousands of GPUs operating efficiently.



---

## **9.2 Memory Wall**

### **9.2.1 Memory Breakdown**

Training requires both **static** and **dynamic** memory:

* **Static memory**:

  * Parameters (FP16)
  * Gradients (FP16)
  * Optimizer states (FP32)

* **Dynamic memory**:

  * Activations (dependent on batch size and sequence length)

For a 175B model:

* Parameters + gradients: ~700 GB

* Adam optimizer states: ~2.1 TB

* Total: **> 2.8 TB**

  

### **9.2.2 The Adam Optimizer**

The **Adam optimizer**, short for Adaptive Moment Estimation, is one of the most widely used optimizers in deep learning. Adam improves on basic gradient descent by maintaining historical information about gradients.

Standard gradient descent updates parameters using the current gradient:

\[
w_{t+1} = w_t - \alpha g_t
\]

where \(w_t\) is the current parameter, \(\alpha\) is the learning rate, and \(g_t\) is the gradient.

Adam keeps two additional quantities for each parameter.

The first is the **first moment estimate**, usually written as \(m_t\):

\[
m_t = \beta_1m_{t-1} + (1-\beta_1)g_t
\]

This behaves like momentum. It smooths the gradient direction over time and helps the optimizer move consistently through the loss landscape.

The second is the **second moment estimate**, written as \(v_t\):

\[
v_t = \beta_2v_{t-1} + (1-\beta_2)g_t^2
\]

This tracks the squared gradient. It allows Adam to adapt the learning rate for each parameter. Parameters with large or noisy gradients receive smaller effective updates, while parameters with smaller gradients may receive relatively larger updates. After bias correction, Adam updates the parameters.



### **9.2.3 The Memory Wall**

GPU memory capacity has not scaled proportionally with model size. Even modern GPUs (e.g., 80GB HBM) require:
\[
\frac{2800 \text{ GB}}{80 \text{ GB}} \approx 35 \text{ GPUs (minimum)}
\]
This fundamental limitation makes **single-node training infeasible**, driving the need for distributed systems.

### **9.2.4 Communication as the New Bottleneck**

As training scales across nodes, the system becomes **communication-bound**.

Key observations:

* Gradient synchronization can reach **hundreds of GB per iteration**
* Inter-node bandwidth (e.g., InfiniBand ~50 GB/s) is far lower than intra-node bandwidth (NVLink ~1.8 TB/s)
* Compute performance is growing much faster than network bandwidth

This imbalance makes **communication optimization critical**.



---

## **9.3 Data Parallelism** and Collective Communications

### **9.3.1 Review: Forward and Backward Propagation**

Before examining distributed training architectures, we should briefly review how a neural network learns.

In **forward propagation**, data moves from the input layer through hidden layers to the output layer. Each layer transforms its input using learned weights. In a simple neural network, an input vector may pass through one or more hidden layers before producing an output prediction.

Training then compares the prediction with the correct answer using a loss function. The goal is to adjust the parameters so that the loss becomes smaller.

In **backward propagation**, the model computes gradients of the loss with respect to each parameter. This is done using the chain rule of calculus. If a parameter affects the loss through multiple paths in the network, the gradient sums the contributions from all of those paths.

For example, suppose a weight \(w_{11}\) affects two later activations. Its total influence on the loss must account for both paths. Backpropagation automatically handles this by propagating error signals backward through the computation graph and combining them.

Distributed training does not change the mathematical goal. Each worker still performs forward and backward propagation. The difference is that the work is split across multiple devices, and the resulting gradients or activations must be communicated.

### **9.3.2 Data Parallelism Basics**

The simplest and most common distributed training strategy is **data parallelism**.

In data parallelism:

1. Each worker GPU stores a full copy of the model.
2. The training batch is split into smaller micro-batches.
3. Each GPU processes a different micro-batch.
4. Each GPU computes local gradients.
5. The GPUs synchronize gradients.
6. Each GPU applies the same parameter update.

Data parallelism is attractive because it is conceptually simple and compute-efficient. If we have 100 GPUs, each GPU can process a different portion of the batch. This increases throughput and allows us to train faster. However, data parallelism is memory-inefficient. Every GPU stores a complete copy of the model and optimizer states. If the model is small, this is fine. If the model has hundreds of billions of parameters, it becomes impossible without additional sharding methods. The second major bottleneck is synchronization. After each backward pass, all GPUs must average their gradients. If one GPU is slower than the others, the faster GPUs must wait. This is known as the **straggler problem**. 

### **9.3.3 Parameter Server Architecture**

An early approach to distributed training used the **parameter-server architecture**.

In this architecture, machines are divided into two roles:

- **Workers**, which perform forward and backward computation.
- **Parameter servers**, which store model weights and optimizer states if they exist.

A typical training step proceeds as follows:

1. Workers read the current model weights.
2. Workers compute forward propagation on local data.
3. Workers compute gradients using backward propagation.
4. Workers push gradients to the parameter servers.
5. Parameter servers aggregate gradients and update weights.
6. Workers pull the updated weights for the next iteration.

The SGD-like parameter update can be written as:

\[
W_{\text{new}} = W - \gamma \Delta W
\]

where \(\gamma\) is the learning rate and \(\Delta W\) is the aggregated gradient.

The parameter-server model is simple and was popular in earlier distributed machine learning systems. However, it has serious communication bottlenecks. Many workers send gradients to a smaller number of servers, creating many-to-one network traffic. Then the servers send updated weights back to all workers, creating one-to-many traffic. This can overwhelm network switches and reduce bandwidth efficiency. Because of these bottlenecks, modern large-scale GPU training often uses decentralized collective communication instead of centralized parameter servers. 



Distributed training can be synchronous or asynchronous.

In **synchronous training**, all workers must finish their local computation before the global update happens. This ensures that every worker uses the same model state. The disadvantage is that fast workers wait for slow workers. Even a small delay on one GPU can cause many other GPUs to idle.

In **asynchronous training**, workers do not wait for one another. A worker computes gradients, sends updates, and immediately continues. This improves hardware utilization, but it introduces **staleness**. A slow worker may compute gradients using an old version of the model. By the time its gradients are applied, the model may have already changed.

Stale gradients can harm convergence, especially for large neural networks. For this reason, large language model training usually favors synchronous training with highly optimized communication.



### **9.3.4 Collective Communication Basics**

The *Encyclopedia of Parallel Computing* defines **collective communication** as communication involving a group of processing elements, or nodes, where data is transferred among all or some of them. This transfer may also include operations such as summation, averaging, reduction, or other transformations. In distributed machine learning, collective communication is essential because many GPUs must cooperate to train one model. They need to exchange gradients, parameters, activations, or intermediate results efficiently.

**Major Benefits of Collective communication**

First, it offers a **simplified programming interface**. Developers do not need to manually implement complex synchronization or data-transfer logic. Instead, they can use standard primitives such as `Broadcast`, `Reduce`, `All-Reduce`, `All-Gather`, and `Reduce-Scatter`.

Second, it improves **scalability**. As training systems grow to hundreds or thousands of GPUs, communication becomes difficult to manage manually. Collective communication libraries are designed to move data efficiently at large scale and often include features for reliability, monitoring, and fault tolerance.

Third, it helps **separate computation from communication**. Machine learning researchers can focus on models and algorithms, while systems engineers optimize network performance, topology, and hardware utilization.

**Collective Communication in Distributed Training**

If you have used PyTorch’s `DistributedDataParallel`, you have already used collective communication. Instead of relying on a traditional parameter-server design, `DistributedDataParallel` synchronizes gradients through collective operations, often using NVIDIA’s NCCL backend. These ideas are closely related to MPI, the Message Passing Interface standard. MPI includes point-to-point operations such as `Send` and `Receive`, but distributed training usually needs group communication. For example, after each GPU computes local gradients, those gradients must be averaged across all GPUs. This is commonly done using **All-Reduce**.

**All-Reduce as a Synchronization Barrier**

An **All-Reduce** combines values from all nodes and distributes the final result back to every node. In data-parallel training, it ensures that every GPU receives the same averaged gradient before the optimizer updates the model. Because every GPU must participate, All-Reduce also acts as a **synchronization barrier**. If Node 0 finishes its backward pass at time \(T_1\), but Node 2 finishes at \(T_3\), Node 0 must wait. This waiting time is wasted compute. Such delays can come from hardware jitter, network variation, or imbalanced workloads. A slow worker becomes a **straggler**, reducing overall cluster efficiency. Therefore, balanced batch sizes and evenly distributed work are critical.

**Communication Patterns**

Collective operations can be grouped by data-flow pattern:

- **One-to-many**: `Broadcast`
- **Many-to-one**: `Gather`, `Reduce`
- **Many-to-many**: `All-Gather`, `Reduce-Scatter`, `All-Reduce`

Many-to-one operations can create **incast congestion**, where too much traffic flows into a single node or switch port. This resembles the bottleneck in parameter-server systems. Modern GPU clusters are therefore optimized for many-to-many operations. Libraries such as NCCL break tensors into smaller chunks and move them through ring, tree, or topology-aware algorithms over high-speed links such as NVLink and InfiniBand. Among these operations, **All-Reduce is the most important for data-parallel training** because it drives gradient synchronization. Its efficiency often determines how well an AI cluster is utilized.



### **9.3.5 Collective Communication Operations**

We hereby describe the most common collective operations: `Broadcast`, `Gather`, `Scatter`, `Reduce`, `All-Reduce`, `All-Gather`, `Reduce-Scatter`, and `All-to-All`.



**Broadcast**

A **Broadcast** operation sends data from one selected GPU, called the **root**, to every other GPU in the group. Suppose we have four GPUs, labeled rank 0 through rank 3. In the initial state, rank 2 is the root. Only rank 2 contains the data blocks \(a_2\), \(b_2\), \(c_2\), and \(d_2\). The other ranks do not yet have this data.

During the broadcast, rank 2 sends a copy of its data to every other rank. After the operation completes, all four GPUs contain the same data blocks. Rank 0, rank 1, rank 2, and rank 3 now all hold identical copies of \(a_2\), \(b_2\), \(c_2\), and \(d_2\). The key idea is simple: **Broadcast copies one GPU’s data to all GPUs**. 



**Gather**

A **Gather** operation performs the opposite type of movement: instead of sending data outward from one GPU, it collects data inward onto one GPU.

Suppose four GPUs each hold a different shard of data:

- GPU0 holds \(A_0\)
- GPU1 holds \(B_0\)
- GPU2 holds \(C_0\)
- GPU3 holds \(D_0\)

These shards may be pieces of a larger tensor or dataset. In a Gather operation, one GPU is chosen as the destination, or root. If GPU0 is the root, then GPUs 1, 2, and 3 send their shards to GPU0. After the operation completes, GPU0 holds the complete collection: \(A_0\), \(B_0\), \(C_0\), and \(D_0\). The other GPUs do not receive the full result.

The key idea is: **Gather collects distributed data shards onto one selected GPU**.

Gather is useful when data has been computed or stored across multiple devices, but one device needs the complete result for further processing.



**Scatter**

A **Scatter** operation is the reverse of Gather. Instead of collecting many shards into one GPU, Scatter distributes shards from one GPU to many GPUs.

Suppose GPU0 initially holds a complete dataset divided into four shards:

- \(A_1\)
- \(B_1\)
- \(C_1\)
- \(D_1\)

The other GPUs are initially empty. During Scatter, GPU0 sends one shard to each GPU. Unlike Broadcast, Scatter does not copy the entire dataset to everyone. Instead, each GPU receives a different piece.

After the operation completes:

- GPU0 keeps \(A_1\)
- GPU1 receives \(B_1\)
- GPU2 receives \(C_1\)
- GPU3 receives \(D_1\)

The key idea is: **Scatter divides data from one GPU and sends different shards to different GPUs**.

This operation is useful when a large tensor needs to be split across multiple GPUs so they can work in parallel.



**Reduce**

A **Reduce** operation not only moves data but also performs a mathematical operation during communication.

Suppose four GPUs each hold corresponding data chunks. For example:

- Rank 0 holds \(A_0, B_0, B_0, D_0\)
- Rank 1 holds \(A_1, B_1, C_1, D_1\)
- Rank 2 holds \(A_2, B_2, C_2, D_2\)
- Rank 3 holds \(A_3, B_3, C_3, D_3\)

These chunks could represent partial gradients computed independently by each GPU. During Reduce, the system applies an aggregation operation to corresponding chunks. Common operations include:

- Sum
- Minimum
- Maximum
- Average

For example, using summation, all \(A\)-chunks are added together:

\[
A_0 + A_1 + A_2 + A_3
\]

The same is done for the \(B\), \(C\), and \(D\) chunks. If rank 2 is the root, the final reduced results are stored only on rank 2:

\[
A_0+A_1+A_2+A_3
\]

\[
B_0+B_1+B_2+B_3
\]

\[
C_0+C_1+C_2+C_3
\]

\[
D_0+D_1+D_2+D_3
\]

The other ranks do not store the final result. The key idea is: **Reduce combines corresponding data across GPUs and stores the result on one selected GPU**.



**All-Reduce**

All-Reduce is one of the most important collective operations in distributed deep learning. It extends Reduce by making the final result available to every GPU.

The initial state is the same as Reduce. Each GPU holds its own local chunks, such as local gradients:

- Rank 0 holds \(A_0, B_0, B_0, D_0\)
- Rank 1 holds \(A_1, B_1, C_1, D_1\)
- Rank 2 holds \(A_2, B_2, C_2, D_2\)
- Rank 3 holds \(A_3, B_3, C_3, D_3\)

The system aggregates corresponding chunks. Unlike Reduce, the final aggregated result is distributed back to every GPU. After All-Reduce completes, every rank holds the same reduced data.

The key idea is: **All-Reduce combines data from all GPUs and gives the final result to all GPUs**. This is essential in data-parallel training. Each GPU computes gradients from its own mini-batch. All-Reduce averages those gradients so that every GPU updates its model using the same global gradient. This keeps all model replicas synchronized.



**All-Gather**

**All-Gather** is an extension of Gather. In Gather, all shards are collected onto one root GPU. In All-Gather, the complete collection is made available to every GPU.

Suppose the initial data is distributed as follows:

- Rank 0 holds \(A_0\)
- Rank 1 holds \(B_1\)
- Rank 2 holds \(C_2\)
- Rank 3 holds \(D_3\)

Each GPU has only one piece of the overall data. During All-Gather, every GPU shares its shard with all other GPUs. After the operation completes, every GPU holds the complete assembled dataset:

\[
A_0, B_1, C_2, D_3
\]

The key idea is: **All-Gather collects distributed shards and places the complete result on every GPU**. This operation is useful when each GPU initially owns only part of a tensor, but all GPUs need the full tensor for the next computation.



**Reduce-Scatter**

**Reduce-Scatter** combines two operations: Reduce followed by Scatter. Like Reduce, it begins with each GPU holding corresponding chunks:

- Rank 0 holds \(A_0, B_0, B_0, D_0\)
- Rank 1 holds \(A_1, B_1, C_1, D_1\)
- Rank 2 holds \(A_2, B_2, C_2, D_2\)
- Rank 3 holds \(A_3, B_3, C_3, D_3\)

First, the operation reduces corresponding chunks:

\[
A_0+A_1+A_2+A_3
\]

\[
B_0+B_1+B_2+B_3
\]

\[
C_0+C_1+C_2+C_3
\]

\[
D_0+D_1+D_2+D_3
\]

Then, instead of storing the full reduced result on one GPU or copying it to all GPUs, the reduced chunks are scattered across the ranks.

For example:

- Rank 0 receives the reduced \(A\)-chunk
- Rank 1 receives the reduced \(B\)-chunk
- Rank 2 receives the reduced \(C\)-chunk
- Rank 3 receives the reduced \(D\)-chunk

The key idea is: **Reduce-Scatter aggregates data and then distributes one reduced shard to each GPU**. This is very useful for memory-efficient distributed training because no GPU needs to store the entire reduced tensor.



**All-to-All**

**All-to-All** is the most general and complex collective communication pattern discussed here. It is a coordinated exchange where every GPU sends data to every other GPU. 

Suppose each GPU starts with multiple chunks:

- GPU0 has \(A_1, B_1, C_1, D_1\)
- GPU1 has \(A_2, B_2, C_2, D_2\)
- GPU2 has \(A_3, B_3, C_3, D_3\)
- GPU3 has \(A_4, B_4, C_4, D_4\)

The goal is to reorganize the data by category. Conceptually, GPU \(i\) sends its \(j\)-th chunk to GPU \(j\). This is similar to performing a *matrix transpose* across the network.

After All-to-All completes:

- GPU0 holds all \(A\)-chunks: \(A_1, A_2, A_3, A_4\)
- GPU1 holds all \(B\)-chunks: \(B_1, B_2, B_3, B_4\)
- GPU2 holds all \(C\)-chunks: \(C_1, C_2, C_3, C_4\)
- GPU3 holds all \(D\)-chunks: \(D_1, D_2, D_3, D_4\)

The key idea is: **All-to-All redistributes data so that every GPU sends to and receives from every other GPU**. This operation is useful when data must be completely reshuffled across devices. It is especially important in workloads such as mixture-of-experts models, where tokens may need to be routed to different expert networks.



### **9.3.6 Ring All-Reduce**

One of the most important collective communication operations in distributed training is **All-Reduce**. It is widely used to synchronize gradients across GPUs in data-parallel training. Several algorithms can implement All-Reduce, including:

- **Ring All-Reduce**
- **Tree All-Reduce**
- **Topology-aware All-Reduce**

Among these, **Ring All-Reduce** is one of the most common and easiest to understand.



**Ring All-Reduce**

Ring All-Reduce organizes GPUs into a logical ring. Suppose we have four GPUs: **A, B, C, and D**. Data moves in one direction around the ring:
\[
A \rightarrow B \rightarrow C \rightarrow D \rightarrow A
\]

This structure avoids the bottleneck of a centralized parameter server and utilizes all the link bandwidths. Instead of every GPU sending gradients to one central machine, each GPU communicates only with its neighbor. This distributes network traffic evenly across the system. After backpropagation, each GPU has computed its own local gradients using its local mini-batch. To synchronize training, these local gradients must be summed or averaged across all GPUs. To make communication efficient, each GPU divides its gradient tensor into equal chunks. With four GPUs, each tensor is split into four chunks. For example:

- GPU 0 holds \(A_0, B_0, B_0, D_0\)
- GPU 1 holds \(A_1, B_1, C_1, D_1\)
- GPU 2 holds \(A_2, B_2, C_2, D_2\)
- GPU 3 holds \(A_3, B_3, C_3, D_3\)

The goal is for every GPU to eventually obtain the complete reduced gradient:

\[
A_i + B_i + C_i + D_i
\]

for every chunk \(i\). Ring All-Reduce achieves this in two phases:

\[
\text{All-Reduce} = \text{Reduce-Scatter} + \text{All-Gather}
\]



**Phase-1: Reduce-Scatter**

The first phase is called **Reduce-Scatter**. This is where the actual reduction, or summation, happens. In each communication step, every GPU sends one chunk to its neighbor and receives one chunk from the previous neighbor. After receiving a chunk, the GPU adds it to its own corresponding chunk.

For example, in the first step:

- GPU A sends \(a_0\) to GPU B.
- GPU B sends \(b_1\) to GPU C.
- GPU C sends \(c_2\) to GPU D.
- GPU D sends \(d_3\) to GPU A.

After receiving the data, each GPU adds the incoming chunk to its local matching chunk. For example, GPU B now has:

\[
a_0 + b_0
\]

This happens simultaneously on all GPUs. Every GPU is sending, receiving, and adding at the same time. Because communication is spread around the ring, no single GPU or network link becomes a central bottleneck.

In the next step, GPUs pass along the partial sums. For example, GPU B sends \((a_0 + b_0)\) to GPU C, and GPU C adds \(c_0\), producing:

\[
a_0 + b_0 + c_0
\]

This process continues for \(N-1\) steps, where \(N\) is the number of GPUs. With four GPUs, the Reduce-Scatter phase takes three steps. At the end of this phase, every GPU holds one fully reduced chunk. For example:

- One GPU holds \(a_0 + b_0 + c_0 + d_0\)
- Another holds \(a_1 + b_1 + c_1 + d_1\)
- Another holds \(a_2 + b_2 + c_2 + d_2\)
- Another holds \(a_3 + b_3 + c_3 + d_3\)

The full reduced tensor now exists, but it is distributed across the GPUs. Each GPU has only one piece of the final answer.



**Phase-2: All-Gather**

The second phase is called **All-Gather**. In this phase, no more addition is needed. The computation is complete. The only remaining task is to share the fully reduced chunks with every GPU. Each GPU passes its completed chunk around the ring. In every step, each GPU sends one reduced chunk to its next neighbor and receives one from its previous neighbor. After another \(N-1\) steps, all GPUs have received all completed chunks. At this point, every GPU has the same fully synchronized gradient tensor. This means every GPU is ready to apply the same optimizer update and continue to the next training iteration.



**Communication Cost of Ring All-Reduce**

Let:

- \(N\) be the number of GPUs.
- \(S\) be the total size of the gradient tensor.

Each tensor is split into \(N\) chunks, so each chunk has size: \[\frac{S}{N}\]. The Reduce-Scatter phase requires: \[N - 1\]communication steps. The All-Gather phase also requires: \[N - 1\] communication steps. Therefore, a complete Ring All-Reduce requires: \[2(N-1)\] communication steps.

In each step, each GPU sends one chunk of size: \[\frac{S}{N}\]. So the total traffic per GPU is:

\[
\frac{2S(N - 1)}{N}.
\]

As \(N\) becomes large, the fraction \((N-1)/N\) approaches 1. Therefore, the total communication per GPU approaches: \[2S\]. This is the key advantage of Ring All-Reduce. Even as the number of GPUs increases, the amount of data sent by each GPU remains almost constant. This makes Ring All-Reduce highly scalable for large distributed training jobs.



**Effectiveness of Ring All-Reduce**

Ring All-Reduce is effective because it avoids the traffic pattern that made parameter-server architectures inefficient. In a parameter-server system, many workers send data to one central server, creating an incast bottleneck. In Ring All-Reduce, communication is evenly distributed. Each GPU communicates only with its neighbors. At every step, all GPUs are active, and all links can be used in parallel. This produces balanced network utilization and avoids overloading a single node.

The algorithm can be summarized as:

\[
\text{All-Reduce} = \text{Reduce-Scatter} + \text{All-Gather}
\]

The first phase performs the reduction. The second phase distributes the completed results. Together, these phases allow a plethora of GPUs to synchronize gradients efficiently during large-scale neural network training. 



### **9.3.7 Alternative All-Reduce Algorithms**

**Tree All-Reduce**

We just covered Ring All-Reduce. Now let’s look at an alternative: Tree All-Reduce. In the diagram, the GPUs are arranged like a **tournament bracket** rather than a continuous circle. Instead of passing data around a ring, GPUs pair up and combine their data in rounds. 

Upward phase:

- In the first round, **GPU 0 → GPU 1** and **GPU 2 → GPU 3**.  
- Each pair adds its data and keeps the accumulated result (“winners”).
- Next, the winners pair again (e.g., **GPU 1 sends to GPU 3**), so GPU 3 ends up holding the fully reduced result at the top of the tree.

Downward phase:

The upward phase only completes half the job. In the downward phase, the final answer must be propagated back down the tree:
- The top GPU sends the result to the next level (e.g., **GPU 3 → GPU 1**),
- and then those nodes pass it further to the remaining GPUs (e.g., to **GPU 0 and GPU 2**).

Performance tradeoff:

- Because it’s a **binary tree**, the number of rounds is about: \[\log_2 N\], and with upward + downward, total rounds become roughly: \[2\log_2 N\].
- The big drawback is data volume per round. In a naive Tree All-Reduce, you often transmit the entire payload \(S\) every round (or some links are idle). This yields total traffic roughly proportional to:\[2S\log_2 N\]. As \(N\) grows, the traffic per GPU increases, and some GPUs remain idle while others work near the tree root.
- That lack of overlap and growing traffic is why Ring often beats naive Tree on bandwidth efficiency for large tensors. 



**Butterfly All-Reduce**

Next is Butterfly All-Reduce, which gets its name from the crossing communication pattern in the diagram. Compared to Tree, Butterfly ensures that all GPUs participate in every round. It does this through pairwise exchanges that change partners over time.

- Step 1: GPU 0 pairs with GPU 1, and GPU 2 pairs with GPU 3 (exchange + add).
- Step 2: partners shift to longer distances (e.g., GPU 0 with GPU 2, GPU 1 with GPU 3).
- After \(\log_2 N\) such steps, the fully reduced result is available everywhere.

Performance tradeoff:

- Number of rounds is still about: \[\log_2 N\].
- But the total per-GPU traffic improves relative to naive Tree because Butterfly avoids a separate downward “broadcast” phase. The overall communication is roughly: \(S\log_2 N\).
- Even so, traffic still grows with \(N\). That’s why Ring remains popular when the number of GPUs is large and bandwidth dominates.



**Rabenseifner: the “best of both worlds”**

Now for Rabenseifner’s algorithm, which combines:
- the latency/step efficiency feel of Butterfly, and
- the bandwidth efficiency idea of Ring.

***--Aggregation phase (top row)***

Like Butterfly, GPUs pair up like a tournament—but instead of sending the full \(S\) each time, they progressively send **smaller halves**:
- First exchanges happen with about \(S/2\)
- then \(S/4\),
- then \(S/8\), etc.

At the end of aggregation, the computation is complete, but the results are **scattered** across GPUs rather than all fully replicated.

***--Collection phase (bottom row)***

Then the same process runs in reverse to redistribute the final result so that every GPU gets what it needs.

***--Why it’s efficient***

- The number of rounds is about: \[2\log_2 N\]. (aggregation + collection).
- The total traffic works out to an approximately flat, optimal bandwidth level—because payload sizes shrink each round and form a **geometric series**.
- The key result: Rabenseifner matches the optimal bandwidth behavior associated with Ring, while using **fewer communication steps** than Ring.



**Comparing algorithms with the \(\alpha\)-\(\beta\) model**

To choose among these algorithms, we need more than “total bytes.” We must account for *how many messages* we send. Using the classical \(\alpha\)-\(\beta\) model:

\[
T = \alpha + \beta S
\]
- \(\alpha\): **startup latency** (fixed overhead per message/round)
- \(\beta\): **time per unit data** (inverse bandwidth)
- \(S\): message size

***Ring vs Rabenseifner***

- **Ring** uses about \(2(N-1)\) rounds, so it pays the \(\alpha\) startup cost many times.
  - If messages are huge, Ring’s bandwidth efficiency can dominate positively.
  - But if there are many GPUs or smaller chunks, startup overhead becomes painful.
- **Rabenseifner** uses about \(2\lceil \log_2 N\rceil\) rounds:
  - far fewer startups (\(\alpha\) paid fewer times),
  - while keeping traffic at roughly the same bandwidth-efficient level as Ring.

***Practical takeaway***

- Use **Ring** when it’s easy and effective (often: smaller clusters, very large tensors).
- Use **Rabenseifner** when you want to reduce the number of communication rounds—especially on massive systems or when message chunk sizes are small.



**Topology-aware Collective Communication**

So far, we implicitly assumed all GPUs are equally connected. In real LLM training clusters, the all-reduce operation is usually more complicated, and is affected by the datacenter network topology.

***The hardware hierarchy***

- **Inside a server**: GPUs connect via very fast links (e.g., NVLink).
- **Across servers**: communication uses slower networking (e.g., RDMA over InfiniBand/Ethernet).

If you build one giant Ring across 1,000+ GPUs, the whole system can get throttled by the **slowest inter-server links**.

***Solution: Hierarchical All-Reduce***

Hierarchical All-Reduce scales by splitting the operation into three stages:

1. **Intra-node reduce**  
   GPUs within each server reduce using fast NVLink.  
   One “master” GPU per server holds the server’s partial result.

2. **Inter-node all-reduce**  
   Only the master GPUs communicate across servers using RDMA.  
   This reduces the volume crossing slow network links.

3. **Intra-node broadcast**  
   Masters broadcast the final reduced result back to the other GPUs inside their servers.

This hierarchy is the practical reason distributed training can scale to **thousands of chips** without the network becoming the bottleneck. 



---

## **9.4 Memory Optimization**



### **9.4.1 Memory Optimization**

A classic joke captures the situation:

> **How do you put an elephant into a fridge?**  
> *Cut the elephant into pieces, and distribute the pieces across fridges.*

In this metaphor:

- The **fridge** is the limited GPU memory (often on the order of 80 GB HBM).
- The **elephant** is the total memory footprint required to train an LLM.

Once models grow to billions or trillions of parameters, the training “elephant” no longer fits into a single GPU’s memory, and training fails immediately with an Out-of-Memory (OOM) error. So the central challenge becomes: *how do we restructure memory use so the model state fits across a cluster?*

In standard Data Parallelism (DP), every GPU keeps an identical copy of the relevant training state:

- **Parameters** (blue),
- **Gradients** (orange),
- **Optimizer states** (green)—the large auxiliary buffers used by the optimizer (e.g., Adam’s momentum and variance terms).

If you try to train a modern LLM with DP, you are effectively trying to place the *entire* elephant into *each* fridge. For small models, this approach can be workable. For modern LLMs, it is not: replication multiplies memory consumption until no GPU can hold its copy. That is why we need a fresh new idea: instead of duplicating the whole model state on every GPU, we must **partition (“shard”) it across GPUs**. During training, the memory footprint is dominated by three objects:

1. **Parameters** \(P\)
2. **Gradients** \(G\)
3. **Optimizer states** \(O\)

To express the scale compactly, let:

- \( \Psi \) = total number of model parameters,
- \( N_d \) = number of GPUs/devices.

For typical optimizers like Adam, the optimizer states are large relative to the parameters—often around **12–14×** the parameter size—so reducing redundancy here yields the biggest gains.



### **9.4.2 ZeRO: Zero Redundancy Optimizer**

ZeRO (Zero Redundancy Optimizer) is designed to eliminate redundant storage in DP by **sharding** model state across GPUs. Instead of every GPU holding a full copy of everything, ZeRO provides a spectrum of stages:

- **ZeRO-1**: shard optimizer states
- **ZeRO-2**: shard optimizer states + gradients
- **ZeRO-3**: shard optimizer states + gradients + parameters (full sharding)



**Memory model across ZeRO stages**

Using a simplified memory accounting, ZeRO’s stages can be summarized as:

**ZeRO-1 (optimizer-state partitioning):**
\[
\text{Memory per GPU} \approx 2\Psi + \frac{14\Psi}{N_d}
\]
- The \(2\Psi\) term corresponds to the replicated pieces (parameters and gradients).
- The \( \frac{14\Psi}{N_d} \) term is the sharded optimizer states.

***ZeRO-2 (optimizer-state + gradient partitioning):***
\[
\text{Memory per GPU} \approx 2\Psi + \frac{14\Psi}{N_d}
\]
The expression may look similar, but the *meaning* changes: ZeRO-2 expands sharding so gradients are also partitioned, further reducing the memory that each GPU must hold during synchronization/update.

***ZeRO-3 (full parameter/gradient/optimizer sharding):***
\[
\text{Memory per GPU} \approx \frac{16\Psi}{N_d}
\]
Here, parameters, gradients, and optimizer states are all sharded. As \(N_d\) grows, the per-GPU footprint can shrink dramatically—small enough that models far beyond a single GPU’s memory become trainable.



**The Fundamental Tradeoff**

- **ZeRO-1 and ZeRO-2** mainly reduce memory while keeping the overall computation structure relatively close to DP.
- **ZeRO-3** reduces memory the most, but it significantly changes how computation is performed: since *no GPU permanently holds the full parameters*, GPUs must temporarily assemble the parameter shards they need for each computation step.

That increased need for communication is the price paid for memory savings.



**ZeRO-1 and ZeRO-2: what changes in the training step?**

A key design choice in ZeRO-1 and ZeRO-2 is that:

> **Model parameters are still replicated across GPUs.**

This keeps the **forward** and **backward** computations uncomplicated. Each GPU can run the standard math locally because it already has the full parameter set. So what changes? Primarily the *gradient synchronization and update* logic. Instead of performing one monolithic **AllReduce** for gradients, ZeRO uses a mathematically equivalent decomposition:
\[
\text{AllReduce} \equiv \text{ReduceScatter} + \text{AllGather}
\]

The benefit is that this structure helps avoid storing whole global gradient objects on every device at once.



**ReduceScatter**: keep only what you need.

During **ReduceScatter**, each GPU:
1. contributes its locally computed gradients,
2. participates in a global reduction,
3. receives only a **shard** of the resulting global gradient.

This is memory-friendly because no GPU must store the entire gradient tensor for every parameter.

**Update**: apply gradient shard to optimizer/parameter shard.

After gradient shards are obtained:
- each GPU updates its relevant optimizer portion,
- and then updates its parameter state accordingly.

Finally, when needed for the next iteration’s replicated computations, ZeRO-1/2 perform an **AllGather** so that all GPUs again have consistent replicated parameter information.

**Summary for ZeRO-1 vs ZeRO-2**

- ZeRO-1 shreds **optimizer states**: gradients still require synchronization, but optimizer memory is partitioned across GPUs.
- ZeRO-2 further shreds **gradients** during synchronization: GPUs receive only the gradient shards required for their parameter/optimizer shard updates, reducing memory pressure even more.

In both cases, the core win is memory reduction in the optimizer-related state and synchronization buffers.



**ZeRO-3**: : full sharding and why forward/backward must change.

ZeRO-3 is where ZeRO becomes truly transformative.

> Parameters are no longer replicated. They are sharded permanently across GPUs.

As a result, the straightforward DP story breaks:

- A GPU cannot compute a layer if it does not already own the parameters for that layer.
- Therefore, GPUs must retrieve parameter shards dynamically **just in time**.

***Forward pass: assemble parameters "layer-by-layer"***

During the forward pass:
1. For a given layer (or parameter block), GPUs determine which parameter shard(s) are owned by which GPU.
2. GPUs perform an **AllGather-like** operation so that every GPU temporarily has the parameter pieces required to compute that layer.
3. After computation, temporary parameter buffers can be discarded to free memory.
4. The process repeats for the next layer.

This is the operational meaning of ZeRO-3’s sharding: *parameters exist across the cluster, but are assembled temporarily for computation.*

***Backward pass: same problem, repeated parameter transmission***

Backward propagation has the same constraint: gradients for a layer require the corresponding parameter values (or activations saved from earlier). Since parameters are sharded, the backward pass again must fetch parameter shards dynamically, compute local gradients, and then release temporary buffers.

***ReduceScatter: gradient aggregation stays sharded***

Once local gradients for parameter shards are computed, ZeRO-3 uses *ReduceScatter* so that:

- gradients are globally aggregated,
- but the result remains distributed as **gradient shards**.

This ensures no single GPU becomes a gradient-memory hotspot. 

***Update phase: shard-local updates only, no final AllGather***

Because both *parameters and optimizer states are sharded*, the update step is naturally distributed:
- each GPU updates only the parameter and optimizer shards it owns,
- and there is no need to reconstruct the entire parameter set afterward.

That absence of a post-update full parameter AllGather is a crucial reason ZeRO-3 achieves its best scaling.

***Practical consideration on inplementations***

Implementations (notably DeepSpeed) typically refine the above “idealized” communication pattern in two ways:

1. **Prefer AllGather to naive broadcast**
   - In practice, broadcast can create bottlenecks because one device does all the heavy communication.
   - AllGather distributes the communication workload across devices more evenly.
   
2. **Intra-layer partitioning**
   - Rather than assigning entire layers to different GPUs (which can lead to idle time),
   - partition large tensor operations *within a layer* so GPUs keep working while the next communications happen, and all these GPUs are sunchronized in their forward and backward propagations. 

These engineering decisions reduce practical stalls while preserving ZeRO-3’s memory efficiency.



### **9.4.3 Further Optimization: ZeRO-Offload/ZeRO-Infinity/ZeRO++**

Sometimes, even ZeRO-3 cannot fit the training state into GPU memory—especially on small numbers of GPUs while the model is large. *ZeRO-Offload* addresses this by moving the most memory-intensive parts (typically optimizer states) to CPU memory. ZeRO-Offload recognizes a hardware imbalance: i) GPUs have *fast* memory (HBM) but limited capacity; ii) CPUs have *large* memory (system RAM) but slower access. The GPU focuses on forward/backward compute; after backward produces gradients, gradients are transferred to the CPU for the optimizer update, then updated parameters are sent back.

**What offload changes**

- GPUs still run the fast **forward/backward** computations.
- After backward, gradients are transferred to the CPU (often over PCIe).
- The CPU performs optimizer updates using the optimizer states stored in system RAM.
- Updated parameters are transferred back to GPUs via PCIe for the next forward pass.

This prevents OOM, but introduces a major cost: *CPU↔GPU communication is much slower than GPU↔GPU communication*.

**The cost: lower GPU utilization**

Offload resolves OOM but introduces a new bottleneck: communication between CPU and GPU. On many systems:

- GPU-to-GPU interconnect (e.g., NVLink) can be extremely fast, while CPU↔GPU transfers across **PCIe** are significantly slower.

As a result, training traces show:

- GPU utilization drops, the GPU becomes idle while waiting for CPU-side updates and parameter transfers.

Offload is therefore a capacity tool: it increases what you can train, but reduces how fast you can train. It is best seen as a *capacity vs speed tradeoff* tool.



**ZeRO-Infinity**

Offload helps when CPU memory is enough. But what if the model is so large that CPU RAM is still insufficient? ZeRO Infinity extends the memory hierarchy one more step: it uses *NVMe SSD storage* as the fallback memory layer. The storage path becomes a hierarchy like:

$$
\text{GPU HBM} \rightarrow \text{CPU RAM} \rightarrow \text{NVMe}.
$$

This effectively offers “virtually infinite” capacity from the perspective of storage size, enabling training models at scales where neither GPU memory nor CPU memory alone would suffice. However, NVMe introduces much higher latency and lower bandwidth than HBM or RAM. Therefore, ZeRO Infinity depends on sophisticated data orchestration to keep GPUs fed with data and minimize idle time. The key message remains consistent:

> ZeRO-style systems trade communication and data movement complexity for memory capacity.



**ZeRO++: Low-Bandwidth Scenario**

In distributed clusters across wide-area networks or with limited interconnect bandwidth, traditional collective communication (AllGather, ReduceScatter) becomes expensive. *ZeRO++* introduces three key optimizations for low-bandwidth scenarios:

***Quantized Weight Communication:***

- Convert FP16 weights to INT8 before All-Gather, reducing data volume by 50%.
- Dequantize back to FP16 after gathering.
- Acceptable accuracy loss for training.

***Quantized Gradient Communication:***

- Apply similar quantization to gradients during Reduce-Scatter.
- Combines with block-wise quantization strategies to minimize precision loss.

***Hierarchical Partitioning:***

- Structure the parameter sharding hierarchy to minimize inter-node communication.
- Local (within-node) communication uses faster interconnects.
- Reduce inter-node all-gather volume by communicating along a hierarchy.

As the result, ZeRO++ reduces communication volume while maintaining training convergence.



### **9.4.3 Activation Recomputation**

Memory optimization extends beyond just managing model parameters, gradients, and optimizer states. While these components constitute the 'static content' of our GPU's High Bandwidth Memory (HBM), there's a dynamic, often equally substantial, consumer of GPU resources: **activations**. These are the intermediate outputs of every single layer in our neural network during the forward pass. For instance, in a Transformer model, activations encompass elements like the Query (Q), Key (K), and Value (V) projections, the attention scores, and the outputs of various feed-forward layers. They are 'dynamic' because their memory footprint is not solely fixed by the model's architecture; instead, it heavily depends on run-time factors like the *batch size* and the *sequence length* of the input data. 

**Why Storing Activations?**

The reason activations consume so much memory, and why they cannot simply be discarded, lies in the fundamental requirements of the training process, specifically backpropagation.

*Referencing the `dL/dW = dL/dy * x` equation):* During Forward Propagation, data flows through the network, generating these intermediate activations at each layer. For the Backward Propagation step, which calculates the gradients needed to update our model weights, these activations are indispensable. The chain rule dictates that to compute the gradient of the loss with respect to a weight (`dL/dW`), we need both the gradient flowing back from the subsequent layer (`dL/dy`) AND the exact input (`x`) that entered the current layer during the forward pass. This is why the caption explicitly states: "Cannot be discarded after FP" (Forward Propagation). These values need to be stored in HBM until the backward pass reaches their respective layer to perform the gradient computation. Without them, the gradient calculation is impossible.

**Quantifying the Activation Memory Footprint**

The memory footprint of activations can be shockingly large, often exceeding the capacity of a single GPU, even after aggressively sharding other model states. The activations (depending on batch size) is a variable yet often enormous memory size. For a deep network like a large Transformer, storing all intermediate activations for every single layer across an entire batch can easily consume a large volume of GPU memory. This creates a distinct and challenging memory bottleneck. Even if ZeRO-3 meticulously shards parameters, gradients, and optimizer states, you can still encounter an Out-Of-Memory (OOM) error simply from the sheer volume of activations. To further quantify this, let's look at the memory cost within a typical *Transformer Layer*, the fundamental building block of modern LLMs. We use our standard variables:

*   `b`: **batch size** (e.g., 3.2M tokens/sequence length for the example scenario, meaning total tokens = batch_size * sequence_length)
*   `s`: **sequence length** (e.g., 2048 tokens)
*   `h`: **hidden layer dimension** (e.g., 12288)
*   `a`: **attention head number**
*   `L`: **number of layers** (e.g., 96 layers)

Tracing the memory accumulation within *one* such layer (assuming FP16 precision):
*   The input `X` to the Self-Attention block has dimensions `bsh`, consuming `bsh * 2 Bytes`.
*   Within the *Attention Block*, the process of projecting Query, Key, Value, and computing attention scores generates significant intermediate activations. The `5abss` term captures attention-score-related activations, which critically scale *quadratically with sequence length (`s^2`)*. For long sequences, this term becomes a dominant factor (late on it is alleviated by attention optimization algorithm such as FlashAttention). Other linear intermediates contribute `11bsh`.
*   The subsequent *MLP Block* further adds to the load. Linear layers and activation functions (like GeLU) generate more activations. For instance, a `Linear (h->4h)` output adds `2bsh`, a `GeLU` output adds `8bsh`, and a final `Linear (4h->h)` output adds `bsh`.

Summing these up for a single Transformer Layer, we find an approximate total of **`5abss + 34sbh`**. Given the example values (`b: 3.2M tokens/sequence length` with `s: 2048`, `h: 12288`, and `L: 96 layers`), this cumulative activation memory across all layers can easily exceed *hundreds of gigabytes or even a terabyte*. This monumental memory requirement for activations renders them a primary bottleneck, even after ZeRO-family optimizations. 



**Activation Recomputation (Gradient Checkpointing)**

When storing all activations becomes impossible, we turn to *Activation Recomputation*, commonly known as *Gradient Checkpointing*. This technique offers a strategic trade-off: **trading computation for memory**.The most aggressive form is **'Full activation re-computation'**. In this approach, during the forward pass, we save only the initial input to the entire network and discard almost all intermediate activations. If an activation is needed during the backward pass but wasn't saved, the network simply recomputes all necessary intermediate steps from the last saved point (or the original input) on the fly. This results in **'Minimal memory occupation'**, but at a significant cost: **Prolonged training time (doing forward propagation once again)** by 30%~40%. For LLMs trained for weeks or months, this overhead is substantial.

To mitigate the computational cost of full recomputation while retaining its memory benefits, **strategic recomputation** (or selective checkpointing) is employed. The goal is to significantly reduce memory occupation while slightly increasing training time.
Let's understand the visual cues from the diagram:

*   **Light green blocks:** Represent parts of the model where activations are **'Recomputed during backprop'**. These are computations whose intermediate outputs are *not* saved during the forward pass.
*   **Purple blocks:** Represent parts where activations are **'Fully saved during fprop'** (forward propagation). Their intermediate outputs *are* stored.
*   **Red triangle:** Indicates a **'Saved activation'** or a checkpoint. This is a strategic point where an activation is explicitly stored to prevent needing to recompute everything before it.

*Full Checkpointing:* This aligns with the 'Full activation re-computation' concept but applied at a block level. Imagine each large rectangle containing 'LN' (Layer Normalization), 'MHA' (Multi-Head Attention), and 'MLP' (Multi-Layer Perceptron) as a complete Transformer block or layer. With full checkpointing:

*   We save only the activation entering each major block (indicated by the red triangle at the input of each block).
*   All intermediate activations *within* that block (the light green components like LN, MHA, MLP) are *not* saved.
*   During the backward pass, if the gradient signal needs an activation from within a light green block, the entire block's forward pass is recomputed from the nearest red triangle checkpoint. This offers maximum memory savings at the cost of recomputing entire blocks.

*Selective Checkpointing:* This strategy provides a more nuanced balance.

*   Instead of only saving the input to the entire block, we might choose to **'Fully save'** activations for certain sub-modules within the block (shown in purple). For example, if the Multi-Head Attention (MHA) block generates particularly large activations or is computationally expensive to recompute, we might save its internal activations (purple).
*   For other sub-modules within the same block, such as some Layer Norms or MLPs (shown in light green), we allow their activations to be **'Recomputed during backprop'** if needed.

This level of control is often exposed through framework settings, as hinted by the phrase *'setting activations_checkpoint_granularity = selective or full'*. Practitioners can choose 'full' for maximum memory savings or 'selective' to fine-tune the trade-off. In a nut shell, we can recompute the layers that possess large memory footprints but are computationally light-weight. 

---

<script type="text/javascript" async
  src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/MathJax.js?config=TeX-MML-AM_CHTML">
</script>
