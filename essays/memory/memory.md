# From Token Sequences to Structured Knowledge Memory

## A First-Principles Derivation of a Knowledge-Structured Transformer Alternative

## 1. Why We Should Not Begin from the Transformer

When we try to design a new architecture for long-context language modeling, it is tempting to begin with the current dominant model and ask: how can we improve the Transformer? We may ask whether attention can be made sparse, whether KV cache can be compressed, whether the sequence length can be extended, or whether memory can be paged to CPU or disk. These are all useful engineering questions, but they may not be the right starting point.

The deeper starting point should be different. We should first ask: what is the inherent structure of the object we are trying to model?

A language model is trained on text, but text is not the true object of interest. Text is the observed surface form. When humans read a textbook, a paper, a codebase, or a long conversation, we do not merely store the sequence of words. We gradually form concepts, relations, dependencies, examples, claims, arguments, and procedures. We build something closer to a structured memory.

For example, when a student reads a chapter on gradient descent, the student does not simply remember the exact token sequence:

$$
x_1, x_2, \ldots, x_N.
$$

The student forms a structure: gradient descent is an optimization method; it updates parameters; the update direction is related to the gradient of the loss; the learning rate controls step size; convergence depends on properties of the objective and the update rule. This is not a flat sequence. It is a structured object.

Therefore, from first principles, we should distinguish between two things:

$$
\text{language surface} \quad \text{and} \quad \text{knowledge structure}.
$$

The language surface is the sequence of tokens the model observes. The knowledge structure is the latent object the model should infer and use.

This distinction is the foundation of the chapter.

⸻

## 2. The Surface Form: Language as a Token Sequence

The input to a standard language model is a sequence of tokens:

$$
x_{1:N} = (x_1, x_2, \ldots, x_N).
$$

Each token is mapped into a vector representation:

$$
X^{(0)} \in \mathbb{R}^{N \times d},
$$

where N is the number of tokens and d is the hidden dimension. Each row of X^{(0)} corresponds to one token position.

This representation is necessary. Language is written and spoken sequentially. A word often depends on nearby words. A phrase is built from consecutive words. A sentence has an order. If we remove sequence entirely, we lose important information.

However, sequence is not the complete structure. It is only the observation format.

Consider the sentence:

Gradient descent minimizes a loss function by updating model parameters in the direction of the negative gradient.

At the token level, this is a sequence. But at the knowledge level, it contains several objects and relations. There is a concept called “gradient descent.” There is a concept called “loss function.” There is a concept called “model parameters.” There is a relation “gradient descent minimizes loss.” There is another relation “gradient descent updates parameters.” There is a procedural relation: the update direction uses the negative gradient.

So even in one sentence, the token sequence is already an encoding of a richer structure.

A long document makes this distinction much stronger. A textbook with one million tokens is not merely a million-token chain. It contains chapters, sections, definitions, examples, cross-references, repeated concepts, prerequisite dependencies, causal relations, and evidence spans. Treating all of this as only a flat token sequence loses the central structure.

⸻

## 3. The Latent Object: Knowledge as a Graph

A natural mathematical object for representing knowledge is a graph:

$$
G = (V, E).
$$

Here V is a set of nodes and E is a set of edges. The nodes may represent concepts, entities, events, claims, definitions, procedures, examples, or states. The edges represent relations between these nodes.

For language knowledge, an edge is usually typed. A typed edge may be written as:

$$
(v_i, r, v_j),
$$

where v_i and v_j are nodes and r is a relation type. The relation type may be “defines,” “causes,” “depends on,” “is an example of,” “contradicts,” “supports,” “uses,” “part of,” or “temporally precedes.”

For example:

$$
(\text{gradient descent}, \text{minimizes}, \text{loss function}),
$$

$$
(\text{learning rate}, \text{controls}, \text{step size}),
$$

$$
(\text{overfitting}, \text{is reduced by}, \text{regularization}).
$$

This structure is fundamentally different from a token sequence. A token sequence is indexed by position. A knowledge graph is indexed by semantic units. A token sequence is linear. A knowledge graph is relational. A token sequence may mention the same concept many times. A knowledge graph can consolidate those mentions into one persistent node.

This gives us a generative view:

$$
G \sim p(G),
$$

$$
x_{1:N} \sim p(x_{1:N} \mid G).
$$

The text is generated from an underlying structured knowledge object. The task of reading is therefore an inverse problem:

$$
x_{1:N} \rightarrow G.
$$

This does not mean the model must build a symbolic graph exactly like a human-authored knowledge graph. The graph may be latent, neural, soft, and learned. But the important point is structural: the model should infer persistent semantic units and relations from the observed token sequence.

⸻

## 4. The Transformer as a Token Graph Learner

It would be wrong to say that the Transformer has no graph structure. A Transformer does construct graphs. In fact, self-attention is naturally interpretable as a dynamic weighted graph over tokens.

Given token representations:

$$
X \in \mathbb{R}^{N \times d},
$$

a self-attention head computes:

$$
Q = XW_Q,\quad K = XW_K,\quad V = XW_V.
$$

Then it forms the score matrix:

$$
S = \frac{QK^\top}{\sqrt{d_k}} \in \mathbb{R}^{N \times N}.
$$

After softmax:

$$
A = \operatorname{softmax}(S),
$$

the output is:

$$
Y = AV.
$$

For token i, this means:

$$
y_i = \sum_{j=1}^{N} A_{ij}v_j.
$$

The entry A_{ij} measures how much token i reads from token j. Therefore, at each layer and each head, attention defines a directed weighted graph:

$$
G_{\text{attn}}^{(\ell,h)} = (V_{\text{tok}}, E_{\text{attn}}^{(\ell,h)}),
$$

where the nodes are token positions and the edge weights are attention weights.

This is one reason Transformers are so powerful. They are not simple sequential models. They allow any token to communicate with any other token. A pronoun can attend to its antecedent. A theorem can attend to a previous definition. A later paragraph can attend to an earlier concept. In this sense, attention is a flexible graph construction mechanism.

Multiple Transformer layers further enrich this process. At layer \ell, the representation is:

$$
X^{(\ell)} \in \mathbb{R}^{N \times d}.
$$

A standard Transformer block updates it approximately as:

$$
X^{(\ell+1)} =
X^{(\ell)}
+
\operatorname{Attn}(X^{(\ell)})
+
\operatorname{MLP}(X^{(\ell)}).
$$

The residual path preserves previous information while each layer adds new transformations. Early layers may capture local lexical or syntactic structure. Middle layers may capture entities, phrases, and relations. Later layers may capture more abstract semantic or task-relevant features.

Therefore, the Transformer does learn structure. It builds dynamic token-level graphs and composes them layer by layer.

The question is not whether the Transformer can learn structure. The question is whether the structure it learns is the right memory object for long-context knowledge.

⸻

## 5. The Core Mismatch: Dynamic Token Graph versus Persistent Knowledge Graph

The attention graph and the knowledge graph are not the same object.

An attention edge:

$$
A_{ij}^{(\ell,h)}
$$

means:

At layer \ell, head h, token i reads information from token j.

A knowledge edge:

$$
(v_a, r, v_b)
$$

means:

Concept v_a has semantic relation r to concept v_b.

The first is an information-routing edge. The second is a semantic relation edge.

The attention graph is token-level, dynamic, dense, soft, and layer-dependent. It is recomputed at every layer. Its nodes are positions in the sequence. It is excellent for routing information during computation.

The knowledge graph is concept-level, sparse, persistent, typed, and reusable. Its nodes are semantic units. Its edges encode stable relations. It is excellent for storing and retrieving structured knowledge.

This gives the central mismatch:

$$
\boxed{
\text{Transformer attention graph}
=
\text{dynamic dense token-routing graph}
}
$$

while:

$$
\boxed{
\text{knowledge graph}
=
\text{persistent sparse semantic-relation graph}.
}
$$

The Transformer can approximate knowledge relations through token attention, but it does not explicitly consolidate repeated mentions into persistent concept nodes. If “gradient descent” appears 500 times in a textbook, the Transformer may build many local representations involving those mentions. But the KV cache still stores token-position memories. It does not naturally create one persistent concept memory node with attached evidence and relations.

This is manageable for short context. It becomes costly for million-token context.

⸻

## 6. The KV Cache as Token-Indexed Episodic Memory

During autoregressive inference, the Transformer stores keys and values from previous tokens. For L layers, sequence length N, KV dimension d_{kv}, and element size b bytes, the KV cache memory is:

$$
M_{\text{KV}}
=
2LNd_{kv}b.
$$

The factor of 2 comes from storing both keys and values.

This memory is token-indexed:

$$
M_{\text{KV}}
=
\{(k_i^{(\ell)}, v_i^{(\ell)})\}_{i=1,\ell=1}^{N,L}.
$$

For one million tokens, this becomes enormous. Suppose:

$$
N = 1{,}000{,}000,\quad L=80,\quad d_{kv}=8192,\quad b=2.
$$

Then:

$$
M_{\text{KV}}
=
2 \cdot 80 \cdot 1{,}000{,}000 \cdot 8192 \cdot 2
\approx 2.62 \text{ TB}.
$$

Even with grouped-query attention, where d_{kv} may be much smaller, the memory remains very large. If d_{kv}=1024, the KV cache is still about:

$$
328 \text{ GB}.
$$

The deeper issue is not only the memory size. The deeper issue is what is being stored. KV cache stores the history of observations. It is a token-level episodic memory. But for long-context reasoning, we often want semantic memory: concepts, relations, dependencies, and evidence pointers.

So the long-context problem is not only:

$$
O(N^2) \text{ attention is expensive}.
$$

It is also:

$$
\text{token-indexed memory is the wrong abstraction for structured knowledge}.
$$

⸻

## Part II: Deriving the Model Through Causal-Chained Questions

The design should now be derived by asking a sequence of causal questions. Each question follows from the previous one. We should not arbitrarily add components. Every component must be explained by the structure of the problem.

⸻

## Question 1: If text is only the surface, what should the model try to recover?

The model should try to recover a structured latent memory from the observed token sequence.

The observed input is:

$$
X \in \mathbb{R}^{N \times d}.
$$

But the desired internal memory should not remain purely token-indexed. It should contain semantic units:

$$
Z = \{z_1,z_2,\ldots,z_M\},
$$

where:

$$
M \ll N.
$$

Each z_a is a latent node representing a concept, entity, claim, event, procedure, or higher-level knowledge unit. The purpose of these nodes is to consolidate information that is distributed across many tokens.

This is the first architectural move:

$$
X \rightarrow Z.
$$

The model should perform token-to-concept compression.

Mathematically, one simple form is:

$$
Z = PX,
$$

where:

$$
P \in \mathbb{R}^{M \times N}.
$$

Each row of P defines how one semantic node aggregates token information:

$$
z_a = \sum_{i=1}^{N} P_{ai}x_i.
$$

If P_{ai} is large, token i contributes strongly to semantic node a. If P_{ai} is small, token i contributes little.

This is not merely pooling for efficiency. It is the mathematical expression of abstraction. When we read, we do not store every word independently. We group words into phrases, phrases into claims, claims into concepts, and concepts into larger topics.

The compression matrix P should not be fixed. A fixed chunking scheme assumes that knowledge boundaries align with token windows, but they often do not. A concept may be introduced in one paragraph, clarified in another, and used again much later. Therefore, P should be content-dependent and learned.

This turns compression into semantic assignment. The model asks:

Which observed token representations belong together because they express the same latent knowledge unit?

That is the first design principle.

$$
\boxed{
\text{The model must learn a token-to-knowledge-unit compression map.}
}
$$

⸻

## Question 2: If compression loses information, how can the model remain faithful?

Compression is necessary, but compression is dangerous. A compressed semantic node may preserve the main idea while losing exact details. In language, exact details often matter.

A number may matter. A variable name may matter. A theorem condition may matter. A code indentation may matter. A citation may matter. A definition may require exact wording. A long-context model that only stores summaries will hallucinate or blur distinctions.

Therefore, the model should not compress text into semantic nodes and discard the original evidence. It should build a two-part memory:

$$
M = M_{\text{semantic}} + M_{\text{episodic}}.
$$

The semantic memory stores compressed knowledge:

$$
M_{\text{semantic}} = Z.
$$

The episodic memory stores evidence pointers, token spans, or low-level residual information:

$$
M_{\text{episodic}} = \{R_a\}_{a=1}^{M},
$$

where R_a is the evidence associated with semantic node z_a.

For example, the node “learning rate” may point back to the span where the exact value 3 \times 10^{-4} appears. The node “gradient descent update rule” may point back to the equation defining the update. The node “regularization reduces overfitting” may point back to an explanatory paragraph and a supporting example.

This gives a key principle:

$$
z_a \rightarrow R_a.
$$

A semantic node should not be an unsupported summary. It should be grounded in evidence.

This is similar to how a careful student studies. The student builds conceptual notes, but also marks the original page where the definition or formula appears. When answering a conceptual question, the student uses the note. When exact wording or a precise value is needed, the student returns to the evidence.

Therefore, the model should compress meaning but preserve evidence.

$$
\boxed{
\text{Compression must be paired with evidence preservation.}
}
$$

⸻

## Question 3: If knowledge is relational, what must exist between semantic nodes?

A set of semantic nodes is not enough. Knowledge is not merely a bag of concepts. Knowledge lies in relations.

The model must infer edges between nodes:

$$
E = \{(z_a, r, z_b)\}.
$$

The relation type r may indicate definition, causality, dependency, contrast, support, example, temporal order, or part-whole structure.

Mathematically, we may define a relation score:

$$
e_{ab}^{r} = \phi_r(z_a,z_b),
$$

where \phi_r is a learned function for relation type r. If e_{ab}^{r} is large, the model believes that relation r exists from node a to node b.

The full relation tensor would be:

$$
E \in \mathbb{R}^{M \times M \times R}.
$$

But this tensor should be sparse. Most concepts are not directly related. A textbook may contain thousands of concepts, but each concept usually has only a limited number of direct dependencies, examples, contrasts, or applications.

Therefore, the model should keep only a small neighborhood for each node:

$$
\mathcal{N}_r(a) = \operatorname{TopK}_b \phi_r(z_a,z_b).
$$

Then message passing can occur over the sparse semantic graph:

$$
z_a' =
z_a +
\sum_r
\sum_{b \in \mathcal{N}_r(a)}
\alpha_{ab}^{r} W_r z_b.
$$

This operation is different from full self-attention over all tokens. It is concept-level graph reasoning. It allows the model to update a concept using its relevant neighbors.

The causal reason is clear. If knowledge is relational, then compressed nodes must not remain isolated. They must be connected by inferred semantic relations.

$$
\boxed{
\text{After token-to-concept compression, the model must infer sparse typed concept relations.}
}
$$

⸻

## Question 4: If knowledge is hierarchical, why is one compression step insufficient?

Human language knowledge is not only graph-like. It is hierarchical.

A textbook is organized as:

$$
\text{book}
\rightarrow
\text{chapter}
\rightarrow
\text{section}
\rightarrow
\text{subsection}
\rightarrow
\text{paragraph}
\rightarrow
\text{sentence}
\rightarrow
\text{phrase}
\rightarrow
\text{word}.
$$

Knowledge also has hierarchy:

$$
\text{field}
\rightarrow
\text{topic}
\rightarrow
\text{concept}
\rightarrow
\text{claim}
\rightarrow
\text{evidence}.
$$

Therefore, a single compression step from tokens directly to high-level concepts may be too abrupt. It may destroy intermediate structure.

Instead, the model should perform progressive abstraction:

$$
Z^{(0)} = X,
$$

$$
Z^{(1)} = C_0(Z^{(0)}),
$$

$$
Z^{(2)} = C_1(Z^{(1)}),
$$

$$
Z^{(3)} = C_2(Z^{(2)}),
$$

with:

$$
N_0 > N_1 > N_2 > N_3.
$$

At level 0, nodes are tokens. At level 1, nodes may represent local phrases or sentence-level units. At level 2, nodes may represent concepts or claims. At level 3, nodes may represent topics, procedures, or chapter-level abstractions.

Each compression step is easier than jumping directly from words to high-level knowledge. Lower levels preserve local linguistic structure. Higher levels consolidate semantic structure.

This also reduces computation gradually. Instead of doing global attention over one million tokens, the model can use local attention at the token level, then broader attention or graph reasoning at compressed levels.

For example:

$$
1{,}000{,}000
\rightarrow
100{,}000
\rightarrow
10{,}000
\rightarrow
1{,}000.
$$

The model reads locally, abstracts progressively, and reasons globally only after the number of nodes has become manageable.

This gives a hierarchical autoencoder-like structure:

$$
\text{tokens}
\rightarrow
\text{phrases}
\rightarrow
\text{claims}
\rightarrow
\text{concepts}
\rightarrow
\text{topics}.
$$

But it should not be a standard autoencoder whose only goal is reconstructing every token. Its goal is to preserve information useful for prediction, reasoning, retrieval, and evidence-grounded generation.

$$
\boxed{
\text{The model should compress progressively because knowledge is hierarchical.}
}
$$

⸻

## Question 5: If compression changes the index space, how should residual information flow?

In a standard Transformer, residual connections are straightforward because every layer has the same token index set:

$$
X^{(\ell)} \in \mathbb{R}^{N \times d}.
$$

The residual connection is:

$$
X^{(\ell+1)} = X^{(\ell)} + F(X^{(\ell)}).
$$

Token i at one layer corresponds to token i at the next layer. The residual is position-aligned.

But in a hierarchical compression model, the number of nodes changes:

$$
N_{\ell+1} < N_{\ell}.
$$

Now the representation at layer \ell+1 is not indexed by the same objects as layer \ell. A node at the higher level may represent many lower-level nodes. Therefore, ordinary residual addition is not enough.

This motivates a different idea: cross-layer memory retrieval.

Instead of passing only the immediate previous representation, the model can store previous layer representations as a multi-scale memory bank:

$$
\mathcal{M}^{(\ell)} =
\{Z^{(0)}, Z^{(1)}, \ldots, Z^{(\ell)}\}.
$$

A higher-level node can then attend back to earlier levels:

$$
\tilde{z}_a^{(\ell+1)}
=
z_a^{(\ell+1)}
+
\operatorname{Attn}
\left(
z_a^{(\ell+1)},
\mathcal{M}^{(\ell)}
\right).
$$

This solves a major problem. Higher-level nodes are semantically rich but coarse. Earlier-level nodes are fine-grained but less abstract. Cross-layer retrieval allows a coarse semantic node to retrieve fine-grained evidence when necessary.

This is related to the idea of passing previous representations as residual memory to later layers. But here the role is especially important because the model changes granularity across layers. Later layers do not merely receive a same-index residual. They query a multi-resolution memory bank.

The result is:

$$
\text{coarse semantic reasoning}
+
\text{fine-grained evidence access}.
$$

This is exactly what we want. A model should reason over compressed concepts but still access token-level or phrase-level details when needed.

$$
\boxed{
\text{When compression changes granularity, residuals should become cross-layer retrieval over multi-scale memory.}
}
$$

⸻

## Question 6: What is the role of multi-head attention in this new model?

Multi-head attention should be understood as multiple relation-extraction channels.

In a standard Transformer, each head computes its own attention matrix:

$$
A_h =
\operatorname{softmax}
\left(
\frac{Q_hK_h^\top}{\sqrt{d_h}}
\right).
$$

Different heads may learn different relational patterns. One head may track local syntax. Another may connect entities. Another may capture positional structure. Another may focus on delimiter tokens. In this sense, each head is a different way of reading the graph.

In the knowledge-structured model, heads can be interpreted even more directly as relation channels. A head may specialize in definition relations. Another may specialize in causality. Another may specialize in examples. Another may specialize in contrast or dependency.

At the concept level, attention becomes:

$$
A_h^Z =
\operatorname{softmax}
\left(
\frac{ZQ_h(ZK_h)^\top}{\sqrt{d_h}}
\right).
$$

But now the nodes are not raw tokens. They are semantic nodes. Therefore, each head builds a relation graph over concepts, not merely over token positions.

This changes the interpretation of multi-head attention. It is no longer just many views over a flat sequence. It becomes many possible relation operators over semantic memory.

However, not all heads are always needed. A factual definition may need only a few relation channels. A mathematical proof may need dependency, implication, and exception relations. A code block may need syntax, data-flow, and control-flow relations. A historical narrative may need temporal and causal relations.

Therefore, the model should not necessarily apply all heads to all nodes. It should select useful heads dynamically:

$$
\mathcal{H}(z_a,q)
=
\operatorname{TopK}_h r_h(z_a,q).
$$

This means the number of active heads becomes adaptive. The model uses different computational channels depending on the node and query.

$$
\boxed{
\text{In the structured model, heads become adaptive relation-extraction operators over semantic nodes.}
}
$$

⸻

## Question 7: What is the role of MoE experts?

MoE experts can be interpreted as specialized feature generators.

In a standard MoE layer, a router chooses a small number of experts for each token:

$$
\mathcal{E}(x_i)
=
\operatorname{TopK}_e g_e(x_i).
$$

Then the token is transformed by selected experts:

$$
x_i' =
x_i +
\sum_{e \in \mathcal{E}(x_i)}
g_e(x_i) f_e(x_i).
$$

This says: not every token needs every computation. Different tokens require different transformations.

In the knowledge-structured model, we should generalize this idea from tokens to semantic nodes. A semantic node representing a mathematical theorem may need proof-oriented experts. A code node may need program-analysis experts. A visual concept may need multimodal alignment experts. A dialogue node may need pragmatic or intent-tracking experts.

So for semantic node z_a:

$$
\mathcal{E}(z_a)
=
\operatorname{TopK}_e g_e(z_a),
$$

and:

$$
z_a' =
z_a +
\sum_{e \in \mathcal{E}(z_a)}
g_e(z_a) f_e(z_a).
$$

This gives adaptive semantic computation.

The deeper question is: how many experts should we keep? The answer should not be a fixed number chosen by convention. From first principles, an expert should be kept if it contributes useful non-redundant information.

Let S be the current memory state and u_j be the feature produced by expert or head j. The ideal criterion is conditional mutual information:

$$
I(u_j; y \mid S) > \epsilon.
$$

In words, the generated feature should help predict the target after accounting for what the memory already knows. If the feature is redundant, it should not be stored. If it is useful but too costly, it may be used only when necessary.

A practical approximation is to evaluate usefulness, diversity, and cost:

$$
\text{score}_j
=
\frac{
\text{usefulness}_j
\cdot
\text{diversity}_j
}{
\text{cost}_j
}.
$$

This gives a principled view of heads and experts. They are not simply architectural decorations. They are candidate computational modes. The model should allocate them dynamically.

$$
\boxed{
\text{Experts are adaptive feature generators; the model should keep only useful non-redundant expert outputs.}
}
$$

⸻

## Question 8: What is the complete memory object?

The complete memory should not be just a KV cache.

A standard Transformer stores:

$$
M_{\text{KV}}
=
\{(k_i^{(\ell)},v_i^{(\ell)})\}_{i,\ell}.
$$

This is large because it scales with token count and layer count:

$$
M_{\text{KV}} \sim LNd_{kv}.
$$

Our structured model should store a different object:

$$
M_{\text{structured}}
=
(Z,E,R,U).
$$

Here:

Z

contains semantic nodes.

E

contains sparse relations between nodes.

R

contains evidence pointers or residual token spans.

U

contains selected generated features from heads or experts.

The memory cost becomes approximately:

$$
M_{\text{structured}}
\sim
Md
+
|E|d_e
+
\sum_a |R_a|
+
\sum_a K_a d_s.
$$

Here M is the number of semantic nodes, |E| is the number of sparse edges, and K_a is the number of active feature slots or expert outputs stored for node a.

This memory can be much smaller than token-level KV cache if:

$$
M \ll N,
$$

$$
|E| \ll M^2,
$$

and:

$$
K_a \ll H,E_{\text{experts}}.
$$

This gives three compression axes:

$$
\text{token compression: } N \rightarrow M,
$$

$$
\text{relation sparsification: } M^2 \rightarrow |E|,
$$

$$
\text{operator sparsification: } H/E \rightarrow K_{\text{active}}.
$$

The model is therefore not only compressing tokens. It is learning a sparse computation-memory graph:

$$
\boxed{
\text{nodes} + \text{edges} + \text{operators} + \text{evidence}.
}
$$

⸻

## Question 9: How should inference work?

At inference time, the model should not scan all tokens. Instead, it should retrieve a query-conditioned subgraph.

Given query q, first score semantic nodes:

$$
s_a = q^\top z_a.
$$

Select seed nodes:

$$
\mathcal{S}_0 = \operatorname{TopK}_a s_a.
$$

Then expand through graph edges:

$$
\mathcal{S}_1 =
\operatorname{Expand}(\mathcal{S}_0,E,h),
$$

where h is a hop budget. This gives a query-relevant subgraph:

$$
G_q = (Z_q,E_q).
$$

Then retrieve evidence:

$$
R_q = \bigcup_{z_a \in Z_q} R_a.
$$

Finally generate the answer:

$$
p(y\mid q,x_{1:N})
\approx
p(y\mid q,G_q,R_q).
$$

This is the central inference assumption. The full token sequence is not directly used every time. Instead, the model uses the structured memory built from the sequence.

This is how humans use long documents. When asked a question about a textbook, we do not reread the entire book. We recall the relevant concept, follow related ideas, and check the original text only if exact evidence is needed.

$$
\boxed{
\text{Inference should be graph retrieval plus evidence-grounded decoding, not full-context scanning.}
}
$$

⸻

## Question 10: How should the model be trained?

Next-token prediction remains valuable. It provides large-scale supervision and teaches language fluency, syntax, world regularities, and many implicit structures.

But next-token prediction alone does not explicitly require the model to build compact structured memory. A model can become very good at predicting tokens while still relying on token-indexed memory.

Therefore, training should include additional objectives.

The total loss may be:

$$
\mathcal{L}
=
\mathcal{L}_{LM}
+
\lambda_1 \mathcal{L}_{QA}
+
\lambda_2 \mathcal{L}_{evidence}
+
\lambda_3 \mathcal{L}_{sparsity}
+
\lambda_4 \mathcal{L}_{compression}
+
\lambda_5 \mathcal{L}_{diversity}.
$$

The language modeling loss is:

$$
\mathcal{L}_{LM}
=
-\log p(x_{t+1}\mid x_{\leq t}).
$$

The QA or prediction loss tests whether the compressed graph preserves useful information:

$$
\mathcal{L}_{QA}
=
-\log p(y\mid q,G_q,R_q).
$$

The evidence loss trains the model to retrieve supporting spans:

$$
\mathcal{L}_{evidence}
=
-\log p(R^\star \mid q,Z,E).
$$

The sparsity loss prevents the graph from becoming dense:

$$
\mathcal{L}_{sparsity}
=
\sum_{a,b,r}|E_{ab}^{r}|.
$$

The compression loss encourages the model to use a limited number of semantic nodes and memory slots.

The diversity loss prevents heads or experts from becoming redundant. If two heads always produce the same features, one is unnecessary. Therefore, the model should reward non-redundant useful computation.

Training should therefore teach the model not only to predict text, but to read, compress, organize, retrieve, and verify.

$$
\boxed{
\text{The training objective should reward structured memory formation, not only next-token prediction.}
}
$$

⸻

## Final Derived Architecture

Putting everything together, the model follows a causally derived pipeline:

$$
\text{tokens}
\rightarrow
\text{local token representations}
\rightarrow
\text{hierarchical token-to-concept compression}
\rightarrow
\text{multi-scale residual memory}
\rightarrow
\text{sparse typed concept graph}
\rightarrow
\text{adaptive head/expert feature generation}
\rightarrow
\text{query-conditioned subgraph retrieval}
\rightarrow
\text{evidence-grounded decoding}.
$$

This is not an arbitrary architecture. Each component follows from a structural requirement.

Because text is sequential, the model begins with token representations.

Because knowledge is more compact than text, the model compresses tokens into semantic nodes.

Because compression can lose exact details, the model preserves evidence.

Because knowledge is relational, the model builds sparse typed edges.

Because knowledge is hierarchical, the model compresses progressively.

Because compression changes granularity, the model uses cross-layer retrieval rather than ordinary same-index residuals.

Because different relations and transformations are needed in different contexts, the model uses adaptive heads and experts.

Because long-context queries usually require only part of the memory, the model retrieves a relevant subgraph rather than scanning all tokens.

The final philosophical distinction is:

$$
\text{standard Transformer}
=
\text{token-indexed computation over observation history},
$$

while:

$$
\text{structured memory model}
=
\text{concept-indexed computation over inferred knowledge}.
$$

A standard Transformer remembers what was written. The proposed model tries to remember what was learned, while still preserving where the learned knowledge came from.

That is the core idea.

<script type="text/javascript" async 
  src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/MathJax.js?config=TeX-MML-AM_CHTML">
</script>
