# Chapter 8: Transformer Basics



## 8.1 The Transformer Ecosystem: A Bird's-Eye View

The field of artificial intelligence has entered an era defined by **Large Language Models (LLMs)**—systems of breathtaking scale and capability that are reshaping how we interact with machines. But to train these models, we need a deep understanding of the underlying architecture and, critically, a mastery of the systems engineering required to distribute the enormous computational workload across hundreds or thousands of accelerators (GPUs or NPUs).


Before diving into the mechanics, it is worth appreciating how pervasive the Transformer architecture has become. Today, a single library—Hugging Face's `transformers`—provides a unified interface for an enormous variety of tasks, all powered by Transformer-based models. These tasks span multiple modalities:

- **Text:** text generation, text classification, summarization, translation, feature extraction
- **Vision:** image-to-text, image classification, object detection
- **Audio:** automatic speech recognition, text-to-speech, audio classification

The key abstraction offered by this library is the `pipeline()` function, which connects a pre-trained model with all of its necessary preprocessing and postprocessing steps. Consider the following minimal example of sentiment analysis **(taken directly from a Huggingface book)**:

```python
from transformers import pipeline

classifier = pipeline("sentiment-analysis")
classifier("I've been waiting for a machine learning systems course my whole life.")
# Output: [{'label': 'POSITIVE', 'score': 0.9598}]
```

This deceptively simple code conceals a remarkable amount of complexity. Under the hood, three distinct stages are executed automatically:

1. **Preprocessing:** Raw text is tokenized and converted into numerical representations that the model can process.
2. **Model Inference:** The numerical inputs are passed through the Transformer network, which performs millions (or billions) of floating-point operations.
3. **Post-processing:** The raw numerical outputs are decoded and mapped back into human-readable labels and scores.

By default, this pipeline selects and downloads a pretrained model if not already cached locally. This illustrates a central philosophy of modern NLP: leverage the immense investment of pre-training on massive datasets and adapt the result to specific tasks with minimal additional effort.



---

## 8.2 A Brief History of Transformer Models

To understand why distributed training is so critical, we must first retrospect how rapidly model scale has grown. The modern era of NLP traces its lineage to a paper (2017): *"Attention is All You Need"* by Vaswani et al., which introduced the base Transformer architecture and proved that attention mechanisms is sufficient to achieve state-of-the-art results in machine translation.

The years that followed saw an explosion of increasingly capable models, organized roughly as follows:

| Period | Key Developments |
|---|---|
| **2017** | Original Transformer (*"Attention is All You Need"*) |
| **2018–2019** | Foundational models: GPT, BERT, GPT-2, T5 |
| **2020** | GPT-3 — the emergence of true *Large* Language Models (175B parameters) |
| **2021–2022** | InstructGPT, FLAN, ChatGPT — alignment and instruction-following |
| **2023–2024** | GPT-4, LLaMA, LLaMA-3.1 (405B), GPT-4o — open and closed frontier models |
| **2025** | Advanced reasoning models: OpenAI-o1, DeepSeek-V3, DeepSeek-R1 |

The critical insight embedded in this timeline is one of *scale*. Moving from left to right, model sizes have grown from tens of millions of parameters to hundreds of billions—and in some cases, trillions. A model like GPT-3 has 175 billion parameters; LLaMA-3.1 reaches 405 billion. **No single GPU in existence can hold these models in memory, let alone train them.** This is precisely the problem that distributed training techniques are designed to solve, and it is the central engineering challenge of this chapter.


---

## 8.3 Transformer Preprocessing

Modern LLMs fall into two broad training paradigms, which differ considerably in how they use context.

### 8.3.1 Causal Language Models

A **causal language model** respects the arrow of time: it can only observe past tokens when predicting the next one. It cannot peek into the future. The training objective is elegantly simple—given a sequence of words, predict the next word:

\(\text{Given: "My"} \rightarrow \text{Predict: "name"}\)
\(\text{Given: "My name"} \rightarrow \text{Predict: "is"}\)
\(\text{Given: "My name is"} \rightarrow \text{Predict: "Sylvain"}\)

During training, this process is repeated over billions or trillions of tokens. At each step, the model makes a prediction, compares it against the ground truth in the training data, and updates its weights via backpropagation. The simplicity of this objective belies its computational cost: performing these operations at the scale required for modern LLMs demands extraordinary infrastructure. Models in this family include GPT-3, GPT-4, LLaMA, and virtually most of modern generative AI systems.

### 8.3.2 Non-Causal Language Models

A **non-causal language model** takes a different approach: it reads the entire sequence simultaneously, with access to context from both the left *and* the right. Rather than predicting the next word, these models are trained using **Masked Language Modeling**. A word in the input is randomly replaced with a special `[MASK]` token, and the model must use the surrounding context to reconstruct it:

\[
\text{"My [MASK] is Sylvain ."} \rightarrow \text{Predict: "name"}
\]
This bidirectional reading comprehension makes non-causal models exceptionally powerful for tasks that require understanding an entire input—such as search, semantic embedding and text classification. The canonical example of this family is **BERT**. Regardless of which training paradigm is employed, the underlying matrix operations are enormous, and the need for distributed infrastructure remains exactly the same.

### 8.3.3 From Text to Tokens: Tokenization

Having established what a Transformer learns, we now turn to *how* it learns—tracing the flow of data through the complete architectural stack, from raw text at the input to probability distributions at the output. Neural networks cannot read English, i.e. they operate exclusively on numbers. The first step of the pipeline is therefore to convert raw text into a sequence of integer IDs, a process called **tokenization**.

Crucially, tokens do not always correspond to complete words. A tokenizer uses a vocabulary of subword units, so that common words map to a single token, while rare or complex words are split into multiple subword pieces. For example:

- `"bought"` → single token
- `"indivisible"` → two tokens: `"indiv"` + `"isible"`
- `"."` (punctuation) → its own token

Each unique token in the vocabulary is assigned a specific integer ID. A vocabulary of 50,000 tokens means that every piece of text can be represented as a sequence of integers drawn from the range $[0, 49{,}999]$.

### 8.3.4 One-Hot Encoding

The integer IDs produced by the tokenizer are still not suitable for direct mathematical manipulation. The next step is **one-hot encoding**: each integer ID $i$ is converted into a sparse vector of length $V$ (the vocabulary size), where every entry is zero except for a single $1$ at position $i$.

For a token with ID 3687 in a vocabulary of 50,000:

\[
\mathbf{v} = [\underbrace{0, 0, \ldots, 0}_{3686}, \underbrace{1}_{3687\text{th}}, 0, \ldots, 0] \in \mathbb{R}^{50{,}000}
\]
While conceptually clean, this representation is extremely inefficient: a vector of 50,000 dimensions where 49,999 entries are zero. Feeding such sparse, high-dimensional vectors directly into a neural network would be computationally prohibitive.

### 8.3.5 Token Embedding

The solution to the sparsity problem is the **embedding layer**. We define a learnable **Embedding Matrix** \[W_E \in \mathbb{R}^{V \times d}\], where \[V\] is the vocabulary size and \[d\] is the (much smaller) embedding dimension. Multiplying the one-hot vector by \[W_E\] is equivalent to performing an index lookup—it selects a single row from the matrix:

\[
\mathbf{e} = \mathbf{v} \cdot W_E \in \mathbb{R}^{d}
\]
The result is a dense, \[d\]-dimensional vector of floating-point numbers—the **embedded token**. Instead of thousands of zeros, the token is now represented as a rich, compact vector. This is the format that flows through the rest of the Transformer.

The embedding space is where the model learns semantic relationships. Through training, the geometry of this space comes to reflect meaning: the vector for "dog" will be placed closer to the vector for "cat" than to the vector for "car". Every single token in the sequence is independently embedded in this way.

### 8.3.6 Positional Encoding

After embedding, we face a subtle but critical problem: attention mechanisms are inherently **permutation invariant**—they do not inherently know the order of tokens. To inject sequential information, we add **Positional Encodings** to the embedded token vectors. These encodings are fixed mathematical functions (or learned parameters) that vary with position, ensuring the model can distinguish "Tom likes Jerry" from "Jerry Likes Tom."



---

## 8.4 The Self-Attention Mechanism

The heart of the Transformer—and the source of both its power and its computational cost—is **self-attention**. This mechanism allows every token in the sequence to directly attend to every other token, capturing long-range dependencies that sequential models like RNNs struggle to learn.

### 8.4.1 Queries, Keys, and Values

To compute attention, each embedded token \[\mathbf{x}_i\] is projected into three distinct roles via three separate learned weight matrices:
\[
\mathbf{q}_i = \mathbf{x}_i W_Q, \quad \mathbf{k}_i = \mathbf{x}_i W_K, \quad \mathbf{v}_i = \mathbf{x}_i W_V
\]
where \[W_Q, W_K \in \mathbb{R}^{d \times d_k}\] and \[W_V \in \mathbb{R}^{d \times d_v}\], with \[d_k\] as the query/key dimension and \[d_v\] as the value dimension. The intuition behind these three roles is best understood through a bigdata analogy:

- **Query \(\mathbf{q}\):** What the token is *looking for*. For example, the word "watch" might query for information about what kind of watch it is.
- **Key \(\mathbf{k}\):** What the token *contains*. The word "apple" holds a key indicating it could be a brand or a fruit.
- **Value \(\mathbf{v}\):** The actual *meaning* that will be passed along if a match is made between a query and a key.

In practice, the projections for all tokens are computed simultaneously. We stack the token vectors into a sequence matrix \[X\] and compute:

\[
Q = X W_Q, \quad K = X W_K, \quad V = X W_V
\]
This is a dense matrix multiplication—exactly the type of workload that GPUs are optimized for.

### 8.4.2 Scaled Dot-Product Attention

Given the \[Q, K\], and \[V\] matrices, the attention output is computed as:

\[
\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{Q K^\top}{\sqrt{d_k}}\right) V
\]
Let us unpack each step:

1. **\(Q K^\top\):** The matrix multiplication of Queries and Keys produces an \[N \times N\] matrix of raw similarity scores, where \[N\] is the sequence length (i.e. the number of input tokens of a single request). Entry \[(i, j)\] measures how strongly token \[i\] should attend to token \[j\].
2. **Scaling by\(\frac{1}{\sqrt{d_k}}\):** Without scaling, the dot products can grow very large in magnitude as \[d_k\] increases, pushing the softmax into regions of extremely small gradients. Dividing by \[\sqrt{d_k}\] helps stabilize training.
3. **Softmax:** Converts the raw scores into a probability distribution along each row, so that the attention weights for each token sum to 1.
4. **Multiplication by \[V\]:** The final output is a weighted sum of Value vectors, where the weights are the attention probabilities. Tokens with high attention scores contribute more to the output.

> **Note:** The \[QK^\top\] computation produces an \[N \times N\] matrix. For a sequence of length \[N\], this requires \[\mathcal{O}(N^2)\] memory. As modern LLMs push context windows to 100,000 or even 200,000 tokens, this quadratic memory requirement becomes a severe bottleneck. Today's training and inference frameworks do not compute the entire attention matrix at once. 

### 8.4.3 Multi-Head Attention

A single attention computation can only look for one type of relationship at a time. But natural language is rich with simultaneous, overlapping structure: grammatical agreement, coreference, semantic similarity, temporal ordering, and more.

**Multi-head attention** addresses this by running \[h\] attention computations in parallel, each with its own smaller set of projection matrices. If the full hidden dimension is \[d\], each head operates on a truncated dimension \[d_k = d/h\]:
\[
\text{head}_i = \text{Attention}(Q W_Q^{(i)},\, K W_K^{(i)},\, V W_V^{(i)})
\]

\[
\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_0, \text{head}_1, \ldots, \text{head}_{h-1}) \cdot W_O
\]

Each head specializes independently, and their outputs are concatenated and linearly projected by \[W_O\] to combine all the diverse contextual information into a single, unified representation. This parallel computation is both the source of the Transformer's representational richness and a significant contributor to its memory footprint.

### 8.4.4 Causal Masking in the Decoder

For autoregressive generation (predicting the next token one at a time), it is essential that the model *cannot* attend to future tokens. This is enforced through a **masking matrix** \[M\] added to the pre-softmax attention scores:

\[
\text{MaskedAttention}(Q, K, V) = \text{softmax}\!\left(\frac{Q K^\top}{\sqrt{d_k}} + M\right) V
\]
The masking matrix \[M\] is constructed so that:
- Entries in the **lower triangular** (past and present positions) are \[0\] — the scores are unaffected.
- Entries in the **upper triangular** (future positions) are \[-\infty\] — since \[e^{-\infty} = 0\], the softmax assigns exactly zero weight to all future tokens.

\[
M = \begin{pmatrix} 0 & -\infty & -\infty & -\infty & -\infty \\ 0 & 0 & -\infty & -\infty & -\infty \\ 0 & 0 & 0 & -\infty & -\infty \\ 0 & 0 & 0 & 0 & -\infty \\ 0 & 0 & 0 & 0 & 0 \end{pmatrix}
\]

This single elegant operation transforms the Transformer from a bidirectional reader into a strict unidirectional generator—a fundamentally different computational behavior achieved with no architectural change, only a mask.



---

## 8.5 Feed-Forward Networks

After the self-attention sub-layer, each token representation passes through a **position-wise Feed-Forward Network (FFN)**. Unlike attention, which operates globally across the sequence, the FFN is applied independently and identically to each token:

\[
\text{FFN}(\mathbf{x}) = \text{ReLU}(\mathbf{x} W_1 + \mathbf{b}_1) W_2 + \mathbf{b}_2
\]
The defining characteristic is the **hidden dimension expansion**. If the input has dimension $d$, the first projection \[W_1 \in \mathbb{R}^{d \times 4d}\] expands it to \[4d\]. After applying the non-linear ReLU activation, the second projection \[W_2 \in \mathbb{R}^{4d \times d}\] maps it back down to \[d\].

This $4\times$ expansion factor has a staggering implication for parameter counts. In a standard dense Transformer, **approximately two-thirds of all model parameters reside in the FFN blocks**. For a 70-billion parameter model, tens of billions of those weights sit inside these FFN matrices \[W_1\] and \[W_2\].



---

## 8.6 Stacking Layers and Architectural Families

A single Transformer block consists of a Multi-Head Self-Attention sub-layer followed by a Feed-Forward sub-layer, each wrapped with a residual connection and layer normalization. The full model is constructed by **stacking many such blocks sequentially**:

\[
\mathbf{x}^{(l+1)} = \text{TransformerBlock}(\mathbf{x}^{(l)})
\]
Modern LLMs stack dozens to hundreds of these blocks. If a single block contains roughly 1 billion parameters, an 80-layer model contains approximately 80 billion parameters. The fundamental elements we have described—embedding, attention (with or without masking), and FFN—can be assembled into three distinct architectural families, each with a different purpose:

### 8.6.1 Encoder-Only Models
These models stack unmasked self-attention blocks. Because every token can attend to every other token bidirectionally, the encoder builds an extraordinarily rich representation of the entire input. This makes encoder-only models ideal for tasks requiring deep comprehension: sentiment analysis, text classification, and semantic search, and an archetypal example is **BERT**.

### 8.6.2 Decoder-Only Models
These models use masked self-attention exclusively. Each token can only attend to its predecessors, enabling autoregressive text generation. A Linear projection and Softmax layer at the top convert the final hidden state into a probability distribution over the vocabulary, from which the next token is sampled. The archetypal examples are **GPT-3**, **GPT-4**, and the **LLaMA** family. Today, this is the dominant architecture for large-scale generative AI.

### 8.6.3 Encoder-Decoder Models
This is the original Transformer design from the 2017 paper, tailored for sequence-to-sequence tasks like machine translation or summarization. The encoder stack reads the full input sequence and produces a set of rich contextual representations \[\mathbf{e}_1, \ldots, \mathbf{e}_n\]. The decoder stack then uses these encoder representations—via a cross-attention mechanism—alongside its own masked self-attention to generate the output sequence one token at a time. The archetypal example is **T5**.

---

## 8.7 Generating Output

After passing through all the Transformer layers, a dense hidden vector \[\mathbf{x} \in \mathbb{R}^d\] emerges from the final block. This vector encodes all the context the model has computed, but must be converted back into a human-readable word. This final stage involves three steps:

**Step 1: Linear Projection to Logits**

The hidden vector is multiplied against a large projection matrix \[W \in \mathbb{R}^{d \times V}\] (often tied to the embedding matrix \[W_E\]):

\[
\text{Logits} = \mathbf{x} \cdot W^\top + \mathbf{b}
\]
The result is a vector of \[V\] raw scores (logits), one for each word in the vocabulary. A logit of 150 means the model considers that word highly likely; a logit of \[-40\] means it is considered extremely unlikely.

**Step 2: Softmax to Probability Distribution**

The raw logits are converted into a proper probability distribution using the Softmax function:

\[
P_{\text{word}_i} = \frac{e^{u_i}}{\sum_{j=1}^{V} e^{u_j}}
\]
This ensures all probabilities are non-negative and sum to exactly 1, giving us a clean distribution over the entire vocabulary.

**Step 3: Decoding/Sampling**

Given the probability distribution, we must select the next token. Two main strategies exist:
- **Greedy decoding:** Always select the word with the highest probability. Simple and deterministic, but tends to produce repetitive, robotic text.

- **Sampling:** Treat the probabilities as a weighted die and sample from the top candidates. This introduces variability and creativity, producing the more natural, diverse outputs we expect from modern LLMs.

  

---

## 8.8 The Imperative for Distributed Training

Having traced the complete computational graph of a Transformer—from tokenization and embedding, through self-attention and FFN layers, to logit projection and sampling—we are now positioned to understand precisely *why* training these models at modern scales requires distributed infrastructure.

Consider the memory requirements for training a single Transformer layer:
- **Model weights** must be stored in GPU memory.
- **Gradients** of the same size as the weights must be maintained for backpropagation.
- **Optimizer states** (e.g., first and second moment estimates in Adam) add further multiples of the weight size.
- **Intermediate activations** from the forward pass must be retained for gradient computation.

For a model with 80 billion parameters, the memory demand runs into the terabytes—far exceeding the ~80 GB VRAM of even the commonly used GPUs today. The distributed training techniques we will cover in subsequent sections—Data Parallelism, Collective Communication Libraries, and Memory Optimization—are the engineering solutions to this fundamental constraint. 

<script type="text/javascript" async
  src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/MathJax.js?config=TeX-MML-AM_CHTML">
</script>
