---

# Mock Question 1 — Scaled Dot-Product Attention & Gradients (NumPy-only)

**1A. Stable softmax.**
Write a numerically stable `softmax(x, axis=-1)` without using any deep-learning frameworks. Prove it sums to 1, and explain why stability matters.

**1B. Single-head attention (vectorized).**
Given `Q, K, V` with shapes `[B, L, d]`, implement
`Attn(Q,K,V) = softmax(QKᵀ/√d) V` using only NumPy and `einsum`. Return shape `[B, L, d]`. Test with randomized inputs. (Expect them to care about shapes and `einsum` fluency.) ([NeurIPS Papers][2])

**1C. Jacobian of softmax.**
Derive the Jacobian `∂softmax(z)/∂z` and implement a function that, given `p = softmax(z)`, returns the Jacobian-vector product for an arbitrary `v`. (Efficiency matters.)

**1D. Backprop sketch.**
Given loss `L = CE(logits, y)` where `logits = Attn(Q,K,V) W_o`, derive expressions for `∂L/∂Q`, `∂L/∂K`, `∂L/∂V` (no code required, but be explicit about the role of the softmax Jacobian).

*(Covers core attention math and gradient literacy.)* ([NeurIPS Papers][2])

---

# Mock Question 2 — LayerNorm & Residual-Stream Geometry

**2A. Implement LayerNorm.**
Code `layer_norm(x, eps=1e-5)` that normalizes across the feature dimension of shape `[B, L, d]`, returning `(x - μ)/√(σ²+eps) * γ + β`. Show that the mean of the normalized activations is 0 and variance is 1 before affine. ([arXiv][3])

**2B. Invariances & effect on dot products.**
Given vectors `u, v` (token residuals), show how LayerNorm’s centering/rescaling changes `u·v`. Under what conditions is the cosine similarity preserved?

**2C. Residual stream as a feature space.**
Suppose a “feature” is represented by unit direction `f` in the residual stream. Implement utilities to (i) project tokens onto `f`, (ii) zero out that component, and (iii) measure change in logits when `f` is ablated (assuming tied unembed). Explain how this corresponds to a simple “feature ablation” in the framework’s geometry. ([Anthropic][1])

**2D. RMSNorm vs LayerNorm (theory).**
State RMSNorm and contrast its invariances vs LN. When might RMSNorm be preferable? (Short essay.) ([ACM Digital Library][4])

---

# Mock Question 3 — QK/OV Circuits & Induction Patterns

**3A. From weights to attention logits (QK circuit).**
Given `W_Q, W_K` and residuals `x_i`, compute attention logits `ℓ_{i→j} = (W_Q x_i)·(W_K x_j)/√d`. Add absolute positional embeddings `p_i` and repeat with `x_i + p_i`. Explain how including positions changes which tokens can attend. (Code + 2–3 sentences.)
*(Matches the “QK circuit” view: queries/keys produce where to attend.)* ([Anthropic][1])

**3B. OV circuit to logits.**
Given `W_V, W_O` and unembedding `W_U` (weight tying allowed), compute the induced logit update from attending to a single token `j`:
`Δlogits_i = W_U · W_O · W_V · x_j * a_{i→j}`.
Implement a function that returns `Δlogits_i` when `i` attends to `j` with weight `a_{i→j}`. (This is the “OV circuit” transforming moved information into output logits.) ([Anthropic][1])

**3C. Toy “skip-trigram” circuit.**
You’re given a one-layer, one-head toy model trained on skip-trigrams `([A] … [B] → [C])`. With small integer-sized vocab and supplied `W_Q, W_K, W_V, W_O, W_U`, write code that:

1. measures attention patterns on a minibatch,
2. computes the OV-induced logit deltas per token, and
3. verifies the model’s behavior corresponds to the intended skip-trigrams.
   Briefly explain “OV-incoherence.” ([Transformer Circuits][5])

**3D. Detect an induction head.**
Given sequences with repeated bigrams (… A B … A ? …), implement a detector that flags heads with (i) previous-token match in QK and (ii) copy-like behavior in OV (value of earlier `B` influences later logits for `B`). Provide a metric like “induction score”. *(Short code + explanation.)* ([arXiv][6])

---

# Mock Question 4 — Superposition & Causal Interventions (last part is intentionally hard)

**4A. Feature interference toy.**
Generate sparse synthetic “features” `f₁,…,f_k ∈ ℝ^d` and create residuals as sparse sums. Show how non-orthogonality (feature superposition) causes interference in simple linear readouts. Provide a plot of error vs pairwise angle. (Code.)

**4B. Mean ablation vs zero ablation.**
For a 2-layer attention-only toy model, implement two ablations on a selected edge/head: (i) set activations to 0, (ii) replace with *mean* over a “corrupted” distribution. Compare downstream effect on logits. Explain why mean ablation can be a better “do-operation” proxy. (Short write-up.) ([arXiv][7])

**4C. Patching experiment.**
Implement activation patching: run the model on a *clean* and a *counterfactual* prompt; copy a chosen intermediate activation (e.g., residual after layer ℓ and token t) from clean into the counterfactual run; report change in a target logit. Interpret the causal role of that node.

**4D. (Hard) Recover latent features via sparse coding.**
Given residual activations `X ∈ ℝ^{N×d}`, fit a small sparse autoencoder or dictionary `X ≈ A S` to decode monosemantic features. Show that some recovered directions align with your hand-crafted features from 4A (correlation > threshold). Discuss pitfalls (e.g., “hallucinated features”, centering issues). *(We don’t expect a perfect solution within time—clean partial credit is fine.)* ([Transformer Circuits][5])

---

## Extra practice prompts (shorter items to mix in)

1. Derive why the `1/√d` scale in attention stabilizes gradients when `Q,K` are approximately i.i.d. Gaussian. ([NeurIPS Papers][2])
2. Show that attention with **no positional info** is permutation-equivariant over tokens.
3. Prove the softmax temperature limit: as `τ→0`, attention tends to hard argmax; as `τ→∞`, it tends to uniform.
4. Given tied `W_U ≈ W_Eᵀ`, show that a logit for token *t* is proportional to the residual’s dot-product with the embedding of *t* (“logit lens” intuition).
5. Implement multi-head attention with `einsum`, including head merge/split.
6. Given `LayerNorm(x)`, derive `∂LN/∂x` (write a clean Jacobian-vector product). ([arXiv][3])
7. Compute attention patterns for a toy sentence and verify they sum to 1 across keys for each query. ([NeurIPS Papers][2])
8. Show how adding absolute positional encodings lets a head specialize to “previous token” vs “next token” attention.
9. Write a function that detects **name-mover heads** by correlating attention to subject positions and OV write-ins to name token logits.
10. For two heads in successive layers, show how a “previous-token head” + “copy head” composes into an **induction circuit**. Code a small example. ([arXiv][6])
11. Implement mask handling for causal self-attention and verify zero leakage from future tokens.
12. Show how centering by LayerNorm changes attention logits if `W_Q`/`W_K` are learned on centered inputs.
13. Given random `W_Q, W_K, W_V, W_O`, empirically verify that the **QK circuit** reproduces raw attention logits and **OV circuit** reproduces output logits under the right isolating conditions. (Hint: remove positional effects and other residual content.) ([Transformer Circuits][8])
14. Implement a simple residual-stream **feature patcher** that replaces a direction `f` at layer ℓ with another direction `g` and measures logit changes.
15. Show that superposition error scales with the cosine of feature overlap; plot vs sparsity. ([Transformer Circuits][5])
16. Write a toy **skip-trigram** generator and train a 1-layer attention model to solve it; then read out the learned OV mappings. ([Transformer Circuits][5])
17. Compute the contribution of a single attention edge `(i→j)` to a target logit via `QK` (selection) and `OV` (write).
18. Explain why mean ablation on *edges* can be preferable to node zeroing for isolating circuits. ([arXiv][7])
19. Show a case where attention appears “local” (head attends mostly to itself) yet OV still performs a non-trivial transformation.
20. Prove that adding a constant bias to all attention logits leaves the softmax unchanged (gauge invariance).

---

### Sources to refresh before you sit the OA

* Anthropic’s **framework overview** (residual stream, heads as linear maps + softmax, circuits): skim and internalize the geometry. ([Anthropic][1])
* **Scaled dot-product attention** and multi-head details (core formulas & shapes). ([NeurIPS Papers][2])
* **LayerNorm** definition and invariances/JVP. ([arXiv][3])
* **Superposition / attentional features / skip-trigrams** (for toy tasks & intuition). ([Transformer Circuits][5])
* **Induction heads** mechanics (previous-token + copy head composition). ([arXiv][6])

If you want, I can turn any subset of these into timed mini-exercises with hidden tests, so you can rehearse CodeSignal-style.

[1]: https://www.anthropic.com/research/a-mathematical-framework-for-transformer-circuits "A Mathematical Framework for Transformer Circuits \ Anthropic"
[2]: https://papers.neurips.cc/paper/7181-attention-is-all-you-need.pdf?utm_source=chatgpt.com "Attention is All you Need"
[3]: https://arxiv.org/abs/1607.06450?utm_source=chatgpt.com "Layer Normalization"
[4]: https://dl.acm.org/doi/pdf/10.5555/3454287.3455397?utm_source=chatgpt.com "Root Mean Square Layer Normalization"
[5]: https://transformer-circuits.pub/2023/may-update/index.html "Circuits Updates — May 2023"
[6]: https://arxiv.org/pdf/2404.07129?utm_source=chatgpt.com "What needs to go right for an induction head? A ..."
[7]: https://arxiv.org/html/2405.17969v3?utm_source=chatgpt.com "Knowledge Circuits in Pretrained Transformers"
[8]: https://transformer-circuits.pub/2024/july-update/index.html?utm_source=chatgpt.com "Circuits Updates - July 2024"
