# SOMA-S1 → Architecture

SOMA-S1 is a 12-layer decoder-only Transformer released through the DeepseekV2 architecture class (`boore1/SOMA-S1/config.json`: `"architectures": ["DeepseekV2ForCausalLM"]`, `"model_type": "deepseek_v2"`). Every figure below is counted directly from the released weight tensors, not estimated.

## Configuration (released `config.json`)

| Group | Key | Value |
|---|---|---|
| Geometry | `hidden_size` | 768 |
| | `num_hidden_layers` | 12 |
| | `num_attention_heads` | 12 |
| | `head_dim` / `qk_head_dim` | 32 / 64 |
| | `qk_nope_head_dim` / `qk_rope_head_dim` | 32 / 32 |
| | `v_head_dim` | 64 |
| | `intermediate_size` (dense MLP) | 2048 |
| MLA | `q_lora_rank` | 128 |
| | `kv_lora_rank` | 128 |
| MoE | `n_routed_experts` | 18 |
| | `n_shared_experts` | 1 |
| | `num_experts_per_tok` | 2 |
| | `moe_intermediate_size` | 128 |
| | `topk_method` | `greedy` |
| | `norm_topk_prob` | `false` (raw top-2 softmax weights, no renormalization) |
| | `routed_scaling_factor` | 1.0 |
| | `router_aux_loss_coef` | 0.001 |
| | `first_k_dense_replace` | 0 (every layer is MoE; no dense-first layers) |
| Norm / act | RMSNorm (`rms_norm_eps` 1e-6, pre-norm), SiLU | |
| Position | RoPE, theta 10,000, `max_position_embeddings` 2048 | |
| Embedding | `tie_word_embeddings` true, `vocab_size` 60,000 | |
| Tokens | `bos_token_id` 2 · `eos_token_id` 3 · `pad_token_id` 1 | |

## Multi-head Latent Attention (MLA)

Standard multi-head attention materializes a key/value cache that grows with sequence length and layers. MLA instead **compresses** KV into a latent space:

- each head's query is low-rank-projected through `q_lora_rank` = 128;
- keys and values share a compressed latent block of rank `kv_lora_rank` = 128;
- each head then splits its Q/Key path into a 32-dim no-position half (`qk_nope_head_dim`) and a 32-dim RoPE half (`qk_rope_head_dim`), with the value head at 64 dimensions (`v_head_dim`).

12 RoPE heads therefore see a latent-KV working set that is the resource MLA shrinks: memory for training, and attention's working state at inference. For a small model trained on 24-GB GPUs, that compression is the point, not incidental.

## Sparse Mixture-of-Experts feed-forward

- The router (a linear head) scores all **18 routed experts**; the top-2 are selected greedily and combined with **raw top-2 softmax weights, no renormalization** (`norm_topk_prob: false`), combined with the always-on shared expert:

  `output = w1·E_a(x) + w2·E_b(x) + E_shared(x)`

- Each routed expert has intermediate size 128 (`moe_intermediate_size`). The shared expert is sized by the dense `intermediate_size` = 2048 MLP path and runs for **every** token.
- Training used a small auxiliary router load-balancing loss (`router_aux_loss_coef` = 0.001) and no auxiliary balancing in decode.

## Parameter accounting (from released weights)

| Component | Stored | Active per token |
|---|---|---|
| Tied embedding (60,000 × 768) | 46.08M | 46.08M (fully engaged via the output projection) |
| Attention projections and norms | 12.70M | 12.70M |
| Routers (18 heads × 12 layers) | 0.17M | 0.17M |
| Shared experts | 3.54M | 3.54M |
| Routed expert stack (18 × 12) | 63.70M | 7.08M (2 of 18 fire) |
| **Total** | **126.19M** | **69.57M** |
| Transformer body only (embedding excluded) | 80.11M | 23.49M |

Downstream work should cite **126.19M total and 69.57M active**, stating the body-only 23.49M explicitly when the embedding is deliberately excluded.

## Why this design at 0.1B

Most 0.1B models are dense. SOMA-S1 stores 63.70M parameters in the routed stack while computing 7.08M per token: total capacity supports more representational diversity than a same-active dense model, while the active path keeps training and reproduction cheap on workstation-class hardware. Whether the extra stored capacity converts into better Somali modeling than an equal-active dense model is an empirical question reserved for the evaluation program (see [EVALUATION.md](EVALUATION.md)).

See `figures/fig_architecture.jpeg` (block diagram), `figures/fig_mla.png` (attention path), and `figures/fig_moe.png` (routing path) for the visual companion.
