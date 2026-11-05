# SOMA-S1 → Evaluation

## What is measured

### Validation loss trajectory

Evaluation ran every 100 optimizer steps on the fixed held-out split (1,024 documents packed to 34 blocks of 2,048 tokens; all-ones masks).

| Phase | Behavior |
|---|---|
| Initial session | 8.91 (step 100) → ≈4.27 (step 5,500), smooth descent |
| Resume | 4.26 at step 7,600, continuous with the pre-interruption state |
| Dataset pass boundary | sharp drop **4.11 → 3.84 within 300 steps** near step 9,800 (the loader revisits data from its first pass) |
| Floor | **3.76** near step 13,000 |
| Final band | flat 3.78–3.79 to step 16,400 (last reading 3.79) |

Training loss: 11.15 → ≈4.0 plateau by step 5,600; 3.774 at the step-7,510 resume → 3.505 at step 16,409. Final train–validation gap ≈ 0.3 nats: the model has not memorized its training partition, and stable validation through the final thousands of steps marks a natural stopping point for this corpus pass.

Perplexity equivalent: ≈ 44 (high in absolute terms, and the number a from-scratch preview at this scale and corpus size should be expected to produce).

Curves: `figures/fig_train_loss.png` and `figures/fig_eval_loss.png`; run-level step timing: `figures/fig_telemetry.pdf`.

### Qualitative comparative run

One fixed Somali prompt (`Soomaaliya waa dal …`) was presented to **SOMA-S1, Qwen2.5-0.5B, Gemma-3-1B, and TinyLlama-1.1B**, each sampled for 128 new tokens, in four independent side-by-side generations.

- **SOMA-S1** was the only model that remained in Somali and on topic in every recorded generation.
- **Gemma-3-1B** produced Somali at all, but its continuations abandon the prompt and carry unsupported assertions; it does not function as a competitor.
- **Qwen2.5-0.5B** drifted into English text: hallucinated etymology and assistant chatter ("The Soma (White Horse) Gurus were considered to be reincarnations of Buddha" as the response to a Somali prompt).
- **TinyLlama-1.1B** drifted into English and emitted no coherent Somali clause.

The Colab output screenshot and a typeset rendering of the four outputs: `figures/fig_eval_run_3.png`. This is a qualitative comparison on a fixed prompt, not a benchmark.

### Coherence and repetition diagnostics (recorded informally)

- Greedy decoding degenerates into repetitive loops after roughly 10–20 tokens.
- Sampling with repetition penalty ≥ 1.2, temperature 0.7–0.8, `top_p` 0.9–0.95 reduces but does not eliminate repetition.
- Local grammaticality is often reasonable over short spans; consistency and logical flow degrade with generation length.

## What is under construction

No leaderboard or downstream-task scores are reported; none have been earned yet. The systematic evaluation program covers:

- held-out perplexity on a disjoint Somali validation split;
- tokenizer fertility (tokens per word) on Somali text versus multilingual tokenizers;
- distinct-n and self-repetition rates across generation lengths;
- downstream tasks: summarization, question answering, Somali information retrieval.

Evaluation prompts are natural Somali prefixes with no role markers, consistent with the content-only template (`Soomaaliya waa dal …` is the reference example).
