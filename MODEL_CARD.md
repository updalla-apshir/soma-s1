# SOMA-S1 Model Card

| Field | Value |
|---|---|
| Model | SOMA-S1 (`boore1/SOMA-S1`) |
| Type | Decoder-only text-completion foundation model (base, untuned) |
| Architecture family | DeepseekV2ForCausalLM (sparse MoE + MLA) |
| Parameters | 126.19M total · 69.57M active per token |
| Context | 2,048 tokens |
| Vocabulary | 60,000 (byte-level BPE trained on Somali) |
| Special tokens | PAD = 1 · BOS = 2 · EOS = 3 |
| Language | Somali (`so`) |
| Pretraining data | Somali NLP Corpus (`somast/somali-nlp-corpus`), ≈1.0758B token positions seen |
| License | Apache-2.0 |
| Author | Abdullahi Abshir Boore (Boore.dev) |

## Intended use

Continuation of natural Somali text: drafting aid for Somali writers, linguistic experimentation on an in-language model, low-resource study baseline, and instruction-tuning substrate. Behavior follows the pretraining objective: given a text prefix, the model produces a continuation.

## Out of scope

- **Instruction following and chat.** The bundled template is a content-only passthrough with no role markers. Direct questions frequently produce EOS and empty output, or a continuation unrelated to the question.
- **Factual reference.** The model generates plots, statistics, and names tied to the prompt but not to the world. It must not be treated as a truthful source.
- **Safety-critical or commercial customer-facing deployment.** None of the characteristics required for that are present.

## Behavior and limitations

1. **Repetition.** Greedy decoding degenerates into repetitive loops after roughly 10–20 tokens. Sampling with a repetition penalty of 1.2 or higher, temperature 0.7–0.8, and `top_p` 0.9–0.95 reduces, but does not eliminate, repetition. This is the most visible quality limitation of the preview.
2. **Coherence horizon.** Local grammaticality is often reasonable over short spans; consistency and logical flow degrade as generation lengthens, matching the model's scale and corpus limits.
3. **Fabricated specifics.** Continuations can invent personal names, dates, and pseudo-statistics about Somalia. Recorded example: a pseudo-statistic about Puntland followed by an unexplained bullet list.
4. **Coverage gaps.** Corpus bias and uneven register coverage propagate into the model; orthographic consistency of Somali in open data is imperfect and the model reflects that.

## Sampling recommendations

```text
do_sample = True
temperature      0.7–0.8
top_p            0.9–0.95
repetition_penalty ≥ 1.2
max_new_tokens   start with 128
```

Greedy decoding is retained only as a diagnostic for degenerate repetition, not as a delivery mode.

## Validation snapshot

Evaluation ran every 100 optimizer steps on a fixed 1,024-document held-out split (packed to 34 blocks of 2,048 tokens). Validation cross-entropy fell from 8.91 (first evaluation, step 100) to ≈4.27 by step 5,500, resumed at 4.26, dropped sharply across the loader's dataset-pass boundary near step 9,800 (4.11 → 3.84 within 300 steps), reached its 3.76 floor near step 13,000, and held a flat 3.78–3.79 band to the end (last reading 3.79, step 16,400). Final train–validation gap ≈ 0.3 nats. Perplexity ≈ 44. See [EVALUATION.md](EVALUATION.md).

## Recorded comparative run

One fixed Somali prompt (`Soomaaliya waa dal …`) was presented to SOMA-S1, Qwen2.5-0.5B, Gemma-3-1B, and TinyLlama-1.1B, each sampled for 128 new tokens, in four independent side-by-side generations. SOMA-S1 was the only model that remained in Somali and on topic in every recorded generation. Gemma-3-1B produced Somali at all but with prompt-abandoning, unsupported content; Qwen2.5-0.5B and TinyLlama-1.1B drifted into English text. A typeset rendering appears in this repository's `figures/`.

## Risks

As an untuned base model trained on open Somali text, SOMA-S1 can produce subjective, one-sided, or offensive content present in the training distribution. Somali text online is comparatively unfiltered; treat all output as unmoderated. Fabrication risk is high and inherent to the scale.
