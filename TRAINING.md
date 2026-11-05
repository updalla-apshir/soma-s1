# SOMA-S1 → Training

The complete training configuration ships as [`experiments/train_soma_s1.py`](experiments/train_soma_s1.py). The run record below sums the training log.

## Recipe

| Item | Setting |
|---|---|
| Optimizer | AdamW via `adamw_torch` |
| Peak learning rate | 3e-4, cosine decay to 0 |
| Warmup | 600 optimizer steps (3% of the 20,000-step schedule) |
| Beta / eps | (0.9, 0.95) · 1e-8 |
| Weight decay | 0.1 |
| Gradient clipping | norm 1.0 |
| Effective batch | micro-batch 1 × accumulation 16 × 2 GPUs = 32 sequences × 2,048 tokens = **65,536 token positions per optimizer step** |
| Precision | BF16 mixed precision with TF32, gradient checkpointing enabled |
| Schedule | 20,000 steps configured, seed 42, data seed 42 |
| Cadence | log every 10 · evaluate every 100 · checkpoint every 500 steps |

## Data pipeline

- Corpus: Somali NLP Corpus (`somast/somali-nlp-corpus`), normalized, quality/language filtered, deduplicated, tokenized with the model's byte-level BPE (60,000 vocab, PAD = 1).
- **Sequence packing** over 1,024-document mapping batches: every trained sequence is fully packed content with all-ones attention masks; documents lacking a terminal EOS receive one; a mapping batch's remainder below a full 2,048-token block is discarded (a bounded overhead controlled by the packing parameter).
- **Validation split:** the first 1,024 documents of the seeded shuffle (`dataset.take(1024)`), the remaining training stream skipping them (seed 42, buffer 100,000), packed to 34 blocks.

## Run record

| | |
|---|---|
| Logged optimizer steps | 16,413 (configured schedule: 20,000; the configured endpoint corresponds to ≈1.31B positions) |
| Positions observed | 1.0758B token positions |
| Step time | ≈2.74 s mean (2.63–2.87 s band, steps 8,700–16,400) |
| Derived throughput | ≈23,900 positions/s across two GPUs · ≈11,950 per GPU |
| Session timers | ≈12.4 h total · final session 6 h 54 m 52 s (steps ≈7,510 → 16,413) |
| Eval loss | 8.91 → 3.79 (floor 3.76 near step 13,000) |
| Loss | 11.15 (first logged step) → 3.505 (step 16,409) |

Device-level telemetry (peak memory, sustained utilization, power) was not recorded; performance figures are step-level timings.

## Session history

The run is a restart chain rather than a single session. The learning-rate anchors tie every segment to global optimizer steps and the segments join seamlessly.

1. The initial 20,000-step configuration launched and fell to an **NCCL watchdog timeout** near step 5,600, after training loss had fallen from 11.15 to a ≈4.0 plateau and validation to ≈4.27.
2. A resumed session carried to ≈6,414 steps.
3. A restart failed with **disk exhaustion** writing checkpoints.
4. A host-side continuation carried to ≈7,500 steps; the checkpoint state at step 7,500 attests the hand-off.
5. The final session resumed at step 7,510 and ran to **16,413**, ending when the host's working volume filled again (≈92% disk usage).

## Recorded interruptions and their resolution

- The epoch field printed in the log is unreliable after a resume shift (offset ≈0.511 steps after step 9,800); global optimizer steps are anchored through the learning-rate schedule, mapping `lr = 3e-4 · 0.5·(1+cos(π·(s−600)/19,400))`.
- One generation of mismatch warnings (`save_steps` 500 in the script vs 100 in a loaded trainer state) at a checkpoint-100-fixed resume; resolved by fixing save cadence.

## What is NOT claimed

The run ended at step 16,413 by exhaustion; the configured 20,000 steps were not completed, and no post-16,413 continuation was recorded. Reports of "20,000 completed steps" or a checkpoint-20000 would be unsupported by the training log.
