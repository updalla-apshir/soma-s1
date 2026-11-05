
import os
import inspect
import torch

from datasets import load_dataset, Dataset
from transformers import (
    DeepseekV2Config,
    DeepseekV2ForCausalLM,
    PreTrainedTokenizerFast,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from huggingface_hub import HfApi
from safetensors.torch import load_file, save_file


# ============================================================
# 1. CONFIGURATION
# ============================================================

MODEL_NAME = "SOMA S1"

DATASET_REPO_ID = "boore1/somali-tokenized-datasetv2"
TOKENIZER_PATH = "/workspace/somali_bpe_60k.json"

CHECKPOINT_DIR = "/workspace/SOMA-S1-checkpoints"
FINAL_MODEL_DIR = "/workspace/SOMA-S1"

HF_MODEL_REPO = "boore1/SOMA-S1"

MAX_STEPS = 20_000
MAX_SEQUENCE_LENGTH = 2048

TRAIN_BATCH_SIZE = 1
EVAL_BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 16

WORLD_SIZE = 2

EFFECTIVE_BATCH_SIZE = (
    WORLD_SIZE
    * TRAIN_BATCH_SIZE
    * GRADIENT_ACCUMULATION_STEPS
)

LEARNING_RATE = 3e-4
WARMUP_STEPS = 600
WEIGHT_DECAY = 0.1

SEED = 42
EOS_TOKEN_ID = 3

VALIDATION_DOCUMENTS = 1024
PACK_BATCH_SIZE = 1024

SAVE_STEPS = 500
SAVE_TOTAL_LIMIT = 3


# ============================================================
# 2. MOE CHECKPOINT CONVERSION
# ============================================================

def fix_moe_checkpoint(checkpoint_dir, config):

    model_file = os.path.join(
        checkpoint_dir,
        "model.safetensors",
    )

    if not os.path.isfile(model_file):
        raise FileNotFoundError(model_file)

    print(
        "Checking checkpoint MoE weights...",
        flush=True,
    )

    state = load_file(
        model_file,
        device="cpu",
    )

    # Support both already-packed and individual expert layouts.
    if any(
        ".mlp.experts.gate_up_proj" in key
        for key in state
    ):
        print(
            "MoE checkpoint is already packed.",
            flush=True,
        )
        return

    num_layers = config.num_hidden_layers
    num_experts = config.num_experts
    hidden_size = config.hidden_size
    intermediate_size = config.moe_intermediate_size

    packed = {
        key: value
        for key, value in state.items()
        if ".mlp.experts." not in key
    }

    for layer in range(num_layers):

        prefix = (
            f"model.layers.{layer}.mlp.experts."
        )

        gate_up = []
        down = []

        for expert in range(num_experts):

            gate_key = (
                prefix + f"{expert}.gate_proj.weight"
            )

            up_key = (
                prefix + f"{expert}.up_proj.weight"
            )

            down_key = (
                prefix + f"{expert}.down_proj.weight"
            )

            for key in (gate_key, up_key, down_key):
                if key not in state:
                    raise KeyError(
                        f"Missing checkpoint tensor: {key}"
                    )

            gate = state[gate_key]
            up = state[up_key]
            down_weight = state[down_key]

            assert gate.shape == (
                intermediate_size,
                hidden_size,
            )

            assert up.shape == (
                intermediate_size,
                hidden_size,
            )

            assert down_weight.shape == (
                hidden_size,
                intermediate_size,
            )

            # The model implementation uses:
            # gate, up = gate_up_proj.chunk(2, dim=-1)
            gate_up.append(
                torch.cat([gate, up], dim=0)
            )

            down.append(down_weight)

        packed[prefix + "gate_up_proj"] = torch.stack(
            gate_up,
            dim=0,
        )

        packed[prefix + "down_proj"] = torch.stack(
            down,
            dim=0,
        )

    # Validate all packed expert tensors.
    expected_count = num_layers * 2

    actual_count = sum(
        ".mlp.experts." in key
        for key in packed
    )

    if actual_count != expected_count:
        raise RuntimeError(
            f"Expected {expected_count} packed expert tensors, "
            f"found {actual_count}"
        )

    expected_gate_shape = (
        num_experts,
        2 * intermediate_size,
        hidden_size,
    )

    expected_down_shape = (
        num_experts,
        hidden_size,
        intermediate_size,
    )

    for layer in range(num_layers):

        prefix = (
            f"model.layers.{layer}.mlp.experts."
        )

        assert packed[
            prefix + "gate_up_proj"
        ].shape == expected_gate_shape

        assert packed[
            prefix + "down_proj"
        ].shape == expected_down_shape

    # Write a temporary file first.
    temporary_file = model_file + ".tmp"

    print(
        "Writing converted MoE checkpoint...",
        flush=True,
    )

    save_file(
        packed,
        temporary_file,
        metadata={"format": "pt"},
    )

    # Verify the written file before replacing the original.
    check = load_file(
        temporary_file,
        device="cpu",
    )

    assert len(check) == len(packed)

    assert torch.equal(
        check[
            "model.layers.0.mlp.experts.gate_up_proj"
        ],
        packed[
            "model.layers.0.mlp.experts.gate_up_proj"
        ],
    )

    os.replace(
        temporary_file,
        model_file,
    )

    print(
        "MoE checkpoint conversion successful.",
        flush=True,
    )

    print(
        f"Layers converted: {num_layers}",
        flush=True,
    )

    print(
        f"Experts per layer: {num_experts}",
        flush=True,
    )

    print(
        "Optimizer, scheduler, trainer state and RNG files untouched.",
        flush=True,
    )


# ============================================================
# 3. HARDWARE CHECK
# ============================================================

assert torch.cuda.is_available(), (
    "CUDA is not available."
)

assert torch.cuda.device_count() == WORLD_SIZE, (
    f"Expected {WORLD_SIZE} GPUs, "
    f"found {torch.cuda.device_count()}."
)

for i in range(WORLD_SIZE):

    assert torch.cuda.is_bf16_supported(), (
        f"GPU {i} does not support BF16."
    )

    print(
        f"GPU {i}: {torch.cuda.get_device_name(i)}",
        flush=True,
    )

print(
    f"Effective batch size: {EFFECTIVE_BATCH_SIZE}",
    flush=True,
)


# ============================================================
# 4. REPRODUCIBILITY
# ============================================================

set_seed(SEED)


# ============================================================
# 5. TOKENIZER
# ============================================================

tokenizer = PreTrainedTokenizerFast(
    tokenizer_file=TOKENIZER_PATH,
    unk_token="[UNK]",
    pad_token="[PAD]",
    bos_token="[BOS]",
    eos_token="[EOS]",
)

assert tokenizer.vocab_size == 60_000, (
    f"Expected vocabulary 60,000, "
    f"got {tokenizer.vocab_size}"
)

assert tokenizer.eos_token_id == EOS_TOKEN_ID
assert tokenizer.pad_token_id == 1

print(
    f"Tokenizer loaded: {tokenizer.vocab_size:,}",
    flush=True,
)


# ============================================================
# 6. LOAD STREAMING DATASET
# ============================================================

print(
    "Loading Hugging Face dataset...",
    flush=True,
)

dataset = load_dataset(
    DATASET_REPO_ID,
    split="train",
    streaming=True,
    token=True,
)

dataset = dataset.shuffle(
    seed=SEED,
    buffer_size=100_000,
)

validation_documents = dataset.take(
    VALIDATION_DOCUMENTS
)

training_documents = dataset.skip(
    VALIDATION_DOCUMENTS
)


# ============================================================
# 7. PACK DOCUMENTS INTO 2048-TOKEN BLOCKS
# ============================================================

def pack_documents(batch):

    all_tokens = []

    for ids, mask in zip(
        batch["input_ids"],
        batch["attention_mask"],
    ):

        tokens = [
            int(token_id)
            for token_id, keep in zip(ids, mask)
            if int(keep) == 1
        ]

        if not tokens:
            continue

        if tokens[-1] != EOS_TOKEN_ID:
            tokens.append(EOS_TOKEN_ID)

        all_tokens.extend(tokens)

    total_length = (
        len(all_tokens) // MAX_SEQUENCE_LENGTH
    ) * MAX_SEQUENCE_LENGTH

    input_ids = []
    attention_masks = []

    for start in range(
        0,
        total_length,
        MAX_SEQUENCE_LENGTH,
    ):

        block = all_tokens[
            start:start + MAX_SEQUENCE_LENGTH
        ]

        input_ids.append(block)

        attention_masks.append(
            [1] * MAX_SEQUENCE_LENGTH
        )

    return {
        "input_ids": input_ids,
        "attention_mask": attention_masks,
    }


print(
    "Preparing streaming training dataset...",
    flush=True,
)

packed_train = training_documents.map(
    pack_documents,
    batched=True,
    batch_size=PACK_BATCH_SIZE,
    remove_columns=training_documents.column_names,
)


# ============================================================
# 8. MATERIALIZE VALIDATION DATASET
# ============================================================

print(
    "Preparing validation dataset...",
    flush=True,
)

packed_validation = validation_documents.map(
    pack_documents,
    batched=True,
    batch_size=PACK_BATCH_SIZE,
    remove_columns=validation_documents.column_names,
)

validation_rows = list(packed_validation)

if not validation_rows:
    raise RuntimeError(
        "Validation dataset produced zero packed blocks."
    )

validation_dataset = Dataset.from_list(
    validation_rows
)

print(
    f"Validation blocks: {len(validation_dataset):,}",
    flush=True,
)


# ============================================================
# 9. MODEL CONFIGURATION
# ============================================================

config = DeepseekV2Config(
    vocab_size=60_000,

    hidden_size=768,
    intermediate_size=2048,
    num_hidden_layers=12,
    num_attention_heads=12,

    max_position_embeddings=MAX_SEQUENCE_LENGTH,

    # Multi-head Latent Attention
    kv_lora_rank=128,
    q_lora_rank=128,
    qk_nope_head_dim=32,
    qk_rope_head_dim=32,
    v_head_dim=64,

    # Mixture of Experts
    n_routed_experts=18,
    n_shared_experts=1,
    num_experts_per_tok=2,
    moe_intermediate_size=128,

    # Normalization and activation
    rms_norm_eps=1e-6,
    hidden_act="silu",

    # Router auxiliary loss
    router_aux_loss_coef=0.001,
    output_router_logits=True,

    tie_word_embeddings=True,
    use_cache=False,
)

print(
    "Initializing SOMA S1 model...",
    flush=True,
)

model = DeepseekV2ForCausalLM(config)

parameter_count = sum(
    parameter.numel()
    for parameter in model.parameters()
)

print(
    f"Total parameters: {parameter_count:,}",
    flush=True,
)

# Validate model/checkpoint architecture compatibility.
assert config.num_experts == 18
assert config.num_hidden_layers == 12


# ============================================================
# 10. DATA COLLATOR
# ============================================================

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)


# ============================================================
# 11. TRAINING ARGUMENTS
# ============================================================

training_kwargs = dict(

    output_dir=CHECKPOINT_DIR,

    max_steps=MAX_STEPS,

    per_device_train_batch_size=TRAIN_BATCH_SIZE,
    per_device_eval_batch_size=EVAL_BATCH_SIZE,

    gradient_accumulation_steps=(
        GRADIENT_ACCUMULATION_STEPS
    ),

    learning_rate=LEARNING_RATE,
    warmup_steps=WARMUP_STEPS,
    lr_scheduler_type="cosine",

    weight_decay=WEIGHT_DECAY,
    adam_beta1=0.9,
    adam_beta2=0.95,
    adam_epsilon=1e-8,

    max_grad_norm=1.0,

    bf16=True,
    tf32=True,

    gradient_checkpointing=True,

    optim="adamw_torch",

    logging_strategy="steps",
    logging_steps=10,
    logging_first_step=True,

    eval_steps=100,

    save_strategy="steps",
    save_steps=SAVE_STEPS,
    save_total_limit=SAVE_TOTAL_LIMIT,

    dataloader_num_workers=0,
    dataloader_pin_memory=True,

    ddp_find_unused_parameters=True,

    # Match the original training configuration.
    accelerator_config={
        "dispatch_batches": True,
    },

    eval_accumulation_steps=16,

    remove_unused_columns=False,

    report_to="none",

    seed=SEED,
    data_seed=SEED,
)


ta_parameters = inspect.signature(
    TrainingArguments.__init__
).parameters

if "eval_strategy" in ta_parameters:
    training_kwargs["eval_strategy"] = "steps"
else:
    training_kwargs["evaluation_strategy"] = "steps"

if "gradient_checkpointing_kwargs" in ta_parameters:
    training_kwargs["gradient_checkpointing_kwargs"] = {
        "use_reentrant": False,
    }

training_args = TrainingArguments(
    **training_kwargs
)

# ============================================================
# 12. EXPLICIT CHECKPOINT RECOVERY
# ============================================================

import json

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

RESUME_CHECKPOINT = os.path.join(
    CHECKPOINT_DIR,
    "checkpoint-7500",
)

required_files = [
    "config.json",
    "model.safetensors",
    "optimizer.pt",
    "scheduler.pt",
    "trainer_state.json",
    "training_args.bin",
]

missing_files = [
    name
    for name in required_files
    if not os.path.isfile(
        os.path.join(RESUME_CHECKPOINT, name)
    )
]

if missing_files:
    raise RuntimeError(
        f"Incomplete recovery checkpoint: {missing_files}"
    )

with open(
    os.path.join(
        RESUME_CHECKPOINT,
        "trainer_state.json",
    ),
    "r",
    encoding="utf-8",
) as f:
    checkpoint_state = json.load(f)

checkpoint_step = int(
    checkpoint_state["global_step"]
)

if checkpoint_step != 7500:
    raise RuntimeError(
        f"Expected step 7500, got {checkpoint_step}"
    )

if checkpoint_step >= MAX_STEPS:
    raise RuntimeError(
        "Checkpoint already reached maximum training steps."
    )

print("=" * 60, flush=True)
print("CHECKPOINT VALIDATION SUCCESSFUL", flush=True)
print(f"Recovery: {RESUME_CHECKPOINT}", flush=True)
print(f"Global step: {checkpoint_step}", flush=True)
print(f"Remaining steps: {MAX_STEPS - checkpoint_step}", flush=True)
print("=" * 60, flush=True)
# ============================================================
# 13. TRAINER
# ============================================================

trainer_kwargs = dict(
    model=model,
    args=training_args,
    train_dataset=packed_train,
    eval_dataset=validation_dataset,
    data_collator=data_collator,
)

trainer_parameters = inspect.signature(
    Trainer.__init__
).parameters

if "processing_class" in trainer_parameters:
    trainer_kwargs["processing_class"] = tokenizer
else:
    trainer_kwargs["tokenizer"] = tokenizer

trainer = Trainer(
    **trainer_kwargs
)

# ============================================================
# 14. CONVERT AND RESUME
# ============================================================

print("\n" + "=" * 65, flush=True)
print(f"RESUMING {MODEL_NAME}", flush=True)
print("=" * 65, flush=True)

print(
    f"Resume checkpoint: {RESUME_CHECKPOINT}",
    flush=True,
)

print(
    f"Resume step: {checkpoint_step:,} / {MAX_STEPS:,}",
    flush=True,
)

# Only global rank zero converts the checkpoint.
if trainer.is_world_process_zero():

    fix_moe_checkpoint(
        RESUME_CHECKPOINT,
        config,
    )

# Other processes must wait until conversion is complete.
trainer.accelerator.wait_for_everyone()

print(
    "Loading checkpoint model, optimizer and scheduler...",
    flush=True,
)

train_result = trainer.train(
    resume_from_checkpoint=RESUME_CHECKPOINT
)

print(
    "Training run returned.",
    flush=True,
)


# ============================================================
# 15. FINAL EVALUATION
# ============================================================

print(
    "\nRunning final evaluation...",
    flush=True,
)

final_metrics = trainer.evaluate()

if trainer.is_world_process_zero():

    print(
        "\nFINAL VALIDATION RESULTS",
        flush=True,
    )

    for key, value in final_metrics.items():
        print(
            f"{key}: {value}",
            flush=True,
        )


# ============================================================
# 16. SAVE FINAL MODEL LOCALLY
# ============================================================

trainer.accelerator.wait_for_everyone()

if trainer.is_world_process_zero():

    print(
        "\nSaving final SOMA S1 model...",
        flush=True,
    )

    os.makedirs(
        FINAL_MODEL_DIR,
        exist_ok=True,
    )

    trainer.model.config.output_router_logits = False
    trainer.model.config.use_cache = True

    trainer.save_model(
        FINAL_MODEL_DIR
    )

    tokenizer.save_pretrained(
        FINAL_MODEL_DIR
    )

    trainer.model.config.save_pretrained(
        FINAL_MODEL_DIR
    )

    print(
        f"Final model saved: {FINAL_MODEL_DIR}",
        flush=True,
    )

trainer.accelerator.wait_for_everyone()


# ============================================================
# 17. UPLOAD FINAL MODEL ONLY TO HUGGING FACE
# ============================================================

if trainer.is_world_process_zero():

    print(
        "\nUploading final SOMA S1 model...",
        flush=True,
    )

    api = HfApi()

    api.create_repo(
        repo_id=HF_MODEL_REPO,
        repo_type="model",
        private=True,
        exist_ok=True,
    )

    api.upload_folder(
        folder_path=FINAL_MODEL_DIR,
        repo_id=HF_MODEL_REPO,
        repo_type="model",
        commit_message=(
            "Release SOMA S1 - Somali language model "
            "trained from scratch"
        ),
    )

    print("\n" + "=" * 65, flush=True)
    print("SOMA S1 TRAINING AND UPLOAD COMPLETED", flush=True)
    print(f"Model: {HF_MODEL_REPO}", flush=True)
    print(f"Local model: {FINAL_MODEL_DIR}", flush=True)
    print(f"Local checkpoints: {CHECKPOINT_DIR}", flush=True)
    print("Hugging Face visibility: PRIVATE", flush=True)
    print("=" * 65, flush=True)
