#!/usr/bin/env python3
"""
train.py — Unsloth + TRL SFTTrainer LoRA training for AlexandriaX, covering
both stages of the pipeline with one script:

    --mode pretrain   NileChat-3B pretrained on MADAR + PADIC (out-of-domain
                       dialect data) to broaden dialect coverage before the
                       task-specific stage.
    --mode finetune    Same model/LoRA/optimizer setup, fine-tuned on the
                       Alex train/dev conversations for the actual task.

Both stages use the identical model, LoRA config, and optimizer
hyperparameters — the only real differences were which data went in, and
checkpoint/logging cadence. Those differences are captured in MODE_DEFAULTS 
below and are all still overridable from the CLI,
so e.g. running finetune starting from a pretrain checkpoint is just
`--mode finetune --model-name /path/to/pretrain/checkpoint`.

Data format
-----------
Every --train-files / --eval-files entry is a JSONL file of
{"messages": [{"role": ..., "content": ...}, ...]} records (the output of
prepare_alex_data.py or prepare_pretraining_data.py). Multiple files are
concatenated, e.g.:
    pretrain: --train-files madar_train_messages.jsonl padic_train_messages.jsonl
              --eval-files  madar_dev_messages.jsonl padic_dev_messages.jsonl
    finetune: --train-files alex_train.jsonl --eval-files alex_dev.jsonl

Requirements
------------
pip install unsloth trl datasets torch wandb

Secrets: set HF_TOKEN and WANDB_API_KEY in the environment. On Kaggle,
these are picked up automatically from Kaggle Secrets if present.

Usage
-----
python train.py --mode pretrain \
    --train-files data/madar_train_messages.jsonl data/padic_train_messages.jsonl \
    --eval-files  data/madar_dev_messages.jsonl data/padic_dev_messages.jsonl

python train.py --mode finetune \
    --model-name outputs/pretrain_madar_padic \
    --train-files data/alex_train.jsonl --eval-files data/alex_dev.jsonl
"""

from __future__ import annotations

import os

# Must be set before torch is imported anywhere in the process.
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

import argparse
import json
import random
from pathlib import Path

DEFAULT_MODEL_NAME = "UBC-NLP/NileChat-3B"
DEFAULT_WANDB_ENTITY = "RosettaAtAlexandriaX"
DEFAULT_WANDB_PROJECT = "DialectalArabicMT"

# Everything that differs by default between the two stages. Every value
# here can still be overridden by an explicit CLI flag.
MODE_DEFAULTS = {
    "pretrain": {
        "checkpoint_dir": "outputs/pretrain_madar_padic",
        "run_name": "Pretrain_MADAR_PADIC",
        "save_steps": 200,
        "eval_steps": 200,
        "save_total_limit": 5,
    },
    "finetune": {
        "checkpoint_dir": "outputs/finetune_alex",
        "run_name": "Finetune_Alex",
        "save_steps": 500,
        "eval_steps": 500,
        "save_total_limit": 10,
    },
}


# ---------------------------------------------------------------------------
# Secrets / environment
# ---------------------------------------------------------------------------

def load_kaggle_secrets() -> None:
    """On Kaggle, populate HF_TOKEN / WANDB_API_KEY from Kaggle Secrets if
    they aren't already set in the environment. No-op elsewhere."""
    try:
        from kaggle_secrets import UserSecretsClient
    except ImportError:
        return

    user_secrets = UserSecretsClient()
    if "HF_TOKEN" not in os.environ:
        try:
            os.environ["HF_TOKEN"] = user_secrets.get_secret("HF_TOKEN")
            print("Hugging Face Hub authenticated via Kaggle Secrets.")
        except Exception:
            print("No HF_TOKEN Kaggle Secret found.")
    if "WANDB_API_KEY" not in os.environ:
        try:
            os.environ["WANDB_API_KEY"] = user_secrets.get_secret("WANDB_API_KEY")
        except Exception:
            print("No WANDB_API_KEY Kaggle Secret found.")


def setup_wandb(entity: str, project: str) -> None:
    os.environ["WANDB_ENTITY"] = entity
    os.environ["WANDB_PROJECT"] = project
    os.environ["WANDB_LOG_MODEL"] = "checkpoint"
    if os.environ.get("WANDB_API_KEY"):
        import wandb
        wandb.login()
    else:
        print("WANDB_API_KEY not set — training will run without W&B logging "
              "unless it's configured elsewhere (pass --report-to none to silence this).")


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_messages_dataset(paths: list[Path]):
    """Concatenate one or more {"messages": [...]} JSONL files into a
    single HF Dataset."""
    from datasets import Dataset

    rows: list[dict] = []
    for path in paths:
        file_rows = read_jsonl(path)
        print(f"  {path}: {len(file_rows)} examples")
        rows.extend(file_rows)
    return Dataset.from_list(rows)


def apply_chat_template(dataset, tokenizer):
    """Render each example's `messages` through the model's chat template
    into a flat `text` column, which SFTTrainer trains on."""
    def _map(example):
        return {"text": tokenizer.apply_chat_template(
            example["messages"], tokenize=False, add_generation_prompt=False
        )}
    return dataset.map(_map)


# ---------------------------------------------------------------------------
# Model / training setup
# ---------------------------------------------------------------------------

def load_model_and_tokenizer(model_name: str, max_seq_length: int, use_4bit: bool):
    import torch
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        load_in_4bit=use_4bit,
        dtype=torch.float16,
    )

    tokenizer.truncation_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    print(f"Model dtype: {next(model.parameters()).dtype}")
    print("EOS token:", tokenizer.eos_token)
    print("PAD token:", tokenizer.pad_token)
    return model, tokenizer


def wrap_with_lora(model, args, seed: int):
    from unsloth import FastLanguageModel

    return FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=seed,
        use_rslora=False,
    )


def build_training_args(args, checkpoint_dir: Path):
    from trl import SFTConfig

    kwargs = dict(
        output_dir=str(checkpoint_dir),
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        max_length=args.max_seq_length,

        # Calculate loss strictly on the assistant's translations
        assistant_only_loss=True,

        logging_steps=10, 
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        bf16=False,
        fp16=True,
        # Unsloth's get_peft_model already handles gradient checkpointing;
        # HF-side checkpointing must stay off or the two conflict.
        gradient_checkpointing=False,
        optim="paged_adamw_8bit" if args.use_4bit else "adamw_torch",
        report_to=args.report_to,
        remove_unused_columns=True,
        packing=False,
        run_name=args.run_name,
    )


    return SFTConfig(**kwargs)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument("--mode", choices=["pretrain", "finetune"], required=True,
                         help="Which stage to run — sets the defaults in MODE_DEFAULTS (all overridable below).")
    parser.add_argument("--train-files", nargs="+", type=Path, required=True,
                         help="One or more JSONL files of {'messages': [...]}, concatenated for training.")
    parser.add_argument("--eval-files", nargs="+", type=Path, required=True,
                         help="One or more JSONL files of {'messages': [...]}, concatenated for eval.")

    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME,
                         help="HF model id, or a local path to a previous checkpoint "
                              "(e.g. resume finetune from a pretrain checkpoint).")
    parser.add_argument("--checkpoint-dir", type=Path, default=None,
                         help="Where to save the LoRA adapter + tokenizer. Defaults per --mode.")
    parser.add_argument("--run-name", default=None, help="W&B run name. Defaults per --mode.")

    parser.add_argument("--max-seq-length", type=int, default=1024)
    parser.add_argument("--num-epochs", type=float, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--use-4bit", action="store_true", default=True)
    parser.add_argument("--no-4bit", dest="use_4bit", action="store_false")

    parser.add_argument("--lora-r", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0)

    parser.add_argument("--save-steps", type=int, default=None, help="Defaults per --mode.")
    parser.add_argument("--eval-steps", type=int, default=None, help="Defaults per --mode.")
    parser.add_argument("--save-total-limit", type=int, default=None, help="Defaults per --mode.")
    parser.add_argument("--report-to", default="wandb", help="Set to 'none' to disable W&B logging.")

    parser.add_argument("--wandb-entity", default=DEFAULT_WANDB_ENTITY)
    parser.add_argument("--wandb-project", default=DEFAULT_WANDB_PROJECT)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    defaults = MODE_DEFAULTS[args.mode]
    if args.checkpoint_dir is None:
        args.checkpoint_dir = Path(defaults["checkpoint_dir"])
    if args.run_name is None:
        args.run_name = defaults["run_name"]
    if args.save_steps is None:
        args.save_steps = defaults["save_steps"]
    if args.eval_steps is None:
        args.eval_steps = defaults["eval_steps"]
    if args.save_total_limit is None:
        args.save_total_limit = defaults["save_total_limit"]

    return args


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    load_kaggle_secrets()
    if args.report_to != "none":
        setup_wandb(args.wandb_entity, args.wandb_project)

    print(f"Mode: {args.mode}")
    print("Loading train files:")
    train_dataset_raw = load_messages_dataset(args.train_files)
    print("Loading eval files:")
    eval_dataset_raw = load_messages_dataset(args.eval_files)
    print(f"Train examples: {len(train_dataset_raw)} | Eval examples: {len(eval_dataset_raw)}")

    model, tokenizer = load_model_and_tokenizer(args.model_name, args.max_seq_length, args.use_4bit)

    train_dataset = apply_chat_template(train_dataset_raw, tokenizer)
    eval_dataset = apply_chat_template(eval_dataset_raw, tokenizer)

    model = wrap_with_lora(model, args, args.seed)

    training_args = build_training_args(args, args.checkpoint_dir)

    from trl import SFTTrainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )

    trainer.train()

    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(args.checkpoint_dir)
    tokenizer.save_pretrained(args.checkpoint_dir)
    print(f"Saved LoRA adapter and tokenizer to: {args.checkpoint_dir}")


if __name__ == "__main__":
    main()
