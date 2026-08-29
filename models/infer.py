#!/usr/bin/env python3
"""
infer.py — Run a fine-tuned NileChat-3B checkpoint over dev (from the HF
dataset) or test (from local private-test JSONL files) data, translating
each conversation turn-by-turn with beam search.

Each conversation is translated autoregressively over its own turns: turn
N's prompt includes the model's own predictions for turns 1..N-1 as history
(not the gold reference), because that's what the model actually has
available at real inference time.

Model loading
-------------
Supply exactly one of:
    --checkpoint-dir   a local directory with adapter_config.json (+ tokenizer)
    --wandb-artifact    a W&B model artifact, e.g. "entity/project/name:v5"
                         (downloaded via wandb.init(job_type="inference"))
The base model itself is always loaded fresh from --model-name and the LoRA
adapter is applied on top (via PEFT), matching how it was trained.

Output format (one JSON object per line):
    {"conv_id": <str>, "country": <str>,
     "turns": [{"turn_order": <int>, "prediction": <str>}, ...]}

This is the input evaluation/score.py expects (for dev) and the shape a
submission zip is built from (for test).

Requirements
------------
pip install torch transformers peft bitsandbytes wandb datasets tqdm

Usage
-----
# Dev, scoring against a local checkpoint
python infer.py --data-source dev --checkpoint-dir outputs/finetune_alex \
    --output-path development_predictions.jsonl

# Test, pulling the checkpoint from a W&B artifact, plus a submission zip
python infer.py --data-source test \
    --wandb-artifact RosettaAtAlexandriaX/DialectalArabicMT/model-FullScaleFT-NCchat:v5 \
    --test-data-dir testdataalex \
    --output-path test_predictions.jsonl --make-submission-zip
"""

from __future__ import annotations

import argparse
import json
import zipfile
from collections import defaultdict
from pathlib import Path

DATASET_NAME = "UBC-NLP/alexandria"
COUNTRIES = ["EG", "JO", "LB", "LY", "MA", "MR", "OM", "PS", "SA", "SD", "SY", "TN", "YE"]
DEFAULT_MODEL_NAME = "UBC-NLP/NileChat-3B"
DEFAULT_WANDB_ENTITY = "RosettaAtAlexandriaX"
DEFAULT_WANDB_PROJECT = "DialectalArabicMT"


# ---------------------------------------------------------------------------
# Data loading (dev via HF, test via local JSONL) — same normalized shape
# and prompt logic as prepare_alex_data.py, minus history noising (never
# applied at inference).
# ---------------------------------------------------------------------------

def discover_hf_splits(dataset_name: str, countries: list[str]) -> dict[str, set[str]]:
    from datasets import get_dataset_config_names, get_dataset_split_names

    available_configs = set(get_dataset_config_names(dataset_name))
    split_map: dict[str, set[str]] = {}
    for country in countries:
        if country not in available_configs:
            split_map[country] = set()
            continue
        split_map[country] = set(get_dataset_split_names(dataset_name, country))
    return split_map


def _turn_order(turn: dict, fallback: int = 0) -> int:
    try:
        return int(turn.get("turn_order", fallback))
    except (TypeError, ValueError):
        return fallback


def sorted_turns(turns: list[dict]) -> list[dict]:
    return sorted(turns or [], key=lambda turn: _turn_order(turn))


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def normalize_hf_record(row: dict) -> dict:
    english_turns = sorted_turns(row.get("english_conversation", []))
    dialect_turns = {
        _turn_order(turn, index + 1): turn
        for index, turn in enumerate(sorted_turns(row.get("dialectal_conversation", [])))
    }

    turns = []
    for index, english_turn in enumerate(english_turns, start=1):
        order = _turn_order(english_turn, index)
        reference_turn = dialect_turns.get(order, {})
        turns.append({
            "turn_order": order,
            "speaker": str(english_turn.get("speaker", "")).strip(),
            "sentence": str(english_turn.get("text", "")).strip(),
            "direction": str(english_turn.get("direction", "")).strip(),
            "reference": str(reference_turn.get("text", "")).strip(),
        })

    return {
        "conv_id": str(row.get("conv_id", "")).strip(),
        "country": str(row.get("country", "")).strip(),
        "domain": str(row.get("domain", "")).strip(),
        "dialect": str(row.get("dialect", "Arabic Dialect")).strip() or "Arabic Dialect",
        "participants": str(row.get("participants", "")).strip(),
        "turns": turns,
    }


def load_hf_dev_records(dataset_name: str, countries: list[str]) -> list[dict]:
    from datasets import load_dataset
    from tqdm import tqdm

    records: list[dict] = []
    for country in tqdm(countries, desc="Loading HF dev countries", unit="country"):
        dataset = load_dataset(dataset_name, country, split="dev")
        country_records = [normalize_hf_record(row) for row in dataset]
        records.extend(country_records)
        print(f"Loaded {len(country_records):>5} dev conversations for {country}")
    return records


def load_test_records(test_data_dir: Path, file_pattern: str, countries: list[str]) -> list[dict]:
    from tqdm import tqdm

    records: list[dict] = []
    for country in tqdm(countries, desc="Loading test countries", unit="country"):
        path = test_data_dir / file_pattern.format(country=country)
        if not path.exists():
            print(f"  {path}: not found, skipping")
            continue
        country_records = read_jsonl(path)
        records.extend(country_records)
        print(f"Loaded {len(country_records):>5} test conversations for {country}")
    return records


def format_history(history: list[dict]) -> str:
    if not history:
        return "No previous turns (Start of conversation)."
    lines = []
    for item in history:
        speaker = item.get("speaker", "Speaker") or "Speaker"
        lines.append(f"{speaker}: {item['sentence']}\nTranslation: {item['translation']}")
    return "\n".join(lines)


def build_prompt(record: dict, turn: dict, history: list[dict]) -> str:
    dialect = record.get("dialect") or "Arabic Dialect"
    domain = record.get("domain") or "Unknown Domain"
    participants = record.get("participants") or "Unknown Participants"
    direction = turn.get("direction") or "Unknown"
    speaker = turn.get("speaker") or "Unknown Speaker"

    return (
        f"Translate the English sentence into {dialect}.\n\n"
        f"### Metadata:\n"
        f"- Country: {record.get('country', '')}\n"
        f"- Domain: {domain}\n"
        f"- Participants: {participants}\n"
        f"- Speaker: {speaker}\n"
        f"- Speaker Direction: {direction}\n\n"
        f"### Conversation History:\n"
        f"{format_history(history)}\n\n"
        f"### Sentence to Translate:\n"
        f"{turn.get('sentence', '').strip()}\n\n"
    )


def system_prompt() -> str:
    return (
        "You are an expert translator.\n\n"
        "- Return only the translated text.\n"
        "- Do not add any code, explanations, comments, or any other extra text.\n"
        "- Keep the meaning and tone and respect the gender direction.\n"
        "- Consider the country, the domain, the participants, and the speaker in your translation.\n"
    )


def format_to_messages(prompt: str) -> list[dict]:
    return [
        {"role": "system", "content": system_prompt()},
        {"role": "user", "content": prompt.strip()},
    ]


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def resolve_checkpoint_dir(args) -> Path:
    if args.checkpoint_dir:
        return Path(args.checkpoint_dir)

    if args.wandb_artifact:
        import wandb

        run = wandb.init(entity=args.wandb_entity, project=args.wandb_project, job_type="inference")
        artifact = run.use_artifact(args.wandb_artifact, type="model")
        artifact_dir = Path(artifact.download())
        print(f"Checkpoint downloaded to: {artifact_dir}")
        return artifact_dir

    raise ValueError("Provide either --checkpoint-dir or --wandb-artifact.")


def build_quantization_config(load_in_4bit: bool):
    from transformers import BitsAndBytesConfig
    import torch

    if not load_in_4bit:
        return None
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,  # fp16, not bf16 - matches T4-class GPUs
        bnb_4bit_use_double_quant=True,
    )


def load_model_and_tokenizer(checkpoint_dir: Path, model_name: str, load_in_4bit: bool):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    adapter_config_path = checkpoint_dir / "adapter_config.json"
    if not adapter_config_path.exists():
        raise FileNotFoundError(
            f"No LoRA adapter found at {checkpoint_dir}. Point --checkpoint-dir at a "
            "completed training checkpoint, or use --wandb-artifact."
        )

    tokenizer_source = checkpoint_dir if (checkpoint_dir / "tokenizer_config.json").exists() else model_name
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.float16,
        quantization_config=build_quantization_config(load_in_4bit),
        trust_remote_code=True,
    )

    model = PeftModel.from_pretrained(base_model, checkpoint_dir)
    if hasattr(model, "gradient_checkpointing_disable"):
        model.gradient_checkpointing_disable()
    model.eval()
    model.config.use_cache = True

    print(f"Loaded fine-tuned model from: {checkpoint_dir}")
    return model, tokenizer


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def count_turns(records: list[dict]) -> int:
    return sum(len(record.get("turns", [])) for record in records)


def chunks(items: list, batch_size: int):
    for start in range(0, len(items), batch_size):
        yield items[start:start + batch_size]


def clean_generation(text: str) -> str:
    text = text.strip()
    for marker in ("\n###", "### Sentence to Translate:", "### Translation:"):
        if marker in text:
            text = text.split(marker, 1)[0].strip()
    return text


def generate_translations(
    prompts: list,
    model,
    tokenizer,
    max_new_tokens: int,
    num_beams: int,
    length_penalty: float,
) -> list[str]:
    import torch

    previous_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    inputs = tokenizer.apply_chat_template(
        prompts,
        tokenize=True,
        add_generation_prompt=True,
        padding=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            do_sample=False,
            num_beams=num_beams,
            length_penalty=length_penalty,  # < 1.0 favors shorter sequences
            early_stopping=True,
        )

    generated_tokens = outputs[:, inputs["input_ids"].shape[1]:]
    decoded = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

    tokenizer.padding_side = previous_padding_side
    return [clean_generation(text) for text in decoded]


def generate_prediction_records(
    records: list[dict],
    model,
    tokenizer,
    batch_size: int,
    max_new_tokens: int,
    num_beams: int,
    length_penalty: float,
    desc: str = "Generating predictions",
) -> list[dict]:
    """Translate every conversation turn-by-turn, feeding each record's OWN
    prior predictions back in as history (not gold references)."""
    from tqdm import tqdm

    histories: dict[int, list[dict]] = defaultdict(list)
    outputs = [
        {"conv_id": record["conv_id"], "country": record["country"], "turns": []}
        for record in records
    ]
    max_turns = max((len(record.get("turns", [])) for record in records), default=0)

    with tqdm(total=count_turns(records), desc=desc, unit="turn") as progress:
        for turn_position in range(max_turns):
            active = []
            for record_index, record in enumerate(records):
                turns = sorted_turns(record.get("turns", []))
                if turn_position >= len(turns):
                    continue
                turn = turns[turn_position]
                prompt = build_prompt(record, turn, histories[record_index])
                messages = format_to_messages(prompt)
                active.append((record_index, turn, messages))

            for batch in chunks(active, batch_size):
                prompts = [item[2] for item in batch]
                translations = generate_translations(
                    prompts, model, tokenizer, max_new_tokens, num_beams, length_penalty
                )
                for (record_index, turn, _), translation in zip(batch, translations):
                    outputs[record_index]["turns"].append({
                        "turn_order": int(turn["turn_order"]),
                        "prediction": translation,
                    })
                    histories[record_index].append({
                        "speaker": turn.get("speaker", ""),
                        "sentence": turn.get("sentence", ""),
                        "translation": translation,
                    })
                progress.update(len(batch))

    return outputs


def write_jsonl(records: list[dict], path: Path, desc: str | None = None) -> Path:
    from tqdm import tqdm

    path.parent.mkdir(parents=True, exist_ok=True)
    iterator = tqdm(records, desc=desc or f"Writing {path.name}", unit="conversation")
    with path.open("w", encoding="utf-8") as handle:
        for record in iterator:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return path


def make_submission_zip(predictions_jsonl: Path, zip_path: Path) -> Path:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(predictions_jsonl, arcname="predictions.jsonl")
    print(f"Wrote {zip_path}")
    return zip_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument("--data-source", choices=["dev", "test"], required=True)
    parser.add_argument("--countries", nargs="+", default=COUNTRIES)
    parser.add_argument("--dataset-name", default=DATASET_NAME)
    parser.add_argument("--test-data-dir", type=Path, default=Path("testdataalex"),
                         help="Directory holding private test input files (--data-source test).")
    parser.add_argument("--test-file-pattern", default="alexandria_{country}_private_test_input.jsonl")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--checkpoint-dir", type=Path, help="Local dir with adapter_config.json (+ tokenizer).")
    group.add_argument("--wandb-artifact", help="W&B model artifact, e.g. entity/project/name:v5")
    parser.add_argument("--wandb-entity", default=DEFAULT_WANDB_ENTITY)
    parser.add_argument("--wandb-project", default=DEFAULT_WANDB_PROJECT)

    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME, help="Base model the LoRA adapter is applied to.")
    parser.add_argument("--load-in-4bit", action="store_true", default=True)
    parser.add_argument("--no-4bit", dest="load_in_4bit", action="store_false")

    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--num-beams", type=int, default=5)
    parser.add_argument("--length-penalty", type=float, default=0.7)

    parser.add_argument("--output-path", type=Path, required=True, help="Where to write the predictions JSONL.")
    parser.add_argument("--make-submission-zip", action="store_true")
    parser.add_argument("--submission-zip-path", type=Path, default=Path("submission.zip"))

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.data_source == "dev":
        hf_split_map = discover_hf_splits(args.dataset_name, args.countries)
        dev_countries = [c for c in args.countries if "train" in hf_split_map.get(c, set())]
        skipped = [c for c in args.countries if c not in dev_countries]
        if skipped:
            print("Countries without an HF train split (skipped for dev):", skipped)
        records = load_hf_dev_records(args.dataset_name, dev_countries)
    else:
        records = load_test_records(args.test_data_dir, args.test_file_pattern, args.countries)
    print(f"Loaded {len(records)} conversations ({args.data_source}).")

    checkpoint_dir = resolve_checkpoint_dir(args)
    model, tokenizer = load_model_and_tokenizer(checkpoint_dir, args.model_name, args.load_in_4bit)

    predictions = generate_prediction_records(
        records, model, tokenizer,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        num_beams=args.num_beams,
        length_penalty=args.length_penalty,
        desc=f"Generating {args.data_source} predictions",
    )

    write_jsonl(predictions, args.output_path, desc=f"Writing {args.data_source} predictions")

    if args.make_submission_zip:
        make_submission_zip(args.output_path, args.submission_zip_path)


if __name__ == "__main__":
    main()
