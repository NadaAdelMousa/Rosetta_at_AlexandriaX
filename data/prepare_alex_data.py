#!/usr/bin/env python3
"""
prepare_alex_data.py — Build fine-tuning-ready chat-format JSONL for the
AlexandriaX shared task from the UBC-NLP/alexandria HF dataset.

Pipeline
--------
1. Discover which country configs actually have a "train" split on the Hub
   (a few countries are test-only).
2. Download the train and dev splits for those countries and normalize each
   HF conversation record: sort English/dialectal turns by turn_order and
   pair each English turn with its dialectal reference.
3. Flatten every conversation into one training pair per turn, where the
   prompt includes the conversation metadata (country/domain/participants/
   speaker/direction) and the dialectal history up to that turn.
4. For the train split only, optionally apply history noising as an
   exposure-bias fix: each history turn's translation has a chance of being
   corrupted (word drop/swap/duplicate), and the whole history has a chance
   of being truncated to a shorter prefix. This simulates the imperfect
   history the model will see at inference time (its own prior outputs)
   instead of always training on clean gold history. The CLEAN reference
   always propagates forward into the next turn's history — only the copy
   shown in the prompt is corrupted. Dev is left clean for stable eval.
5. Wrap each pair into the {"messages": [system, user, assistant]} chat
   schema and write alex_train.jsonl / alex_dev.jsonl.

Output format (one JSON object per line):
    {"messages": [
        {"role": "system", "content": "<translator instructions>"},
        {"role": "user", "content": "<prompt with metadata + history + sentence>"},
        {"role": "assistant", "content": "<dialectal reference>"}
    ]}

Requirements
------------
pip install datasets huggingface_hub tqdm

If the dataset is gated, set HF_TOKEN in the environment (or pass
--hf-token) before running.

Usage
-----
python prepare_alex_data.py --output-dir ./data/processed
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

from tqdm import tqdm

DATASET_NAME = "UBC-NLP/alexandria"
COUNTRIES = ["EG", "JO", "LB", "LY", "MA", "MR", "OM", "PS", "SA", "SD", "SY", "TN", "YE"]

# ---- Exposure-bias fix: history noising (train split only) ----------------
HISTORY_NOISE_P = 0.25       # P(a history translation gets corrupted during training)
HISTORY_TRUNCATE_P = 0.15    # P(history randomly truncated to a shorter prefix)
NOISE_SEED = 1234


# ---------------------------------------------------------------------------
# HF dataset loading and normalization
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


def normalize_hf_record(row: dict) -> dict:
    """Pair each English turn with its dialectal reference by turn_order."""
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


def load_hf_records(dataset_name: str, countries: list[str], split: str) -> list[dict]:
    from datasets import load_dataset

    records: list[dict] = []
    for country in tqdm(countries, desc=f"Loading HF {split} countries", unit="country"):
        dataset = load_dataset(dataset_name, country, split=split)
        country_records = [normalize_hf_record(row) for row in dataset]
        records.extend(country_records)
    return records


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

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


def corrupt_translation(text: str, rng: random.Random) -> str:
    """Word-level corruption (drop/swap/duplicate) used to simulate an
    imperfect translation history during training."""
    words = text.split()
    if len(words) < 3:
        return text

    mode = rng.choice(["drop", "swap", "duplicate"])
    words = list(words)

    if mode == "drop":
        k = max(1, int(len(words) * rng.uniform(0.10, 0.25)))
        for _ in range(k):
            if len(words) > 2:
                words.pop(rng.randrange(len(words)))
    elif mode == "swap":
        k = max(1, int(len(words) * rng.uniform(0.10, 0.20)))
        for _ in range(k):
            i = rng.randrange(len(words) - 1)
            words[i], words[i + 1] = words[i + 1], words[i]
    else:  # duplicate
        i = rng.randrange(len(words))
        words.insert(i, words[i])

    return " ".join(words)


def noise_history(
    history: list[dict],
    rng: random.Random,
    noise_p: float = HISTORY_NOISE_P,
    truncate_p: float = HISTORY_TRUNCATE_P,
) -> list[dict]:
    """Randomly truncate the history and/or corrupt individual turns'
    translations, as shown in the prompt (does not mutate the input)."""
    if not history:
        return history

    if rng.random() < truncate_p and len(history) > 1:
        history = history[: rng.randrange(1, len(history) + 1)]

    noised = []
    for item in history:
        translation = item["translation"]
        if rng.random() < noise_p:
            translation = corrupt_translation(translation, rng)
        noised.append({**item, "translation": translation})
    return noised


def create_finetuning_pairs(
    record: dict,
    noise: bool = False,
    rng: random.Random | None = None,
    noise_p: float = HISTORY_NOISE_P,
    truncate_p: float = HISTORY_TRUNCATE_P,
) -> list[dict]:
    """Flatten a conversation into one (prompt, response) pair per turn.

    When `noise` is True, the history shown in the prompt is randomly
    corrupted/truncated (exposure-bias fix) via `rng` — but the CLEAN
    reference always propagates forward into the next turn's history, so
    corruption never compounds across turns.
    """
    pairs = []
    history: list[dict] = []

    for turn in sorted_turns(record.get("turns", [])):
        sentence = turn.get("sentence", "").strip()
        reference = turn.get("reference", "").strip()
        if not sentence or not reference:
            continue

        prompt_history = noise_history(history, rng, noise_p, truncate_p) if noise else history
        pairs.append({
            "prompt": build_prompt(record, turn, list(prompt_history)),
            "response": reference,
            "country": record.get("country", ""),
            "conv_id": record.get("conv_id", ""),
            "turn_order": turn.get("turn_order"),
        })
        history.append({"speaker": turn.get("speaker", ""), "sentence": sentence, "translation": reference})

    return pairs


def format_to_messages(pair: dict) -> dict:
    return {
        "messages": [
            {"role": "system", "content": system_prompt()},
            {"role": "user", "content": pair["prompt"]},
            {"role": "assistant", "content": pair["response"]},
        ]
    }


def write_jsonl(records: list[dict], path: Path, desc: str | None = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in tqdm(records, desc=desc or f"Writing {path.name}", unit="example"):
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory to write alex_train.jsonl / alex_dev.jsonl")
    parser.add_argument("--dataset-name", default=DATASET_NAME, help="HF dataset to load.")
    parser.add_argument("--countries", nargs="+", default=COUNTRIES, help="Candidate country configs to check for a train split.")
    parser.add_argument("--hf-token", default=None, help="HF token (falls back to HF_TOKEN env var if set).")
    parser.add_argument("--no-history-noise", action="store_true",
                         help="Disable history noising on the train split (on by default).")
    parser.add_argument("--history-noise-p", type=float, default=HISTORY_NOISE_P,
                         help="P(a history translation gets corrupted) when noising is enabled.")
    parser.add_argument("--history-truncate-p", type=float, default=HISTORY_TRUNCATE_P,
                         help="P(history randomly truncated to a shorter prefix) when noising is enabled.")
    parser.add_argument("--noise-seed", type=int, default=NOISE_SEED, help="Seed for the history-noising RNG.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    hf_token = args.hf_token or os.environ.get("HF_TOKEN")
    if hf_token:
        from huggingface_hub import login
        login(token=hf_token)

    hf_split_map = discover_hf_splits(args.dataset_name, args.countries)
    train_countries = [c for c in args.countries if "train" in hf_split_map.get(c, set())]
    missing = [c for c in args.countries if c not in train_countries]
    print("Countries with an HF train split:", train_countries)
    if missing:
        print("Countries without an HF train split (skipped):", missing)

    noise_rng = random.Random(args.noise_seed)

    for split, out_name in [("train", "alex_train.jsonl"), ("dev", "alex_dev.jsonl")]:
        records = load_hf_records(args.dataset_name, train_countries, split)

        # History noising (exposure-bias fix) is only applied to the train
        # split; dev is kept on clean gold history for stable evaluation.
        noise = split == "train" and not args.no_history_noise
        pairs = [
            pair
            for record in records
            for pair in create_finetuning_pairs(
                record, noise=noise, rng=noise_rng,
                noise_p=args.history_noise_p, truncate_p=args.history_truncate_p,
            )
        ]
        messages = [format_to_messages(pair) for pair in pairs]
        print(f"{split}: {len(records)} conversations -> {len(messages)} turns (history noising: {noise})")
        write_jsonl(messages, args.output_dir / out_name, desc=f"Writing {out_name}")


if __name__ == "__main__":
    main()