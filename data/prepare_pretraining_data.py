#!/usr/bin/env python3
"""
prepare_pretraining_data.py — Turn the MADAR/PADIC parallel files produced by
preprocess_madar_padic.py into chat-format JSONL for unconstrained-track
pretraining.

Pipeline
--------
1. Read madar_train.jsonl, madar_dev.jsonl, padic_train.jsonl, padic_dev.jsonl
   — each a list of {"country": <dialect name>, "sentence": <English>,
   "reference": <dialect text>} records.
2. Build a translation prompt per record (English -> dialect, with the
   short country code looked up from the full dialect name for the
   metadata line) and a fixed pretraining system prompt.
3. Wrap into {"messages": [system, user, assistant]} chat schema.
4. Write the four *_messages.jsonl files, one per input file.

This mirrors prepare_alex_data.py's chat schema but uses the simpler
pretraining prompt (no domain/participants/speaker/history — MADAR and
PADIC are isolated sentence pairs, not multi-turn dialogues).

Output format (one JSON object per line, per file):
    {"messages": [
        {"role": "system", "content": "<translator instructions>"},
        {"role": "user", "content": "<prompt with country + sentence>"},
        {"role": "assistant", "content": "<dialectal reference>"}
    ]}

Usage
-----
python prepare_pretraining_data.py \
    --input-dir ./data/processed \
    --output-dir ./data/processed
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tqdm import tqdm

# Full dialect name (as stored in the "country" field by
# preprocess_madar_padic.py) -> short country code shown in the prompt.
DIALECT_TO_COUNTRY_CODE = {
    "Egyptian Arabic": "EG",
    "Jordanian Arabic": "JO",
    "Lebanese Arabic": "LB",
    "Libyan Arabic": "LY",
    "Moroccan Arabic": "MA",
    "Mauritanian Arabic": "MR",
    "Omani Arabic": "OM",
    "Palestinian Arabic": "PS",
    "Saudi Arabic": "SA",
    "Sudanese Arabic": "SD",
    "Syrian Arabic": "SY",
    "Tunisian Arabic": "TN",
    "Yemeni Arabic": "YE",
}

INPUT_FILES = ["madar_train.jsonl", "madar_dev.jsonl", "padic_train.jsonl", "padic_dev.jsonl"]


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(records: list[dict], path: Path, desc: str | None = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in tqdm(records, desc=desc or f"Writing {path.name}", unit="example"):
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return path


def build_prompt(record: dict) -> str:
    dialect = record.get("country") or "Arabic Dialect"
    country_code = DIALECT_TO_COUNTRY_CODE.get(dialect, "")
    return (
        f"Translate the English sentence into {dialect}.\n\n"
        f"### Metadata:\n"
        f"- Country: {country_code}\n"
        f"### Sentence to Translate:\n"
        f"{record.get('sentence', '').strip()}\n\n"
    )


def system_prompt() -> str:
    return (
        "You are an expert translator.\n\n"
        "- Return only the translated text.\n"
        "- Do not add any code, explanations, comments, or any other extra text.\n"
        "- Consider the country in your translation.\n"
    )


def format_to_messages(record: dict) -> dict:
    return {
        "messages": [
            {"role": "system", "content": system_prompt()},
            {"role": "user", "content": build_prompt(record)},
            {"role": "assistant", "content": record.get("reference", "").strip()},
        ]
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-dir", type=Path, required=True,
                         help="Directory containing madar_train.jsonl, madar_dev.jsonl, padic_train.jsonl, padic_dev.jsonl")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory to write the *_messages.jsonl files")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for filename in INPUT_FILES:
        input_path = args.input_dir / filename
        if not input_path.exists():
            print(f"Skipping {filename}: not found in {args.input_dir}")
            continue

        records = read_jsonl(input_path)
        messages = [format_to_messages(record) for record in records]

        output_name = input_path.stem + "_messages.jsonl"
        write_jsonl(messages, args.output_dir / output_name, desc=f"Writing {output_name}")
        print(f"{filename}: {len(records)} pairs -> {output_name}")


if __name__ == "__main__":
    main()
