#!/usr/bin/env python3
"""
score.py — Score a predictions JSONL (from models/infer.py, --data-source
dev) against the AlexandriaX HF dev references, computing per-country and
average spBLEU (flores200 tokenizer) and chrF++ (word_order=2).

This is dev-only: the test split has no public references, so there is
nothing to score --data-source test predictions against here — those are
scored by the shared-task organizers on submission.

Output
------
A scores.json with per-country spbleu_<COUNTRY> / chrfpp_<COUNTRY>, plus
spbleu_avg / chrfpp_avg / num_countries / num_conversations / num_turns,
and a console table (via pandas if available, else JSON).

Requirements
------------
pip install sacrebleu sentencepiece datasets tqdm pandas

Usage
-----
python score.py --predictions-path development_predictions.jsonl \
    --output-path development_scores.json
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

DATASET_NAME = "UBC-NLP/alexandria"
COUNTRIES = ["EG", "JO", "LB", "LY", "MA", "MR", "OM", "PS", "SA", "SD", "SY", "TN", "YE"]


# ---------------------------------------------------------------------------
# Reference loading (same normalization as models/infer.py)
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
            "reference": str(reference_turn.get("text", "")).strip(),
        })

    return {
        "conv_id": str(row.get("conv_id", "")).strip(),
        "country": str(row.get("country", "")).strip(),
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
    return records


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def prediction_map(prediction_records: list[dict]) -> dict[tuple[str, str, int], str]:
    mapped = {}
    for record in prediction_records:
        country = str(record.get("country", "")).strip()
        conv_id = str(record.get("conv_id", "")).strip()
        for index, turn in enumerate(record.get("turns", []), start=1):
            turn_order = int(turn.get("turn_order", index))
            key = (country, conv_id, turn_order)
            if key in mapped:
                raise ValueError(f"Duplicate prediction key: {key}")
            mapped[key] = str(turn.get("prediction", "")).strip()
    return mapped


def reference_map(reference_records: list[dict]) -> dict[tuple[str, str, int], str]:
    mapped = {}
    for record in reference_records:
        country = str(record.get("country", "")).strip()
        conv_id = str(record.get("conv_id", "")).strip()
        for index, turn in enumerate(sorted_turns(record.get("turns", [])), start=1):
            reference = str(turn.get("reference", "")).strip()
            if not reference:
                continue
            turn_order = int(turn.get("turn_order", index))
            key = (country, conv_id, turn_order)
            if key in mapped:
                raise ValueError(f"Duplicate reference key: {key}")
            mapped[key] = reference
    return mapped


def score_prediction_records(
    prediction_records: list[dict],
    reference_records: list[dict],
    output_path: Path | None = None,
    desc: str = "Scoring countries",
) -> dict:
    try:
        from sacrebleu.metrics import BLEU, CHRF
    except ImportError as exc:
        raise ImportError("Install scoring dependencies with: pip install sacrebleu sentencepiece") from exc
    from tqdm import tqdm

    predictions = prediction_map(prediction_records)
    references = reference_map(reference_records)

    missing = sorted(set(references) - set(predictions))
    extra = sorted(set(predictions) - set(references))
    if missing or extra:
        raise ValueError(f"Prediction/reference mismatch. Missing={missing[:5]}, extra={extra[:5]}")

    try:
        bleu = BLEU(tokenize="flores200", effective_order=False)
    except Exception as exc:
        raise RuntimeError(
            "Could not initialize SacreBLEU's flores200 tokenizer. "
            "Install/upgrade sacrebleu and sentencepiece."
        ) from exc
    chrf = CHRF(word_order=2)

    by_country: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for key in tqdm(sorted(references), desc="Aligning predictions", unit="turn"):
        country = key[0]
        by_country[country].append((predictions[key], references[key]))

    scores: dict[str, float | int] = {}
    spbleu_values = []
    chrfpp_values = []
    for country, rows in tqdm(sorted(by_country.items()), desc=desc, unit="country"):
        hypotheses = [prediction for prediction, _ in rows]
        refs = [reference for _, reference in rows]
        spbleu = bleu.corpus_score(hypotheses, [refs]).score
        chrfpp = chrf.corpus_score(hypotheses, [refs]).score
        spbleu_values.append(spbleu)
        chrfpp_values.append(chrfpp)
        scores[f"spbleu_{country}"] = round(spbleu, 6)
        scores[f"chrfpp_{country}"] = round(chrfpp, 6)

    scores["spbleu_avg"] = round(sum(spbleu_values) / len(spbleu_values), 6)
    scores["chrfpp_avg"] = round(sum(chrfpp_values) / len(chrfpp_values), 6)
    scores["num_countries"] = len(by_country)
    scores["num_turns"] = len(references)
    scores["num_conversations"] = len({key[:2] for key in references})

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(scores, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
        print(f"Scores saved to {output_path}")

    return scores


def display_scores(scores: dict) -> None:
    summary_keys = ["spbleu_avg", "chrfpp_avg", "num_countries", "num_conversations", "num_turns"]
    country_rows = []
    for key, value in scores.items():
        if not key.startswith("spbleu_") or key == "spbleu_avg":
            continue
        country = key.removeprefix("spbleu_")
        country_rows.append({
            "country": country,
            "spbleu": value,
            "chrfpp": scores.get(f"chrfpp_{country}"),
        })

    try:
        import pandas as pd

        summary_df = pd.DataFrame([{key: scores[key] for key in summary_keys}])
        country_df = pd.DataFrame(country_rows).sort_values("country").reset_index(drop=True)
        print(summary_df.to_string(index=False))
        print(country_df.to_string(index=False))
    except ImportError:
        print(json.dumps({key: scores[key] for key in summary_keys}, indent=2, ensure_ascii=False))
        print(json.dumps(sorted(country_rows, key=lambda row: row["country"]), indent=2, ensure_ascii=False))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--predictions-path", type=Path, required=True,
                         help="JSONL from models/infer.py --data-source dev.")
    parser.add_argument("--references-path", type=Path, default=None,
                         help="Optional local JSONL of dev records with 'reference' fields "
                              "(same normalized shape infer.py uses). If omitted, references "
                              "are reloaded from the HF dataset directly.")
    parser.add_argument("--countries", nargs="+", default=COUNTRIES,
                         help="Only used when --references-path is not given.")
    parser.add_argument("--dataset-name", default=DATASET_NAME)
    parser.add_argument("--output-path", type=Path, default=Path("development_scores.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    predictions = read_jsonl(args.predictions_path)
    print(f"Loaded {len(predictions)} prediction conversations from {args.predictions_path}")

    if args.references_path is not None:
        references = read_jsonl(args.references_path)
    else:
        hf_split_map = discover_hf_splits(args.dataset_name, args.countries)
        dev_countries = [c for c in args.countries if "train" in hf_split_map.get(c, set())]
        references = load_hf_dev_records(args.dataset_name, dev_countries)
    print(f"Loaded {len(references)} reference conversations")

    scores = score_prediction_records(predictions, references, args.output_path)
    display_scores(scores)


if __name__ == "__main__":
    main()
