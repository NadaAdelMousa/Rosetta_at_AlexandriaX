#!/usr/bin/env python3
"""
embeddings.py — Build a retrieval index over the Alex train set (embedded
with a fine-tuned NileChat checkpoint) and retrieve top-k similar examples
for dev/test/single queries, for few-shot-augmented inference.

This was an exploratory retrieval-augmented-inference trial and is NOT part
of the official AlexandriaX submission pipeline — but it plugs into
models/infer.py via --fewshot-examples-path if you want to use it.

Two modes
---------
--mode index
    Embed every turn of the train set (last-token pooled, L2-normalized
    hidden states from the fine-tuned checkpoint) and save:
        <index-dir>/train_records.jsonl   flattened train turns + metadata
        <index-dir>/train_embeddings.pt   matching (N, dim) embedding tensor
    At retrieval time these are loaded and re-grouped into an in-memory
    dialect -> domain -> FAISS(IndexFlatIP) nested index (cheap to rebuild,
    so nothing FAISS-specific is serialized to disk).

--mode retrieve
    Embed queries from --query-source {dev, test, single} and, for each,
    retrieve the top-k nearest train examples — preferring same
    dialect+domain, falling back to same-dialect-only when the domain
    sub-index is too sparse (< --min-domain-examples). Writes:
        {"conv_id": <str>, "turn_order": <int>, "country": <str>,
         "fewshot_examples_text": "<formatted English/translation blocks>"}
    one line per query turn — this is the --fewshot-examples-path file
    models/infer.py consumes.
    --query-source single instead prints the retrieved examples for one
    ad-hoc sentence (useful for sanity-checking retrieval quality).

Requirements
------------
pip install torch transformers peft faiss-cpu pandas datasets tqdm

Usage
-----
# 1. Build the index once from the train set
python embeddings.py --mode index --checkpoint-dir outputs/finetune_alex \
    --index-dir ./retrieval_index

# 2. Retrieve few-shot examples for the dev set
python embeddings.py --mode retrieve --query-source dev \
    --checkpoint-dir outputs/finetune_alex --index-dir ./retrieval_index \
    --output-path dev_fewshot_examples.jsonl

# Ad-hoc single-sentence sanity check
python embeddings.py --mode retrieve --query-source single \
    --checkpoint-dir outputs/finetune_alex --index-dir ./retrieval_index \
    --sentence "How are you doing today?" --dialect "Egyptian Arabic" --domain "Casual Chat"
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

DATASET_NAME = "UBC-NLP/alexandria"
COUNTRIES = ["EG", "JO", "LB", "LY", "MA", "MR", "OM", "PS", "SA", "SD", "SY", "TN", "YE"]
DEFAULT_MODEL_NAME = "UBC-NLP/NileChat-3B"
DEFAULT_WANDB_ENTITY = "RosettaAtAlexandriaX"
DEFAULT_WANDB_PROJECT = "DialectalArabicMT"


# ---------------------------------------------------------------------------
# Data loading / normalization (same shape as models/infer.py)
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


def write_jsonl(records: list[dict], path: Path, desc: str | None = None) -> Path:
    from tqdm import tqdm

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in tqdm(records, desc=desc or f"Writing {path.name}", unit="record"):
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return path


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


def load_hf_records(dataset_name: str, countries: list[str], split: str) -> list[dict]:
    from datasets import load_dataset
    from tqdm import tqdm

    records: list[dict] = []
    for country in tqdm(countries, desc=f"Loading HF {split} countries", unit="country"):
        dataset = load_dataset(dataset_name, country, split=split)
        country_records = [normalize_hf_record(row) for row in dataset]
        records.extend(country_records)
        print(f"Loaded {len(country_records):>5} {split} conversations for {country}")
    return records


def load_test_records(test_data_dir: Path, file_pattern: str, countries: list[str]) -> list[dict]:
    from tqdm import tqdm

    records: list[dict] = []
    for country in tqdm(countries, desc="Loading test countries", unit="country"):
        path = test_data_dir / file_pattern.format(country=country)
        if not path.exists():
            print(f"  {path}: not found, skipping")
            continue
        records.extend(read_jsonl(path))
    return records


def flatten_conversations(raw_data: list[dict], context_window: int = 2) -> list[dict]:
    """One record per turn, with the embedding text = the preceding
    `context_window` turns' English sentences plus the current sentence."""
    records = []
    for conv in raw_data:
        eng_turns = sorted_turns(conv["turns"])
        for i, eng_turn in enumerate(eng_turns):
            context_turns = eng_turns[max(0, i - context_window):i]
            context_str = "\n".join(f"{t['speaker']}: {t['sentence']}" for t in context_turns)
            records.append({
                "conv_id": conv["conv_id"],
                "turn_order": eng_turn["turn_order"],
                "country": conv["country"],
                "domain": conv["domain"],
                "dialect": conv["dialect"],
                "speaker": eng_turn["speaker"],
                "direction": eng_turn["direction"],
                "context_text": context_str,
                "source_text": eng_turn["sentence"],
                "target_text": eng_turn.get("reference", ""),
                "embed_text": (context_str + "\n" if context_str else "") + f"{eng_turn['speaker']}: {eng_turn['sentence']}",
            })
    return records


def build_single_query_record(sentence: str, dialect: str, domain: str, speaker: str = "", context: str = "") -> dict:
    return {
        "conv_id": "__single__",
        "turn_order": 1,
        "country": "",
        "domain": domain,
        "dialect": dialect,
        "speaker": speaker,
        "direction": "",
        "context_text": context,
        "source_text": sentence,
        "target_text": "",
        "embed_text": (context + "\n" if context else "") + (f"{speaker}: {sentence}" if speaker else sentence),
    }


def resolve_checkpoint_dir(args) -> Path:
    if args.checkpoint_dir:
        return Path(args.checkpoint_dir)

    if args.wandb_artifact:
        import wandb

        run = wandb.init(entity=args.wandb_entity, project=args.wandb_project, job_type="embeddings")
        artifact = run.use_artifact(args.wandb_artifact, type="model")
        artifact_dir = Path(artifact.download())
        print(f"Checkpoint downloaded to: {artifact_dir}")
        return artifact_dir

    raise ValueError("Provide either --checkpoint-dir or --wandb-artifact.")


def load_model_and_tokenizer(checkpoint_dir: Path, model_name: str, load_in_4bit: bool):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    adapter_config_path = checkpoint_dir / "adapter_config.json"
    if not adapter_config_path.exists():
        raise FileNotFoundError(f"No LoRA adapter found at {checkpoint_dir}.")

    tokenizer_source = checkpoint_dir if (checkpoint_dir / "tokenizer_config.json").exists() else model_name
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    quantization_config = None
    if load_in_4bit:
        from transformers import BitsAndBytesConfig
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True,
        )

    base_model = AutoModelForCausalLM.from_pretrained(
        model_name, device_map="auto", torch_dtype=torch.float16,
        quantization_config=quantization_config, trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base_model, checkpoint_dir)
    if hasattr(model, "gradient_checkpointing_disable"):
        model.gradient_checkpointing_disable()
    model.eval()
    model.config.use_cache = True

    print(f"Loaded fine-tuned model from: {checkpoint_dir}")
    return model, tokenizer


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

def build_embed_input(tokenizer, record: dict) -> str:
    messages = [{"role": "user", "content": record["embed_text"]}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def embed(records: list[dict], model, tokenizer, batch_size: int = 16, max_length: int = 512):
    """Last-token pooled, L2-normalized hidden states for a list of
    flattened turn records (as produced by flatten_conversations)."""
    import torch
    from tqdm import tqdm

    embeddings = []
    with torch.no_grad():
        for i in tqdm(range(0, len(records), batch_size), desc="Embedding", unit="batch"):
            batch = [build_embed_input(tokenizer, r) for r in records[i:i + batch_size]]
            inputs = tokenizer(batch, padding=True, truncation=True, return_tensors="pt", max_length=max_length).to(model.device)
            outputs = model(**inputs, output_hidden_states=True)
            hidden = outputs.hidden_states[-1]

            seq_lengths = inputs["attention_mask"].sum(dim=1) - 1
            pooled = hidden[torch.arange(hidden.size(0)), seq_lengths]
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
            embeddings.append(pooled.cpu())
    return torch.cat(embeddings)


# ---------------------------------------------------------------------------
# Nested dialect -> domain FAISS index
# ---------------------------------------------------------------------------

def build_dialect_domain_indices(records_df, embeddings) -> dict:
    """dialect -> {'_all': {index, records}, 'domains': {domain: {index, records}}}.
    Retrieval always stays within the query's dialect; the domain sub-index
    is used when it has enough examples, else falls back to the full
    dialect pool."""
    import faiss

    nested_indices: dict = {}
    for dialect, dialect_group in records_df.groupby("dialect"):
        dialect_idxs = dialect_group.index.to_numpy()
        dialect_embs = embeddings[dialect_idxs].numpy().astype("float32")
        all_index = faiss.IndexFlatIP(dialect_embs.shape[1])
        all_index.add(dialect_embs)
        nested_indices[dialect] = {
            "_all": {"index": all_index, "records": dialect_group.reset_index(drop=True)},
            "domains": {},
        }

        for domain, domain_group in dialect_group.groupby("domain"):
            domain_idxs = domain_group.index.to_numpy()
            domain_embs = embeddings[domain_idxs].numpy().astype("float32")
            domain_index = faiss.IndexFlatIP(domain_embs.shape[1])
            domain_index.add(domain_embs)
            nested_indices[dialect]["domains"][domain] = {
                "index": domain_index, "records": domain_group.reset_index(drop=True),
            }

    return nested_indices


def retrieve_fewshot_nested(nested_indices: dict, query_embedding, dialect: str, domain: str, k: int, min_domain_examples: int):
    """query_embedding: (1, dim) float32 numpy array. Returns a DataFrame of
    the top-k matches (with a retrieval_score column), or None if the
    dialect has no index at all."""
    dialect_entry = nested_indices.get(dialect)
    if dialect_entry is None:
        return None

    domain_entry = dialect_entry["domains"].get(domain)
    entry = domain_entry if (domain_entry is not None and domain_entry["index"].ntotal >= min_domain_examples) else dialect_entry["_all"]

    search_k = min(k, entry["index"].ntotal)
    if search_k == 0:
        return None

    scores, idxs = entry["index"].search(query_embedding, search_k)
    return entry["records"].iloc[idxs[0]].assign(retrieval_score=scores[0])


def format_fewshot_examples(fewshot_df) -> str:
    if fewshot_df is None or len(fewshot_df) == 0:
        return ""
    blocks = [
        f"English: {row['source_text'].strip()}\nTranslation: {row['target_text'].strip()}"
        for _, row in fewshot_df.iterrows()
    ]
    return "\n\n".join(blocks)


# ---------------------------------------------------------------------------
# Mode: index
# ---------------------------------------------------------------------------

def run_index_mode(args) -> None:
    import pandas as pd

    hf_split_map = discover_hf_splits(args.dataset_name, args.countries)
    train_countries = [c for c in args.countries if "train" in hf_split_map.get(c, set())]
    train_conversations = load_hf_records(args.dataset_name, train_countries, "train")

    records = flatten_conversations(train_conversations, context_window=args.context_window)
    print(f"Flattened to {len(records)} train turns to embed.")

    checkpoint_dir = resolve_checkpoint_dir(args)
    model, tokenizer = load_model_and_tokenizer(checkpoint_dir, args.model_name, args.load_in_4bit)

    embeddings = embed(records, model, tokenizer, batch_size=args.embed_batch_size)

    args.index_dir.mkdir(parents=True, exist_ok=True)
    embed_only_records = [{k: v for k, v in r.items() if k != "embed_text"} for r in records]
    write_jsonl(embed_only_records, args.index_dir / "train_records.jsonl", desc="Writing train_records.jsonl")

    import torch
    torch.save(embeddings, args.index_dir / "train_embeddings.pt")
    print(f"Saved {embeddings.shape[0]} embeddings (dim={embeddings.shape[1]}) to {args.index_dir / 'train_embeddings.pt'}")


# ---------------------------------------------------------------------------
# Mode: retrieve
# ---------------------------------------------------------------------------

def load_index(index_dir: Path):
    import pandas as pd
    import torch

    records = read_jsonl(index_dir / "train_records.jsonl")
    records_df = pd.DataFrame(records)
    embeddings = torch.load(index_dir / "train_embeddings.pt")
    print(f"Loaded index: {len(records_df)} train records, embeddings {tuple(embeddings.shape)}")
    return records_df, embeddings


def run_retrieve_mode(args) -> None:
    records_df, train_embeddings = load_index(args.index_dir)

    checkpoint_dir = resolve_checkpoint_dir(args)
    model, tokenizer = load_model_and_tokenizer(checkpoint_dir, args.model_name, args.load_in_4bit)

    nested_indices = build_dialect_domain_indices(records_df, train_embeddings)

    if args.query_source == "single":
        if not args.sentence or not args.dialect or not args.domain:
            raise ValueError("--query-source single requires --sentence, --dialect, and --domain.")
        query_record = build_single_query_record(args.sentence, args.dialect, args.domain, args.speaker or "", args.context or "")
        query_embedding = embed([query_record], model, tokenizer, batch_size=1).numpy().astype("float32")
        matches = retrieve_fewshot_nested(nested_indices, query_embedding, args.dialect, args.domain, args.top_k, args.min_domain_examples)
        if matches is None:
            print(f"No index available for dialect '{args.dialect}'.")
            return
        print(matches[["source_text", "target_text", "dialect", "domain", "retrieval_score"]].to_string(index=False))
        print("\nFormatted few-shot block:\n")
        print(format_fewshot_examples(matches))
        return

    if args.query_source == "dev":
        hf_split_map = discover_hf_splits(args.dataset_name, args.countries)
        query_countries = [c for c in args.countries if "train" in hf_split_map.get(c, set())]
        conversations = load_hf_records(args.dataset_name, query_countries, "dev")
    else:  # test
        conversations = load_test_records(args.test_data_dir, args.test_file_pattern, args.countries)

    query_records = flatten_conversations(conversations, context_window=args.context_window)
    print(f"Flattened to {len(query_records)} {args.query_source} turns to retrieve for.")

    query_embeddings = embed(query_records, model, tokenizer, batch_size=args.embed_batch_size).numpy().astype("float32")

    output_records = []
    for i, record in enumerate(query_records):
        matches = retrieve_fewshot_nested(
            nested_indices, query_embeddings[i:i + 1], record["dialect"], record["domain"],
            args.top_k, args.min_domain_examples,
        )
        output_records.append({
            "conv_id": record["conv_id"],
            "turn_order": record["turn_order"],
            "country": record["country"],
            "fewshot_examples_text": format_fewshot_examples(matches),
        })

    write_jsonl(output_records, args.output_path, desc=f"Writing {args.query_source} few-shot examples")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=["index", "retrieve"], required=True)

    parser.add_argument("--dataset-name", default=DATASET_NAME)
    parser.add_argument("--countries", nargs="+", default=COUNTRIES)
    parser.add_argument("--context-window", type=int, default=2,
                         help="Preceding turns folded into each embedding's context.")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--checkpoint-dir", type=Path)
    group.add_argument("--wandb-artifact")
    parser.add_argument("--wandb-entity", default=DEFAULT_WANDB_ENTITY)
    parser.add_argument("--wandb-project", default=DEFAULT_WANDB_PROJECT)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--load-in-4bit", action="store_true", default=True)
    parser.add_argument("--embed-batch-size", type=int, default=16)

    parser.add_argument("--index-dir", type=Path, required=True,
                         help="Where to write (--mode index) or read (--mode retrieve) train_records.jsonl / train_embeddings.pt")

    # --mode retrieve options
    parser.add_argument("--query-source", choices=["dev", "test", "single"], default="dev")
    parser.add_argument("--test-data-dir", type=Path, default=Path("testdataalex"))
    parser.add_argument("--test-file-pattern", default="alexandria_{country}_private_test_input.jsonl")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--min-domain-examples", type=int, default=3,
                         help="Fall back to the dialect-wide pool if the dialect+domain sub-index has fewer than this many examples.")
    parser.add_argument("--output-path", type=Path, default=None,
                         help="Required for --query-source dev/test. Ignored for 'single'.")

    # --query-source single options
    parser.add_argument("--sentence", default=None)
    parser.add_argument("--dialect", default=None)
    parser.add_argument("--domain", default=None)
    parser.add_argument("--speaker", default=None)
    parser.add_argument("--context", default=None, help="Optional preceding-turns text.")

    args = parser.parse_args()
    if args.mode == "retrieve" and args.query_source != "single" and args.output_path is None:
        parser.error("--output-path is required for --query-source dev/test.")
    return args


def main() -> None:
    args = parse_args()
    if args.mode == "index":
        run_index_mode(args)
    else:
        run_retrieve_mode(args)


if __name__ == "__main__":
    main()
