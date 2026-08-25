#!/usr/bin/env python3
"""
preprocess_madar_padic.py — Build parallel (English -> Dialect) MADAR and
PADIC datasets for AlexandriaX unconstrained-track pretraining.

Pipeline
--------
MADAR (per split, "train" and "dev"):
    1. Read the MSA TSV for that split and translate every MSA sentence to
       English with NLLB-200.
    2. For each target city's dialect TSV, keep that split's rows and merge
       them with the English translations on the shared sentence ID.
    3. Tag every row with its country (via CITY_TO_DIALECT) and write
       madar_<split>.jsonl.
    MADAR's own "dev" split is used as-is for madar_dev.jsonl — there is no
    extra held-out split.

PADIC:
    1. Parse PADIC.xml into one row per <sentence> element.
    2. PADIC's raw XML repeats several columns' values on one extra row per
       9-row block. Drop those duplicate rows and
       re-index each block so the columns line up as true aligned sentences.
    3. Translate the aligned MODERN-STANDARD-ARABIC column into English.
    4. Melt the wide dialect columns (ALGIERS, ANNABA, SYRIAN, PALESTINIAN,
       MOROCCAN) into long (country, sentence, reference) rows, keeping
       only dialects in TARGET_DIALECTS.
    5. Stratified 90/10 split by country -> padic_train.jsonl / padic_dev.jsonl.
       (PADIC has no official dev split, so this script creates one.)

Output format (all 4 files, one JSON object per line):
    {"country": <str>, "sentence": <English text>, "reference": <dialect text>}

These are the inputs consumed downstream by the pretraining prompt-building
script, which turns them into {"messages": [...]} chat-format JSONL.

Requirements
------------
pip install pandas torch transformers tqdm scikit-learn camel-tools 


Usage
-----
python preprocess_madar_padic.py \
    --madar-dir <path/to/MADAR_Corpus> \
    --padic-xml <path/to/PADIC.xml> \
    --output-dir <path/to/output_dir>
"""

from __future__ import annotations

import argparse
import glob
import logging
import os
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


# MADAR city -> country/dialect label used throughout the pipeline.
CITY_TO_DIALECT = {
    "Rabat": "Moroccan Arabic",
    "Fes": "Moroccan Arabic",
    "Cairo": "Egyptian Arabic",
    "Alexandria": "Egyptian Arabic",
    "Aswan": "Egyptian Arabic",
    "Tunis": "Tunisian Arabic",
    "Sfax": "Tunisian Arabic",
    "Tripoli": "Libyan Arabic",
    "Benghazi": "Libyan Arabic",
    "Beirut": "Lebanese Arabic",
    "Damascus": "Syrian Arabic",
    "Aleppo": "Syrian Arabic",
    "Amman": "Jordanian Arabic",
    "Salt": "Jordanian Arabic",
    "Jerusalem": "Palestinian Arabic",
    "Riyadh": "Saudi Arabic",
    "Jeddah": "Saudi Arabic",
    "Sanaa": "Yemeni Arabic",
    "Khartoum": "Sudanese Arabic",
    "Muscat": "Omani Arabic",
}

# PADIC XML column -> country/dialect label.
PADIC_COLUMN_TO_COUNTRY = {
    "ALGIERS": "Algerian Arabic",
    "ANNABA": "Algerian Arabic",
    "SYRIAN": "Syrian Arabic",
    "PALESTINIAN": "Palestinian Arabic",
    "MOROCCAN": "Moroccan Arabic",
}

# Only these dialects are kept for pretraining (matches the AlexandriaX
# shared-task country set).
TARGET_DIALECTS = {
    "Moroccan Arabic", "Lebanese Arabic", "Jordanian Arabic",
    "Palestinian Arabic", "Omani Arabic", "Tunisian Arabic",
    "Sudanese Arabic", "Libyan Arabic", "Mauritanian Arabic",
    "Syrian Arabic", "Saudi Arabic", "Egyptian Arabic", "Yemeni Arabic",
}

# PADIC's XML repeats each of these columns' value on one extra row per
# 9-row block, at this fixed position within the block (0-indexed). E.g.
# "MOROCCAN" duplicates onto position 1 of every block. Reverse-engineered
# from the raw corpus by checking which rows equal the row directly above
# them, and confirmed to recur every 9 rows.
PADIC_ROW_OFFSETS = {
    "MOROCCAN": 1,
    "SYRIAN": 2,
    "PALESTINIAN": 3,
    "ANNABA": 5,
    "ALGIERS": 6,
    "MODERN-STANDARD-ARABIC": 7,
}
PADIC_BLOCK_SIZE = 9

DEFAULT_NLLB_MODEL = "facebook/nllb-200-3.3B"


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def fix_tokenized_spaces(text: str) -> str:
    """Undo NLLB/tokenizer detokenization artifacts, e.g. "It 's" -> "It's"."""
    if not isinstance(text, str):
        return text
    text = re.sub(r"(\w+)\s+'(s|m|re|ve|ll|d|t)\b", r"\1'\2", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+'\s+", "'", text)
    text = re.sub(r"\s+([.,!?;:]+)", r"\1", text)
    text = re.sub(r"\(\s+", "(", text)
    text = re.sub(r"\s+\)", ")", text)
    return re.sub(r"\s+", " ", text).strip()


def write_jsonl(df: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_json(path, orient="records", lines=True, force_ascii=False)
    logger.info("Wrote %d records to %s", len(df), path)
    return path


def load_nllb(model_name: str, device: str):
    """Load an NLLB checkpoint in fp16, source language fixed to Arabic."""
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    logger.info("Loading %s on %s", model_name, device)
    tokenizer = AutoTokenizer.from_pretrained(model_name, src_lang="arb_Arab")
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, torch_dtype=torch.float16).to(device)
    model.eval()
    return tokenizer, model


def translate_batch(
    texts: list[str],
    tokenizer,
    model,
    device: str,
    batch_size: int = 32,
    max_length: int = 256,
    num_beams: int = 4,
) -> list[str]:
    """Translate Arabic (MSA) sentences to English with NLLB, batched."""
    import torch
    from tqdm import tqdm

    target_lang_id = tokenizer.convert_tokens_to_ids("eng_Latn")
    outputs: list[str] = []

    for i in tqdm(range(0, len(texts), batch_size), desc="Translating MSA -> English"):
        batch = texts[i:i + batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to(device)
        with torch.no_grad():
            generated = model.generate(
                **inputs,
                forced_bos_token_id=target_lang_id,
                max_length=max_length,
                num_beams=num_beams,
            )
        outputs.extend(tokenizer.batch_decode(generated, skip_special_tokens=True))

    return outputs


# ---------------------------------------------------------------------------
# MADAR
# ---------------------------------------------------------------------------


def build_madar_split(
    madar_dir: Path,
    split: str,
    tokenizer,
    model,
    device: str,
    batch_size: int,
) -> pd.DataFrame:
    """Build a (country, sentence, reference) frame for one MADAR split."""
    msa_path = madar_dir / "MADAR.corpus.MSA.tsv"
    df_msa = pd.read_csv(
        msa_path, sep="\t", names=["sentID.BTEC", "split", "lang", "msa_sent"], header=None
    )
    df_msa = df_msa[df_msa["split"].str.contains(split, na=False)].reset_index(drop=True)
    logger.info("MADAR %s: %d MSA sentences to translate", split, len(df_msa))

    english = translate_batch(df_msa["msa_sent"].tolist(), tokenizer, model, device, batch_size)
    df_en = df_msa[["sentID.BTEC"]].copy()
    df_en["english_sent"] = [fix_tokenized_spaces(s) for s in english]

    dialect_frames = []
    for file_path in sorted(glob.glob(str(madar_dir / "*.tsv"))):
        filename = os.path.basename(file_path)
        city = filename.replace("MADAR.corpus.", "").replace(".tsv", "").replace(".index", "")
        if city not in CITY_TO_DIALECT:
            continue

        df_dialect = pd.read_csv(
            file_path, sep="\t", names=["sentID.BTEC", "split", "lang", "dialect_sent"], header=None
        )
        df_dialect = df_dialect[df_dialect["split"].str.contains(split, na=False)]

        merged = pd.merge(df_dialect, df_en, on="sentID.BTEC", how="inner")
        merged["country"] = CITY_TO_DIALECT[city]
        dialect_frames.append(merged[["country", "english_sent", "dialect_sent"]])

    combined = pd.concat(dialect_frames, ignore_index=True)
    combined = combined.rename(columns={"english_sent": "sentence", "dialect_sent": "reference"})
    logger.info("MADAR %s: %d parallel rows across %d dialects", split, len(combined), combined["country"].nunique())
    return combined[["country", "sentence", "reference"]]


# ---------------------------------------------------------------------------
# PADIC
# ---------------------------------------------------------------------------

def decode_buckwalter(text: str | None) -> str:
    """Clean whitespace and map Buckwalter-encoded XML text to Arabic script."""
    from camel_tools.utils.charmap import CharMapper

    if not text or not isinstance(text, str):
        return ""
    cleaned = re.sub(r"\s+", " ", text).strip()
    if not cleaned:
        return ""
    mapper = decode_buckwalter._mapper
    if mapper is None:
        mapper = CharMapper.builtin_mapper("bw2ar")
        decode_buckwalter._mapper = mapper
    try:
        return mapper.map_string(cleaned)
    except Exception:
        return cleaned


decode_buckwalter._mapper = None


def parse_padic_xml(xml_path: Path) -> pd.DataFrame:
    """Parse PADIC.xml into one row per <sentence>, decoding Buckwalter text."""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    rows = []
    for sentence in root:
        row = {}
        for child in sentence:
            text = child.text.strip() if child.text else None
            row[child.tag] = decode_buckwalter(text)
        rows.append(row)

    df = pd.DataFrame(rows)
    logger.info("Parsed PADIC XML: %s", df.shape)
    return df


def align_padic_blocks(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop PADIC's duplicated-row artifacts and re-align every 9-row block
    into one row per aligned sentence.

    See PADIC_ROW_OFFSETS for how the offsets were derived: for each
    affected column, the row at that fixed position within a block repeats
    the value from the row directly above it. We validate that assumption
    on every full block before trusting the drop.
    """
    df = df.copy()
    df["block_id"] = df.index // PADIC_BLOCK_SIZE
    df["pos_in_block"] = df.index % PADIC_BLOCK_SIZE

    block_sizes = df.groupby("block_id").size()
    full_blocks = block_sizes[block_sizes == PADIC_BLOCK_SIZE].index
    df_full = df[df["block_id"].isin(full_blocks)].copy()
    logger.info(
        "PADIC alignment: dropped %d incomplete block(s), %d row(s) total",
        df["block_id"].nunique() - len(full_blocks), len(df) - len(df_full),
    )

    validation_issues = {}
    for col, offset in PADIC_ROW_OFFSETS.items():
        at_offset = df_full[df_full["pos_in_block"] == offset]
        prev_rows = df_full.loc[at_offset.index - 1, col].values
        mismatches = at_offset.index[at_offset[col].values != prev_rows]
        if len(mismatches) > 0:
            validation_issues[col] = mismatches.tolist()
    if validation_issues:
        logger.warning("PADIC alignment: unexpected mismatches found: %s", validation_issues)
    else:
        logger.info("PADIC alignment: all full blocks validated against expected offset pattern")

    clean_cols = {}
    for col, offset in PADIC_ROW_OFFSETS.items():
        keep = df_full[df_full["pos_in_block"] != offset].copy()
        keep["new_pos"] = keep.groupby("block_id").cumcount()
        clean_cols[col] = keep.set_index(["block_id", "new_pos"])[col]

    aligned = pd.concat(clean_cols.values(), axis=1, keys=clean_cols.keys())
    aligned.columns = list(clean_cols.keys())
    return aligned.reset_index()


def build_padic_dataset(
    padic_xml: Path,
    tokenizer,
    model,
    device: str,
    batch_size: int,
) -> pd.DataFrame:
    """Build the full (country, sentence, reference) frame for PADIC, pre-split."""
    raw_df = parse_padic_xml(padic_xml)
    aligned_df = align_padic_blocks(raw_df)

    english = translate_batch(
        aligned_df["MODERN-STANDARD-ARABIC"].tolist(), tokenizer, model, device, batch_size
    )
    aligned_df["english_sent"] = [fix_tokenized_spaces(s) for s in english]

    combined_rows = []
    for col, country in PADIC_COLUMN_TO_COUNTRY.items():
        if country not in TARGET_DIALECTS or col not in aligned_df.columns:
            continue
        subset = aligned_df[[col, "english_sent"]].rename(columns={col: "reference", "english_sent": "sentence"})
        subset["country"] = country
        subset = subset.dropna(subset=["reference", "sentence"])
        subset = subset[subset["reference"].str.strip() != ""]
        subset = subset[subset["sentence"].str.strip() != ""]
        combined_rows.append(subset[["country", "sentence", "reference"]])

    combined = pd.concat(combined_rows, ignore_index=True)
    logger.info("PADIC: %d parallel rows across %d dialects", len(combined), combined["country"].nunique())
    return combined


def split_padic(df: pd.DataFrame, dev_fraction: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stratified train/dev split by country (PADIC has no official dev split)."""
    from sklearn.model_selection import train_test_split

    train_df, dev_df = train_test_split(
        df, test_size=dev_fraction, stratify=df["country"], random_state=seed
    )
    logger.info("PADIC split: %d train / %d dev (%.0f%% dev)", len(train_df), len(dev_df), dev_fraction * 100)
    return train_df, dev_df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--madar-dir", type=Path, default=None,
                         help="Path to MADAR_Corpus/ (containing the .tsv files).")
    parser.add_argument("--padic-xml", type=Path, default=None,
                         help="Path to PADIC.xml.")
    parser.add_argument("--output-dir", type=Path, required=True,
                         help="Directory to write madar_train.jsonl, madar_dev.jsonl, padic_train.jsonl, padic_dev.jsonl")
    parser.add_argument("--model-name", default=DEFAULT_NLLB_MODEL, help="NLLB checkpoint for MSA->English translation.")
    parser.add_argument("--device", default="cuda", help="torch device for translation (cuda/cpu).")
    parser.add_argument("--batch-size", type=int, default=32, help="Translation batch size.")
    parser.add_argument("--madar-splits", nargs="+", default=["train", "dev"], help="MADAR splits to build.")
    parser.add_argument("--padic-dev-fraction", type=float, default=0.10, help="Fraction of filtered PADIC held out as dev.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for the PADIC train/dev split.")
    parser.add_argument("--skip-madar", action="store_true", help="Skip MADAR processing.")
    parser.add_argument("--skip-padic", action="store_true", help="Skip PADIC processing.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_madar and args.madar_dir is None:
        raise ValueError("--madar-dir must be provided unless --skip-madar is set.")
    if not args.skip_padic and args.padic_xml is None:
        raise ValueError("--padic-xml is required unless --skip-padic is set.")

    need_translation = not args.skip_madar or not args.skip_padic
    tokenizer = model = None
    if need_translation:
        tokenizer, model = load_nllb(args.model_name, args.device)

    if not args.skip_madar:
        for split in args.madar_splits:
            df = build_madar_split(args.madar_dir, split, tokenizer, model, args.device, args.batch_size)
            write_jsonl(df, args.output_dir / f"madar_{split}.jsonl")

    if not args.skip_padic:
        full_df = build_padic_dataset(args.padic_xml, tokenizer, model, args.device, args.batch_size)
        train_df, dev_df = split_padic(full_df, args.padic_dev_fraction, args.seed)
        write_jsonl(train_df, args.output_dir / "padic_train.jsonl")
        write_jsonl(dev_df, args.output_dir / "padic_dev.jsonl")

    logger.info("Done.")


if __name__ == "__main__":
    main()
