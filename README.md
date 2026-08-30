# Rosetta at AlexandriaX-2026

LoRA-adapted NileChat-3B system for **Subtask 1** of the [AlexandriaX-2026 shared task](https://alexandriax.dlnlp.ai/) — context-aware English-to-dialectal-Arabic dialogue translation. Submitted to both the constrained and unconstrained tracks.

Full system description, related work, and error analysis: see our paper, *"Rosetta at AlexandriaX-2026: LoRA-Adapted NileChat for Context-Aware Dialectal Arabic Dialogue Translation."*

## Approach

- Freeze [UBC-NLP/NileChat-3B](https://huggingface.co/UBC-NLP/NileChat-3B-Base) and attach a LoRA adapter (r=32, α=32, dropout=0) targeting the attention and MLP projections.
- Structured system/user prompts condition generation on dialect, country, domain, participants, speaker, and gender direction, with prior turns supplied as dialogue history.
- **Constrained track:** fine-tuned only on the official [Alexandria](https://huggingface.co/datasets/UBC-NLP/alexandria) training set.
- **Unconstrained track:** additionally pretrained on MADAR and PADIC (mapped to country-level dialect labels, MSA translated to English via NLLB-200) before task fine-tuning.
- History noising during fine-tuning (word-level drop/swap/duplicate + random truncation) to mitigate exposure bias.
- Beam search decoding (5 beams, length penalty 0.7) at inference, with previously generated turns fed back as history.
- Trained on a single Kaggle Tesla T4 using Unsloth + 4-bit NF4 quantization.

## Results (official test set)

| Track | spBLEU | chrF++ | Rank |
|---|---|---|---|
| Constrained | 26.10 | 41.79 | 4th |
| Unconstrained | 25.09 | 41.02 | 5th |

Per-dialect breakdown, the negative-transfer finding for the unconstrained track, and the qualitative error analysis are in the paper. Egyptian, Jordanian, Lebanese, Palestinian, Saudi, and Syrian score highest; Mauritanian, Libyan, Sudanese, and Moroccan trail.

## Repository structure

```
data/         preprocessing scripts (not the dataset itself)
models/       fine-tuning and inference scripts
evaluation/   spBLEU / chrF++ scoring scripts
baselines/    the official shared-task baseline, adapted to run on a Kaggle T4
notebooks/    exploratory work (MBR decoding trial, error analysis)
docs/         decisions log, meeting notes, paper draft
```

### Pipeline

1. **Data prep** (`data/`)
   - `prepare_alex_data.py` — downloads Alexandria from HF, builds chat-format train/dev JSONL with history noising.
   - `preprocess_madar_padic.py` — downloads MADAR/PADIC, translates MSA→English via NLLB, filters/aligns dialects, splits train/dev. (Unconstrained track only.)
   - `prepare_pretraining_data.py` — turns the MADAR/PADIC files above into chat-format pretraining JSONL.
2. **Training** (`models/train.py`) — one script, two modes:
   - `--mode pretrain` — MADAR + PADIC (unconstrained track only)
   - `--mode finetune` — Alexandria (both tracks; `--model-name` points at a pretrain checkpoint to chain the two stages)
3. **Inference** (`models/infer.py`) — generates predictions for `dev` (scored locally) or `test` (submission), turn-by-turn, using the model's own prior predictions as history.
4. **Evaluation** (`evaluation/score.py`) — spBLEU (flores200) and chrF++ against dev references.

Exploratory extensions, not part of the official submission, also live in `models/` and plug into `infer.py` via flags: MBR decoding (`--decoding mbr`) and FAISS-based few-shot retrieval (`embeddings.py` + `--fewshot-examples-path`). See `notebooks/Try_MBR.ipynb` for the write-up of the MBR trial.

Every script takes `--help` for its full argument list and documents its expected input/output format in its module docstring.

## Setup

```bash
pip install -r requirements.txt
```

Set these environment variables (or Kaggle Secrets, if running there) before running anything that touches the Hub or W&B:

- `HF_TOKEN` — Hugging Face Hub access (dataset + model)
- `WANDB_API_KEY` — experiment tracking and checkpoint artifacts (project: `RosettaAtAlexandriaX/DialectalArabicMT`)

Training and the Kaggle baseline notebook additionally need a GPU (developed and tested on a single Tesla T4).

## Citation

```bibtex
@inproceedings{rosetta-alexandriax-2026,
  title     = {Rosetta at AlexandriaX-2026: LoRA-Adapted NileChat for Context-Aware Dialectal Arabic Dialogue Translation},
  author    = {Esmaeil, Nada and Rena, Fathima and Subhash, Sibi and Elgendy, Osama and Naguib, Mina and Omar, Salma and Arif, Muhammad},
  year      = {2026},
  note      = {AlexandriaX-2026 Shared Task system description paper}
}
```

(Update with the final venue/proceedings entry once published.)

## Acknowledgments

Built for the [AlexandriaX-2026](https://alexandriax.dlnlp.ai/) shared task, organized by El Mekki, Elmadany, Magdy, Ezzini, El-Haj, Jarrar, Alyafeai, Ghanem, and Abdul-Mageed. Uses the [Alexandria dataset](https://huggingface.co/datasets/UBC-NLP/alexandria) and [NileChat-3B](https://huggingface.co/UBC-NLP/NileChat-3B-Base).