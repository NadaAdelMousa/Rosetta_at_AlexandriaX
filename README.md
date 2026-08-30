# Rosetta at AlexandriaX-2026

LoRA-adapted NileChat-3B system for **Subtask 1** of the [AlexandriaX-2026 shared task](https://alexandriax.dlnlp.ai/) — context-aware English-to-dialectal-Arabic dialogue translation. Submitted to both the constrained and unconstrained tracks.

Full system description, related work, and error analysis: see our paper, *"Rosetta at AlexandriaX-2026: LoRA-Adapted NileChat for Context-Aware Dialectal Arabic Dialogue Translation."*

## Approach

- Freeze [UBC-NLP/NileChat-3B](https://huggingface.co/UBC-NLP/NileChat-3B) and attach a LoRA adapter (r=32, α=32, dropout=0) targeting the attention and MLP projections.
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
(Will be Updated with the final venue/proceedings entry once published.)

If you use the Alexandria dataset or NileChat-3B, please also cite the papers that introduced them:

```bibtex
@inproceedings{el-mekki-etal-2026-alexandria,
    title = "Alexandria: A Multi-Domain Dialectal {A}rabic Machine Translation Dataset for Culturally Inclusive and Linguistically Diverse {LLM}s",
    author = "EL Mekki, Abdellah and Magdy, Samar M. and Atou, Houdaifa and AbuHweidi, Ruwa and
      Qawasmeh, Baraah and Nacar, Omer and Al-hibiri, Thikra and Saadie, Razan and
      Alsayadi, Hamzah A. and Hammouda, Nadia Ghezaiel and Alkhazimi, Alshima Mohammed and
      Hamod, Aya and Al-Ghafri, Al-Yas Yaqoob and El-Sayed, Wesam and al Sharji, Asila Ismail and
      Ballout, Mohamad and Belfathi, Anas and Ghaddar, Karim and Sibaee, Serry and Aoun, Alaa and
      Aseri, Aeej Mohammed and Abureesh, Lina and Bashiti, Ahlam and Yousef, Majdal and
      Hafiz, Abdulaziz and Mohamed, Yehdih and Hamedtou, Emira and Emehah, Brakehe and
      Alhamouri, Rahaf and Nafea, Youssef and El Aatar, Aya and Al-Dhabyani, Walid and
      Hamed, Emhemed S. and Shatnawi, Sara and Alwajih, Fakhraddin and Elkhidir, Khalid and
      Alasmari, Ashwag and Gerrio, Abdurrahman and Alshahri, Omar Said and Elmadany, AbdelRahim A. and
      Berrada, Ismail and Al-kathiri, Amir Azad Adli and Zaraket, Fadi and Jarrar, Mustafa and
      EL Hadj, Yahya Mohamed and Alhuzali, Hassan and Abdul-Mageed, Muhammad",
    booktitle = "Proceedings of the 64th Annual Meeting of the {A}ssociation for {C}omputational {L}inguistics (Volume 1: Long Papers)",
    month = jul,
    year = "2026",
    address = "San Diego, California, United States",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2026.acl-long.1503/",
    pages = "32567--32592",
    ISBN = "979-8-89176-390-6"
}

@inproceedings{el-mekki-etal-2025-nilechat,
    title = "{N}ile{C}hat: Towards Linguistically Diverse and Culturally Aware {LLM}s for Local Communities",
    author = "El Mekki, Abdellah and Atou, Houdaifa and Nacar, Omer and Shehata, Shady and
      Abdul-Mageed, Muhammad",
    booktitle = "Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2025",
    address = "Suzhou, China",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.emnlp-main.556/",
    doi = "10.18653/v1/2025.emnlp-main.556",
    pages = "10978--11002",
    ISBN = "979-8-89176-332-6"
}
```

## Licensing

The code in this repository (all scripts and notebooks) is released under the [MIT License](LICENSE).

That covers the code only. The data and model this project builds on carry their own, separate — and more restrictive — licenses:

- **[Alexandria dataset](https://huggingface.co/datasets/UBC-NLP/alexandria)**: CC BY-NC 4.0 (non-commercial, attribution required). This repo never redistributes the dataset itself — `data/` holds only preprocessing scripts.
- **[NileChat-3B](https://huggingface.co/UBC-NLP/NileChat-3B)**: released under Qwen's research-only license (it continues pretraining from Qwen2.5-3B). Fine-tuned adapter weights are a derivative of this model and inherit the same research/non-commercial restriction — if we publish them, they're for research use only, not relicensed under MIT.

If you build on this repo, make sure your use of the dataset and model stays within those upstream terms independently of the MIT license on the code.

## Acknowledgments

Built for the [AlexandriaX-2026](https://alexandriax.dlnlp.ai/) shared task, organized by El Mekki, Elmadany, Magdy, Ezzini, El-Haj, Jarrar, Alyafeai, Ghanem, and Abdul-Mageed. Uses the [Alexandria dataset](https://huggingface.co/datasets/UBC-NLP/alexandria) and [NileChat-3B](https://huggingface.co/UBC-NLP/NileChat-3B).