# BiLSTM Denoising & Constrained Mutant Generation
### SARS-CoV-2 Omicron Spike Sequences

A two-stage deep learning pipeline using a **Bidirectional LSTM (BiLSTM)** to:

1. **Train** a masked-sequence denoiser on Omicron spike nucleotide FASTA data
2. **Infer** constrained mutant sequences from a seed (e.g. Wuhan reference) using intelligent Top-K/Temperature sampling — targeted only at uncertain or disagreement sites

---

## Repository Structure

```
bilstm-omicron-denoising/
│
├── bilstm_denoising_pipeline.py       # STAGE 1 — Training
├── inference_constrained_sampling.py  # STAGE 2 — Inference / Mutant Generation
├── requirements.txt
├── .gitignore
├── README.md
│
└── data/                              # (not tracked by git — add your own)
    ├── merged_sequences.fasta         # Training corpus (Omicron spike, ~6873 seqs)
    └── wuhan_reference.fasta          # Seed for inference
```

---

## Architecture Overview

```
Input (int32 token IDs)
    └── Embedding (dim=32)
        └── LayerNormalization
            └── Bidirectional LSTM (128 units, return_sequences=True)
                └── Dropout (0.2)
                    └── Dense (vocab_size, softmax)
```

- **Task:** Masked language modelling / sequence denoising (20% masking)
- **Loss:** Custom `masked_sparse_cce` — ignores PAD, BOS, EOS, non-canonical tokens
- **Metric:** Custom `masked_accuracy` — evaluated only on A/T/G/C positions
- **Optimizer:** AdamW + CosineDecay LR schedule

---

## Setup

```bash
git clone https://github.com/YOUR_USERNAME/bilstm-omicron-denoising.git
cd bilstm-omicron-denoising
pip install -r requirements.txt
```

---

## Stage 1 — Training

```bash
python bilstm_denoising_pipeline.py \
    --fasta_path data/merged_sequences.fasta \
    --run_dir    runs/experiment_01
```

| Argument | Description | Default |
|---|---|---|
| `--fasta_path` | Path to training FASTA file | *(required)* |
| `--run_dir` | Output directory | `./runs/bilstm_run` |

### Outputs saved to `--run_dir`

| File | Description |
|---|---|
| `final_denoising_model.keras` | Trained model **→ needed for Stage 2** |
| `vocab_mapping.json` | Character→integer vocab **→ needed for Stage 2** |
| `train_split.fasta` / `val_split.fasta` / `test_split.fasta` | 80/10/10 splits |
| `encoded_arrays.npz` | Compressed numeric matrices |
| `epoch_results.csv` | Per-epoch training log |
| `training_history_plot.png` | Loss & accuracy curves |
| `training_log.txt` | Timestamped run log |

---

## Stage 2 — Inference (Constrained Mutant Generation)

```bash
python inference_constrained_sampling.py \
    --seed_fasta             data/wuhan_reference.fasta \
    --model_path             runs/experiment_01/final_denoising_model.keras \
    --vocab_json             runs/experiment_01/vocab_mapping.json \
    --output_dir             runs/inference_output \
    --num_sequences          1000 \
    --top_k                  4 \
    --temperature            1.0 \
    --uncertainty_threshold  0.90
```

| Argument | Description | Default |
|---|---|---|
| `--seed_fasta` | Seed sequence FASTA (e.g. Wuhan ref) | *(required)* |
| `--model_path` | `.keras` model from Stage 1 | *(required)* |
| `--vocab_json` | `vocab_mapping.json` from Stage 1 | `None` |
| `--training_fasta` | Fallback vocab source if no JSON | `None` |
| `--output_dir` | Where to save outputs | `./runs/inference_output` |
| `--num_sequences` | Mutants to generate | `1000` |
| `--top_k` | Top-K for sampling at targeted sites | `4` |
| `--temperature` | Sampling temperature | `1.0` |
| `--uncertainty_threshold` | Confidence threshold for targeting | `0.90` |

### How Sampling Works

Each position in the seed is classified as:

- **Fidelity site** — model is confident AND agrees with seed → kept unchanged (deterministic)
- **Mutation site** — model uncertain (`prob < threshold`) OR disagrees → sampled via Top-K + Temperature

The first and last 3 bases are always frozen (structural constraint).

### Inference Outputs

| File | Description |
|---|---|
| `*_Batch_K4_T1.0_Thresh0.9.fasta` | All generated mutant sequences |
| `*_Batch_K4_T1.0_Thresh0.9_Analysis.csv` | Per-sequence base composition + mutation list |

---

## End-to-End Connection

```
bilstm_denoising_pipeline.py
    │
    ├── saves ──► final_denoising_model.keras ──┐
    └── saves ──► vocab_mapping.json ───────────┤
                                                ▼
                        inference_constrained_sampling.py
                            ├── loads model + vocab
                            ├── loads seed FASTA
                            └── outputs mutant FASTA + analysis CSV
```

> Always pass `--vocab_json` from Stage 1 into Stage 2 to guarantee vocabulary consistency and prevent leakage.

---

## Key Hyperparameters

| Parameter | Value |
|---|---|
| Embedding dim | 32 |
| LSTM units | 128 |
| Dropout | 0.2 |
| Learning rate | 2e-3 |
| Weight decay | 1e-4 |
| Batch size | 32 |
| Max epochs | 50 (early stopping patience=5) |
| Masking probability | 0.20 |
| Random seed | 42 |

---

## Notes

- Google Colab `drive.mount` and `!pip install` calls removed — all paths are CLI arguments
- Vocabulary built from **training split only** — no data leakage into val/test
- `<PAD>` is always at index 0 for Keras masking compatibility
- `files.download()` (Colab-only) removed — outputs are saved to `--output_dir`

---

## Push to GitHub

```bash
git init
git add bilstm_denoising_pipeline.py inference_constrained_sampling.py requirements.txt .gitignore README.md
git commit -m "Initial commit: BiLSTM training + constrained inference pipeline"
git remote add origin https://github.com/YOUR_USERNAME/bilstm-omicron-denoising.git
git push -u origin main
```

---

## Citation

If you use this code, please cite the associated publication (details to be added upon acceptance).

---

## License

MIT License
