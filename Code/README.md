# BiLSTM Denoising Pipeline for SARS-CoV-2 Omicron Spike Sequences

A deep learning pipeline using a **Bidirectional LSTM (BiLSTM)** architecture to denoise SARS-CoV-2 Omicron spike protein sequences. The model is trained as a masked-sequence denoiser on nucleotide-level FASTA data.

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

- **Task:** Masked language modelling / denoising (20% canonical bases masked)
- **Loss:** Custom `masked_sparse_cce` — ignores PAD, BOS, EOS tokens
- **Metric:** Custom `masked_accuracy` — evaluated only on canonical bases (A/T/G/C)
- **Optimizer:** AdamW with CosineDecay LR schedule

---

## Dataset

- Input: Deduplicated, filtered FASTA file of Omicron spike sequences  
  (sourced from NCBI Virus & GISAID, filtered for <10 ambiguous bases)
- Split: **80% train / 10% validation / 10% test** (stratified by unique sequence)

---

## Setup

```bash
git clone https://github.com/YOUR_USERNAME/bilstm-denoising-omicron.git
cd bilstm-denoising-omicron
pip install -r requirements.txt
```

---

## Usage

```bash
python bilstm_denoising_pipeline.py \
    --fasta_path data/merged_sequences.fasta \
    --run_dir    runs/experiment_01
```

| Argument | Description | Default |
|---|---|---|
| `--fasta_path` | Path to input `.fasta` file | *(required)* |
| `--run_dir` | Directory for outputs | `./runs/bilstm_run` |

---

## Outputs

All outputs are saved to `--run_dir`:

| File | Description |
|---|---|
| `train_split.fasta` | Training sequences (80%) |
| `val_split.fasta` | Validation sequences (10%) |
| `test_split.fasta` | Hold-out test sequences (10%) |
| `encoded_arrays.npz` | Compressed numeric matrices |
| `vocab_mapping.json` | Character → integer vocabulary |
| `epoch_results.csv` | Per-epoch loss/accuracy log |
| `training_log.txt` | Timestamped pipeline log |
| `training_history_plot.png` | Loss & accuracy curves |
| `final_denoising_model.keras` | Saved Keras model |

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
| Epochs | 50 (early stopping patience=5) |
| Masking probability | 0.20 |
| Random seed | 42 |

---

## Notes

- Vocabulary is built **from training sequences only** to prevent data leakage
- `<PAD>` is always assigned index 0 for Keras masking compatibility


---

## Citation

If you use this code, please cite the associated publication (details to be added upon acceptance).

---

## License

MIT License
