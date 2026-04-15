# BiLSTM Genomic Sequence Denoising

This repository implements a BiLSTM-based deep learning framework for genomic sequence modeling and denoising. The model learns nucleotide-level dependencies using masked training and reconstructs original sequences from noisy inputs.

## Features
- BiLSTM architecture
- Masked denoising objective
- AdamW optimizer with cosine annealing
- Train/validation/test split (80/10/10)

## Usage

```bash
python code/training/train.py
