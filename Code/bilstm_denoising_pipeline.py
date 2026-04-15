# -*- coding: utf-8 -*-
"""
BiLSTM Denoising Pipeline for SARS-CoV-2 Omicron Spike Sequences
=================================================================
Original: Colab Notebook (Denoising_BILSTM_MODULAR_12APril.3)
GitHub Version: Paths made configurable; Colab-specific calls removed.
Foundation (model architecture, loss, training logic) unchanged.

Usage:
    python bilstm_denoising_pipeline.py \
        --fasta_path /path/to/your/sequences.fasta \
        --run_dir   /path/to/output_directory
"""

# --- [Step 0] Dependencies ---
# Install via: pip install -r requirements.txt

import os
import json
import datetime
import argparse
import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt

from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import (
    Input, Embedding, Bidirectional, LSTM, Dropout, Dense,
    LayerNormalization
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.optimizers.schedules import CosineDecay
from tensorflow.keras.callbacks import EarlyStopping, CSVLogger


# ==============================================================================
# [Module 2] CONFIG — paths now come from CLI args (see __main__ block below)
# ==============================================================================
CONFIG = {
    # Model Hyperparameters
    "EMBED_DIM": 32,
    "LSTM_UNITS": 128,
    "DROPOUT_RATE": 0.2,

    # Training Hyperparameters
    "LEARNING_RATE": 2e-3,
    "WEIGHT_DECAY": 1e-4,
    "BATCH_SIZE": 32,
    "EPOCHS": 50,
    "CLIPNORM": 0.7,
    "LR_SCHEDULE_ALPHA": 0.1,

    # Denoising Task Parameters
    "MASKING_PROBABILITY": 0.2,
    "RANDOM_SEED": 42,

    # File Names for Persistence
    "TRAIN_FASTA": "train_split.fasta",
    "VAL_FASTA": "val_split.fasta",
    "TEST_FASTA": "test_split.fasta",
    "ENCODED_DATA": "encoded_arrays.npz",
    "VOCAB_JSON": "vocab_mapping.json",
    "EPOCH_LOG": "epoch_results.csv",
}


# ==============================================================================
# [Module 3] Environment Setup (Colab drive.mount replaced with os.makedirs)
# ==============================================================================
def setup_environment(run_dir):
    print("--- [Step 2] Setting up environment ---")
    os.makedirs(run_dir, exist_ok=True)
    np.random.seed(CONFIG['RANDOM_SEED'])
    tf.random.set_seed(CONFIG['RANDOM_SEED'])


LOG_FILE = None  # Set dynamically in main()

def save_log(text, also_print=True):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{datetime.datetime.now().isoformat()} - {text}\n")
    if also_print:
        print(text)


# ==============================================================================
# [Module 4] Data Engine: Loading, Splitting, & Persistence
# ==============================================================================
def save_fasta_split(sequences, file_path, prefix="seq"):
    """Saves a list of strings as a FASTA file for external use."""
    records = [
        SeqRecord(Seq(s), id=f"{prefix}_{i}", description="")
        for i, s in enumerate(sequences)
    ]
    with open(file_path, "w") as output_handle:
        SeqIO.write(records, output_handle, "fasta")


def encode_seq(sequence_string, char_map, max_len):
    """Converts a DNA string into a padded integer array with BOS/EOS tokens."""
    encoded = np.full(max_len, char_map['<PAD>'], dtype=np.int32)
    encoded[0] = char_map['<BOS>']
    for idx, char in enumerate(sequence_string):
        encoded[idx + 1] = char_map.get(char, char_map['<MASK>'])
    if len(sequence_string) + 1 < max_len:
        encoded[len(sequence_string) + 1] = char_map['<EOS>']
    return encoded


def load_and_preprocess_data(fasta_path, run_dir):
    save_log("\n--- [Step 3] Data Preprocessing & 80/10/10 Split ---")
    try:
        all_sequences = [
            str(record.seq).upper()
            for record in SeqIO.parse(fasta_path, 'fasta')
        ]
        unique_sequences = sorted(list(set(all_sequences)))
        np.random.shuffle(unique_sequences)
    except FileNotFoundError:
        save_log(f"FATAL: FASTA file not found at {fasta_path}. Halting.")
        return None

    # 1. 80/10/10 Split Logic
    train_seqs, temp_seqs = train_test_split(
        unique_sequences, test_size=0.20, random_state=CONFIG['RANDOM_SEED']
    )
    val_seqs, test_seqs = train_test_split(
        temp_seqs, test_size=0.50, random_state=CONFIG['RANDOM_SEED']
    )
    save_log(f"Data Segregated: Train={len(train_seqs)}, Val={len(val_seqs)}, Test={len(test_seqs)}")

    # 2. Save Human-Readable FASTA splits
    train_path = os.path.join(run_dir, CONFIG['TRAIN_FASTA'])
    val_path   = os.path.join(run_dir, CONFIG['VAL_FASTA'])
    test_path  = os.path.join(run_dir, CONFIG['TEST_FASTA'])

    save_fasta_split(train_seqs, train_path, "train")
    save_fasta_split(val_seqs,   val_path,   "val")
    save_fasta_split(test_seqs,  test_path,  "test")
    save_log(f"FASTA splits saved to: {run_dir}")

    # 3. Build & Save Vocabulary (No Leakage — vocab built from train only)
    special_tokens = ['<PAD>', '<BOS>', '<EOS>', '<MASK>']
    all_chars_in_train = sorted(list(set(''.join(train_seqs))))
    full_vocab = special_tokens + [c for c in all_chars_in_train if c not in special_tokens]

    if '<PAD>' in full_vocab:
        full_vocab.remove('<PAD>')
    full_vocab.insert(0, '<PAD>')  # Ensure <PAD> is at index 0

    char_to_int = {c: i for i, c in enumerate(full_vocab)}

    vocab_path = os.path.join(run_dir, CONFIG['VOCAB_JSON'])
    with open(vocab_path, 'w') as f:
        json.dump(char_to_int, f)
    save_log(f"Vocabulary mapping (Size: {len(full_vocab)}) saved to: {vocab_path}")

    return train_seqs, val_seqs, test_seqs, full_vocab, char_to_int


# ==============================================================================
# [Module 5] Custom Loss & Metric Functions
# ==============================================================================
loss_obj = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False, reduction='none')


def build_loss_and_metric(full_vocab, char_to_int):
    """Returns masked_sparse_cce and masked_accuracy compiled against the vocab."""
    core_bases = {'A', 'T', 'G', 'C'}
    special_tokens = ['<PAD>', '<BOS>', '<EOS>']
    chars_to_mask = [
        char for char in full_vocab
        if char not in core_bases and char != '<MASK>'
    ]
    tokens_to_ignore = list(set(special_tokens + chars_to_mask))
    loss_mask_indices = sorted([
        char_to_int[tok] for tok in tokens_to_ignore if tok in char_to_int
    ])

    @tf.keras.utils.register_keras_serializable()
    def masked_accuracy(y_true, y_pred):
        y_pred_class = tf.cast(tf.argmax(y_pred, axis=-1), dtype=y_true.dtype)
        loss_indices_tensor = tf.convert_to_tensor(loss_mask_indices, dtype=y_true.dtype)

        def compute_mask():
            is_ignored = tf.reduce_any(
                tf.equal(y_true[..., tf.newaxis], loss_indices_tensor), axis=-1
            )
            return tf.logical_not(is_ignored)

        def no_mask():
            return tf.ones_like(y_true, dtype=tf.bool)

        mask = tf.cond(tf.size(loss_indices_tensor) > 0, compute_mask, no_mask)
        matches = tf.cast(tf.equal(y_true, y_pred_class), tf.float32)
        masked_matches = tf.boolean_mask(matches, mask)
        return tf.cond(
            tf.size(masked_matches) > 0,
            lambda: tf.reduce_mean(masked_matches),
            lambda: tf.constant(0.0)
        )

    @tf.keras.utils.register_keras_serializable()
    def masked_sparse_cce(y_true, y_pred):
        per_pos_loss = loss_obj(y_true, y_pred)
        loss_indices_tensor = tf.convert_to_tensor(loss_mask_indices, dtype=y_true.dtype)

        def compute_mask():
            is_ignored = tf.reduce_any(
                tf.equal(y_true[..., tf.newaxis], loss_indices_tensor), axis=-1
            )
            return tf.cast(tf.logical_not(is_ignored), tf.float32)

        def no_mask():
            return tf.ones_like(y_true, dtype=tf.float32)

        mask = tf.cond(tf.size(loss_indices_tensor) > 0, compute_mask, no_mask)
        per_pos_loss *= mask
        valid_mask_sum = tf.reduce_sum(mask)
        return tf.cond(
            valid_mask_sum > 0,
            lambda: tf.reduce_sum(per_pos_loss) / valid_mask_sum,
            lambda: tf.constant(0.0)
        )

    return masked_sparse_cce, masked_accuracy


# ==============================================================================
# [Module 6] Model Builder
# ==============================================================================
def build_editor_model(seq_len, vocab_size, params, total_train_steps,
                       masked_sparse_cce, masked_accuracy):
    inputs  = Input(shape=(seq_len,), dtype='int32', name='inputs')
    x       = Embedding(vocab_size, params['EMBED_DIM'], mask_zero=False)(inputs)
    x       = LayerNormalization()(x)
    x       = Bidirectional(LSTM(params['LSTM_UNITS'], return_sequences=True))(x)
    x       = Dropout(params['DROPOUT_RATE'])(x)
    outputs = Dense(vocab_size, activation='softmax')(x)

    model = Model(inputs, outputs)
    lr_schedule = CosineDecay(
        initial_learning_rate=params['LEARNING_RATE'],
        decay_steps=total_train_steps,
        alpha=params['LR_SCHEDULE_ALPHA']
    )
    optimizer = AdamW(
        learning_rate=lr_schedule,
        weight_decay=params['WEIGHT_DECAY'],
        clipnorm=params['CLIPNORM']
    )
    model.compile(
        optimizer=optimizer,
        loss=masked_sparse_cce,
        metrics=[masked_accuracy]
    )
    return model


# ==============================================================================
# [Module 7] Masking Helper & Training
# ==============================================================================
def create_masked_input(sequences, mask_id, canonical_ids, prob):
    inputs = np.copy(sequences)
    is_canonical = np.isin(inputs, list(canonical_ids))
    should_mask  = np.random.rand(*inputs.shape) < prob
    final_mask   = is_canonical & should_mask
    inputs[final_mask] = mask_id
    return inputs


# ==============================================================================
# [Main] Entry Point
# ==============================================================================
def main(args):
    global LOG_FILE

    run_dir  = args.run_dir
    LOG_FILE = os.path.join(run_dir, 'training_log.txt')

    setup_environment(run_dir)
    save_log("--- [START] Denoising Full Training Pipeline ---")

    # --- Data ---
    result = load_and_preprocess_data(args.fasta_path, run_dir)
    if result is None:
        return
    train_seqs, val_seqs, test_seqs, full_vocab, char_to_int = result
    vocab_size = len(full_vocab)

    max_len_corpus = max(len(s) for s in (train_seqs + val_seqs + test_seqs))
    padded_seq_len = max_len_corpus + 2

    X_train_full = np.array([encode_seq(s, char_to_int, padded_seq_len) for s in train_seqs])
    X_val_orig   = np.array([encode_seq(s, char_to_int, padded_seq_len) for s in val_seqs])
    X_test_orig  = np.array([encode_seq(s, char_to_int, padded_seq_len) for s in test_seqs])

    npz_path = os.path.join(run_dir, CONFIG['ENCODED_DATA'])
    np.savez_compressed(npz_path, xtrain=X_train_full, xval=X_val_orig, xtest=X_test_orig)
    save_log(f"Numeric arrays saved to: {npz_path}")

    # --- Loss / Metric ---
    masked_sparse_cce, masked_accuracy = build_loss_and_metric(full_vocab, char_to_int)

    mask_token_id         = char_to_int['<MASK>']
    canonical_base_indices = {char_to_int[c] for c in {'A', 'T', 'G', 'C'} if c in char_to_int}

    # --- Masked inputs ---
    save_log("\n--- [Step 5] Training on 80% Split ---")
    X_train_masked = create_masked_input(X_train_full, mask_token_id, canonical_base_indices, CONFIG["MASKING_PROBABILITY"])
    X_val_masked   = create_masked_input(X_val_orig,   mask_token_id, canonical_base_indices, CONFIG["MASKING_PROBABILITY"])
    X_test_masked  = create_masked_input(X_test_orig,  mask_token_id, canonical_base_indices, CONFIG["MASKING_PROBABILITY"])

    # --- Model ---
    steps_per_epoch    = max(1, len(X_train_full) // CONFIG['BATCH_SIZE'])
    total_train_steps  = steps_per_epoch * CONFIG['EPOCHS']
    model = build_editor_model(
        padded_seq_len, vocab_size, CONFIG, total_train_steps,
        masked_sparse_cce, masked_accuracy
    )
    model.summary()

    # --- Callbacks ---
    epoch_log_path = os.path.join(run_dir, CONFIG['EPOCH_LOG'])
    csv_logger     = CSVLogger(epoch_log_path, append=True)
    early_stopping = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

    history = model.fit(
        X_train_masked, X_train_full,
        epochs=CONFIG['EPOCHS'],
        batch_size=CONFIG['BATCH_SIZE'],
        validation_data=(X_val_masked, X_val_orig),
        callbacks=[early_stopping, csv_logger],
        verbose=1
    )

    # --- Save Model ---
    save_log("\n--- [Step 6] Saving Model and Plotting ---")
    final_model_path = os.path.join(run_dir, 'final_denoising_model.keras')
    model.save(final_model_path)

    # --- Plot ---
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    ax1.plot(history.history['masked_accuracy'],     label='Train Acc', color='#1f77b4', lw=2)
    ax1.plot(history.history['val_masked_accuracy'], label='Val Acc',   color='#ff7f0e', lw=2)
    ax1.set_title('BiLSTM Training Accuracy', fontsize=14, weight='bold')
    ax1.legend()

    ax2.plot(history.history['loss'],     label='Train Loss', color='#1f77b4', lw=2)
    ax2.plot(history.history['val_loss'], label='Val Loss',   color='#ff7f0e', lw=2)
    ax2.set_title('BiLSTM Training Loss', fontsize=14, weight='bold')
    ax2.legend()

    plt.tight_layout()
    plot_path = os.path.join(run_dir, 'training_history_plot.png')
    plt.savefig(plot_path)
    plt.show()

    # --- Final Test Evaluation ---
    test_results = model.evaluate(X_test_masked, X_test_orig, verbose=0)
    save_log(f"Final Test Loss: {test_results[0]:.4f}, Test Accuracy: {test_results[1]:.4f}")
    save_log("--- [END] Pipeline Finished Successfully ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="BiLSTM Denoising Pipeline for Omicron Spike Sequences"
    )
    parser.add_argument(
        "--fasta_path",
        type=str,
        required=True,
        help="Full path to input FASTA file (e.g. data/sequences.fasta)"
    )
    parser.add_argument(
        "--run_dir",
        type=str,
        default="./runs/bilstm_run",
        help="Output directory for model, logs, and splits (default: ./runs/bilstm_run)"
    )
    args = parser.parse_args()
    main(args)
