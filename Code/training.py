# -*- coding: utf-8 -*-
"""
BiLSTM Denoising Training Pipeline (GitHub Version)
No change in model foundation
"""

import os
import datetime
import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

from sklearn.model_selection import train_test_split

from tensorflow.keras.layers import (
    Input, Embedding, Bidirectional, LSTM,
    Dropout, Dense, LayerNormalization
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.optimizers.schedules import CosineDecay
from tensorflow.keras.callbacks import EarlyStopping, CSVLogger

# ================= CONFIG =================
CONFIG = {
    "DATA_PATH": "./data/your_sequences.fasta",  # <-- change this
    "RUN_NAME": "bilstm_run",

    "EMBED_DIM": 32,
    "LSTM_UNITS": 128,
    "DROPOUT_RATE": 0.2,

    "LEARNING_RATE": 2e-3,
    "WEIGHT_DECAY": 1e-4,
    "BATCH_SIZE": 32,
    "EPOCHS": 50,
    "CLIPNORM": 0.7,
    "LR_SCHEDULE_ALPHA": 0.1,

    "MASKING_PROBABILITY": 0.2,
    "RANDOM_SEED": 42,
}

RUN_DIR = os.path.join("./results", CONFIG["RUN_NAME"])
os.makedirs(RUN_DIR, exist_ok=True)

# ================= LOGGING =================
def save_log(text):
    print(text)
    with open(os.path.join(RUN_DIR, "log.txt"), "a") as f:
        f.write(f"{datetime.datetime.now()} - {text}\n")

# ================= DATA =================
def save_fasta_split(sequences, file_path):
    records = [SeqRecord(Seq(s), id=f"seq_{i}", description="") for i, s in enumerate(sequences)]
    with open(file_path, "w") as f:
        SeqIO.write(records, f, "fasta")

def encode_seq(seq, char_map, max_len):
    encoded = np.full(max_len, char_map['<PAD>'])
    encoded[0] = char_map['<BOS>']

    for i, c in enumerate(seq):
        encoded[i + 1] = char_map.get(c, char_map['<MASK>'])

    if len(seq) + 1 < max_len:
        encoded[len(seq) + 1] = char_map['<EOS>']

    return encoded

def load_data():
    sequences = [str(r.seq).upper() for r in SeqIO.parse(CONFIG["DATA_PATH"], "fasta")]
    sequences = list(set(sequences))
    np.random.shuffle(sequences)

    train, temp = train_test_split(sequences, test_size=0.2, random_state=CONFIG["RANDOM_SEED"])
    val, test = train_test_split(temp, test_size=0.5, random_state=CONFIG["RANDOM_SEED"])

    return train, val, test

# ================= MASKING =================
def create_masked_input(sequences, mask_id, canonical_ids, prob):
    inputs = np.copy(sequences)
    mask = (np.isin(inputs, list(canonical_ids))) & (np.random.rand(*inputs.shape) < prob)
    inputs[mask] = mask_id
    return inputs

# ================= LOSS =================
loss_obj = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False, reduction='none')

def masked_sparse_cce(y_true, y_pred):
    return tf.reduce_mean(loss_obj(y_true, y_pred))

def masked_accuracy(y_true, y_pred):
    return tf.reduce_mean(tf.cast(tf.equal(y_true, tf.argmax(y_pred, axis=-1)), tf.float32))

# ================= MODEL =================
def build_model(seq_len, vocab_size, total_steps):
    inputs = Input(shape=(seq_len,))
    x = Embedding(vocab_size, CONFIG["EMBED_DIM"])(inputs)
    x = LayerNormalization()(x)
    x = Bidirectional(LSTM(CONFIG["LSTM_UNITS"], return_sequences=True))(x)
    x = Dropout(CONFIG["DROPOUT_RATE"])(x)
    outputs = Dense(vocab_size, activation="softmax")(x)

    lr_schedule = CosineDecay(CONFIG["LEARNING_RATE"], total_steps, alpha=CONFIG["LR_SCHEDULE_ALPHA"])
    optimizer = AdamW(learning_rate=lr_schedule, weight_decay=CONFIG["WEIGHT_DECAY"], clipnorm=CONFIG["CLIPNORM"])

    model = Model(inputs, outputs)
    model.compile(optimizer=optimizer, loss=masked_sparse_cce, metrics=[masked_accuracy])
    return model

# ================= MAIN =================
if __name__ == "__main__":

    np.random.seed(CONFIG["RANDOM_SEED"])
    tf.random.set_seed(CONFIG["RANDOM_SEED"])

    save_log("Loading data...")
    train_seqs, val_seqs, test_seqs = load_data()

    vocab = ['<PAD>', '<BOS>', '<EOS>', '<MASK>', 'A', 'T', 'G', 'C']
    char_to_int = {c: i for i, c in enumerate(vocab)}

    max_len = max(len(s) for s in train_seqs + val_seqs + test_seqs) + 2

    X_train = np.array([encode_seq(s, char_to_int, max_len) for s in train_seqs])
    X_val   = np.array([encode_seq(s, char_to_int, max_len) for s in val_seqs])
    X_test  = np.array([encode_seq(s, char_to_int, max_len) for s in test_seqs])

    mask_id = char_to_int['<MASK>']
    canonical_ids = {char_to_int[c] for c in ['A','T','G','C']}

    X_train_masked = create_masked_input(X_train, mask_id, canonical_ids, CONFIG["MASKING_PROBABILITY"])
    X_val_masked   = create_masked_input(X_val, mask_id, canonical_ids, CONFIG["MASKING_PROBABILITY"])

    steps = max(1, len(X_train) // CONFIG["BATCH_SIZE"]) * CONFIG["EPOCHS"]

    model = build_model(max_len, len(vocab), steps)
    model.summary()

    csv_logger = CSVLogger(os.path.join(RUN_DIR, "training_log.csv"))

    history = model.fit(
        X_train_masked, X_train,
        validation_data=(X_val_masked, X_val),
        epochs=CONFIG["EPOCHS"],
        batch_size=CONFIG["BATCH_SIZE"],
        callbacks=[EarlyStopping(patience=5, restore_best_weights=True), csv_logger]
    )

    model.save(os.path.join(RUN_DIR, "model.keras"))

    save_log("Training complete.")
