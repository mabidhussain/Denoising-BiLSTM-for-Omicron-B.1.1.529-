# -*- coding: utf-8 -*-
"""
BATCH-MODE CONSTRAINED SAMPLING (Intelligent Targeting)
========================================================
Loads a trained denoising BiLSTM model and generates mutant sequences
by applying sampling (Top-K/Temp) ONLY at uncertain or disagreement sites.
All other sites are kept identical to the seed.

Connects to: bilstm_denoising_pipeline.py (training script)

Usage:
    python inference_constrained_sampling.py \
        --seed_fasta     data/seedOFrefseqWUHAN.fasta \
        --training_fasta data/merged_sequences.fasta \
        --model_path     runs/bilstm_run/final_denoising_model.keras \
        --vocab_json     runs/bilstm_run/vocab_mapping.json \
        --output_dir     runs/inference_output \
        --num_sequences  1000 \
        --top_k          4 \
        --temperature    1.0 \
        --uncertainty_threshold 0.90
"""

import os
import json
import datetime
import argparse
import numpy as np
import tensorflow as tf
import pandas as pd
from collections import Counter
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import SparseCategoricalCrossentropy


# ==============================================================================
# [Step 3] Vocabulary Builder
# Rebuilds vocab from training FASTA OR loads from saved vocab_mapping.json
# (vocab_mapping.json is saved automatically by bilstm_denoising_pipeline.py)
# ==============================================================================
def build_vocabulary(training_fasta_path=None, vocab_json_path=None):
    """
    Priority: Load from vocab_json if available (no leakage risk).
    Fallback: Rebuild from training FASTA (matches pipeline behaviour exactly).
    """
    if vocab_json_path and os.path.exists(vocab_json_path):
        print(f"[Step 3] Loading vocabulary from JSON: {vocab_json_path}")
        with open(vocab_json_path, 'r') as f:
            char_to_int = json.load(f)
        int_to_char = {v: k for k, v in char_to_int.items()}
        full_vocab   = [int_to_char[i] for i in range(len(char_to_int))]
        print(f"✅ Vocabulary loaded. Size: {len(full_vocab)}")
        return char_to_int, int_to_char, full_vocab

    if training_fasta_path and os.path.exists(training_fasta_path):
        print(f"[Step 3] Rebuilding vocabulary from training FASTA: {training_fasta_path}")
        all_sequences    = [str(r.seq).upper() for r in SeqIO.parse(training_fasta_path, 'fasta')]
        unique_sequences = sorted(list(set(all_sequences)))
        train_seqs, _    = train_test_split(unique_sequences, test_size=0.20, random_state=42)

        special_tokens       = ['<PAD>', '<BOS>', '<EOS>', '<MASK>']
        all_chars_in_train   = sorted(list(set(''.join(train_seqs))))
        full_vocab           = special_tokens + [c for c in all_chars_in_train if c not in special_tokens]
        if '<PAD>' in full_vocab:
            full_vocab.remove('<PAD>')
        full_vocab.insert(0, '<PAD>')

        char_to_int = {c: i for i, c in enumerate(full_vocab)}
        int_to_char = {i: c for c, i in char_to_int.items()}
        print(f"✅ Vocabulary rebuilt. Size: {len(full_vocab)}")
        return char_to_int, int_to_char, full_vocab

    raise FileNotFoundError(
        "Must provide either --vocab_json (preferred) or --training_fasta to build vocabulary."
    )


def get_padded_seq_len(training_fasta_path):
    """Computes padded_seq_len from the full corpus (matches training pipeline)."""
    all_sequences = [str(r.seq).upper() for r in SeqIO.parse(training_fasta_path, 'fasta')]
    return max(len(s) for s in all_sequences) + 2


# ==============================================================================
# [Step 4] Custom Loss & Metric (identical to bilstm_denoising_pipeline.py)
# ==============================================================================
loss_obj = SparseCategoricalCrossentropy(from_logits=False, reduction='none')


def build_custom_objects(char_to_int, full_vocab):
    """Returns custom loss/metric keyed to the loaded vocabulary."""
    core_bases    = {'A', 'T', 'G', 'C'}
    special_toks  = ['<PAD>', '<BOS>', '<EOS>']
    chars_to_mask = [c for c in full_vocab if c not in core_bases and c != '<MASK>']
    tokens_to_ignore  = list(set(special_toks + chars_to_mask))
    loss_mask_indices = sorted([char_to_int[t] for t in tokens_to_ignore if t in char_to_int])

    @tf.keras.utils.register_keras_serializable()
    def masked_sparse_cce(y_true, y_pred):
        per_pos_loss = loss_obj(y_true, y_pred)
        mask = tf.ones_like(y_true, dtype=tf.float32)
        for idx in loss_mask_indices:
            mask *= tf.cast(tf.not_equal(y_true, idx), tf.float32)
        per_pos_loss *= mask
        valid_mask_sum = tf.reduce_sum(mask)
        return tf.cond(
            valid_mask_sum > 0,
            lambda: tf.reduce_sum(per_pos_loss) / valid_mask_sum,
            lambda: tf.constant(0.0)
        )

    @tf.keras.utils.register_keras_serializable()
    def masked_accuracy(y_true, y_pred):
        y_pred_class = tf.cast(tf.argmax(y_pred, axis=-1), dtype=y_true.dtype)
        mask = tf.ones_like(y_true, dtype=tf.bool)
        for token_id in loss_mask_indices:
            mask &= tf.not_equal(y_true, token_id)
        matches        = tf.cast(y_true == y_pred_class, dtype=tf.int32)
        masked_matches = tf.boolean_mask(matches, mask)
        return tf.cond(
            tf.size(masked_matches) > 0,
            lambda: tf.reduce_mean(tf.cast(masked_matches, tf.float32)),
            lambda: tf.constant(0.0)
        )

    return masked_sparse_cce, masked_accuracy


# ==============================================================================
# [Step 5] Model Loader — builds logits-only model on top of loaded weights
# ==============================================================================
def load_logits_model(model_path, vocab_size, masked_sparse_cce, masked_accuracy):
    print(f"[Step 5] Loading model from: {model_path}")
    model = load_model(
        model_path,
        custom_objects={
            'masked_sparse_cce': masked_sparse_cce,
            'masked_accuracy':   masked_accuracy
        },
        compile=False
    )
    print("✅ Model loaded.")

    # Build logits-only head (removes softmax for numerically stable sampling)
    dense_weights    = model.layers[-1].get_weights()
    input_to_last    = model.layers[-2].output
    logits_layer     = Dense(vocab_size, activation=None, name="logits_output")
    logits_output    = logits_layer(input_to_last)
    logits_model     = Model(model.input, logits_output, name="logits_model")
    logits_layer.set_weights(dense_weights)
    print("✅ Logits-only model built.")
    return logits_model


# ==============================================================================
# [Step 6] Helper Functions
# ==============================================================================
def encode_seq(seq, char_map, max_len):
    encoded    = np.full(max_len, char_map['<PAD>'], dtype=np.int32)
    encoded[0] = char_map['<BOS>']
    for idx, char in enumerate(seq):
        encoded[idx + 1] = char_map.get(char, char_map['<MASK>'])
    if len(seq) + 1 < max_len:
        encoded[len(seq) + 1] = char_map['<EOS>']
    return encoded


def create_masked_input(sequences, mask_id, canonical_ids, prob):
    inputs       = np.copy(sequences)
    is_canonical = np.isin(inputs, list(canonical_ids))
    should_mask  = np.random.rand(*inputs.shape) < prob
    final_mask   = is_canonical & should_mask
    inputs[final_mask] = mask_id
    return inputs


def decode_ids_to_string(ids, id_to_char_map, special_tokens):
    chars = []
    for i in ids:
        char = id_to_char_map.get(i, "")
        if char == '<EOS>':
            break
        if char not in special_tokens:
            chars.append(char)
    return "".join(chars)


def get_mutation_analysis(original_seq, mutant_seq, clean_bases=('A', 'T', 'G', 'C')):
    comp  = Counter(mutant_seq)
    total = len(mutant_seq)
    composition_data = {base: comp.get(base, 0) / total * 100 for base in clean_bases}
    other_count = total - sum(comp.get(b, 0) for b in clean_bases)
    composition_data['Other'] = other_count / total * 100
    mutation_list = []
    min_len = min(len(original_seq), len(mutant_seq))
    for i in range(3, min_len - 3):
        if original_seq[i] != mutant_seq[i]:
            mutation_list.append(f"{original_seq[i]}{i+1}{mutant_seq[i]}")
    return composition_data, mutation_list


# ==============================================================================
# [Step 7 core] Targeted Sampler — unchanged from original
# ==============================================================================
def generate_targeted_mutant(
    model_logits, all_probs, y_true_encoded, eos_index,
    k, T, char_to_int, freeze_length=3, uncertainty_threshold=0.95
):
    best_guess_ids = np.argmax(model_logits[0], axis=-1)
    seq_len   = y_true_encoded.shape[0]
    start_pos = freeze_length + 1
    end_pos   = eos_index - freeze_length
    final_ids = np.zeros(seq_len, dtype=np.int32)

    # Freeze ends
    final_ids[0]           = y_true_encoded[0]
    final_ids[1:start_pos] = y_true_encoded[1:start_pos]
    final_ids[end_pos:eos_index] = y_true_encoded[end_pos:eos_index]
    final_ids[eos_index]   = y_true_encoded[eos_index]
    final_ids[eos_index+1:] = y_true_encoded[eos_index+1:]

    def local_sampler(logits, k_val, T_val):
        logits = logits / T_val
        top_k_logits, top_k_indices = tf.nn.top_k(logits, k=k_val)
        min_logits   = tf.fill(logits.shape, -1e9)
        row_indices  = tf.zeros((1, k_val), dtype=tf.int32)
        full_indices = tf.stack([row_indices, top_k_indices], axis=-1)
        masked_logits = tf.tensor_scatter_nd_update(min_logits, full_indices, top_k_logits)
        return tf.random.categorical(masked_logits, num_samples=1)[0, 0].numpy()

    for i in range(start_pos, end_pos):
        original_base_id   = y_true_encoded[i]
        best_guess_id      = best_guess_ids[i]
        original_base_prob = all_probs[0, i, original_base_id]
        is_disagreement    = (original_base_id != best_guess_id)
        is_uncertain       = (original_base_prob < uncertainty_threshold)

        if is_disagreement or is_uncertain:
            single_pos_logits = model_logits[0, i, :][tf.newaxis, :]
            final_ids[i]      = local_sampler(single_pos_logits, k, T)
        else:
            final_ids[i] = best_guess_id

    mutation_count  = int(np.sum(final_ids[start_pos:end_pos] != y_true_encoded[start_pos:end_pos]))
    return final_ids, mutation_count


# ==============================================================================
# [Main] Entry Point
# ==============================================================================
def main(args):
    special_tokens = ['<PAD>', '<BOS>', '<EOS>', '<MASK>']

    # --- Vocab ---
    char_to_int, int_to_char, full_vocab = build_vocabulary(
        training_fasta_path=args.training_fasta,
        vocab_json_path=args.vocab_json
    )
    vocab_size = len(full_vocab)

    core_bases             = {'A', 'T', 'G', 'C'}
    canonical_base_indices = {char_to_int[c] for c in core_bases if c in char_to_int}
    mask_token_id          = char_to_int['<MASK>']

    # --- Padded length: use training FASTA if given, else fall back to vocab JSON path hint ---
    if args.training_fasta and os.path.exists(args.training_fasta):
        padded_seq_len = get_padded_seq_len(args.training_fasta)
    else:
        # If no training FASTA, derive from seed + generous buffer
        seed_len       = len(str(next(SeqIO.parse(args.seed_fasta, 'fasta')).seq))
        padded_seq_len = seed_len + 50
        print(f"Warning: No training FASTA provided. Using padded_seq_len={padded_seq_len} (seed+50).")
    print(f"Padded sequence length: {padded_seq_len}")

    # --- Custom objects ---
    masked_sparse_cce, masked_accuracy = build_custom_objects(char_to_int, full_vocab)

    # --- Model ---
    logits_model = load_logits_model(
        args.model_path, vocab_size, masked_sparse_cce, masked_accuracy
    )

    # --- Seed ---
    print(f"\n[Step 7] Loading seed from: {args.seed_fasta}")
    seed_record        = next(SeqIO.parse(args.seed_fasta, 'fasta'))
    future_seed_string = str(seed_record.seq).upper()
    future_seed_id     = seed_record.id.replace("/", "_").replace(":", "_")
    print(f"✅ Seed loaded: {seed_record.id} (Length: {len(future_seed_string)})")

    y_true_encoded = encode_seq(future_seed_string, char_to_int, padded_seq_len)
    y_true_batch   = np.array([y_true_encoded])
    X_noisy_batch  = create_masked_input(
        y_true_batch, mask_token_id, canonical_base_indices, prob=0.2
    )

    # --- Logits + probs (computed once) ---
    print("Running logits_model.predict()...")
    all_logits = logits_model.predict(X_noisy_batch)
    all_probs  = tf.nn.softmax(all_logits).numpy()
    print(f"✅ Logits shape: {all_logits.shape} | Probs shape: {all_probs.shape}")

    eos_arr = np.where(y_true_encoded == char_to_int['<EOS>'])[0]
    eos_index = int(eos_arr[0]) if len(eos_arr) > 0 else len(future_seed_string) + 1

    # --- Output directory ---
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp       = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    run_name_prefix = (
        f"K{args.top_k}_T{args.temperature:.1f}_Thresh{args.uncertainty_threshold}"
    )
    fasta_out_path  = os.path.join(
        args.output_dir, f"{timestamp}_Batch_{run_name_prefix}.fasta"
    )
    csv_out_path    = os.path.join(
        args.output_dir, f"{timestamp}_Batch_{run_name_prefix}_Analysis.csv"
    )

    # --- Generation loop ---
    print(f"\n[Step 8] Generating {args.num_sequences} sequences → {fasta_out_path}")
    analysis_results = []

    # Baseline
    orig_comp, _ = get_mutation_analysis(future_seed_string, future_seed_string)
    orig_comp.update({'Run': 'Original Seed', 'Mutations': 0, 'Mutation_List': ''})
    analysis_results.append(orig_comp)

    with open(fasta_out_path, 'w') as f_out:
        for i in range(1, args.num_sequences + 1):
            if i % 100 == 0:
                print(f"  ...sequence {i}/{args.num_sequences}")

            final_ids, mutation_count = generate_targeted_mutant(
                all_logits, all_probs, y_true_encoded, eos_index,
                k=args.top_k,
                T=args.temperature,
                char_to_int=char_to_int,
                freeze_length=3,
                uncertainty_threshold=args.uncertainty_threshold
            )
            mutant_sequence = decode_ids_to_string(final_ids, int_to_char, special_tokens)

            comp_data, mutation_list = get_mutation_analysis(future_seed_string, mutant_sequence)
            run_name = f"{run_name_prefix}_Run{i}"
            comp_data.update({
                'Run':           run_name,
                'Mutations':     mutation_count,
                'Mutation_List': ", ".join(mutation_list[:5]) + (", ..." if len(mutation_list) > 5 else "")
            })
            analysis_results.append(comp_data)

            header = f">gen_mutant_{run_name}_from_{future_seed_id} | MUTATIONS={mutation_count}"
            f_out.write(f"{header}\n{mutant_sequence}\n")

    print(f"✅ {args.num_sequences} sequences saved to: {fasta_out_path}")

    # --- Report ---
    print("\n[Step 9] Generating analysis report...")
    df = pd.DataFrame(analysis_results)
    df = df[['Run', 'A', 'T', 'G', 'C', 'Other', 'Mutations', 'Mutation_List']].round(3)
    df.to_csv(csv_out_path, index=False)
    print(f"✅ Report saved to: {csv_out_path}")

    gen_df = df[df['Run'] != 'Original Seed']
    if not gen_df.empty:
        print(f"\n## Analysis Summary")
        print(f"  Total sequences : {len(gen_df)}")
        print(f"  Avg mutations   : {gen_df['Mutations'].mean():.2f}")
        print(f"  Mutation range  : {gen_df['Mutations'].min()} – {gen_df['Mutations'].max()}")

    print("\n--- Batch Inference Complete ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Constrained BiLSTM Inference — Targeted Mutant Generation"
    )
    parser.add_argument("--seed_fasta",    type=str, required=True,
                        help="Path to seed FASTA (e.g. data/wuhan_reference.fasta)")
    parser.add_argument("--model_path",    type=str, required=True,
                        help="Path to trained .keras model (output of bilstm_denoising_pipeline.py)")
    parser.add_argument("--vocab_json",    type=str, default=None,
                        help="Path to vocab_mapping.json saved by training script (preferred)")
    parser.add_argument("--training_fasta", type=str, default=None,
                        help="Path to original training FASTA (used if --vocab_json not provided)")
    parser.add_argument("--output_dir",    type=str, default="./runs/inference_output",
                        help="Directory to save FASTA and CSV outputs")
    parser.add_argument("--num_sequences", type=int, default=1000,
                        help="Number of mutant sequences to generate (default: 1000)")
    parser.add_argument("--top_k",         type=int, default=4,
                        help="Top-K for sampling (default: 4)")
    parser.add_argument("--temperature",   type=float, default=1.0,
                        help="Sampling temperature (default: 1.0)")
    parser.add_argument("--uncertainty_threshold", type=float, default=0.90,
                        help="Confidence threshold below which a site is targeted (default: 0.90)")
    args = parser.parse_args()
    main(args)
