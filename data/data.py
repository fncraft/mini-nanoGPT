"""
data/data.py

This file handles data processing, including reading raw text, 
splitting into train/validation sets, tokenizing (char-level or GPT-2), 
and saving to binary files.
"""

import os
import math
import pickle
import numpy as np
import tiktoken
from multiprocessing import Pool, cpu_count

# We import our global config and integer type definition
from config.default import DEFAULT_CONFIG, IntegerTypes


def get_chunks(text, n):
    """
    Splits the text into 'n' roughly equal chunks for parallel processing.
    :param text: The full text string to split.
    :param n: The number of chunks to split into.
    :return: A list of text chunks.
    """
    chunk_size = math.ceil(len(text) / n)
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]


def get_unique_chars(text):
    """
    Returns a set of unique characters found in the text.
    :param text: A string from which to collect unique characters.
    :return: A set of unique characters.
    """
    return set(text)


def encode_text_chunk(chunk, stoi):
    """
    Encodes a chunk of text at the character level using the 'stoi' dictionary.
    :param chunk: A substring of text.
    :param stoi: A dict mapping characters to their integer IDs.
    :return: A list of integer IDs representing the chunk.
    """
    return [stoi.get(ch, 0) for ch in chunk]


def encode_gpt2_chunk(chunk, tokenizer):
    """
    Encodes a chunk of text using GPT-2 tokenizer.
    :param chunk: A substring of text.
    :param tokenizer: A GPT-2 tokenizer from 'tiktoken'.
    :return: A list of token IDs.
    """
    return tokenizer.encode(chunk, allowed_special={"<|endoftext|>"})


def process_data(
    input_text="",
    input_dir="",
    raw_data_dir=DEFAULT_CONFIG["data_process"]["raw_data_dir"],
    processed_data_dir=DEFAULT_CONFIG["data_process"]["processed_data_dir"],
    train_split_ratio=DEFAULT_CONFIG["data_process"]["train_split_ratio"],
    no_validation=DEFAULT_CONFIG["data_process"]["no_validation"],
    use_gpt2_tokenizer=DEFAULT_CONFIG["data_process"]["use_gpt2_tokenizer"],
    num_proc=DEFAULT_CONFIG["data_process"]["num_proc"]
):
    """
    Splits data into train and val sets (unless 'no_validation' is True),
    supports both GPT-2 or char-level tokenization, and utilizes multiprocessing 
    for encoding large datasets.

    :param input_text: Directly provided text to process.
    :param input_dir: Directory containing .txt files if no direct text is given.
    :param raw_data_dir: Where to store the merged raw text.
    :param processed_data_dir: Where to store the processed binary data (train.bin, val.bin).
    :param train_split_ratio: The fraction of data to allocate for training.
    :param no_validation: If True, skip creating a validation set.
    :param use_gpt2_tokenizer: Whether to use GPT-2 tokenizer or char-level.
    :param num_proc: Number of processes for parallel encoding.
    :return: A dict containing information about the processed data, 
             including processed_data_dir, vocab_size, train_size, and optionally val_size.
    """
    os.makedirs(raw_data_dir, exist_ok=True)
    os.makedirs(processed_data_dir, exist_ok=True)

    data = ""
    # Priority 1: use 'input_text' if provided
    if input_text.strip():
        data = input_text
    # Priority 2: if 'input_dir' is specified, read .txt files from it
    elif input_dir.strip():
        txt_files = [f for f in os.listdir(input_dir) if f.endswith('.txt')]
        for file_name in txt_files:
            file_path = os.path.join(input_dir, file_name)
            with open(file_path, 'r', encoding='utf-8') as f:
                data += f.read()
    else:
        raise ValueError("No text input or directory provided.")

    # Save raw text for reference
    raw_text_file = os.path.join(raw_data_dir, 'merged_input.txt')
    with open(raw_text_file, 'w', encoding='utf-8') as f:
        f.write(data)

    # Estimate data size in MB to decide whether to use multiprocessing
    data_size = len(data.encode('utf-8')) / (1024 * 1024)
    suggested_proc = min(num_proc, cpu_count())
    # For smaller data, using multiple processes is often overkill
    actual_proc = suggested_proc if data_size > 100 else 1

    # ------------------------------
    # GPT-2 Tokenization Workflow
    # ------------------------------
    if use_gpt2_tokenizer:
        enc = tiktoken.get_encoding("gpt2")
        vocab_size = enc.n_vocab

        # Parallel or single-process encoding
        if actual_proc > 1:
            chunks = get_chunks(data, actual_proc)
            with Pool(actual_proc) as pool:
                token_chunks = pool.starmap(encode_gpt2_chunk, [(chunk, enc) for chunk in chunks])
            tokens = []
            for chunk in token_chunks:
                tokens.extend(chunk)
        else:
            tokens = encode_gpt2_chunk(data, enc)

        # Append end-of-text token if missing
        if tokens and tokens[-1] != enc.eot_token:
            tokens.append(enc.eot_token)

        # Split train/val
        if not no_validation:
            split_idx = int(len(tokens) * train_split_ratio)
            splits = {
                "train": tokens[:split_idx],
                "val": tokens[split_idx:]
            }
        else:
            splits = {"train": tokens}

        # Save to .bin files
        for split, tokens_ in splits.items():
            filename = os.path.join(processed_data_dir, f'{split}.bin')
            arr = np.array(tokens_, dtype=np.uint32)
            arr.tofile(filename)

        # Save metadata (important for reconstructing the tokenizer state)
        meta_path = os.path.join(processed_data_dir, 'meta.pkl')
        meta = {
            'vocab_size': vocab_size,
            'itos': {i: enc.decode([i]) for i in range(vocab_size)},
            'stoi': {enc.decode([i]): i for i in range(vocab_size)},
            'tokenizer': 'gpt2'
        }
        with open(meta_path, 'wb') as f:
            pickle.dump(meta, f)

        return {
            "processed_data_dir": processed_data_dir,
            "vocab_size": vocab_size,
            "train_size": len(splits["train"]),
            "val_size": len(splits.get("val", "")) if not no_validation else None
        }

    # ------------------------------
    # Char-level Tokenization
    # ------------------------------
    else:
        # Collect all unique characters
        if actual_proc > 1:
            chunks = get_chunks(data, actual_proc)
            with Pool(actual_proc) as pool:
                char_sets = pool.map(get_unique_chars, chunks)
            chars = sorted(list(set().union(*char_sets)))
        else:
            chars = sorted(list(set(data)))

        vocab_size = len(chars)
        stoi = {ch: i for i, ch in enumerate(chars)}
        itos = {i: ch for i, ch in enumerate(chars)}

        # Encode data
        if actual_proc > 1:
            chunks = get_chunks(data, actual_proc)
            with Pool(actual_proc) as pool:
                encoded_chunks = pool.starmap(encode_text_chunk, [(chunk, stoi) for chunk in chunks])
            encoded_data = []
            for chunk in encoded_chunks:
                encoded_data.extend(chunk)
        else:
            encoded_data = encode_text_chunk(data, stoi)

        # Split train/val
        if not no_validation:
            split_idx = int(len(encoded_data) * train_split_ratio)
            train_ids = np.array(encoded_data[:split_idx], dtype=IntegerTypes)
            val_ids = np.array(encoded_data[split_idx:], dtype=IntegerTypes)
        else:
            train_ids = np.array(encoded_data, dtype=IntegerTypes)
            val_ids = None

        train_bin_path = os.path.join(processed_data_dir, 'train.bin')
        val_bin_path = os.path.join(processed_data_dir, 'val.bin')
        meta_path = os.path.join(processed_data_dir, 'meta.pkl')

        # Write binary files
        train_ids.tofile(train_bin_path)
        if not no_validation and val_ids is not None:
            val_ids.tofile(val_bin_path)

        # Save meta info
        meta = {
            'vocab_size': vocab_size,
            'itos': itos,
            'stoi': stoi,
        }
        with open(meta_path, 'wb') as f:
            pickle.dump(meta, f)

        print(f"Used {actual_proc} process(es) for data processing.")
        result = {
            "processed_data_dir": processed_data_dir,
            "vocab_size": vocab_size,
            "train_size": len(train_ids),
        }
        if not no_validation:
            result["val_size"] = len(val_ids)
        return result
