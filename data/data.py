import os
import math
import pickle
import numpy as np
from multiprocessing import Pool, cpu_count

import tiktoken  # GPT-2 Tokenizer
from config.default import DEFAULT_CONFIG, IntegerTypes


def get_chunks(text, n):
    """
    Divide text into n chunks for multiprocessing.
    """
    chunk_size = math.ceil(len(text) / n)
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

def get_unique_chars(text):
    """
    Get unique characters in text, which will be used as the vocabulary.
    """
    return set(text)

def encode_text_chunk(chunk, stoi):
    """
    Encode text chunk into integer IDs using the provided stoi mapping.
    If a character is not in the mapping, it will be encoded as 0.
    """
    return [stoi.get(ch, 0) for ch in chunk]

def encode_gpt2_chunk(chunk, tokenizer):
    """
    Encode text chunk using GPT-2 tokenizer.
    """
    return tokenizer.encode(chunk, allowed_special={"<|endoftext|>"})

##############################################################################
# 核心的数据处理函数
##############################################################################

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
    1. Reads text from either an input file or .txt files in a directory (required)
    2. Splits data into training and validation sets (optional) 
    3. Tokenization: Uses either GPT-2 tokenizer or character-level tokenization
    4. Leverages multiprocessing for faster tokenization on large datasets
    5. Saves processed data as train.bin, val.bin, and meta.pkl in the output directory
    
    The processed files will be stored in processed_data_dir.
    """
    os.makedirs(raw_data_dir, exist_ok=True)
    os.makedirs(processed_data_dir, exist_ok=True)

    data = ""
    # If input_text is provided, use it directly
    if input_text.strip():
        data = input_text
    # If input_dir is provided, read all .txt files in the directory
    elif input_dir.strip():
        txt_files = [f for f in os.listdir(input_dir) if f.endswith('.txt')]
        for file_name in txt_files:
            file_path = os.path.join(input_dir, file_name)
            with open(file_path, 'r', encoding='utf-8') as f:
                data += f.read()
    else:
        raise ValueError("No text input or directory provided.")

    # Save raw text to a file for reference
    raw_text_file = os.path.join(raw_data_dir, 'merged_input.txt')
    with open(raw_text_file, 'w', encoding='utf-8') as f:
        f.write(data)

    # Calculate data size and suggested number of processes
    data_size = len(data.encode('utf-8')) / (1024 * 1024)
    suggested_proc = min(num_proc, cpu_count())
    # For small datasets, use only 1 process
    actual_proc = suggested_proc if data_size > 100 else 1

    # -----------------------------------------------
    # Process data using GPT-2 tokenizer
    # -----------------------------------------------
    if use_gpt2_tokenizer:
        enc = tiktoken.get_encoding("gpt2")
        vocab_size = enc.n_vocab

        # Multi-process tokenization
        if actual_proc > 1:
            chunks = get_chunks(data, actual_proc)
            with Pool(actual_proc) as pool:
                token_chunks = pool.starmap(encode_gpt2_chunk, [(chunk, enc) for chunk in chunks])
            tokens = []
            for chunk in token_chunks:
                tokens.extend(chunk)
        else:
            tokens = encode_gpt2_chunk(data, enc)

        # Append EOT token to the end of each document
        if tokens and tokens[-1] != enc.eot_token:
            tokens.append(enc.eot_token)

        # Split train / val
        if not no_validation:
            split_idx = int(len(tokens) * train_split_ratio)
            splits = {
                "train": tokens[:split_idx],
                "val": tokens[split_idx:]
            }
        else:
            splits = {"train": tokens}

        # Save to binary files
        for split, tokens_ in splits.items():
            filename = os.path.join(processed_data_dir, f'{split}.bin')
            arr = np.array(tokens_, dtype=np.uint32)
            arr.tofile(filename)

        # Save meta information
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

    # -----------------------------------------------
    # Process data using character-level tokenization
    # -----------------------------------------------
    else:
        # If actual_proc > 1, merge character sets first and then unify encoding
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

        # Encode text using multiprocessing
        if actual_proc > 1:
            chunks = get_chunks(data, actual_proc)
            with Pool(actual_proc) as pool:
                encoded_chunks = pool.starmap(encode_text_chunk, [(chunk, stoi) for chunk in chunks])
            encoded_data = []
            for chunk in encoded_chunks:
                encoded_data.extend(chunk)
        else:
            encoded_data = encode_text_chunk(data, stoi)

        # Split train / val
        if not no_validation:
            split_idx = int(len(encoded_data) * train_split_ratio)
            train_ids = np.array(encoded_data[:split_idx], dtype=IntegerTypes)
            val_ids = np.array(encoded_data[split_idx:], dtype=IntegerTypes)
        else:
            train_ids = np.array(encoded_data, dtype=IntegerTypes)
            val_ids = None

        # Save to binary files
        train_bin_path = os.path.join(processed_data_dir, 'train.bin')
        val_bin_path = os.path.join(processed_data_dir, 'val.bin')
        meta_path = os.path.join(processed_data_dir, 'meta.pkl')

        train_ids.tofile(train_bin_path)
        if not no_validation and val_ids is not None:
            val_ids.tofile(val_bin_path)

        # Save meta information
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
