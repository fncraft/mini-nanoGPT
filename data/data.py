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
from pathlib import Path

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

def _encode_custom_chunk(chunk: str, tok_path: str):
    """
    子进程用：每次按路径重新加载 tokenizer.json，避免 Pickle 报错。
    """
    from tokenizers import Tokenizer   # 局部 import，主进程无需硬依赖
    tk = Tokenizer.from_file(tok_path)
    return tk.encode(chunk).ids

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
    1. 若勾选 `use_gpt2_tokenizer`，将 **优先** 寻找根目录
       的 `tokenizer.json`（多语种），使用 HuggingFace Tokenizers
       编码；若文件缺失则自动回退到 GPT-2 分词器 (tiktoken)。
    2. 无论自定义 tokenizer 还是 GPT-2，都会**裁剪词表**
       —— 仅保留输入文本中出现过的 token，并重新映射为
       连续 id（0…N-1），极大减少磁盘与显存占用。

    其余逻辑、返回字段与老版本完全一致。
    """
    # -----------------------------------------------------------
    # 0. 目录准备
    # -----------------------------------------------------------
    os.makedirs(raw_data_dir, exist_ok=True)
    os.makedirs(processed_data_dir, exist_ok=True)

    # -----------------------------------------------------------
    # 1. 读取输入文本
    # -----------------------------------------------------------
    data = input_text.strip()
    if not data and input_dir.strip():
        txt_files = [f for f in os.listdir(input_dir) if f.endswith('.txt')]
        for fn in txt_files:
            with open(os.path.join(input_dir, fn), 'r', encoding='utf-8') as f:
                data += f.read()
    if not data:
        raise ValueError("No text input or directory provided.")

    with open(os.path.join(raw_data_dir, 'merged_input.txt'), 'w', encoding='utf-8') as f:
        f.write(data)

    # -----------------------------------------------------------
    # 2. 进程数决策
    # -----------------------------------------------------------
    data_size_mb = len(data.encode('utf-8')) / (1024 * 1024)
    actual_proc = min(num_proc, cpu_count()) if data_size_mb > 100 else 1

    # -----------------------------------------------------------
    # 3-A. 分词器路径优先：根目录 tokenizer.json
    # -----------------------------------------------------------
    if use_gpt2_tokenizer:
        json_path = Path.cwd() / "tokenizer.json"
        is_custom = json_path.exists()

        # ---------- 3-A-1. 自定义 tokenizer.json ----------
        if is_custom:
            try:
                from tokenizers import Tokenizer
            except ImportError as e:
                raise ImportError(
                    "检测到 tokenizer.json，但未安装 `tokenizers`：\n"
                    "    pip install tokenizers\n"
                ) from e

            if actual_proc == 1:
                tok = Tokenizer.from_file(str(json_path))
                tokens = tok.encode(data).ids
            else:
                chunks = get_chunks(data, actual_proc)
                with Pool(actual_proc) as pool:
                    token_chunks = pool.starmap(
                        _encode_custom_chunk,
                        [(ck, str(json_path)) for ck in chunks]
                    )
                tokens = [t for ck in token_chunks for t in ck]

            # — 结束符处理（若定义了）
            eot_old = tok.token_to_id("")
            if eot_old is None:
                eot_old = tok.token_to_id("<|endoftext|>")
            if eot_old is not None and (not tokens or tokens[-1] != eot_old):
                tokens.append(eot_old)

            tokenizer_tag = "custom_json"

        # ---------- 3-A-2. 回退 GPT-2 (tiktoken) ----------
        else:
            import tiktoken
            enc = tiktoken.get_encoding("gpt2")
            tokenizer_tag = "gpt2"

            if actual_proc == 1:
                tokens = encode_gpt2_chunk(data, enc)
            else:
                chunks = get_chunks(data, actual_proc)
                with Pool(actual_proc) as pool:
                    token_chunks = pool.starmap(encode_gpt2_chunk, [(ck, enc) for ck in chunks])
                tokens = [t for ck in token_chunks for t in ck]

            if tokens and tokens[-1] != enc.eot_token:
                tokens.append(enc.eot_token)
            eot_old = enc.eot_token

        # ---------- 3-A-3. 词表裁剪 old→new ----------
        old2new = {old_id: new_id for new_id, old_id in enumerate(sorted(set(tokens)))}
        tokens_new = [old2new[t] for t in tokens]
        vocab_size = len(old2new)

        # 拆分
        if not no_validation:
            cut = int(len(tokens_new) * train_split_ratio)
            splits = {"train": tokens_new[:cut], "val": tokens_new[cut:]}
        else:
            splits = {"train": tokens_new}

        # 写 .bin
        for sp, seq in splits.items():
            np.array(seq, dtype=np.uint32).tofile(os.path.join(processed_data_dir, f"{sp}.bin"))

        # 构建 itos / stoi
        if is_custom:
            tok_meta = Tokenizer.from_file(str(json_path))
            itos = {nid: tok_meta.decode([oid]) for oid, nid in old2new.items()}
        else:
            itos = {nid: enc.decode([oid]) for oid, nid in old2new.items()}
        stoi = {s: i for i, s in itos.items()}

        meta = {
            "vocab_size": vocab_size,
            "itos": itos,
            "stoi": stoi,
            "tokenizer": tokenizer_tag,
            "old2new": old2new,
            "eot_id_new": old2new.get(eot_old, None)
        }
        with open(os.path.join(processed_data_dir, 'meta.pkl'), 'wb') as f:
            pickle.dump(meta, f)

        result = {
            "processed_data_dir": processed_data_dir,
            "vocab_size": vocab_size,
            "train_size": len(splits["train"])
        }
        if not no_validation:
            result["val_size"] = len(splits["val"])
        return result

    # -----------------------------------------------------------
    # 3-B. 字符级（原逻辑，无任何变化）
    # -----------------------------------------------------------
    # 收集字符
    if actual_proc > 1:
        chunks = get_chunks(data, actual_proc)
        with Pool(actual_proc) as pool:
            char_sets = pool.map(get_unique_chars, chunks)
        chars = sorted(set().union(*char_sets))
    else:
        chars = sorted(set(data))

    vocab_size = len(chars)
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}

    # 编码
    if actual_proc > 1:
        chunks = get_chunks(data, actual_proc)
        with Pool(actual_proc) as pool:
            enc_chunks = pool.starmap(encode_text_chunk, [(ck, stoi) for ck in chunks])
        encoded = [e for ck in enc_chunks for e in ck]
    else:
        encoded = encode_text_chunk(data, stoi)

    # 切分
    if not no_validation:
        cut = int(len(encoded) * train_split_ratio)
        train_ids = np.array(encoded[:cut], dtype=IntegerTypes)
        val_ids = np.array(encoded[cut:], dtype=IntegerTypes)
    else:
        train_ids = np.array(encoded, dtype=IntegerTypes)
        val_ids = None

    # 写文件
    train_ids.tofile(os.path.join(processed_data_dir, 'train.bin'))
    if not no_validation:
        val_ids.tofile(os.path.join(processed_data_dir, 'val.bin'))

    with open(os.path.join(processed_data_dir, 'meta.pkl'), 'wb') as f:
        pickle.dump({"vocab_size": vocab_size, "itos": itos, "stoi": stoi}, f)

    result = {
        "processed_data_dir": processed_data_dir,
        "vocab_size": vocab_size,
        "train_size": len(train_ids)
    }
    if not no_validation:
        result["val_size"] = len(val_ids)
    print(f"Used {actual_proc} process(es) for data processing.")
    return result
