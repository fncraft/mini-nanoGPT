"""
modules/inference.py

This file contains the function to load a GPT model checkpoint and generate text 
based on a given prompt. It now properly handles CPU-only devices by mapping 
the checkpoint storage to CPU when CUDA is not available.
"""

import os
import torch
import torch.nn.functional as F
import pickle
from contextlib import nullcontext

from config.default import DEFAULT_CONFIG
from modules.gpt import GPT, GPTConfig


def generate_text(
    data_dir=DEFAULT_CONFIG["inference"]["data_dir"],
    out_dir=DEFAULT_CONFIG["inference"]["out_dir"],
    prompt=DEFAULT_CONFIG["inference"]["prompt"],
    num_samples=DEFAULT_CONFIG["inference"]["num_samples"],
    max_new_tokens=DEFAULT_CONFIG["inference"]["max_new_tokens"],
    temperature=DEFAULT_CONFIG["inference"]["temperature"],
    top_k=DEFAULT_CONFIG["inference"]["top_k"],
    seed=DEFAULT_CONFIG["inference"]["seed"],
    device=DEFAULT_CONFIG["inference"]["device"],
    dtype=DEFAULT_CONFIG["inference"]["dtype"],
    compile_model=DEFAULT_CONFIG["inference"]["compile_model"]
):
    """
    Generates text from a single checkpoint. If the checkpoint is not found,
    yields an error message. For each sample, tokens are generated one at a time 
    until 'max_new_tokens' is reached.

    :yield: Intermediate strings or final generated strings for each sample.
    """
    if not prompt.strip():
        yield "Prompt is empty, please provide a starting text."
        return

    try:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        # Determine the appropriate device (CPU or CUDA)
        device_obj = torch.device(device) if torch.cuda.is_available() and "cuda" in device else torch.device('cpu')
        ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
        ctx = nullcontext() if device_obj.type == 'cpu' else torch.amp.autocast(device_type=device_obj.type, dtype=ptdtype)

        # Determine checkpoint path
        if out_dir.endswith('.pt'):
            ckpt_path = out_dir
        else:
            ckpt_path = os.path.join(out_dir, 'ckpt.pt')
        if not os.path.exists(ckpt_path):
            yield f"Error: checkpoint not found at {ckpt_path}."
            return

        # Load the checkpoint with appropriate map_location
        checkpoint = torch.load(ckpt_path, map_location=device_obj)
        gptconf = GPTConfig(**checkpoint['model_args'])
        model = GPT(gptconf)
        state_dict = checkpoint['model']
        unwanted_prefix = '_orig_mod.'
        for k, v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
        model.load_state_dict(state_dict)
        model.eval()
        model.to(device_obj)
        if compile_model:
            model = torch.compile(model)

        # Load metadata (tokenizer information)
        meta_path = os.path.join(data_dir, 'meta.pkl')
        if not os.path.exists(meta_path):
            yield f"Error: meta.pkl not found at {meta_path}."
            return
        with open(meta_path, 'rb') as f:
            meta = pickle.load(f)
        stoi, itos = meta['stoi'], meta['itos']

        # Define encode and decode helper functions
        def encode(s):
            return [stoi.get(ch, 0) for ch in s]

        def decode(l):
            # decoded_chars = [itos.get(i, '') for i in l]
            # print(f"Decoding tokens: {l}", flush=True)
            # print(f"Decoded string: {''.join(decoded_chars)}", flush=True)
            return ''.join([itos.get(i, '') for i in l])

        xids = torch.tensor(encode(prompt), dtype=torch.long, device=device_obj)[None, ...]
        block_size = gptconf.block_size
        if xids.size(1) > block_size:
            yield f"Error: input length ({xids.size(1)}) exceeds block size ({block_size})."
            return

        # Generate text
        with torch.no_grad():
            with ctx:
                for s_i in range(num_samples):
                    idx = xids.clone()
                    generated = prompt
                    for token_iter in range(max_new_tokens):
                        if idx.size(1) == 0:
                            yield "Can't generate an empty sequence."
                            return
                        idx_cond = idx[:, -block_size:]
                        logits, _ = model(idx_cond)
                        logits = logits[:, -1, :] / temperature
                        if top_k is not None and top_k > 0:
                            v, _ = torch.topk(logits, top_k)
                            top_value = v[:, -1].unsqueeze(-1)
                            logits[logits < top_value] = -float('Inf')
                        probs = F.softmax(logits, dim=-1)
                        idx_next = torch.multinomial(probs, num_samples=1)
                        idx = torch.cat((idx, idx_next), dim=1)
                        generated_tokens = idx[0].tolist()
                        generated = decode(generated_tokens)
                        yield f"Sample {s_i+1} (iteration {token_iter+1}):\n{generated}"
                    if s_i < num_samples - 1:
                        yield "-" * 20

        final_output = generated.strip() if generated else ""
        if final_output:
            yield final_output
        else:
            yield "No text generated."

    except Exception as ex:
        yield f"An unexpected error occurred: {str(ex)}"
