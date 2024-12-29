##############################################################################
# inference/inference.py
##############################################################################
import os
import pickle
from contextlib import nullcontext

import torch
import torch.nn.functional as F
import torch._dynamo
torch._dynamo.config.suppress_errors = True

from config.default import DEFAULT_CONFIG
from models.gpt import GPT, GPTConfig

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
    从指定的 ckpt.pt 中加载模型，并对 prompt 进行文本生成。
    通过 yield 返回多轮结果，便于实时显示生成的文本。
    
    -- 修改点：如果 out_dir 以 .pt 结尾，则视为路径，否则自动拼接 'ckpt.pt' --
    """
    if not prompt.strip():
        yield "Prompt is empty, please provide a starting text."
        return

    try:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        device_type = 'cuda' if 'cuda' in device else 'cpu'
        ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
        ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

        # ----------------------------- 关键修改 -----------------------------
        # 判断 out_dir 是否直接是 .pt 文件
        if out_dir.endswith('.pt'):
            ckpt_path = out_dir
        else:
            ckpt_path = os.path.join(out_dir, 'ckpt.pt')
        # -----------------------------------------------------------------

        if not os.path.exists(ckpt_path):
            yield f"Error: checkpoint not found at {ckpt_path}."
            return

        checkpoint = torch.load(ckpt_path, map_location=device)
        gptconf = GPTConfig(**checkpoint['model_args'])
        model = GPT(gptconf)
        state_dict = checkpoint['model']
        unwanted_prefix = '_orig_mod.'
        for k, v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
        model.load_state_dict(state_dict)

        model.eval()
        model.to(device)
        if compile_model:
            model = torch.compile(model)

        meta_path = os.path.join(data_dir, 'meta.pkl')
        if not os.path.exists(meta_path):
            yield f"Error: meta.pkl not found at {meta_path}."
            return

        with open(meta_path, 'rb') as f:
            meta = pickle.load(f)

        stoi, itos = meta['stoi'], meta['itos']

        def encode(s):
            return [stoi.get(ch, 0) for ch in s]

        def decode(l):
            return ''.join([itos.get(i, '') for i in l])

        xids = torch.tensor(encode(prompt), dtype=torch.long, device=device)[None, ...]
        block_size = gptconf.block_size
        if xids.size(1) > block_size:
            yield f"Error: input length ({xids.size(1)}) exceeds block size ({block_size})."
            return

        with torch.no_grad():
            with ctx:
                for s_i in range(num_samples):
                    idx = xids.clone()
                    generated = prompt
                    for _ in range(max_new_tokens):
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
                        yield f"Sample {s_i+1}:\n{generated}"

                    if s_i < num_samples - 1:
                        yield "-" * 20

    except Exception as ex:
        yield f"An unexpected error occurred: {str(ex)}"
