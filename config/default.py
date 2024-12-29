import numpy as np
from multiprocessing import cpu_count

IntegerTypes = np.uint32

# ------------------- default config -------------------
DEFAULT_CONFIG = {
    "data_process": {
        "raw_data_dir": "./data/raw",
        "processed_data_dir": "./data/processed",
        "train_split_ratio": 0.9,
        "no_validation": False,
        "use_gpt2_tokenizer": False,
        "num_proc": cpu_count() // 2
    },
    "training": {
        "data_dir": "./data/processed",
        "out_dir": "out",
        "plot_interval": 10,
        "log_interval": 10,
        "num_eval_seeds": 0,
        "save_best_val_checkpoint": True,
        "init_from": "scratch",
        "gradient_accumulation_steps": 1,
        "batch_size": 32,
        "block_size": 512,
        "n_layer": 6,
        "n_head": 6,
        "n_embd": 384,
        "dropout": 0.1,
        "bias": True,
        "learning_rate": 1e-3,
        "max_iters": 300,
        "weight_decay": 1e-2,
        "beta1": 0.9,
        "beta2": 0.999,
        "lr_scheduler_type": "cosine",
        "warmup_iters": 10,
        "lr_decay_iters": 300,
        "min_lr": 1e-5,
        "step_size": 150,
        "step_gamma": 0.1,
        "polynomial_power": 2.0,
        "backend": "nccl",
        "device": "cuda",
        "dtype": "float16",
        "compile_model": False,
        "seed": 1024,
        "save_interval": 50
    },
    "inference": {
        "data_dir": "./data/processed",
        "out_dir": "out",
        "prompt": "",
        "num_samples": 1,
        "max_new_tokens": 50,
        "temperature": 0.7,
        "top_k": 50,
        "seed": 1024,
        "device": "cuda",
        "dtype": "float16",
        "compile_model": False
    }
}
