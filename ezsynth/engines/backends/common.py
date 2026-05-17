import gc

import torch


def get_auto_torch_device():
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch, "mps") and torch.mps.is_available():
        return "mps"
    else:
        return "cpu"


def clear_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
