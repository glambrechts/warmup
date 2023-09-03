import os
import torch
import random
import numpy as np


def set_seed(seed):
    """
    Set the random number generators seed for reproducibility.
    """
    if seed is None:
        seed = int.from_bytes(os.urandom(4), 'big')

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    return seed
