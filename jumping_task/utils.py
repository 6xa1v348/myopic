import os

import random
import numpy as np
import torch


def set_seed_everywhere(seed):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True


def mkdir(path):
    try:
        os.mkdir(path)
    except FileExistsError:
        pass
    return path
