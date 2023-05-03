
import os
import random
import torch
import numpy as np


def is_cuda_available():
    return torch.cuda.is_available()


def set_seed(seed):
    """
    Referred from:
    - https://stackoverflow.com/questions/38469632/tensorflow-non-repeatable-results
    """
    # Reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = True
    try:
        os.environ['PYTHONHASHSEED'] = str(seed)
    except:
        pass
