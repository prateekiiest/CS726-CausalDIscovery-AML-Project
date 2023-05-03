
import os
import random
import torch
import numpy as np
import networkx as nx


def is_cuda_available():
    return torch.cuda.is_available()


def set_seed(seed):
    """
    Set random seed for reproducibility.

    Parameters
    ----------
    seed: int
        Random seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    try:
        os.environ['PYTHONHASHSEED'] = str(seed)
    except:
        pass


def is_dag(B):
    """
    Check whether B corresponds to a DAG.

    Parameters
    ----------
    B: numpy.ndarray
        [d, d] binary or weighted matrix.
    """
    return nx.is_directed_acyclic_graph(nx.DiGraph(B))

