

from castle.backend import backend

if backend == 'pytorch':
    from .torch import MCSL