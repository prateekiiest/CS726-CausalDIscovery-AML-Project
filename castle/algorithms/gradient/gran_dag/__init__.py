

from castle.backend import backend

if backend == 'pytorch':
    from .torch import GraNDAG
elif backend == 'mindspore':
    from .mindspore import GraNDAG
