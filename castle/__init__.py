__version__ = "1.0.3"


import sys
if sys.version_info < (3, 6):
    sys.exit('Sorry, Python < 3.6 is not supported.')
import os
import logging

from castle.common import GraphDAG
from castle.metrics import MetricsDAG


logging.basicConfig(format='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s',
                    level=logging.INFO)


def _import_algo(algo):
    """
    import algorithm corresponding to `algo`

    Parameters
    ----------
    algo: str
        lowercase letters of the algorithm `algo`

    Returns
    -------
    out: class object
        castle algorithm
    """

    if algo.lower() == 'pc':
        from castle.algorithms import PC as Algorithm
    elif algo.lower() == 'anm':
        from castle.algorithms import ANMNonlinear as Algorithm
    elif algo.lower() == 'icalingam':
        from castle.algorithms import ICALiNGAM as Algorithm
    elif algo.lower() == 'directlingam':
        from castle.algorithms import DirectLiNGAM as Algorithm
    elif algo.lower() == 'notears':
        from castle.algorithms import Notears as Algorithm
    elif algo.lower() == 'notearslowrank':
        from castle.algorithms import NotearsLowRank as Algorithm
    elif algo.lower() == 'notearsnonlinear':
        from castle.algorithms import NotearsNonlinear as Algorithm
    elif algo.lower() == 'corl':
        from castle.algorithms import CORL as Algorithm
    elif algo.lower() == 'rl':
        from castle.algorithms import RL as Algorithm
    elif algo.lower() == 'gae':
        from castle.algorithms import GAE as Algorithm
    elif algo.lower() == 'ges':
        from castle.algorithms import GES as Algorithm
    elif algo.lower() == 'golem':
        from castle.algorithms import GOLEM as Algorithm
    elif algo.lower() == 'grandag':
        from castle.algorithms import GraNDAG as Algorithm
    elif algo.lower() == 'pnl':
        from castle.algorithms import PNL as Algorithm
    else:
        raise ValueError('Unknown algorithm.==========')

    logging.info(f"import algorithm corresponding to {algo} complete!")

    return Algorithm
