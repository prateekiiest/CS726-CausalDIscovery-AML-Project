#@title CORL Demo
import os
os.environ['CASTLE_BACKEND'] ='pytorch'

from castle.common import GraphDAG
from castle.metrics import MetricsDAG
from castle.datasets import DAG, IIDSimulation
from castle.algorithms import CORL

type = 'ER'  # or `SF`
h = 2  # ER2 when h=5 --> ER5
n_nodes = 5
n_edges = h * n_nodes
method = 'linear'
sem_type = 'gauss'

if type == 'ER':
    weighted_random_dag = DAG.erdos_renyi(n_nodes=n_nodes, n_edges=n_edges,
                                          weight_range=(0.5, 2.0), seed=300)
elif type == 'SF':
    weighted_random_dag = DAG.scale_free(n_nodes=n_nodes, n_edges=n_edges,
                                         weight_range=(0.5, 2.0), seed=300)
else:
    raise ValueError('Just supported `ER` or `SF`.')

dataset = IIDSimulation(W=weighted_random_dag, n=2000,
                        method=method, sem_type=sem_type)
true_dag, X = dataset.B, dataset.X

# rl learn
rl = CORL(encoder_name='transformer',
          decoder_name='lstm',
          reward_mode='episodic',
          reward_regression_type='LR',
          batch_size=64,
          input_dim=64,
          embed_dim=64,
          iteration=2000,
          device_type='gpu')
rl.learn(X)

# plot est_dag and true_dag
GraphDAG(rl.causal_matrix, true_dag)

# calculate accuracy
met = MetricsDAG(rl.causal_matrix, true_dag)
print(met.metrics)
