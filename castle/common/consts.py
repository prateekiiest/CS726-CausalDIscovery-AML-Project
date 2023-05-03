
LOG_FREQUENCY = 200

LOG_FORMAT = ("[%(asctime)s][%(filename)s - line %(lineno)s] "
              "- %(levelname)s - %(message)s")


# ==========================================================
# const variable used to check arguments for each algorithm

# key of dict denotes argument name of function;
# corresponding value denotes valid value for key.
# ==========================================================

# CORL
CORL_VALID_PARAMS = {
    'encoder_name': ['transformer', 'lstm', 'mlp'],
    'decoder_name': ['lstm', 'mlp'],
    'reward_mode': ['episodic', 'dense'],
    'reward_score_type': ['BIC', 'BIC_different_var'],
    'reward_regression_type': ['LR', 'GPR']
}

# RL
RL_VALID_PARAMS = {
    'encoder_type': ['TransformerEncoder', 'GATEncoder'],
    'decoder_type': ['SingleLayerDecoder', 'TransformerDecoder',
                     'BilinearDecoder', 'NTNDecoder'],
    'decoder_activation': ['tanh', 'relu', 'none'],
    'score_type': ['BIC', 'BIC_different_var'],
    'reg_type': ['LR', 'QR']
}

# GraNDAG
GRANDAG_VALID_PARAMS = {
    'model_name': ['NonLinGaussANM', 'NonLinGauss'],
    'nonlinear': ['leaky-relu', 'sigmoid'],
    'optimizer': ['rmsprop', 'sgd'],
    'norm_prod': ['paths', 'none']
}

# Notears
NOTEARS_VALID_PARAMS = {
    'loss_type': ['l2', 'logistic', 'poisson']
}

# nonlinear Notears
NONLINEAR_NOTEARS_VALID_PARAMS = {
    'model_type': ['mlp', 'sob']
}

# mcsl
MCSL_VALID_PARAMS = {
    'model_type': ['nn', 'qr']
}

# direct lingam
DIRECT_LINGAM_VALID_PARAMS = {
    'measure': ['pwling' , 'kernel']
}

# pc
PC_VALID_PARAMS = {
    'variant': ['original', 'stable', 'parallel'],
    'ci_test': ['fisher', 'g2', 'chi2']
}

# TTPM
TTPM_VALID_PARAMS = {
    'penalty': ['BIC', 'AIC']
}

# DAG_GNN
GNN_VALID_PARAMS = {
    'encoder_type': ['mlp', 'sem'],
    'decoder_type': ['mlp', 'sem'],
    'optimizer': ['adam', 'sgd']
}
