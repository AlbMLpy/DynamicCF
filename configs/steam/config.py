from itertools import product

import numpy as np

from load_data import (
    DATA_DIR_STEAM, 
    RESULTS_DIR, 
    DIR_STEAM_NAME,
)
from source.rp_hooi import tucker_rank_is_valid

dataset = DIR_STEAM_NAME
prepared_data_path = DATA_DIR_STEAM / 'prepared_data.csv'
init_ratio = 0.8
quantile_train = 0.99
hm_actions_min_stream = None
how_many_iterations = 50
max_len_user_history = 8

topk = 5
### SVD based models params: ###
optional_config_svd = {
    'rank': np.arange(10, 400, 10),
}
fixed_config_svd = {
    'n_power_iter': 1,
    'oversampling_factor': 2,
    'seed': 13,
}
### TD based models params: ###
_rank_0 = [64, 128, 256]
_rank_1 = [64, 100]
_rank_2 = [2, 4]
_rank = [
    rank for rank in product(_rank_0, _rank_1, _rank_2) 
    if tucker_rank_is_valid(rank)
]
optional_config_tdrec = {
    'rank': _rank,
    'att_f': [0, 2],
}
fixed_config_tdrec = {
    'seq_len': max_len_user_history,
    'n_power_iter': 1,
    'oversampling_factor': 2,
    'seed': 13,
    'parallel': True,
    'force_n_iter': False,
}

n_chunks_list = [1, 10]
swl_list = [20,]
plot_models_list = ['SVD', 'PSIRec', 'TDRec', 'TDRecReinit', 'TIRecA']

hyperparams_dir = RESULTS_DIR / DIR_STEAM_NAME / 'hyperparams'
metric_dynamics_dir = RESULTS_DIR / DIR_STEAM_NAME / 'metric_dynamics'
graphics_dir = RESULTS_DIR / DIR_STEAM_NAME / 'graphics'
ablation_study_dir = RESULTS_DIR / DIR_STEAM_NAME / 'ablation_study'
