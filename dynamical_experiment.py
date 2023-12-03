import argparse

import os
os.environ["SCIPY_USE_PROPACK"] = "True"

from source.experiments.hyperparameters_search import import_config
from source.experiments.dynamical_experiment import full_dynamical_experiment

DATASET_HELP = 'Choose dataset name from: "ml_20m", "steam", "amz_b", "amz_g"'
MODEL_NAME_HELP = 'Choose model_type name from:\
    "SVD", "PSIRec", "TDRec", "TDRecReinit", "TIRec", "TIRecA", "Random"'
TQDM_HELP = 'Disable (True) or not (False) tqdm interactive progress line'

if __name__ == '__main__':
    # Parse script arguments:
    parser = argparse.ArgumentParser(description='Dynamical experiment')
    parser.add_argument('dataset', type=str, help=DATASET_HELP)
    parser.add_argument('model_name', type=str, help=MODEL_NAME_HELP)
    parser.add_argument('disable_tqdm', type=str, help=TQDM_HELP)
    args = parser.parse_args()
    # Get a particular config for a dataset:
    conf = import_config(args.dataset)
    disable_tqdm = True if args.disable_tqdm == "True" else False
    config = {
        'prepared_data_path': conf.prepared_data_path,
        'init_ratio': conf.init_ratio,
        'hm_actions_min_stream': conf.hm_actions_min_stream,
        'how_many_iterations': conf.how_many_iterations,
        'topk': conf.topk,
        'metric_dynamics_dir': conf.metric_dynamics_dir,
        'dataset': conf.dataset,
        'max_len_user_history': conf.max_len_user_history,
    }
    if args.model_name in ["SVD", "PSIRec"]:
        config['fixed_config_svd'] = conf.fixed_config_svd
        data_dim = '2d'
    elif args.model_name in ["TDRec", "TDRecReinit", "TIRec", "TIRecA", "Random"]:
        config['fixed_config_tdrec'] = conf.fixed_config_tdrec
        data_dim = '3d'
    else:
        raise RuntimeError(f"Bad model_type name - {args.model_name}")
    # Start dynamical experiment:
    print(f"Start dynamical experiment for {args.model_name} model on {args.dataset} dataset.")
    full_dynamical_experiment(
        model_name=args.model_name, 
        data_dim=data_dim, 
        config=config, 
        disable_tqdm=disable_tqdm,
    )
