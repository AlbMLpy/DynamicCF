import argparse

from source.experiments.hyperparameters_search import (
    import_config,
    hyperparameters_search_svd, 
    hyperparameters_search_tdrec,
)

DATASET_HELP = 'Choose dataset name from: "ml_20m", "steam", "amz_b", "amz_g"'
MODEL_TYPE_HELP = 'Choose model_type name from: "svd", "tdrec"'
TQDM_HELP = 'Disable (True) or not (False) tqdm interactive progress line'

if __name__ == '__main__':
    # Parse script arguments:
    parser = argparse.ArgumentParser(description='Hyperparameters - Metrics Calculations')
    parser.add_argument('dataset', type=str, help=DATASET_HELP)
    parser.add_argument('model_type', type=str, help=MODEL_TYPE_HELP)
    parser.add_argument('disable_tqdm', type=str, help=TQDM_HELP)
    args = parser.parse_args()
    # Get a particular config for a dataset:
    conf = import_config(args.dataset)
    disable_tqdm = True if args.disable_tqdm == "True" else False
    if args.model_type == 'svd':
        config = {
            'prepared_data_path': conf.prepared_data_path,
            'init_ratio': conf.init_ratio,
            'quantile_train': conf.quantile_train,
            'topk': conf.topk,
            'optional_config_svd': conf.optional_config_svd,
            'fixed_config_svd': conf.fixed_config_svd,
            'hyperparams_dir': conf.hyperparams_dir,
        }
        print(
        f"Start hyperparameters-metrics calculations for {args.model_type} type model on {args.dataset} dataset.")
        hyperparameters_search_svd(config, disable_tqdm=disable_tqdm)
    elif args.model_type == 'tdrec':
        config = {
            'prepared_data_path': conf.prepared_data_path,
            'init_ratio': conf.init_ratio,
            'quantile_train': conf.quantile_train,
            'topk': conf.topk,
            'optional_config_tdrec': conf.optional_config_tdrec,
            'fixed_config_tdrec': conf.fixed_config_tdrec,
            'max_len_user_history': conf.max_len_user_history,
            'hyperparams_dir': conf.hyperparams_dir,
        }
        print(f"Start hyperparameters-metrics calculations for {args.model_type} type model on {args.dataset} dataset.")
        hyperparameters_search_tdrec(config, disable_tqdm=disable_tqdm)
    else:
        raise RuntimeError(f"Bad model_type name - {args.model_type}")
