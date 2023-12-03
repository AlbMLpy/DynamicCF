import json
import argparse

import os
os.environ["SCIPY_USE_PROPACK"] = "True"

from source.ti_data_processing import (
    get_df_with_cropped_pos_column,
)
from source.experiments.prepare_data import (
    prepare_data_for_experiment, 
    get_users_stability,
)
from source.experiments.prepare_model import prepare_model
from source.experiments.hyperparameters_search import import_config
from source.experiments.dynamical_experiment import dynamical_experiment

DATASET_HELP = 'Choose dataset name from: "ml_20m", "steam", "amz_b", "amz_g"'
MODEL_NAME_HELP = 'Choose model_type name from: "PSIRec", "TIRecA"'
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
    if args.model_name in ["PSIRec",]:
        config['fixed_config_svd'] = conf.fixed_config_svd
        data_dim = '2d'
        options = [
            (0, True), 
            (1e-8, False), (1e-5, False), (1e-3, False),
            (1e-8, True), (1e-5, True), (1e-3, True),
        ]
    elif args.model_name in ["TIRecA",]:
        config['fixed_config_tdrec'] = conf.fixed_config_tdrec
        data_dim = '3d'
        options = [
            (0, True), 
            (1e-8, False), (1e-5, False), (1e-3, False),
            (1e-8, True), (1e-5, True), (1e-3, True),
        ]
    else:
        raise RuntimeError(f"Bad model_type name - {args.model_name}")
    
    model_name = args.model_name
    
    for init_new_emb, nu_ni_proc_int in options:
    
        # Start dynamical experiment:
        print(
            f"Start dynamical experiment for {args.model_name} model "
            + f"(init_new_emb={init_new_emb}, nu_ni_proc_int={nu_ni_proc_int}) on {args.dataset} dataset."
        )
        # Get initial training data, data stream:
        initial_data, data_stream, left_data = prepare_data_for_experiment(
            config['prepared_data_path'], 
            config['init_ratio'], 
            config['hm_actions_min_stream'],
        )
        if data_dim == '3d':
            # Leave the last user-item interactions:
            initial_data = get_df_with_cropped_pos_column(
                initial_data, config['max_len_user_history'],
            ).reset_index(drop=True)
        elif data_dim == '2d':
            pass
        else:
            raise RuntimeError(f'Bad data_dim - {data_dim}')
        # Find users to test stability of recommendations:
        test_users_wji_similarity = get_users_stability(
            initial_data, left_data, config['how_many_iterations']
        )
        # Prepare model and extra parameters:
        dyn_exp_dict = prepare_model(model_name, config)
        dyn_exp_dict['rec_model']._init_new_embeddings = init_new_emb
        dyn_exp_dict['rec_model']._nu_ni_process_integrator = nu_ni_proc_int
        
        # Start dynamical experiment:
        results = dynamical_experiment(
            initial_data=initial_data,
            data_stream=data_stream, 
            test_users_wji_similarity=test_users_wji_similarity,
            k_recs=config['topk'],
            how_many_iterations=config['how_many_iterations'],
            disable_tqdm=disable_tqdm,
            **dyn_exp_dict,
        )
        # Save results:
        with open(config['metric_dynamics_dir'] / f'{model_name}_std_{init_new_emb}_int_{nu_ni_proc_int}.json', 'w') as f:
            json.dump(results, f)
