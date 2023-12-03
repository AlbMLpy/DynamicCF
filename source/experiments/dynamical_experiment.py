import json
from typing import Callable, Any

import numpy as np
import pandas as pd
from tqdm import tqdm

from ..data_preparation import (
    POSITION_ID,
    get_users_to_recommend_test_items
)
from ..ti_data_processing import (
    get_df_with_cropped_pos_column,
)
from ..general_functions import elapsed_time, rel_norm
from ..evaluation import hr, mrr, wji_sim
from .prepare_model import prepare_model
from .prepare_data import (
    prepare_data_for_experiment, 
    get_users_stability,
)

def dynamical_experiment(
    rec_model, 
    initial_data: pd.DataFrame,
    data_stream,
    test_users_wji_similarity: np.ndarray,
    k_recs: int, 
    max_len_history: int,
    how_many_iterations: int,
    update_fuction: Callable,
    upd_data_mode: str,
    disable_tqdm: bool = False,
    reinit_model_bool: bool = False,
    three_dimensional_bool: bool = False,
) -> dict[str, list[float]]:
    """
    Explore the dynamical characteristics of RecSys model.

    Parameters
    ----------
    rec_model : object
        Recommender model having interface like SVD from 
        source.model.svd for example 
    initial_data : pd.DataFrame
        Initial training data having the following columns: 
        'user_id', 'item_id', 'timestamp', 'relevance', 
        ['position'] - optional.
    data_stream : pandas.core.groupby.generic.DataFrameGroupBy
        Data stream of days or groups for dynamical learning.
    test_users_wji_similarity : np.ndarray
        Array of users to test stability of recommendations on.
    k_recs : int
        K in top-K recommndation problem.
    max_len_history : int
        Maximum number of actions in a user history a model can 
        take into account. (Used for three-dimensional models)
    how_many_iterations : int
        How many days/groups to use from data_stream.
    update_fuction : Callable
        Function used for parameters update: rec_model.train or
        rec_model.update depending on model class.
    upd_data_mode : str
        'chunk' or 'concatenated' used for parameters update.
    disable_tqdm : bool, default: False
        Disable tqdm line or not.
    reinit_model_bool : bool, default: False
        Parameter used only for tensor model with reinitialization.
    three_dimensional_bool : bool, default: False
        Use three-dimensional data or not.
    Returns
    -------
    output : dict[str, list[float]]
        Dictionary with dynamics of a model.
    """
    # Prepare structure to save dynamics of the experiments:
    dynamics_dict = {
        'calculation_time': [], 
        'hr': [], 
        'mrr': [], 
        'wji': [],
        'rnd_users': [],
        'rnd_items': [],
        'nd_users': [],
        'nd_items': [],
    }
    # Train model the first time:
    _, ts = elapsed_time(rec_model.train, initial_data)
    dynamics_dict['calculation_time'].append(ts)
    # Prepare structure for data concatenation:
    concatenated = initial_data.copy()
    # Start dynamical experiment:
    for iter_number, (_, chunk) in tqdm(enumerate(data_stream, 1), total=how_many_iterations, disable=disable_tqdm):
        # Calculate RecSys quality metrics:
        user_to_recommend, test_items = get_users_to_recommend_test_items(
            chunk, rec_model.mappings
        )
        rec_array = rec_model.recommend(user_to_recommend, k_recs)
        _hr, _mrr = hr(rec_array, test_items), mrr(rec_array, test_items)
        dynamics_dict['hr'].append(_hr)
        dynamics_dict['mrr'].append(_mrr)
        # Prepare and calculate wji similarity (stability of RecSys):
        if iter_number == 1:
            rec_array_t1 = rec_model.recommend(test_users_wji_similarity, k_recs, internal=True)
        elif iter_number == 2:
            rec_array_t2 = rec_model.recommend(test_users_wji_similarity, k_recs, internal=True)
            dynamics_dict['wji'].append(wji_sim(rec_array_t1, rec_array_t2, rec_model.get_n_items()))
        else:
            rec_array_t1 = rec_array_t2
            rec_array_t2 = rec_model.recommend(test_users_wji_similarity, k_recs, internal=True)
            dynamics_dict['wji'].append(wji_sim(rec_array_t1, rec_array_t2, rec_model.get_n_items()))
        

        # Relative norm difference user/item embeddings:
        if iter_number == 1:
            u1, v1 = rec_model.get_u(), rec_model.get_v()
        elif iter_number == 2:
            u2, v2 = rec_model.get_u(), rec_model.get_v()
            if u2.shape[0] > u1.shape[0]:
                _u = np.zeros(u2.shape)
                _u[:u1.shape[0], :] = u1
                u1 = _u
            if v2.shape[0] > v1.shape[0]:
                _v = np.zeros(v2.shape)
                _v[:v1.shape[0], :] = v1
                v1 = _v
            dynamics_dict['rnd_users'].append(rel_norm(u2 - u1, u2))
            dynamics_dict['rnd_items'].append(rel_norm(v2 - v1, v2))
        else:
            u1, v1 = u2, v2
            u2, v2 = rec_model.get_u(), rec_model.get_v()
            if u2.shape[0] > u1.shape[0]:
                _u = np.zeros(u2.shape)
                _u[:u1.shape[0], :] = u1
                u1 = _u
            dynamics_dict['rnd_users'].append(rel_norm(u2 - u1, u2))
            if v2.shape[0] > v1.shape[0]:
                _v = np.zeros(v2.shape)
                _v[:v1.shape[0], :] = v1
                v1 = _v
            if v2.shape[0] < v1.shape[0]:
                dynamics_dict['rnd_items'].append(-1)
            else:
                dynamics_dict['rnd_items'].append(rel_norm(v2 - v1, v2))
        # Norm dynamics user/item embeddings:
        dynamics_dict['nd_users'].append(
            np.linalg.norm(rec_model.get_u(), ord='fro')
        )
        dynamics_dict['nd_items'].append(
            np.linalg.norm(rec_model.get_v(), ord='fro')
        )
        
    
        # Update data:
        if upd_data_mode != 'chunk':
            if three_dimensional_bool:
                concatenated = concatenated[[i for i in concatenated.columns if i != POSITION_ID]]
                concatenated = pd.concat([concatenated, chunk], axis=0)
                # Add order column:
                concatenated = get_df_with_cropped_pos_column(concatenated, max_len_history)
            else:
                concatenated = pd.concat([concatenated, chunk], axis=0)
        # Train the model iteratively:
        upd_data = chunk if upd_data_mode == 'chunk' else concatenated
        if reinit_model_bool:
            factors_init_list = rec_model.get_factors()
            previous_mappings = rec_model.mappings
            _, ts = elapsed_time(update_fuction, upd_data, factors_init_list, previous_mappings)
        else:
            _, ts = elapsed_time(update_fuction, upd_data)
        dynamics_dict['calculation_time'].append(ts)
        if iter_number == how_many_iterations:
            break
    return dynamics_dict

def full_dynamical_experiment(
    model_name: str,
    data_dim: str, 
    config: dict[str, Any], 
    disable_tqdm: bool = True
) -> dict[str, list[float]]:
    """
    Get/Save dictionaty with metrics in dynamics.

    Parameters
    ----------
    model_name : str
        Choose model name from:
            - 'SVD'
            - 'PSIRec'
            - 'TDRec'
            - 'TDRecReinit'
            - 'TIRec'
            - 'TIRecA'
            - 'Random'
    data_dim : str
        Choose '2d' or '3d'.
    config : dict[str, Any]
        Dictionary with all necessary parameters for hyperparameters search:
            - 'prepared_data_path' - str or Path
            - 'init_ratio' - float
            - 'hm_actions_min_stream' - int
            - 'how_many_iterations' - int
            - 'topk' - int
            - 'fixed_config_svd' or 'fixed_config_tdrec' - dict 
            - 'metric_dynamics_dir' -  str or Path
            - 'dataset' - str
            - 'max_len_user_history' - int
    Returns
    -------
    output : dict[str, list[float]]
        Dictionary with dynamical results.
    """
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
    with open(config['metric_dynamics_dir'] / f'{model_name}.json', 'w') as f:
        json.dump(results, f)
    return results
