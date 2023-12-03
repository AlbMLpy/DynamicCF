from typing import Any, Callable

import numpy as np
import pandas as pd
from tqdm import tqdm

from ..random_grid import full_grid
from ..data_preparation import (
    get_users_to_recommend_test_items,
)
from .prepare_data import (
    prepare_data_for_experiment, 
    train_test_split,
)
from ..evaluation import hr
from ..models.svd import SVD
from ..models.tdrec import TDRec
from ..ti_data_processing import (
    get_df_with_cropped_pos_column,
)
from source.rp_hooi import TuckerFactorsCore


ML20M_NAME = 'ml_20m'
AMZ_B_NAME = 'amz_b'
AMZ_G_NAME = 'amz_g'
STEAM_NAME = 'steam'

def import_config(dataset: str):
    if dataset == ML20M_NAME:
        import configs.ml_20m.config as conf
    elif dataset == AMZ_B_NAME:
        import configs.amz_b.config as conf
    elif dataset == AMZ_G_NAME:
        import configs.amz_g.config as conf
    elif dataset == STEAM_NAME:
        import configs.steam.config as conf
    else:
        raise RuntimeError(f'Bad dataset name - {dataset}')
    return conf

########## SVD MODEL ##########

def get_hparams_metric_grid_svd(
    model_class,
    train: pd.DataFrame,
    valid: pd.DataFrame,
    metric: Callable,
    topk: int,
    optional_config: dict, 
    fixed_config: dict, 
    disable_tqdm: bool = True,
) -> pd.DataFrame:
    params_list = []
    grid, param_names = full_grid(optional_config)
    grid = list(grid)
    for i in tqdm(range(len(grid)), disable=disable_tqdm):
        loose_config = dict(zip(param_names, grid[i]))
        model = model_class(**loose_config, **fixed_config)
        model.train(train)
        # calculate metrics:
        user_to_recommend, test_items = get_users_to_recommend_test_items(valid, model.mappings)
        rec_array = model.recommend(user_to_recommend, topk)
        params_list.append((*grid[i], metric(rec_array, test_items)))

    res_df = pd.DataFrame(params_list, columns=[*param_names, f'{metric.__name__}_{topk}'])
    return res_df

def hyperparameters_search_svd(config: dict[str, Any], disable_tqdm: bool = True) -> pd.DataFrame:
    """
    Get/Save pandas.DataFrame with hyperparameters and corresponding metrics results.

    Parameters
    ----------
    config : dict[str, Any]
        Dictionary with all necessary parameters for hyperparameters search:
            - 'prepared_data_path' - str or Path
            - 'init_ratio' - float
            - 'quantile_train' - float
            - 'topk' - int
            - 'optional_config_svd' - dict
            - 'fixed_config_svd' - dict
            - 'hyperparams_dir' - str or Path
    Returns
    -------
    output : pd.DataFrame
        Pandas DataFrame with results. 
        Also there is a side effect -> save this dataframe 
        to the dir defined by 'hyperparams_dir'.
    """
    # Get initial data and divide it into train and validation:
    initial_data, _, _ = prepare_data_for_experiment(
        config['prepared_data_path'], config['init_ratio'], None)
    train, valid = train_test_split(initial_data, config['quantile_train'])
    # Start hyperparameters-metric calculations:
    params_df = get_hparams_metric_grid_svd(
        SVD,
        train,
        valid,
        metric=hr,
        topk=config['topk'],
        optional_config=config['optional_config_svd'],
        fixed_config=config['fixed_config_svd'],
        disable_tqdm=disable_tqdm
    )
    # Save results:
    params_df.to_csv(config['hyperparams_dir'] / 'svd.csv')
    return params_df

def get_optimal_svd_rank(dataset: str, metric_name: str) -> int:
    params_df = pd.read_csv(import_config(dataset).hyperparams_dir / 'svd.csv', index_col=0)
    if dataset == ML20M_NAME:
        mask = (params_df[metric_name] >= params_df[metric_name].max())
        res_config = params_df[mask].iloc[params_df[mask]['rank'].argmin()]
        rank = int(res_config['rank'])
    elif dataset == AMZ_B_NAME:
        mask = (params_df[metric_name] >= params_df[metric_name].max())
        res_config = params_df[mask].iloc[params_df[mask]['rank'].argmin()]
        rank = int(res_config['rank'])
    elif dataset == AMZ_G_NAME:
        mask = (params_df[metric_name] >= params_df[metric_name].max())
        res_config = params_df[mask].iloc[params_df[mask]['rank'].argmin()]
        rank = int(res_config['rank'])
    elif dataset == STEAM_NAME:
        mask = (params_df[metric_name] >= params_df[metric_name].max())
        res_config = params_df[mask].iloc[params_df[mask]['rank'].argmin()]
        rank = int(res_config['rank'])
    else:
        raise RuntimeError(f'Bad dataset name - {dataset}')
    return rank

########## SVD MODEL ##########

########## TDRec MODEL ##########

def prepare_exit_callback(
    model, 
    valid: pd.DataFrame, 
    metric: Callable,
    topk: int, 
    max_n_iter_relax: int = 3
) -> Callable[[TuckerFactorsCore], bool]:
    def exit_callback(u, v, w, core) -> bool:
        model.u, model.v, model.w, model.core = u, v, w, core
        model.aw = model.attention_mtx.dot(model.w)
        model.last_pos_emb = model.attention_mtx_inv.T.dot(model.w)[-1]
        user_to_recommend, test_items = get_users_to_recommend_test_items(valid, model.mappings)
        rec_array = model.recommend(user_to_recommend, topk)
        score = metric(rec_array, test_items)
        if (score <= exit_callback.max_metric) and (exit_callback.n_iter > max_n_iter_relax):
            return True
        else:
            if score > exit_callback.max_metric:
                exit_callback.max_metric = score
                exit_callback.n_iter = 1
            else:
                exit_callback.n_iter += 1
            exit_callback.metric_list.append(score)
            return False
    exit_callback.metric_list = []
    exit_callback.max_metric = 0.0
    exit_callback.n_iter = 1
    return exit_callback

def get_hparams_metric_grid_tdrec(
    model_class,
    train: pd.DataFrame,
    valid: pd.DataFrame,
    metric: Callable,
    topk: int,
    optional_config: dict, 
    fixed_config: dict, 
    disable_tqdm: bool = True,
) -> pd.DataFrame:
    metric_list = []
    params_list = []

    grid, param_names = full_grid(optional_config)
    grid = list(grid)
    for i in tqdm(range(len(grid)), disable=disable_tqdm):
        loose_config = dict(zip(param_names, grid[i]))
        model = model_class(**loose_config, **fixed_config)

        exit_callback = prepare_exit_callback(model, valid, metric, topk=topk)
        model.train(train, exit_callback)
        
        metric_list.extend(exit_callback.metric_list)
        for j in range(1, len(exit_callback.metric_list) + 1):
            params_list.append((*grid[i], j))
    res_df = pd.DataFrame(params_list, columns=[*param_names, 'n_iter'])
    res_df[f'{metric.__name__}_{topk}'] = metric_list
    return res_df

def hyperparameters_search_tdrec(config: dict[str, Any], disable_tqdm: bool = True) -> pd.DataFrame:
    """
    Get/Save pandas.DataFrame with hyperparameters and corresponding metrics results.

    Parameters
    ----------
    config : dict[str, Any]
        Dictionary with all necessary parameters for hyperparameters search:
            - 'prepared_data_path' - str or Path
            - 'init_ratio' - float
            - 'quantile_train' - float
            - 'topk' - int
            - 'optional_config_tdrec' - dict
            - 'fixed_config_tdrec' - dict
            - 'max_len_user_history' - int
            - 'hyperparams_dir' - str or Path
    Returns
    -------
    output : pd.DataFrame
        Pandas DataFrame with results. 
        Also there is a side effect -> save this dataframe 
        to the dir defined by 'hyperparams_dir'.
    """
    # Get initial data and divide it into train and validation:
    initial_data, _, _ = prepare_data_for_experiment(
        config['prepared_data_path'], config['init_ratio'], None)
    # Leave the last user-item interactions:
    initial_data = get_df_with_cropped_pos_column(
        initial_data, config['max_len_user_history'],
    ).reset_index(drop=True)
    # Train/Validation data split:
    train, valid = train_test_split(initial_data, config['quantile_train'])
    # Start hyperparameters-metric calculations:
    params_df = get_hparams_metric_grid_tdrec(
        TDRec,
        train,
        valid,
        metric=hr,
        topk=config['topk'],
        optional_config=config['optional_config_tdrec'],
        fixed_config=config['fixed_config_tdrec'],
        disable_tqdm=disable_tqdm
    )
    # Save results:
    params_df.to_csv(config['hyperparams_dir'] / 'tdrec.csv')
    return params_df

def get_optimal_tdrec_params(dataset: str, metric_name: str) -> tuple[int, float, int]:
    params_df = pd.read_csv(import_config(dataset).hyperparams_dir / 'tdrec.csv', index_col=0)
    params_df['rank'] = params_df['rank'].apply(lambda x: eval(x))
    if dataset == ML20M_NAME:
        max_n_iter = 5
        params_df = params_df[params_df['n_iter'] <= max_n_iter]
        mask = (
            (params_df[metric_name] >= params_df[metric_name].max())
            & (params_df['att_f'] != 0)
        )
        res_config = params_df[mask].iloc[params_df[mask]['rank'].apply(lambda x: np.prod(x)).argmin()]
        rank, att_f, n_iter = res_config['rank'], res_config['att_f'], res_config['n_iter']
    elif dataset == AMZ_B_NAME:
        max_n_iter = 5
        params_df = params_df[params_df['n_iter'] <= max_n_iter]
        mask = (
            (params_df[metric_name] >= params_df[metric_name].max())
        )
        res_config = params_df[mask].iloc[params_df[mask]['rank'].apply(lambda x: np.prod(x)).argmin()]
        rank, att_f, n_iter = res_config['rank'], res_config['att_f'], res_config['n_iter']
    elif dataset == AMZ_G_NAME:
        max_n_iter = 5
        params_df = params_df[params_df['n_iter'] <= max_n_iter]
        mask = (
            (params_df[metric_name] >= params_df[metric_name].sort_values(ascending=False).unique()[1])
        )
        res_config = params_df[mask].iloc[params_df[mask]['rank'].apply(lambda x: np.prod(x)).argmin()]
        rank, att_f, n_iter = res_config['rank'], res_config['att_f'], res_config['n_iter']
    elif dataset == STEAM_NAME:
        max_n_iter = 5
        params_df = params_df[params_df['n_iter'] <= max_n_iter]
        mask = (
            (params_df[metric_name] >= params_df[metric_name].max())
        )
        res_config = params_df[mask].iloc[params_df[mask]['rank'].apply(lambda x: np.prod(x)).argmin()]
        rank, att_f, n_iter = res_config['rank'], res_config['att_f'], res_config['n_iter']
    else:
        raise RuntimeError(f'Bad dataset name - {dataset}')
    return rank, att_f, n_iter

########## TDRec MODEL ##########
