from typing import Tuple, Sequence, Dict, Any

import numpy as np
import pandas as pd

USER_ID = 'user_id'
ITEM_ID = 'item_id'
POSITION_ID = 'position'
RELEVANCE_COLUMN = 'relevance'
TIMESTAMP = 'timestamp'
DATE_DAYS = 'day'

OLD_NEW_MAP_NAME = 'old2new'
NEW_OLD_MAP_NAME = 'new2old'
N_INDICES = 'n_indices'

PAD_VALUE = -1

def get_cont_mapping_struct(
    values: Sequence[Any],
    start: int = 0,
) -> Dict[str, Any]:
    """
    TODO
    """
    result = {}
    result[OLD_NEW_MAP_NAME] = {v: i for i, v in enumerate(values, start)}
    result[NEW_OLD_MAP_NAME] = {i: v for i, v in enumerate(values, start)}
    result[N_INDICES] = len(result[NEW_OLD_MAP_NAME].keys())
    return result

def update_cont_mapping_struct(mapping_struct: Dict[Any, Any], extra_values: Sequence[int]) -> None:
    """
    TODO
    """
    extra_mapping = get_cont_mapping_struct(extra_values, start=mapping_struct[N_INDICES])
    mapping_struct[OLD_NEW_MAP_NAME].update(extra_mapping[OLD_NEW_MAP_NAME])
    mapping_struct[NEW_OLD_MAP_NAME].update(extra_mapping[NEW_OLD_MAP_NAME])
    mapping_struct[N_INDICES] += extra_mapping[N_INDICES]

def get_df_with_updated_indices(
    df: pd.DataFrame, 
    col_names: Tuple[str, str]
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    TODO
    """
    df_copy = df.copy(deep=True)
    mappings = {}
    for col_name in col_names:
        mappings[col_name] = get_cont_mapping_struct(df_copy[col_name].unique())
        df_copy.loc[:, col_name] = df_copy[col_name].apply(
            lambda x: mappings[col_name][OLD_NEW_MAP_NAME][x]
        )
    return df_copy, mappings

def map_df_columns(
    df: pd.DataFrame, 
    col_names: Tuple[str, str],
    mappings: Dict[str, Any],
) -> pd.DataFrame:
    df_copy = df.copy(deep=True)
    for col_name in col_names:
        df_copy.loc[:, col_name] = df_copy[col_name].apply(lambda x: mappings[col_name][OLD_NEW_MAP_NAME][x])
    return df_copy

def get_inds_vals(data: pd.DataFrame, mappings=None):
    """
    TODO
    """
    rows, cols = data[USER_ID], data[ITEM_ID]
    if mappings is not None:
        rows = rows.apply(lambda x: mappings[USER_ID][OLD_NEW_MAP_NAME].get(x, PAD_VALUE))
        cols = cols.apply(lambda x: mappings[ITEM_ID][OLD_NEW_MAP_NAME].get(x, PAD_VALUE))
    indices = pd.concat([rows, cols], axis=1).to_numpy()
    return indices, data[RELEVANCE_COLUMN].values

def mask_first_occurence(a):
    mask = np.ones(len(a), dtype=bool)
    mask[np.unique(a, return_index=True)[1]] = False
    return mask

def print_recsys_df_stats(df: pd.DataFrame, save_path=None) -> None:
    nu_users = df[USER_ID].nunique()
    nu_items = df[ITEM_ID].nunique()
    nu_dates = df[DATE_DAYS].nunique()
    n_interactions, _ = df.shape
    n_items_mean = np.round(df.groupby(USER_ID)[ITEM_ID].count().mean(), 2)
    n_items_median = np.round(df.groupby(USER_ID)[ITEM_ID].count().median(), 2)
    density = np.round(n_interactions / (nu_users * nu_items) * 100, 2)
    print(
        f'# Unique users = {nu_users}\n'
        f'# Unique items = {nu_items}\n'
        f'# Interactions = {n_interactions}\n'
        f'# Unique dates(days) = {nu_dates}\n'
        f'# Items per user (mean) = {n_items_mean}\n'
        f'# Items per user (median) = {n_items_median}\n'
        f'Density% = {density}\n'

    )
    if save_path is not None:
        pd.Series(
            {
                'n_users': nu_users,
                'n_items': nu_items,
                'nu_dates': nu_dates,
                'n_interactions': n_interactions,
                'n_items_mean': n_items_mean,
                'n_items_median': n_items_median,
                'density%': density,
            }
        ).to_csv(save_path)

def get_known_ui_df_mask(df: pd.DataFrame, mappings):
    users = df[USER_ID].apply(lambda x: mappings[USER_ID][OLD_NEW_MAP_NAME].get(x, PAD_VALUE))
    items = df[ITEM_ID].apply(lambda x: mappings[ITEM_ID][OLD_NEW_MAP_NAME].get(x, PAD_VALUE))
    old_users_mask = ~(users == PAD_VALUE)
    old_items_mask = ~(items == PAD_VALUE)
    mask = (old_users_mask) & (old_items_mask)
    return mask

def get_users_to_recommend_test_items(data: pd.DataFrame, mappings):
    inds, _ = get_inds_vals(
        data[get_known_ui_df_mask(data, mappings)]
    )
    mask = ~mask_first_occurence(inds[:, 0])
    user_to_recommend = inds[mask][:, 0]
    test_items = inds[mask][:, 1]
    return user_to_recommend, test_items
