from typing import Optional
from datetime import datetime

import numpy as np
import pandas as pd

from ..data_preparation import (
    DATE_DAYS, USER_ID
)

def get_date_to_group_mapping(date_user_count: pd.Series, hm_actions_min: int):
    new_start_date = True
    date_to_group_id = {}
    local_sum = 0
    group_n = None
    for i, (d, v) in enumerate(date_user_count.items()):
        if new_start_date:
            local_sum = v
            group_n = d
            new_start_date = True if local_sum > hm_actions_min else False
        else:
            local_sum += v
            if local_sum > hm_actions_min:
                new_start_date = True
        date_to_group_id[d] = group_n
        if i == len(date_user_count) - 1:
            if local_sum <= hm_actions_min:
                date_to_group_id[d] = np.sort(date_user_count.index)[i-1]
    return date_to_group_id

def prepare_data_for_experiment(
    prepared_data_path, 
    init_ratio: float,
    hm_actions_min: Optional[int],
):
    data = pd.read_csv(prepared_data_path, index_col=[0])
    mask_init = data.index < int(data.shape[0] * init_ratio)
    initial_data = data[mask_init]
    if hm_actions_min is None:
        data_stream = data[~mask_init].groupby(DATE_DAYS)
        left_data = data[~mask_init]
    else:
        left_data = data[~mask_init].sort_values(by=DATE_DAYS, ascending=True)
        date_user_count = left_data.groupby(DATE_DAYS)[USER_ID].count()
        date_to_group_id = get_date_to_group_mapping(date_user_count, hm_actions_min)
        left_data['group'] = left_data[DATE_DAYS].apply(lambda x: date_to_group_id[x])
        left_data = left_data.drop([DATE_DAYS], axis=1).rename({'group': DATE_DAYS}, axis=1)
        data_stream = left_data.groupby(DATE_DAYS)
    return initial_data, data_stream, left_data

def get_users_stability(
    initial_data: pd.DataFrame,
    left_data: pd.DataFrame,
    hm_unique_days: int,
):
    mask = (
        left_data[DATE_DAYS].isin(np.sort(left_data[DATE_DAYS].unique())[:hm_unique_days])
        & left_data[USER_ID].isin(initial_data[USER_ID].unique())
    )
    return np.array(
        left_data[mask].groupby([USER_ID])[DATE_DAYS].nunique().sort_values().tail(50).index
    )

def train_test_split(data: pd.DataFrame, quantile_train: float):
    int_dd = data[DATE_DAYS].apply(lambda x: datetime.fromisoformat(x).timestamp())
    quantile_dd = int_dd.quantile(q=quantile_train, interpolation='nearest')
    mask_last = int_dd >= quantile_dd
    train, test = data[~mask_last], data[mask_last] 
    return train, test
