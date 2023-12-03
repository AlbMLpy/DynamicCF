from typing import Union
from collections import deque

import numpy as np
import pandas as pd

from .data_preparation import (
    USER_ID, POSITION_ID, TIMESTAMP,
)

PAD_VALUE = -1

def update_user_seq_history(
    new_interactions: np.ndarray, 
    user_seq_history: dict[int, deque[int]], 
    max_len_history: int
) -> None:
    """
    Update user_seq_history structure with newly added interactions.

    Parameters
    ----------
    new_interactions : numpy.ndarray[int, int, int]
        Numpy array of (User, Item, Position) triples sorted by [User, Position] in ascending order.
    user_seq_history : dict[int, deque[int]]
        Dictionary structure that stores users' sequential history.
    max_len_history: int
        Maximum length of user's history of interactions.

    Returns
    -------
    output : None 

    """
    for user, item, _ in new_interactions:
        if user not in user_seq_history:
            user_seq_history[user] = deque([PAD_VALUE] * max_len_history, maxlen=max_len_history)
        user_seq_history[user].append(item)

def calculate_delta(
    user_seq_history_new: dict[int, deque[int]], 
    user_seq_history_old: dict[int, Union[deque[int], list[int]]],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate delta between two sequential histories of users in the form of COO tensor.
    Parameters
    ----------
    user_seq_history_new : dict[int, deque[int]]
        Dictionary structure that stores newly updated users' sequential history.
    user_seq_history_old : dict[int, deque[int]]
        Dictionary structure that stores previous users' sequential history.

    Returns
    -------
    output : (np.ndarray[int, int, int], np.ndarray[float])
        Return indices and values of a tensor in COO format: inds[User, Item, Postion], vals

    """
    inds = []
    vals = []
    for user in user_seq_history_new:
        uh_new = user_seq_history_new[user]
        # new user:
        if user not in user_seq_history_old:
            for pos, item_new in enumerate(uh_new):
                if item_new != PAD_VALUE:
                    inds.append((user, item_new, pos))
                    vals.append(1)
        else:
            uh_old = user_seq_history_old[user]
            for pos, (item_new, item_old) in enumerate(zip(uh_new, uh_old)):
                if item_new != item_old:
                    if item_new != PAD_VALUE:
                        inds.append((user, item_new, pos))
                        vals.append(1)

                    if item_old != PAD_VALUE:
                        inds.append((user, item_old, pos))
                        vals.append(-1)
    return np.array(inds), np.array(vals)

def add_order_cropped(x: pd.DataFrame, max_len: int):
    len_x = len(x)
    if max_len >= len_x:
        x[POSITION_ID] = np.arange(len_x)
    else:
        order_values = np.array([PAD_VALUE] * len_x)
        order_values[-max_len:] = np.arange(max_len)
        x[POSITION_ID] = order_values
    return x

def get_df_with_cropped_pos_column(
    df: pd.DataFrame, 
    max_seq_len: int, 
    sort_bool: bool = True
) -> pd.DataFrame:
    # Add order column:
    if sort_bool:
        df = df.sort_values(by=[USER_ID, TIMESTAMP])
    res_df = (
        df
        .groupby(USER_ID)
        .apply(add_order_cropped, max_seq_len)
    )
    # Leave the latest user-item interactions:
    return res_df[res_df[POSITION_ID] != PAD_VALUE].reset_index(drop=True)
