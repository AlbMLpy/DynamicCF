import json

import pandas as pd
import numpy as np
import matplotlib as mpl

from source.plotting import PARAMS, get_fig_ax, set_ax, add_plot_ax

from source.data_preparation import (
    USER_ID, 
    ITEM_ID, 
    DATE_DAYS,
    RELEVANCE_COLUMN,
    TIMESTAMP,
    print_recsys_df_stats,
)
from source.experiments.prepare_data import prepare_data_for_experiment
from source.experiments.hyperparameters_search import import_config
from load_data import (
    RESULTS_DIR,
    DATA_DIR_AMZ_B, 
    DATA_DIR_AMZ_G, 
    DATA_DIR_ML20M, 
    DATA_DIR_STEAM,
    DIR_ML20M_NAME,
    DIR_AMZ_B_NAME,
    DIR_AMZ_G_NAME,
    DIR_STEAM_NAME,
)

def encode_column(data: pd.DataFrame, column: str) -> pd.DataFrame:
    return (
        data
        .assign(_temp=pd.Categorical(data[column]).codes)
        .drop([column], axis=1)
        .rename({'_temp': column}, axis=1)
    )

def prepare_ml20m():
    ratio = 0.2
    data = (
        pd.read_csv(DATA_DIR_ML20M / 'ratings.csv')
        .rename({'userId': USER_ID, 'movieId': ITEM_ID}, axis=1)
        [[USER_ID, ITEM_ID, TIMESTAMP]]
    )
    data = data.sort_values(by=TIMESTAMP)
    data = data.tail(int(data.shape[0] * ratio)).reset_index(drop=True)
    data[TIMESTAMP] = pd.to_datetime(data[TIMESTAMP], unit='s', utc=True)
    data['day'] = data[TIMESTAMP].round('d')

    data[RELEVANCE_COLUMN] = np.ones(len(data))

    print_recsys_df_stats(data, DATA_DIR_ML20M / 'data_stats.csv')
    data.to_csv(DATA_DIR_ML20M / 'prepared_data.csv')

def prepare_amazon(file_dir, file_name):
    data = (
        pd.read_csv(file_dir / file_name)
        .rename(
            {'reviewerID': USER_ID, 'asin': ITEM_ID, 'unixReviewTime': TIMESTAMP}, 
            axis=1
        )
        [[USER_ID, ITEM_ID, TIMESTAMP]]
    )
    data = data.sort_values(by=TIMESTAMP)
    data = data.reset_index(drop=True)
    data[TIMESTAMP] = pd.to_datetime(data[TIMESTAMP], unit='s', utc=True)
    data['day'] = data[TIMESTAMP].round('d')

    data[RELEVANCE_COLUMN] = np.ones(len(data))
    data = encode_column(data, USER_ID)
    data = encode_column(data, ITEM_ID)

    print_recsys_df_stats(data, file_dir / 'data_stats.csv')
    data.to_csv(file_dir / 'prepared_data.csv')

def prepare_steam():
    data = (
        pd.read_csv(DATA_DIR_STEAM / 'steam.csv')
        .rename(
            {'username': USER_ID, 'product_id': ITEM_ID}, 
            axis=1
        )
        [[USER_ID, ITEM_ID, TIMESTAMP]]
    )
    data = data.sort_values(by=TIMESTAMP)
    data = data.reset_index(drop=True)
    data[TIMESTAMP] = pd.to_datetime(data[TIMESTAMP], unit='s', utc=True)
    data['day'] = data[TIMESTAMP].round('d')

    data[RELEVANCE_COLUMN] = np.ones(len(data))
    data = encode_column(data, USER_ID)
    data = encode_column(data, ITEM_ID)

    print_recsys_df_stats(data, DATA_DIR_STEAM / 'data_stats.csv')
    data.to_csv(DATA_DIR_STEAM / 'prepared_data.csv')

def prepare_data():
    prepare_ml20m()
    prepare_amazon(DATA_DIR_AMZ_B, 'amz_b.gz')
    prepare_amazon(DATA_DIR_AMZ_G, 'amz_g.gz')
    prepare_steam()

def save_data_dynamics():
    for dataset in [DIR_ML20M_NAME, DIR_AMZ_B_NAME, DIR_AMZ_G_NAME, DIR_STEAM_NAME]:
        conf = import_config(dataset)
        _, _, left_data = prepare_data_for_experiment(
            conf.prepared_data_path, 
            conf.init_ratio, 
            conf.hm_actions_min_stream,
        )

        target_day = np.sort(left_data[DATE_DAYS].unique())[:conf.how_many_iterations][-1]
        left_data = left_data[left_data[DATE_DAYS] < target_day]

        uniq_users = left_data.groupby(DATE_DAYS)[USER_ID].nunique().sort_index().values
        uniq_interactions = left_data.groupby(DATE_DAYS)[USER_ID].count().sort_index().values
        res = {
            'n_users_dynamics': uniq_users.tolist(),
            'n_interactions_dynamics': uniq_interactions.tolist(),
        }

        with open(RESULTS_DIR / dataset / 'metric_dynamics' / 'data_dynamics.json', 'w') as f:
            json.dump(res, f)

if __name__ == '__main__':
    prepare_data()
    save_data_dynamics()
