import os
import json
import gzip
import zipfile
import urllib.request
from pathlib import Path
from ast import literal_eval

import pandas as pd

DATA_DIR = Path('data')
RESULTS_DIR = Path('results')
CONFIGS_DIR = Path('configs')
DIR_ML20M_NAME = "ml_20m"
DIR_AMZ_B_NAME = "amz_b"
DIR_AMZ_G_NAME = "amz_g"
DIR_STEAM_NAME = "steam"
DATA_DIR_ML20M = DATA_DIR / DIR_ML20M_NAME
DATA_DIR_AMZ_B = DATA_DIR / DIR_AMZ_B_NAME
DATA_DIR_AMZ_G = DATA_DIR / DIR_AMZ_G_NAME
DATA_DIR_STEAM = DATA_DIR / DIR_STEAM_NAME

### Movielens 20M data loader: ###
def load_ml20m():
    ml_20m_url = 'https://files.grouplens.org/datasets/movielens/ml-20m.zip'
    # Create data dir if not exists:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    # Load the data:
    print('Download of ML20M data started')
    temp_file = DATA_DIR / 'temp.zip'
    urllib.request.urlretrieve(ml_20m_url, filename=temp_file)
    print('Download of ML20M data finished')
    # Unzip file:
    with zipfile.ZipFile(temp_file, 'r') as zip_ref:
        zip_ref.extractall(DATA_DIR)
    # Remove temp file:
    temp_file.unlink(missing_ok=True)
    # Rename dir:
    os.rename(str(DATA_DIR / 'ml-20m'), str(DATA_DIR_ML20M))

### Amazon data parser and loader: ###
def parse_lines_amz(path, fields):
    with gzip.open(path, 'rt') as gz:
        for line in gz:
            yield json.loads(line, object_hook=lambda dct: tuple(dct[key] for key in fields))

def load_amazon(dataset_name: str):
    """
    dataset_name : str
        Use 'amz-b' to load Beauty data. Use 'amz-g' to load Toys_and_Games data.
    """
    dsname = {'amz_b': 'Beauty', 'amz_g': 'Toys_and_Games'}
    name = dsname[dataset_name]
    url = f'http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_{name}_5.json.gz'
    # Create data dir if not exists:
    (DATA_DIR / dataset_name).mkdir(parents=True, exist_ok=True)
    # Load the data:
    print(f'Download of {dataset_name} data started')
    temp_file = DATA_DIR / ('temp_' + dataset_name)
    urllib.request.urlretrieve(url, filename=temp_file) 
    print(f'Download of {dataset_name} data finished')
    # Parse the data:
    fields = ['reviewerID', 'asin', 'unixReviewTime']
    data = pd.DataFrame.from_records(parse_lines_amz(temp_file, fields), columns=fields)
    dest = DATA_DIR / dataset_name / (dataset_name + '.gz')
    data.to_csv(dest, index=False)
    # Remove temp file:
    temp_file.unlink(missing_ok=True)

### Steam data parser and loader: ###
def parse_lines_steam(path, fields):
    with gzip.open(path, 'rt') as gz:
        for line in gz:
            dct = literal_eval(line.strip())
            yield {key: dct[key] for key in fields}

def pcore_filter(data, pcore, userid, itemid):
    while pcore: # do only if pcore is specified
        item_check = True
        valid_items = data[itemid].value_counts() >= pcore
        if not valid_items.all():
            data = data.query(
                f'{itemid} in @valid_items.index[@valid_items]'
            )
            item_check = False
            
        user_check = True
        valid_users = data[userid].value_counts() >= pcore
        if not valid_users.all():
            data = data.query(
                f'{userid} in @valid_users.index[@valid_users]'
            )
            user_check = False
        
        if user_check and item_check:
            break
    return data.copy()

def load_steam():
    url = f'http://cseweb.ucsd.edu/~wckang/steam_reviews.json.gz'
    # Create data dir if not exists:
    steam_data_path = DATA_DIR_STEAM / 'steam.csv'
    DATA_DIR_STEAM.mkdir(parents=True, exist_ok=True)
    # Load the data:
    print(f'Download of Steam data started')
    temp_file = DATA_DIR / 'temp_steam'
    urllib.request.urlretrieve(url, filename=temp_file) 
    print(f'Download of Steam data finished')
    fields = ['username', 'product_id', 'date']
    raw_data = pd.DataFrame.from_records(parse_lines_steam(temp_file, fields), columns=fields)
    data_dedup = raw_data.drop_duplicates(subset=['username', 'product_id'], keep='last')
    data_clean = pcore_filter(data_dedup, 5, 'username', 'product_id')

    data_clean.loc[:, 'timestamp'] = (
        pd.to_datetime(data_clean['date']) - pd.Timestamp("1970-01-01")
    ) // pd.Timedelta('1s')
    (
        data_clean
        .loc[:, ['username', 'product_id', 'timestamp']]
        .to_csv(steam_data_path, index=False)
    )
    # Remove temp file:
    temp_file.unlink(missing_ok=True)

def prepare_extra_dirs():
    data_dirs = [DIR_ML20M_NAME, DIR_AMZ_B_NAME, DIR_AMZ_G_NAME, DIR_STEAM_NAME]
    target_dirs = ['graphics', 'hyperparams', 'metric_dynamics', 'ablation_study']
    for data_dir in data_dirs:
        for target_dir in target_dirs:
            (RESULTS_DIR / data_dir / target_dir).mkdir(parents=True, exist_ok=True)

    for data_dir in data_dirs:
        (CONFIGS_DIR / data_dir).mkdir(parents=True, exist_ok=True)
    
def load_data():
    load_ml20m()
    load_amazon('amz_b')
    load_amazon('amz_g')
    load_steam()
    prepare_extra_dirs()

if __name__ == '__main__':
    load_data()
