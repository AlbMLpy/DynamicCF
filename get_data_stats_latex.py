import pandas as pd
from load_data import (
    DATA_DIR,
    DIR_ML20M_NAME,
    DIR_AMZ_B_NAME,
    DIR_AMZ_G_NAME,
    DIR_STEAM_NAME,
)

map_data_names = {
    DIR_ML20M_NAME: 'ML-20M',
    DIR_AMZ_B_NAME: 'AMZ-B',
    DIR_AMZ_G_NAME: 'AMZ-G',
    DIR_STEAM_NAME: 'Steam'
}

data_list = []
for data_name in [DIR_ML20M_NAME, DIR_AMZ_B_NAME, DIR_AMZ_G_NAME, DIR_STEAM_NAME]:
    data = (
        pd.read_csv(DATA_DIR / data_name / 'data_stats.csv')
        .rename({'Unnamed: 0': 'key', '0': 'value'}, axis=1)
    )
    data = data.pivot_table('value', columns=['key'])
    data['dataset'] = map_data_names[data_name]
    data = data[['dataset', 'n_users', 'n_items', 'n_interactions', 'n_items_median', 'density%']].reset_index(drop=True)
    data.columns.name = None
    data_list.append(data)

dd = (
    pd.concat(data_list)
    .reset_index(drop=True)
    .astype(
        {
            'n_users': int, 
            'n_items': int,
            'n_interactions': int, 
            'n_items_median': int, 
        }
    )
)

print(dd.to_latex(index=False))
