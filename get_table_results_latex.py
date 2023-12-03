import json

import numpy as np
import pandas as pd

from load_data import (
    RESULTS_DIR,
    DIR_ML20M_NAME,
    DIR_AMZ_B_NAME,
    DIR_AMZ_G_NAME,
    DIR_STEAM_NAME,
)

data_names = [DIR_ML20M_NAME, DIR_AMZ_B_NAME, DIR_AMZ_G_NAME, DIR_STEAM_NAME]
models = ['SVD', 'PSIRec', 'TDRec', 'TDRecReinit', 'TIRec', 'TIRecA']
metrics = ['hr', 'mrr', 'wji', 'calculation_time']

data_to_metrics = []
for data_name in data_names:
    model_to_metric = {}
    for model_name in models:
        model_to_metric[model_name] = {}
        try:
            with open(RESULTS_DIR / data_name / 'metric_dynamics' / (model_name + '.json')) as f:
                metrics_dict = json.load(f)
                for metric in metrics:
                    model_to_metric[model_name][metric] = np.mean(metrics_dict[metric])
        except:
            for metric in metrics:
                model_to_metric[model_name][metric] = '-'
    data_to_metrics.append(pd.DataFrame(model_to_metric))

df = pd.concat(data_to_metrics)
tuples = list(zip(np.repeat(data_names, 4), df.index))
index = pd.MultiIndex.from_tuples(tuples)
df.index = index
print(df.to_latex(float_format="%.3f"))

_data_to_metrics = [data_to_metrics[i].unstack().reorder_levels([1, 0]).sort_index() for i in range(len(data_to_metrics))]
_df = pd.concat(_data_to_metrics, axis=1)
_df.columns = data_names
print(
    _df
    .to_latex(
        float_format="%.3f", 
        column_format='||l|l|c|c|c|c||',
        caption="Results of evaluation. All metrics are averaged and computed for top-n recommendations with n = 5",
        label="table:ev_results",
        multirow=False
    )
)
