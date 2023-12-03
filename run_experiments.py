import os
import argparse

DATASET_HELP = 'Choose dataset name from: "ml_20m", "steam", "amz_b", "amz_g"'

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='All experiments')
    parser.add_argument('dataset', type=str, help=DATASET_HELP)
    args = parser.parse_args()

    datasets = [args.dataset,]#['ml_20m', 'steam', 'amz_b', 'amz_g']
    models = ['SVD', 'PSIRec', 'TIRec', 'TIRecA', 'TDRec', 'TDRecReinit']
    h_models = ['svd', 'tdrec']
    as_models = ['PSIRec', 'TIRecA']

    for data in datasets:
        for model in h_models:
            os.system(f"python find_hyperparams.py {data} {model} False")
        for model in models:
            os.system(f"python dynamical_experiment.py {data} {model} False")
        for model in as_models:
            os.system(f"python dynamical_ablation_study.py {data} {model} False")
        os.system(f"python prepare_graphs.py {data}")
