# Dynamic Collaborative Filtering

Official repository for ["Dynamic Collaborative Filtering for Matrix- and Tensor-based Recommender Systems"](https://arxiv.org/abs/2312.10064) paper.

## Project Description:
In production applications of recommender systems, a continuous data flow is employed to update models in real-time. Many recommender models often require complete retraining to adapt to new data. In this work, we introduce a novel collaborative filtering model for sequential problems known as Tucker Integrator Recommender - TIRecA. TIRecA efficiently updates its parameters using only the new data segment, allowing incremental addition of new users and items to the recommender system. To demonstrate the effectiveness of the proposed model, we conducted experiments on four publicly available datasets: MovieLens 20M, Amazon Beauty, Amazon Toys and Games, and Steam. Our comparison with general matrix and tensor-based baselines in terms of prediction quality and computational time reveals that TIRecA achieves comparable quality to the baseline methods, while being 10-20 times faster in training time.

## Datasets
In this work, we use the following publicly available datasets in recommender systems field: Movielens-20M (ML-20M), 
Amazon Beauty (AMZ-B), Amazon Toys and Games
(AMZ-G), and Steam (Self-Attentive Sequential Recommendation. Wang-Cheng Kang and Julian McAuley. 2018). We follow the same data initial preparation procedure as in (Tensor-based Sequential Learning via
Hankel Matrix Representation for Next
Item Recommendations. Evgeny Frolov and Ivan Oseledets. 2022). Namely, we leave no less than five interactions per each user and each item for all datasets except Movielens-20M. The explicit values of ratings are transformed into an implicit binary signal indicating the presence of a rating. As in the Steam data some users assigned more than one review to the same items, such duplicates were removed.
In Movielens-20M data we use 20\% of the last interactions.

Datasets statistics after initial preprocessing:

| Dataset  | #users | #items | #interactions | matrix density% |
| ------------- | ------------- | ------------- | ------------- | ------------- |
| ML-20M  | 31670 | 26349 | 4000052 | 0.48 |
| AMZ-B   | 22363 | 12101 | 198502 | 0.07 |
| AMZ-G  | 19412 | 11924 | 167597 | 0.07 |
| Steam  | 281206 | 11961 | 3483694 | 0.10 |

## The reported results and corresponding configurations
### Hyper-parameters
|           |          |     AMZ-B |     AMZ-G |    ML-20M |    Steam |
|:----------|:---------|----------:|----------:|----------:|---------:|
| PSIRec    |      rank|    190    |    160    |    100    |    10    |
| PureSVD   |          |    190    |    160    |    100    |    10    |
|           |          |           |           |           |          |
| TIRec     | user rank|     64    |    100    |    100    |   128    |
| TIRecA    | item rank|    256    |    128    |    128    |    64    |
| TDRec     | pos rank |      5    |     10    |      5    |     2    |
|TDRecReinit| scaling-f|      0    |      0    |      2    |     2    |

### Final scores
|        | metric@5    | PSIRec       | PureSVD      | TIRec        | TIRecA       | TDRec        | TDRecReinit  |
|:-------|:------------|:-------------|:-------------|:-------------|:-------------|:-------------|:-------------|
| ML-20M | Time        |     0.376    |     7.129    |     27.681   |     1.729    |     22.064   |     10.223   | 
|        | HR          |     0.019    |     0.020    |     0.025    |     0.025    |     0.026    |     0.027    | 
|        | MRR         |     0.008    |     0.009    |     0.012    |     0.013    |     0.013    |     0.013    | 
|        | WJI         |     0.668    |     0.649    |     0.236    |     0.429    |     0.092    |     0.241    | 
|  AMZ-B | Time        |     0.722    |     4.420    |     39.237   |     1.253    |     10.028   |     5.288    | 
|        | HR          |     0.013    |     0.015    |     0.012    |     0.013    |     0.020    |     0.020    | 
|        | MRR         |     0.007    |     0.008    |     0.006    |     0.007    |     0.010    |     0.011    | 
|        | WJI         |     0.673    |     0.641    |     0.214    |     0.224    |     0.030    |     0.231    |
|  AMZ-G | Time        |     0.591    |     1.513    |     31.858   |     1.284    |     12.550   |     5.303    | 
|        | HR          |     0.016    |     0.017    |     0.011    |     0.010    |     0.012    |     0.013    | 
|        | MRR         |     0.008    |     0.009    |     0.006    |     0.005    |     0.006    |     0.006    | 
|        | WJI         |     0.663    |     0.624    |     0.229    |     0.210    |     0.029    |     0.167    |
|  Steam | Time        |     0.657    |     5.413    |       -      |    17.410    |     130.491  |     52.141   | 
|        | HR          |     0.022    |     0.022    |       -      |     0.009    |     0.015    |     0.015    | 
|        | MRR         |     0.011    |     0.011    |       -      |     0.004    |     0.007    |     0.010    | 
|        | WJI         |     0.683    |     0.672    |       -      |     0.539    |     0.111    |     0.488    |


## Environment
We use `conda` package manager to install required python packages. In order to improve speed and reliability of package version resolution it is advised to use `mamba-forge` ([installation](https://github.com/conda-forge/miniforge#mambaforge)) that works over `conda`. Once `mamba is installed`, run the following command (while in the root of the repository):
```
mamba env create -f environment/environment.yaml
```
This will create new environment named `dcf` with all required packages already installed. You can install additional packages by running:
```
mamba install <package name>
```
To activate the virtual environment:
```
mamba activate dcf
```

In order to read and run `Jupyter Notebooks` you may follow either of two options:
1. [*recommended*] using notebook-compatibility features of modern IDEs, e.g. via `python` and `jupyter` extensions of [VS Code](https://code.visualstudio.com/).
2. install jupyter notebook packages:
  either with `mamba install jupyterlab` or with `mamba install jupyter notebook`

*Note*: If you prefer to use `conda`, just replace `mamba` commands with `conda`, e.g. instead of `mamba install` use `conda install`.

## Docker
To reproduce the experiment in Docker Contaner use the following commands (sequentially):
To build a Docker Image run:
```
docker build -t aw/dcf .
```
To run a working container run:
```
docker run -it --name dcf -v $(pwd):/dcf aw/dcf /bin/bash
```
After that you can reproduce the experimental results using the commands below.

## Reproduction of the experimental results

Note: all the configurations/settings for experiments with explanation of parameters can be found in './configs/{DATA}/config.py' where DATA=amz_b/amz_g/ml_20m/steam.

1. Run:
   ```shell
   python load_data.py
   ```
   to load all the datasets locally and configure internal directories.

2. Run: 
   ```shell
   python prepare_data.py
   ```
   to prepare all the data for further experiments.

3. Run (Advised):
   ```shell
   python run_experiments.py DATA
   ```
   where DATA=amz_b/amz_g/ml_20m/steam, to run all the essential experiments for a particular dataset (It can take several hours).

Note: You can find all the plots with experimental results in './results/{DATA}' where DATA=amz_b/amz_g/ml_20m/steam

Moreover, if you want to run all the experiments youself you can use the following steps (However, you have to run experiment for all the datasets and models in order to run prepare_graphs.py):

3. Run:
   ```shell
   python find_hyperparams.py DATA MODEL DISABLE_TQDM
   ```
   where DATA=amz_b/amz_g/ml_20m/steam, MODEL=svd/tdrec, DISABLE_TQDM=False/True, to find the optimal hyperparameters for a model on a dataset.

4. Run:
   ```shell
   python dynamical_experiment.py DATA MODEL DISABLE_TQDM
   ```
   where DATA=amz_b/amz_g/ml_20m/steam, MODEL=SVD/PSIRec/TDRec/TDRecReinit/TIRec/TIRecA, DISABLE_TQDM=False/True, to calculate the dynamical results of the experiments.

5. Run:
   ```shell
   python dynamical_ablation_study.py DATA MODEL DISABLE_TQDM
   ```
   where DATA=amz_b/amz_g/ml_20m/steam, MODEL=PSIRec/TIRecA, DISABLE_TQDM=False/True, to calculate experiments with different initializations for new users/items factors.

6. Run:
   ```shell
   python prepare_graphs.py DATA
   ```
   where DATA=amz_b/amz_g/ml_20m/steam, to prepare the graphs of the experiments.

# Acknowledgements
This work is supported by the [RSCF Grant 22-21-00911](https://rscf.ru/en/project/22-21-00911/).
