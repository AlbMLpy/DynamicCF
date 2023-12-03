import json
import argparse
from pathlib import Path

from load_data import (
    RESULTS_DIR,
)
from source.experiments.hyperparameters_search import (
    import_config,
)
from source.experiments.prepare_graphs import (
    plot_stability,
    plot_cum_metrics,
    plot_data_dynamics,
    plot_new_entities_dynamics,
    plot_sliding_window_metrics,
    plot_computational_time_dynamics,
)
DATASET_HELP = 'Choose dataset name from: "ml_20m", "steam", "amz_b", "amz_g"'

def dynamical_experiment(graphics_dir, dynamics_dict, extra_str, topk):
    plot_computational_time_dynamics(graphics_dir, dynamics_dict, extra_str)
    plot_cum_metrics(graphics_dir, topk, dynamics_dict, extra_str, ('hr',))
    plot_stability(graphics_dir, topk, dynamics_dict, extra_str)
    for swl in conf.swl_list:
        plot_sliding_window_metrics(
            graphics_dir,
            topk=topk,
            dynamics_dict=dynamics_dict,
            swl=swl,
            extra_str=extra_str,
            modes=('hr',)
        )

def ablation_study(ablation_study_dir, dynamics_dict, extra_str, topk):
    plot_stability(ablation_study_dir, topk, dynamics_dict, extra_str, 'cum')
    plot_cum_metrics(ablation_study_dir, topk, dynamics_dict, extra_str, ('hr',))
    plot_sliding_window_metrics(ablation_study_dir, topk, dynamics_dict, 20, extra_str, modes=('hr',))

def map_tensor_name(model_name):
    if model_name == 'TIRec':
        model_name += ''
    elif model_name == 'TIRecA':
        model_name += ''
    else:
        left, int_mode = model_name.split('_int_')
        model_name, std_mode = left.split('_std_')
        if std_mode == '0':
            model_name += '(0;'
        else:
            model_name += f'({std_mode};'
        if int_mode == 'True':
            model_name += '0)'
        else:
            model_name += f'B)'
    return model_name

def map_matrix_name(model_name):
    if model_name == 'PSIRec':
        model_name += ''
    else:
        left, int_mode = model_name.split('_int_')
        model_name, std_mode = left.split('_std_')
        if std_mode == '0':
            model_name += '(0;'
        else:
            model_name += f'({std_mode};'
        if int_mode == 'True':
            model_name += '0)'
        else:
            model_name += f'B)'
    return model_name

if __name__ == '__main__':
    # Parse script arguments:
    parser = argparse.ArgumentParser(description='Prepare plots')
    parser.add_argument('dataset', type=str, help=DATASET_HELP)
    args = parser.parse_args()
    # Get a particular config for a dataset:
    conf = import_config(args.dataset)

    ### Dynamical Experiment: ###
    models_list = conf.plot_models_list
    dynamics_dict = {model: None for model in models_list}
    for name in dynamics_dict:
        with open(conf.metric_dynamics_dir / (name + ".json"), 'r') as f:
            dynamics_dict[name] = json.load(f)

    extra_str = args.dataset + '_'
    dynamical_experiment(conf.graphics_dir, dynamics_dict, extra_str, conf.topk)
    plot_new_entities_dynamics(
        conf.graphics_dir,
        conf.how_many_iterations,
        conf.prepared_data_path,
        conf.init_ratio,
        conf.hm_actions_min_stream,
        extra_str=args.dataset + '_'
    )
    with open(RESULTS_DIR / args.dataset / 'metric_dynamics' / 'data_dynamics.json', 'r') as f:
        dyn_dict = json.load(f)
        plot_data_dynamics(
            conf.graphics_dir,
            dyn_dict,
            extra_str=args.dataset + '_'
        )
    ### Dynamical Experiment: ###

    ### Ablation study: ###
    options = [('TIRec', map_tensor_name), ('PSIRec', map_matrix_name)]
    for target_model, mapper in options:
        res_path_list = [
            str(p) for p in Path(conf.metric_dynamics_dir).iterdir() 
            if (p.is_file()) and (target_model in str(p))
        ]
        dynamics_dict = {}
        for res_path in res_path_list:
            with open(res_path, 'r') as f:
                model_name = res_path.split('/')[-1].split('.json')[0]
                model_name = mapper(model_name)
                dynamics_dict[model_name] = json.load(f)
        extra_str = target_model + '_' + args.dataset + '_'
        ablation_study(
            conf.ablation_study_dir, 
            dynamics_dict, 
            extra_str,
            conf.topk
        )
    ### Ablation study: ###
