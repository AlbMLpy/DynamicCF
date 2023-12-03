import numpy as np
import matplotlib as mpl

from ..general_functions import sliding_window
from ..plotting import (
    PARAMS, 
    set_ax,
    save_fig, 
    get_fig_ax,
    add_plot_ax,
    plot_results, 
)
from .prepare_data import (
    prepare_data_for_experiment, 
)
from ..data_preparation import (
    USER_ID, ITEM_ID,
)

X_LABEL = 'Time step'
DPI = 300
EXT = '.pdf'

def plot_computational_time_dynamics(
    graphics_dir,
    dynamics_dict: dict[str, dict[str, list[float]]],
    extra_str: str = '',
    mode: str = 'regular',
) -> None:
    calc_time_list = []
    for _, dynamics_model in dynamics_dict.items():
        res = dynamics_model['calculation_time']
        if mode == 'cum':
            res = np.cumsum(res)
        calc_time_list.append(res)
    ylabel = r'$\Delta$'+'Computational Time (sec)' if mode == 'cum' else 'Computational Time (sec)'
    title = 'Computational Time Dynamics (Cumulative)' if mode == 'cum' else 'Computational Time Dynamics'
    save_path_ext = f'cum_calc_time{EXT}' if mode == 'cum' else f'calc_time{EXT}'
    plot_results(
        np.arange(len(calc_time_list[0])), 
        calc_time_list, 
        list(dynamics_dict.keys()), 
        X_LABEL, 
        ylabel, 
        title, 
        graphics_dir / (extra_str + save_path_ext),
        DPI,
    )

def plot_stability(
    graphics_dir,
    topk: int,
    dynamics_dict: dict[str, dict[str, list[float]]],
    extra_str: str = '',
    mode: str = 'regular',
) -> None: 
    metric_list = []
    for _, dynamics_model in dynamics_dict.items():
        res = dynamics_model['wji']
        if mode == 'cum':
            res = np.cumsum(res)
        metric_list.append(res)
    ylabel = r'$\Delta$' + f'{mode.upper()}@{topk}' if mode == 'cum' else f'{mode.upper()}@{topk}'
    title = 'Stability of Recommendations (Cumulative)' if mode == 'cum' else 'Stability of Recommendations'
    save_path_ext = f'cum_stability{EXT}' if mode == 'cum' else f'stability{EXT}'
    plot_results(
        np.arange(len(metric_list[0])), 
        metric_list, 
        list(dynamics_dict.keys()),
        X_LABEL, 
        ylabel, 
        title, 
        save_path=graphics_dir / (extra_str + save_path_ext),
        dpi=DPI,
    )

def plot_cum_metrics(
    graphics_dir,
    topk: int,
    dynamics_dict: dict[str, dict[str, list[float]]],
    extra_str: str = '',
    modes=('hr', 'mrr'),
) -> None:     
    for mode in modes:
        cum_list = []
        for _, dynamics_model in dynamics_dict.items():
            cum_list.append(np.cumsum(dynamics_model[mode]))
        plot_results(
            np.arange(len(cum_list[0])), 
            cum_list, 
            list(dynamics_dict.keys()),  
            X_LABEL, 
            r'$\Delta$'+f'{mode.upper()}@{topk}', 
            f'{mode.upper()} metric dynamics (Cumulative)', 
            save_path=graphics_dir / (extra_str + f'cum_{mode}{EXT}'),
            dpi=DPI,
        )

def plot_sliding_window_metrics(
    graphics_dir,
    topk: int,
    dynamics_dict: dict[str, dict[str, list[float]]],
    swl: int,
    extra_str: str = '',
    modes=('hr', 'mrr'),
) -> None:
    for mode in modes:
        cum_list = []
        for _, dynamics_model in dynamics_dict.items():
            cum_list.append(sliding_window(dynamics_model[mode], swl, 'mean'))
        plot_results(
            np.arange(len(cum_list[0])), 
            cum_list, 
            list(dynamics_dict.keys()),  
            X_LABEL, 
            f'{mode.upper()}@{topk}', 
            f'{mode.upper()} metric dynamics (SW - {swl})', 
            save_path=graphics_dir / (extra_str + f'sw_{swl}_{mode}{EXT}'),
            dpi=DPI,
        )

def plot_factors_dynamics(
    graphics_dir,
    dynamics_dict: dict[str, dict[str, list[float]]],
    extra_str: str = '',
) -> None: 
    modes = ['rnd_users', 'rnd_items']
    for mode in modes:
        metric_list = []
        for _, dynamics_model in dynamics_dict.items():
            metric_list.append(dynamics_model[mode])  
        plot_results( 
            np.arange(len(metric_list[0])), 
            metric_list, 
            list(dynamics_dict.keys()),
            X_LABEL, 
            r'$\dfrac{\|U_i - U_{i-1}\|}{\|U_i\|}$', 
            f'Factors Dynamics', 
            save_path=graphics_dir / (extra_str + f'{mode}{EXT}'),
            dpi=DPI,
        )

def plot_new_entities_dynamics(
    graphics_dir,
    how_many_iterations,
    prepared_data_path, 
    init_ratio, 
    hm_actions_min_stream,
    extra_str: str = '',
):
    initial_data, data_stream, _ = prepare_data_for_experiment(
        prepared_data_path, 
        init_ratio, 
        hm_actions_min_stream,
    )
    users_set = set(initial_data[USER_ID].unique())
    items_set = set(initial_data[ITEM_ID].unique())
    
    n_new_users = []
    n_new_items = []
    n_new_users_ni = []
    n_new_items_nu = []
    
    for _, chunk in data_stream:
        # Process both new users and new items:
        mask = (
            ~chunk[USER_ID].isin(users_set)
            & ~chunk[ITEM_ID].isin(items_set)
        )
        new_users, new_items = set(chunk[mask][USER_ID].unique()), set(chunk[mask][ITEM_ID].unique())
        n_new_users_ni.append(len(new_users))
        n_new_items_nu.append(len(new_items))
        users_set |= new_users
        items_set |= new_items
        # Process new users:
        chunk_users = set(chunk[USER_ID].unique())
        new_users = chunk_users - users_set
        n_new_users.append(len(new_users))
        users_set |= new_users
        # Process new items:
        chunk_items = set(chunk[ITEM_ID].unique())
        new_items = chunk_items - items_set
        n_new_items.append(len(new_items))
        items_set |= new_items
    plot_results(
        np.arange(1, how_many_iterations + 1), 
        [
            n_new_users[:how_many_iterations], 
            n_new_items[:how_many_iterations],
            n_new_users_ni[:how_many_iterations],
            n_new_items_nu[:how_many_iterations],
        ], 
        ['#users', '#items', '#new_users_ni', '#new_items_nu'], 
        'Day', 
        'Amount', 
        'New Users, New Items Dynamics',
        dpi=DPI,
        save_path=graphics_dir / (extra_str + f'dynamics_new_ui{EXT}'),
        markers=["v", "s", "*", "^"]
    )

def plot_data_dynamics(
    graphics_dir,
    dyn_dict, 
    extra_str: str = '',
):
    y = dyn_dict['n_users_dynamics']
    x = np.arange(len(y))
    with mpl.rc_context(PARAMS):
        fig, ax1 = get_fig_ax()

        color = 'tab:red'
        ax1.set_xlabel('Day')
        ax1.set_ylabel('#Unique Users', color=color)
        ax1.set_title('Data Dynamics')
        ax1.grid()
        ax1.plot(x, dyn_dict['n_users_dynamics'], color=color, marker="*")
        ax1.tick_params(axis='y', labelcolor=color)

        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

        color = 'tab:blue'
        ax2.set_ylabel('#Interactions', color=color)  # we already handled the x-label with ax1
        ax2.plot(x, dyn_dict['n_interactions_dynamics'], color=color, marker="^")
        ax2.tick_params(axis='y', labelcolor=color)
        #fig.tight_layout()  # otherwise the right y-label is slightly clipped
        save_fig(fig, graphics_dir / (extra_str + f'data_dynamics{EXT}'), dpi=DPI)
