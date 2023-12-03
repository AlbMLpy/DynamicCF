from typing import Any

from ..models.svd import SVD
from ..models.psi import PSI
from ..models.tirec import TIRec
from ..models.tdrec import TDRec
from ..models.tdrec_reinit import TDRecRe
from ..models.random_rec import RandomRec
from ..models.tirec_accelerated import TIRecA
from .hyperparameters_search import (
    get_optimal_svd_rank,
    get_optimal_tdrec_params,
)

def _prepare_svd_model(config: dict[str, Any]) -> dict[str, Any]:
    rank = get_optimal_svd_rank(config['dataset'], f'hr_{config["topk"]}')
    model = SVD(rank=rank, **config['fixed_config_svd'])
    return {
        'rec_model': model,
        'update_fuction': model.train,
        'upd_data_mode': 'concatenated',
        'reinit_model_bool': False, 
        'three_dimensional_bool': False,
        'max_len_history': None, 
    }

def _prepare_psirec_model(config: dict[str, Any]) -> dict[str, Any]:
    rank = get_optimal_svd_rank(config['dataset'], f'hr_{config["topk"]}')
    model = PSI(rank=rank, **config['fixed_config_svd'])
    return {
        'rec_model': model,
        'update_fuction': model.update,
        'upd_data_mode': 'chunk',
        'reinit_model_bool': False, 
        'three_dimensional_bool': False,
        'max_len_history': None, 
    }

def _prepare_tdrec_model(config: dict[str, Any]) -> dict[str, Any]:
    rank, att_f, _ = get_optimal_tdrec_params(config['dataset'], f'hr_{config["topk"]}')
    model = TDRec(rank=rank, att_f=att_f, **config['fixed_config_tdrec'])
    return {
        'rec_model': model,
        'update_fuction': model.train,
        'upd_data_mode': 'concatenated',
        'reinit_model_bool': False, 
        'three_dimensional_bool': True,
        'max_len_history': config['max_len_user_history'], 
    }

def _prepare_tdrec_reinit_model(config: dict[str, Any]) -> dict[str, Any]:
    rank, att_f, _ = get_optimal_tdrec_params(config['dataset'], f'hr_{config["topk"]}')
    model = TDRecRe(rank=rank, att_f=att_f, **config['fixed_config_tdrec'])
    return {
        'rec_model': model,
        'update_fuction': model.train,
        'upd_data_mode': 'concatenated',
        'reinit_model_bool': True, 
        'three_dimensional_bool': True,
        'max_len_history': config['max_len_user_history'], 
    }

def _prepare_tirec_model(config: dict[str, Any]) -> dict[str, Any]:
    rank, att_f, _ = get_optimal_tdrec_params(config['dataset'], f'hr_{config["topk"]}')
    model = TIRec(rank=rank, att_f=att_f, **config['fixed_config_tdrec'])
    return {
        'rec_model': model,
        'update_fuction': model.update,
        'upd_data_mode': 'chunk',
        'reinit_model_bool': False, 
        'three_dimensional_bool': True,
        'max_len_history': config['max_len_user_history'], 
    }

def _prepare_tireca_model(config: dict[str, Any]) -> dict[str, Any]:
    rank, att_f, _ = get_optimal_tdrec_params(config['dataset'], f'hr_{config["topk"]}')
    model = TIRecA(rank=rank, att_f=att_f, **config['fixed_config_tdrec'])
    return {
        'rec_model': model,
        'update_fuction': model.update,
        'upd_data_mode': 'chunk',
        'reinit_model_bool': False, 
        'three_dimensional_bool': True,
        'max_len_history': config['max_len_user_history'], 
    }

def _prepare_random_model(config: dict[str, Any]) -> dict[str, Any]:
    model = RandomRec(seed=13)
    return {
        'rec_model': model,
        'update_fuction': model.train,
        'upd_data_mode': 'concatenated',
        'reinit_model_bool': False, 
        'three_dimensional_bool': True,
        'max_len_history': config['max_len_user_history'], 
    }

def prepare_model(model_name: str, config: dict[str, Any]) -> dict[str, Any]:
    if model_name == 'SVD':
        dyn_exp_dict = _prepare_svd_model(config)
    elif model_name == 'PSIRec':
        dyn_exp_dict = _prepare_psirec_model(config)
    elif model_name == 'TDRec':
        dyn_exp_dict = _prepare_tdrec_model(config)
    elif model_name == 'TDRecReinit':
        dyn_exp_dict = _prepare_tdrec_reinit_model(config)
    elif model_name == 'TIRec':
        dyn_exp_dict = _prepare_tirec_model(config)
    elif model_name == 'TIRecA':
        dyn_exp_dict = _prepare_tireca_model(config)
    elif model_name == 'Random':
        dyn_exp_dict = _prepare_random_model(config)
    else:
        raise RuntimeError(f'Bad model name - {model_name}')
    return dyn_exp_dict
