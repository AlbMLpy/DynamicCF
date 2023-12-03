from operator import mul as mul_op
from functools import reduce
import random
from random import choice
from itertools import product

def _random_chooser():
    while True:
        values = yield
        yield choice(values)

def random_grid(params, n=60, grid_cache=None, skip_config=None, seed=None):
    """
    Randomized search on hyper parameters.
    The function produces hyper parameters grid and names.
    
    Parameters
    ----------
    params : dict
        Dictionary of parameters names and variable values,
        e.g. {"A": [1, 2, 3], "B": [3, 4, 5]}.
    n : int, optional
        The number of points in a grid. If n=0, 
        then the function gives you all the possible nodes.
        Default is 60.
    grid_cache : list, optional
        Cached grid configurations(points),
        e.g. [(6, 7), (8, 9)], for example above.
        Default is None.
    skip_config : function, optional
        Function that takes config and returns True/False
        to skip/not the config.
        Default is None.

    Returns
    -------
    grid, param_names : tuple
        Tuple-like object, with the following attributes.
    grid : set
        Set of configurations.
    param_names : tuple
        Tuple of all the parameters names.
    """
    if seed is not None:
        random.seed(seed)
    if not isinstance(n, int):
        raise TypeError('n must be an integer, not {}'.format(type(n)))
    if n < 0:
        raise ValueError('n should be >= 0')
    # fix names and order of parameters
    param_names, param_values = zip(*params.items())
    grid = set(grid_cache) if grid_cache is not None else set()
    max_n = reduce(mul_op, [len(vals) for vals in param_values])
    n = min(n if n > 0 else max_n, max_n)

    skipped = set()
    if skip_config is None:
        def never_skip(config): return False
        skip_config = never_skip

    param_chooser = _random_chooser()
    try:
        while len(grid) < (n-len(skipped)):
            level_choice = []
            for param_val in param_values:
                next(param_chooser)
                level_choice.append(param_chooser.send(param_val))
            level_choice = tuple(level_choice)
            if skip_config(level_choice):
                skipped.add(level_choice)
                continue
            grid.add(level_choice)
    except KeyboardInterrupt:
        print('Interrupted by user. Providing current results.')
    return grid, param_names


def full_grid(params):
    """
    Full grid search on hyper parameters.
    The function produces hyper parameters grid and names.
    
    Parameters
    ----------
    params : dict
        Dictionary of parameters names and variable values,
        e.g. {"A": [1, 2, 3], "B": [3, 4, 5]}.
    
    Returns
    -------
    grid, param_names : tuple
        Tuple-like object, with the following attributes.
    grid : set
        Set of configurations.
    param_names : tuple
        Tuple of all the parameters names.
    """
    param_names, param_values = zip(*params.items())
    grid = set(product(*param_values))
    return grid, param_names 
