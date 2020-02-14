"""Routines for constructing an array of qprop objects"""

from os.path import isdir, join
from os import listdir

from ..core import np
from ..core import Qprop20
from .file import ParameterFileSet

def construct_qprop_obj_arr(parent_dir, sort_key=None, reverse=False):
    """Construct an array of `Qprop20` objects
    with optional sorting

    Parameters
    ----------
    parent_dir : str
        path to a directory under which calculation directories reside
    sort_key : one of None, callable, 2-len tuple
        if None: no sorting is performed
        if callable: it is used as a sorting key
        if 2-len tuple: should of form ('file_name', 'param_name')

    Returns
    -------
    q_list : (N,) numpy.ndarray
        an array of qprop objects of length N
    """
    assert isdir(parent_dir)
    _calc_dir_paths = [join(parent_dir, _name) for _name in listdir(parent_dir)]
    _q_pool = [Qprop20(path) for path in _calc_dir_paths]
    
    if sort_key is None: return _q_pool
    elif sort_key is callable:  
        # the sorting key is given by a function
        _q_pool.sort(key=sort_key, reverse=reverse)
    elif hasattr(sort_key, '__getitem__') and hasattr(sort_key, '__len__'):
        # the sorting key is given by an iterable of strings
        assert len(sort_key) == 2
        _param_file_name, _param_name = sort_key
        def _get_param(q):
            _param_file_set = ParameterFileSet.from_calc_dir(q.home)
            return _param_file_set[_param_file_name][_param_name]
        _q_pool.sort(key=_get_param, reverse=reverse)
    else: raise Exception("Unknown type of `sort_key`: {}".form(type(sort_key)))
    
    return _q_pool
