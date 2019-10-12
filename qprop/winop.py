"""package for analyzing window operator method"""

import numpy as np

from os.path import join
from qprop.core import Qprop20


class Winop(object):
    
    _spectrum_file_name = "spectrum_polar.dat"
    _qprop_class = Qprop20
    
    def __init__(self, q):
        """
        # Argument
        
        """
        self.q = None
        if isinstance(q, self._qprop_class): self.q = q
        else: raise TypeError(
            "'q' should be of type {}".format(self._qprop_class))
        
        try: self.has_winop_param_file(q)
        except: raise Exception("Cannot find WINOP parameter file")
            
        self._polar_data_loading_is_done = False
        
    
    @classmethod
    def has_winop_param_file(cls, q):
        _it_has = True
        assert isinstance(q, cls._qprop_class)
        _attr = 'paramFileList'
        if hasattr(q, _attr):
            _it_has &= 'winop.param' in getattr(q, _attr)
        else:
            _err = "No such attribute as '{:s}' in {}"
            raise AttributeError(_err.format(_attr, Qprop20))
        return _it_has
        
            
    @classmethod
    def get_spectrum_file_path(cls, q):
        assert isinstance(q, cls._qprop_class) and hasattr(q, 'home')
        return join(q.home, cls._spectrum_file_name)
    
    
    @staticmethod
    def read_polar_spectrum_file(data_file_path):
        """Compatible Qprop version: 2.0"""
        _raw = np.loadtxt(data_file_path)
        _ind = np.lexsort(_raw[:,-2:0:-1].transpose())
        _raw_sorted = _raw[_ind]
        return _raw_sorted
    
    
    def construct_polar_spectrum_arr(self):
        _polar_spec_arr = None
        try:
            _data_file_path = self.get_spectrum_file_path(self.q)
            _polar_spec_arr = self.read_polar_spectrum_file(_data_file_path)
        except: raise Exception("Failed to construct polar spectrum array")
        
        self.polar = _polar_spec_arr
        
        _k_prob_shape = self.get_polar_spectrum_ndarr_shape()
        self.k_prob = self.polar[:,-1].reshape(_k_prob_shape)
        self.E_arr = np.unique(self.polar[:,0])
        self.k_arr = np.unique(self.polar[:,1])
        self.k_theta_arr = np.unique(self.polar[:,2])
        
        self._polar_data_loading_is_done = True
    
    
    def reset_loading(self):
        del self.polar
        self._polar_data_loading_is_done = False
        
        
    def get_polar_spectrum_ndarr_shape(self):
        _shape = (
            self.q.winopParam['num-energy'], 
            self.q.winopParam['num-theta']
        )
        return _shape
    
    
    def get_k_theta_grid_arr(self):
        if not self._polar_data_loading_is_done:
            self.construct_polar_spectrum_arr()
        
        _k_theta_grid_arr = np.empty((self.k_theta_arr.size+1,), dtype=float)
        _k_theta_grid_arr[1:-1] = 0.5 * (self.k_theta_arr[:-1] + self.k_theta_arr[1:])
        _k_theta_grid_arr[0] = 0.0
        _k_theta_grid_arr[-1] = np.pi
        return _k_theta_grid_arr

    
    def get_E_grid_arr(self):
        if not self._polar_data_loading_is_done:
            self.construct_polar_spectrum_arr()
        _E_grid_arr = np.empty((self.E_arr.size + 1,), dtype=float)
        _E_grid_arr[1:-1] = 0.5 * (self.E_arr[1:] + self.E_arr[:-1])
        _E_grid_arr[0] = 0.0
        _E_grid_arr[-1] = self.E_arr[-1]
        return _E_grid_arr

    
    def get_k_grid_arr(self):
        if not self._polar_data_loading_is_done:
            self.construct_polar_spectrum_arr()
        _k_grid_arr = np.empty((self.k_arr.size + 1,), dtype=float)
        _k_grid_arr[1:-1] = 0.5 * (self.k_arr[1:] + self.k_arr[:-1])
        _k_grid_arr[0] = 0.0
        _k_grid_arr[-1] = self.k_arr[-1] + 0.5 * (self.k_arr[-1] - self.k_arr[-2])
        return _k_grid_arr
    
    
    def get_polar_k_XY_grid(self):
        _k_grid_arr = self.get_k_grid_arr()
        _k_theta_grid_arr = self.get_k_theta_grid_arr()
        _THETA, _K = np.meshgrid(_k_theta_grid_arr, _k_grid_arr)
        _Xk, _Yk = _K*np.cos(_THETA), _K*np.sin(_THETA)
        return _Xk, _Yk
    
    
    def get_polar_E_XY_grid(self):
        _E_grid_arr = self.get_E_grid_arr()
        _k_theta_grid_arr = self.get_k_theta_grid_arr()
        _THETA, _E = np.meshgrid(_k_theta_grid_arr, _E_grid_arr)
        _E_X, _E_Y = _E*np.cos(_THETA), _E*np.sin(_THETA)
        return _E_X, _E_Y

