"""package for analyzing window operator method"""

import numpy as np

from os.path import join
from qprop.core import Qprop20

from .exception import UnexpectedQpropDim


class Winop(object):
    
    _spectrum_file_name = "spectrum_polar.dat"
    _qprop_class = Qprop20

    _spec_polar_column_name_dict = {
            34 : ["energy", "momentum", "theta", "amplitude"],
            44 : ["energy", "momentum", "theta", "phi", "amplitude"]
    }
    
    def __init__(self, q):
        """
        # Argument
        
        """
        self.q = None
        if isinstance(q, self._qprop_class):
            if q.dimension in Qprop20.dimension_list:
                self.q = q
            else: raise UnexpectedQpropDim
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
        """Construct the polar spectrum array 
        and corresponding coordinate arrays"""

        ## Read polar spectrum data from file
        _polar_spec_arr = None
        try:
            _data_file_path = self.get_spectrum_file_path(self.q)
            _polar_spec_arr = self.read_polar_spectrum_file(_data_file_path)
        except: raise Exception("Failed to construct polar spectrum array")
        assert _polar_spec_arr is not None
        self.polar = _polar_spec_arr

        ## Check the number of columns in the polar spectrum files
        _num_of_columns_in_polar_spectrum_file = self.polar.shape[1]
        _expected_col_num = len(self._spec_polar_column_name_dict[self.q.dimension])
        assert _num_of_columns_in_polar_spectrum_file == _expected_col_num
        
        ## Construct coordinate arrays
        self.phi_arr = None
        if self.q.dimension == 44:
            self.phi_arr = np.unique(self.polar[:,3])
        _k_prob_shape = self.get_polar_spectrum_ndarr_shape()
        self.k_prob = self.polar[:,-1].reshape(_k_prob_shape)
        self.E_arr = np.unique(self.polar[:,0])
        self.k_arr = np.unique(self.polar[:,1])
        self.k_theta_arr = np.unique(self.polar[:,2])
        
        ## Set flag
        self._polar_data_loading_is_done = True
    
    
    def reset_loading(self):
        del self.polar
        self._polar_data_loading_is_done = False
        
        
    def get_polar_spectrum_ndarr_shape(self):
        _shape = None
        if self.q.dimension == 34:
            _shape = (
                self.q.winopParam['num-energy'], 
                self.q.winopParam['num-theta']
            )
        elif self.q.dimension == 44:
            _shape = (
                self.q.winopParam['num-energy'],
                self.q.winopParam['num-theta'],
                self.q.winopParam['num-phi']
            )
        else: raise Exception(
                "Unexpected qprop grid dimension: {}".format(self.q.dimension))
        assert _shape is not None

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
        _k_grid_arr[-1] = self.k_arr[-1] + 0.5*(self.k_arr[-1]-self.k_arr[-2])
        return _k_grid_arr


    def get_phi_grid_arr(self):
        assert isinstance(self.phi_arr, np.ndarray)
        _phi_grid_arr = np.empty((self.phi_arr.size + 1,), dtype=float)
        _phi_grid_arr[:-1]=self.phi_arr-0.5*(self.phi_arr[1]-self.phi_arr[0])
        _phi_grid_arr[-1]=self.phi_arr[-1]+0.5*(self.phi_arr[-1]-self.phi_arr[-2])
        return _phi_grid_arr
    
    
    def get_polar_k_XY_grid(self):
        assert self.q.dimension == 34  # for linearly polarized pulse
        _k_grid_arr = self.get_k_grid_arr()
        _k_theta_grid_arr = self.get_k_theta_grid_arr()
        _THETA, _K = np.meshgrid(_k_theta_grid_arr, _k_grid_arr)
        _Xk, _Yk = _K*np.cos(_THETA), _K*np.sin(_THETA)
        return _Xk, _Yk


    def get_k_XY_grid_at_pol_plane(self):
        assert self.q.dimension == 44
        _k_grid_arr = self.get_k_grid_arr()
        _phi_grid_arr = self.get_phi_grid_arr()
        _PHI, _K = np.meshgrid(_phi_grid_arr, _k_grid_arr)
        _Xk, _Yk = _K*np.cos(_PHI), _K*np.sin(_PHI)
        return _Xk, _Yk


    def get_k_prob_grid_at_pol_plane(self):
        if not self._polar_data_loading_is_done:
            self.construct_polar_spectrum_arr()
        _num_of_theta_grid = self.k_theta_arr.size
        _has_odd_num_of_theta_grid =  _num_of_theta_grid % 2 == 1
        if not _has_odd_num_of_theta_grid:
            raise Exception("There may be no data at polarization plane")
        _mid_theta_index = _num_of_theta_grid // 2
        return self.k_prob[:,_mid_theta_index,:]
    
    
    def get_polar_E_XY_grid(self):
        _E_grid_arr = self.get_E_grid_arr()
        _k_theta_grid_arr = self.get_k_theta_grid_arr()
        _THETA, _E = np.meshgrid(_k_theta_grid_arr, _E_grid_arr)
        _E_X, _E_Y = _E*np.cos(_THETA), _E*np.sin(_THETA)
        return _E_X, _E_Y

