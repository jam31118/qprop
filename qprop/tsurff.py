"""Objects for tsurff data files etc."""

from os.path import join, isfile, isdir, getsize
import numpy as np

from qprop.core import Qprop20


class TSURFF_raw(object):
    
    size_of_complex = 16
    dtype=complex
    
    def __init__(self, data_file_path, num_of_ell_m_grid, R_at_grid_point):
        
        ## Process arguments
        if not isfile(data_file_path): raise FileNotFoundError("Given file path: {}".format(data_file_path))
        self.data_file_path = data_file_path
        if not (int(num_of_ell_m_grid) == num_of_ell_m_grid): raise TypeError("The `num_of_ell_m_grid` should be integer")
        self.num_of_ell_m_grid = int(num_of_ell_m_grid)
        if not R_at_grid_point > 0: raise ValueError("`R_at_grid_point`(={}) should be positive number".format(R_at_grid_point))
        self.R_at_grid_point = R_at_grid_point
        
        ## Evalulate basic variables
        self.size_per_time_step = self._get_size_per_time_step(self.num_of_ell_m_grid)
        self.num_of_time_step = self._get_num_of_time_step(self.data_file_path, self.size_per_time_step)
    
    def _get_size_per_time_step(self, num_of_ell_m_grid):
        size_per_number = self.size_of_complex
        num_of_numbers_per_time_step = num_of_ell_m_grid
        size_per_time_step = size_per_number * num_of_numbers_per_time_step
        return size_per_time_step
    
    def _get_num_of_time_step(self, data_file_path, size_per_time_step):
        data_file_size = getsize(data_file_path)
        num_of_time_step, remainder = divmod(data_file_size, size_per_time_step)
        assert remainder == 0
        return num_of_time_step
    
    def get_ell_m_array_at_n_th_time_step(self, n):
        arr = None
        with open(self.data_file_path, "rb") as f:
            f.seek(self.size_per_time_step*n)
            arr = np.fromfile(f, dtype=self.dtype, count=self.num_of_ell_m_grid)
        assert arr is not None
        return arr
    
    def __getitem__(self, n):
        return self.get_ell_m_array_at_n_th_time_step(n)

    
class TSURFF_raw_with_default(TSURFF_raw):
    default_file_name = ""
    @classmethod
    def from_dir(cls, data_dir_path, num_of_ell_m_grid, R_at_grid_point):
        assert isdir(data_dir_path)
        default_file_path = join(data_dir_path, cls.default_file_name)
        if not isfile(data_file_path): raise FileNotFoundError("Given file path: {}".format(default_file_path))
        return cls(default_file_path, num_of_ell_m_grid, R_at_grid_point)
    
    @classmethod
    def from_qprop_obj(cls, qprop_obj):
        assert isinstance(qprop_obj, Qprop20)
        default_file_path = join(qprop_obj.home, cls.default_file_name)
        
        ## Determine R_at_grid_point (not 'R-tsurff')
        R_tsurff, delta_rho = qprop_obj.get_parameter("tsurff.param", "R-tsurff"), qprop_obj.get_parameter("initial.param", "delta-r")
        index_of_R = int(R_tsurff / delta_rho - 0.5)
        R_at_grid_point = (index_of_R + 1) * delta_rho
                
        return cls(default_file_path, qprop_obj.grid.numOfEllGrid, R_at_grid_point)
    

class TSURFF_psi(TSURFF_raw_with_default):
    default_file_name = "tsurffpsi.raw"


class TSURFF_dpsidr(TSURFF_raw_with_default):
    default_file_name = "tsurff-dpsidr.raw"



