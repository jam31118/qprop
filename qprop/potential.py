"""Collection of several types of potentials"""

import numpy as np

from .grid import Grid
from .default import default_config


class Imagpot(object):
    def __init__(self, width, grid, ampl=None):
        assert isinstance(grid, Grid)
        if ampl is None: ampl = default_config['imag-pot-ampl']
        assert ampl > 1.0
        assert width > 0.0
        self.width, self.grid, self.ampl = width, grid, ampl
        self.width_in_num_of_grid_points = int(self.width // self.grid.delta_r)

    def __getitem__(self, rho_index_exp):
        rho_indices = np.arange(self.grid.numOfRadialGrid)[rho_index_exp]
        result_array = np.empty_like(rho_indices, dtype=complex)
        nonzero_start_index = self.grid.numOfRadialGrid - self.width_in_num_of_grid_points
        zero_mask = rho_indices < nonzero_start_index
        result_array[zero_mask] = 0.0
        result_array[~zero_mask] = self.ampl * ((rho_indices[~zero_mask] - nonzero_start_index) / self.width_in_num_of_grid_points) ** 16
        result_array *= -1.0j
        return result_array

