"""Utiltity for evaluating probability flux etc."""

import numpy as np

from .tsurff import TSURFF_psi, TSURFF_dpsidr

class Probability_Flux(object):
    pass

class Probability_Flux_at_R(Probability_Flux):
    def __init__(self, tsurff_psi, tsurff_dpsidr):
        ## Process input arguments
        # Check Types
        assert isinstance(tsurff_psi, TSURFF_psi)
        assert isinstance(tsurff_dpsidr, TSURFF_dpsidr)
        # Check consistency
        list_of_attributes_to_be_same = ["size_per_time_step","num_of_ell_m_grid","R_at_grid_point", "num_of_time_step"]
        for attribute_name in list_of_attributes_to_be_same:
            if not hasattr(tsurff_psi, attribute_name) or not hasattr(tsurff_dpsidr, attribute_name):
                err_mesg = "An attribute '{}' does not exist in either tsurff_psi or tsurff_dpsidr"
                raise AttributeError(err_mesg.format(attribute_name))
            else:
                if not (getattr(tsurff_psi, attribute_name) == getattr(tsurff_dpsidr, attribute_name)):
                    err_mesg = "Inconsistent values of attribute '{}' betweeen `tsurff_psi` and `tsurff_dpsidr`"
                    raise ValueError(err_mesg.format(attribute_name))
        # Assign to member variables
        self.tsurff_psi, self.tsurff_dpsidr = tsurff_psi, tsurff_dpsidr
        self.size_per_time_step, self.num_of_ell_m_grid, self.R_at_grid_point, self.num_of_time_step = [
          getattr(tsurff_psi, attribute_name) for attribute_name in list_of_attributes_to_be_same
        ] # same as `tsurff_dpsidr.attribute_name`
        
        ## Prepare vectorized function
        self.get_probabilty_dissipation_rate_vectorized = np.vectorize(self.get_probabilty_dissipation_rate)
        
    @classmethod
    def from_qprop_obj(cls, qprop_obj):
        tsurff_psi = TSURFF_psi.from_qprop_obj(qprop_obj)
        tsurff_dpsidr = TSURFF_dpsidr.from_qprop_obj(qprop_obj)
        return cls(tsurff_psi, tsurff_dpsidr)
        
    @staticmethod
    def _get_probabilty_dissipation_rate_single_time_step(psi_at_R, dpsidr_at_R, R):
        for arg in [psi_at_R, dpsidr_at_R]: assert isinstance(arg, np.ndarray)
        temp_array = psi_at_R.conj()
        temp_array *= ( - psi_at_R + R * dpsidr_at_R )
        dissipation_rate = temp_array.sum().imag
        dissipation_rate /= pow(R, 3)
        return dissipation_rate
    
    def get_probabilty_dissipation_rate(self, time_step_index):
        idx = time_step_index  # aliasing
        rate = self._get_probabilty_dissipation_rate_single_time_step(
            self.tsurff_psi[idx], self.tsurff_dpsidr[idx], self.R_at_grid_point
        )
        return rate

    def get_probabilty_dissipation_rate_at_once(self):
        _dissipation_rate_per_lm = self.get_probabilty_dissipation_rate_per_lm_at_once()
        assert _dissipation_rate_per_lm.shape == (self.num_of_time_step, self.num_of_ell_m_grid)
        _dissipation_rate_arr = np.sum(_dissipation_rate_per_lm, axis=1)
        return _dissipation_rate_arr

    def get_probabilty_dissipation_rate_per_lm_at_once(self):
        _2d_shape = (self.num_of_time_step, self.num_of_ell_m_grid)

        _tsurff_psi_arr_1d = self.tsurff_psi.load()
        _tsurff_psi_arr_2d = _tsurff_psi_arr_1d.reshape(*_2d_shape)

        _tsurff_dpsidr_arr_1d = self.tsurff_dpsidr.load()
        _tsurff_dpsidr_arr_2d = _tsurff_dpsidr_arr_1d.reshape(*_2d_shape)
        
        _dissipation_rate_per_lm = ( _tsurff_psi_arr_2d.conj() * ( - _tsurff_psi_arr_2d + self.R_at_grid_point * _tsurff_dpsidr_arr_2d ) ).imag
        _dissipation_rate_per_lm *= pow(self.R_at_grid_point, -3)

        return _dissipation_rate_per_lm
        
    def __getitem__(self, time_step_index_exp):
        time_step_index_array = np.arange(self.num_of_time_step, dtype=int)[time_step_index_exp]
        return self.get_probabilty_dissipation_rate_vectorized(time_step_index_array)

    def plot(self, ax, num_of_color_range_order=17, data=None):
        from numbers import Integral
        from matplotlib.axes import Axes
        assert isinstance(num_of_color_range_order, Integral) and num_of_color_range_order > 0
        assert isinstance(ax, Axes)

        from matplotlib.colors import LogNorm

        _dissip_rate_lm_arr = data
        if _dissip_rate_lm_arr is None:
            _dissip_rate_lm_arr = self.get_probabilty_dissipation_rate_per_lm_at_once()
        _vmax = _dissip_rate_lm_arr.max()
        _vmin = _vmax * pow(10,-num_of_color_range_order)
        im = ax.imshow(_dissip_rate_lm_arr, norm=LogNorm(), vmin=_vmin, vmax=_vmax, cmap='jet', aspect='auto')
        cb = ax.figure.colorbar(im, ax=ax)
        cb.set_label("amplitude")
        ax.set_xlabel("ell-m-index")
        ax.set_ylabel("time-index")

        return im

