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
    
    def __getitem__(self, time_step_index_exp):
        time_step_index_array = np.arange(self.num_of_time_step, dtype=int)[time_step_index_exp]
        return self.get_probabilty_dissipation_rate_vectorized(time_step_index_array)

