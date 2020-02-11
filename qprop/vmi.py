"""VMI data analysis routine"""

from os.path import isdir, join
from os import listdir

import numpy as np

class VMI_data_elliptical_pulse(object):
    data_file_name_list = ["dataPlot", "dataPlotFiltered", "xAxisCartesianPixel", "yAxisCartesianPixel"]
    data_pol_file_name_list = ["dataPlotPol", "dataPlotPolFiltered", "xAxisPolar", "yAxisPolar"]
    
    def __init__(self, data_dir_path, load=True):
        assert isdir(data_dir_path)
        for _fname in self.data_file_name_list: assert _fname in listdir(data_dir_path)
        self.data_dir_path = data_dir_path
        
        self.k_map_data, self.k_map_data_filtered, self.x_pixel_arr, self.y_pixel_arr = (
            np.loadtxt(join(data_dir_path, _fname), delimiter=',') for _fname in self.data_file_name_list)



def get_true_train(arr, min_train_length=100):
    """Get start and end index of the continous train of True values 
    in given array and minimum length of the True train.

    This routine can be used to find the position of MCP (multi-channel plate)
    in a given VMI (velocity map imaging) raw image.
    
    Returns
    -------
    i1: int
        start index of the true train
    i2: int
        end index of the true train
        so that arr[i1:i2+1] is the true train
    
    If such true train was not found, `None` is returned instead of indices.
    
    Notes
    -----
    Courtesy of Kim, Jinhwi (진휘) as the developer of this routine 
    at 2020, Daejun, Korea
    """
    t1, t2 = 0, 0
    for i, tf in enumerate(arr):
        if tf: t1 = i+1
        else:
            if t1-t2>min_train_length: return t2, t1-1
            t2=i+1

