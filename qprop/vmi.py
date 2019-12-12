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


