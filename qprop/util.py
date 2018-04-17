"""A collection of utility functions"""

import numpy as np


def get_index_of_nearest_element(array, value, threshold = 1e-3):
    ## Check and pre-process input arguments
    if type(array) is not np.ndarray:
        try: array = np.array(array, dtype=float)
        except: raise TypeError("The 'array' should be of type numpy.ndarray")
    assert array.ndim == 1

    #if value == 0: value = 1e-50
    mask = np.abs(array - value) <= (threshold * abs(value))
    indice = np.where(mask)[0]

    # Check possible anomalies
    if len(indice) < 1:
        raise ValueError("no matching theta value")
    if len(indice) is not 1:
        raise ValueError("number of matched values should be one."
            + "Consider modulating threshold.")

    # Determine theta index
    nearest_index = indice[0]

    return nearest_index

