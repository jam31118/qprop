"""Grid handler"""

import numpy as np


## Define Grid object
class Grid(object):
    def __init__(self, numOfRadialGrid=None, numOfEllGrid=None, numOfOrbitalGrid=None, 
            dimension=None, delta_r=None):

        self.dimension = None
        if (delta_r is not None): self.delta_r = delta_r
        self.initial_m = None
        if (numOfRadialGrid != None): self.numOfRadialGrid = int(numOfRadialGrid)
        if (numOfEllGrid != None): self.numOfEllGrid = int(numOfEllGrid)
        if (numOfOrbitalGrid != None): self.numOfOrbitalGrid = int(numOfOrbitalGrid)
        if (dimension != None): self.dimension = int(dimension)

    def set_delta_r(self, delta_r):
        self.delta_r = delta_r

    def getArrayOfRadialValue(self):
        max_radial_value = self.numOfRadialGrid * self.delta_r
        return np.arange(0, max_radial_value, self.delta_r) + self.delta_r

    def set_dimension(self, dimension):
        self.dimension = int(dimension)

    def sizeOf_ell_m_unified_grid(self):
        return self.get_num_of_phi_lm(self.dimension, self.numOfEllGrid)

    @staticmethod
    def get_num_of_phi_lm(qprop_dimension, N_l):
        num_of_phi_lm = None
        if (qprop_dimension == 44): num_of_phi_lm = N_l * N_l
        elif (qprop_dimension == 34): num_of_phi_lm = N_l
        else: raise ValueError("Unknown qprop dimension")
        if num_of_phi_lm is None: raise Exception("Unexpected error during evaluating `num_of_phi_lm`")
        return num_of_phi_lm

    def printNumOfGridPoints(self):
        print("numOfRadialGrid: ", self.numOfRadialGrid)
        print("numOfEllGrid: ", self.numOfEllGrid)
        print("numOfOrbitalGrid: ", self.numOfOrbitalGrid)

    def set_initial_m(self, initial_m):
        self.initial_m = initial_m

    def calculate_listOf_l_m_tuples(self):
        # Check Required variables
        if ((self.dimension == 34) and (self.initial_m == None)):
            raise IOError("initial_m should be initialized before getting (l,m) tuples")
        if (self.dimension == None): 
            raise IOError("dimension should be initialized before getting (l,m) tuples")

        # Construct the list of (l,m) tuples, depending on qprop dimension (self.dimension)
        listOf_l_m_tuples = []
        if (self.dimension == 44):
            for l in range(self.numOfEllGrid):
                for m in range(-l,l+1):
                    listOf_l_m_tuples.append((l,m))
        elif (self.dimension == 34):
            for l in range(self.numOfEllGrid):
                listOf_l_m_tuples.append((l,self.initial_m))
        else: raise Exception("Unknown qprop dimension")

        self.listOf_l_m_tuples = listOf_l_m_tuples

        # Return the list of (l,m) tuples
        return listOf_l_m_tuples

    def get_valid_lm_mask(self):
        valid_lm_mask = None        
        if self.dimension == 34:
            valid_lm_mask = np.arange(self.numOfEllGrid) >= int(abs(self.initial_m))
        elif self.dimension == 44:
            valid_lm_mask = np.full(self.numOfEllGrid, True, dtype=bool)
        else: raise Exception("Unknown qprop dimension")
        return valid_lm_mask

    def get_l_m_iterator(self):
        return self.get_l_m_iterator_static(self.dimension, self.numOfEllGrid, self.initial_m)

    @staticmethod
    def get_l_m_iterator_static(dimension, N_l, initial_m=None):
        if dimension == 34:
            assert initial_m is not None
            return ((l,initial_m) for l in range(N_l))
        elif dimension == 44:
            return ((l,m) for l in range(N_l) for m in range(-l,l+1))

    @staticmethod
    def get_l_m_array(dimension, N_l, initial_m=None):
        _l_m_iter = Grid.get_l_m_iterator_static(dimension, N_l, initial_m=initial_m)
        _l_m_arr = np.array(list(_l_m_iter), dtype=int)
        return _l_m_arr

    def size(self):
        if (self.dimension == 44):
            return self.numOfRadialGrid * self.numOfEllGrid * self.numOfEllGrid
        elif (self.dimension == 34):
            return self.numOfRadialGrid * self.numOfEllGrid * self.numOfOrbitalGrid
        else:
            raise IOError("Unknown qprop dimension")



