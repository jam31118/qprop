"""Grid handler"""

import numpy as np


## Define Grid object
class Grid(object):
    def __init__(self, numOfRadialGrid=None, numOfEllGrid=None, numOfOrbitalGrid=None, dimension=None):
        self.dimension = None
        self.delta_r = None
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
        if (self.dimension == 44):
            return self.numOfEllGrid * self.numOfEllGrid
        elif (self.dimension == 34):
            return self.numOfEllGrid
        else:
            raise IOError("Unknown qprop dimension")

    def printNumOfGridPoints(self):
        print("numOfRadialGrid: ", self.numOfRadialGrid)
        print("numOfEllGrid: ", self.numOfEllGrid)
        print("numOfOrbitalGrid: ", self.numOfOrbitalGrid)

    def set_initial_m(self, initial_m):
        self.initial_m = initial_m

    def calculate_listOf_l_m_tuples(self):
        # Check Required variables
        if ((self.dimension == 34) and (self.initial_m == None)): raise IOError("initial_m should be initialized before getting (l,m) tuples")
        if (self.dimension == None): raise IOError("dimension should be initialized before getting (l,m) tuples")

        # Construct the list of (l,m) tuples, depending on qprop dimension (self.dimension)
        listOf_l_m_tuples = []
        if (self.dimension == 44):
            for l in range(self.numOfEllGrid):
                for m in range(-l,l+1):
                    listOf_l_m_tuples.append((l,m))
        elif (self.dimension == 34):
            for l in range(self.numOfEllGrid):
                listOf_l_m_tuples.append((l,self.initial_m))
        else: raise IOError("Unknown qprop dimension")

        self.listOf_l_m_tuples = listOf_l_m_tuples

        # Return the list of (l,m) tuples
        return listOf_l_m_tuples

    def get_l_m_iterator(self):
        if self.dimension == 34:
            assert self.initial_m is not None
            return ((l,self.initial_m) for l in range(self.numOfEllGrid))
        elif self.dimension == 44:
            return ((l,m) for l in range(self.numOfEllGrid) for m in range(-l,l+1))

    def size(self):
        if (self.dimension == 44):
            return self.numOfRadialGrid * self.numOfEllGrid * self.numOfEllGrid
        elif (self.dimension == 34):
            return self.numOfRadialGrid * self.numOfEllGrid * self.numOfOrbitalGrid
        else:
            raise IOError("Unknown qprop dimension")



