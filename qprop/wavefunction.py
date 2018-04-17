"""Wavefunction handler"""

from os.path import abspath, join, isfile, isdir

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import sph_harm
from matplotlib.colors import Normalize, LogNorm

from vis.plot import construct_polar_mesh_for_colormesh

from .grid import Grid
from .default import default_config
from .util import get_index_of_nearest_element


class Wavefunction(object):
    def __init__(self, grid):
        ## Process input arguments
        # Check type
        if (type(grid) != Grid):
            raise TypeError("Required 'Grid', not %s" % type(grid))
        # Assign input argument(s) into member variable(s)
        self.grid = grid

        ## Construct 1D array with the size of given grid
        self.data_raw = np.empty((grid.size(),2))  # second '2' is for real and imaginary part of complex wavefunction value

        ## Initialize flags to be Flase
        self.wavefucntion_have_been_loaded = False
        #self.have_been_reconstructed_in_real_space = False

        ## Initialize angle grid
        # [NOTE 171204] It might be better to gather these kind of configuration variables into one object such as dictionary
        # .. similar object can be rcParams of matplotlib.pyplot
        self.grid.defaultNumOfThetaPoints = 9
        self.grid.defaultNumOfPhiPoints = 50
        self.grid.thetaValues = np.linspace(0, np.pi, self.grid.defaultNumOfThetaPoints)
        self.grid.phiValues = np.linspace(0, 2*np.pi, self.grid.defaultNumOfPhiPoints)
    
    @staticmethod
    def check_or_set_valid_dimension_order(dimension_order):
        """Check and return valid dimension_order
        
        If 'dimension_order' is an emtpy sequence, returns default dimension_order, defined in `default` module.
        """
        if dimension_order == []: dimension_order = default_config['wf_file_dimension_order']
        assert type(dimension_order) in [list, tuple]
        assert len(dimension_order) == 2
        for element in dimension_order: assert element in ['rho','ell-m']
        assert dimension_order in [['rho','ell-m'], ['ell-m','rho']]
        return dimension_order


    def load(self, dataFileName, indexOfWavefuncInFile, binary=False, dimension_order=[]):
        self.dimension_order = Wavefunction.check_or_set_valid_dimension_order(dimension_order)

        numOfLineToRead = self.grid.size()
        indexOfFirstLineToRead = int(indexOfWavefuncInFile * self.grid.size())
        numOfLineBeforeLineToBeRead = indexOfFirstLineToRead

        if binary:
            size_of_complex_number = 16
            with open(dataFileName, 'rb') as f:
                f.seek(size_of_complex_number * numOfLineBeforeLineToBeRead)
                self.data = np.fromfile(f, dtype=complex, count=numOfLineToRead)
        else:
            self.data_raw[:] = pd.read_table(dataFileName, sep='\s+', header=None,
                    skiprows=numOfLineBeforeLineToBeRead, nrows=numOfLineToRead, skip_blank_lines=True).values

            if self.data_raw.shape != (numOfLineToRead, 2):
                raise IOError("Data misread from %s" % (dataFileName))

            # Convert two real-valud arrays to one complex-valued array
            real_2_tuple_dimension_index = 1
            self.data = np.apply_along_axis(lambda real_2_tuple: complex(*real_2_tuple), real_2_tuple_dimension_index, self.data_raw)

        # Reshape original 1D array to 2D array
        data_2d_shape = None
        if self.dimension_order == ['ell-m','rho']:
            data_2d_shape = (self.grid.sizeOf_ell_m_unified_grid(), self.grid.numOfRadialGrid)
        elif self.dimension_order == ['rho','ell-m']:
            data_2d_shape = (self.grid.numOfRadialGrid, self.grid.sizeOf_ell_m_unified_grid())
        else: raise Exception("Unexpected value for dimension order: {0}".format(dimension_order))
        
        self.data_2d = self.data.view().reshape(data_2d_shape)

        self.wavefucntion_have_been_loaded = True


    def _generateSphericalHarmonicsOnGrid(self, arrayOfThetaValue, arrayOfPhiValue):
        self.grid.thetaValues = arrayOfThetaValue
        self.grid.phiValues = arrayOfPhiValue
        gridOfTheta, gridOfPhi = np.meshgrid(self.grid.thetaValues, self.grid.phiValues)
        self.grid.shapeOfSpheriHarmGrid = (self.grid.sizeOf_ell_m_unified_grid(), len(self.grid.phiValues), len(self.grid.thetaValues))

        # 'm' corresponds to 'm' and 'n' corresponds to 'l' (orbital quantum number)
        vectorized_spheriHarm = np.vectorize(sph_harm, excluded=['m','n'])

        self.arrayOfSpheriHarm = np.empty(self.grid.shapeOfSpheriHarmGrid, dtype=complex)
        for lm_tuple_idx, lm_tuple in enumerate(self.grid.calculate_listOf_l_m_tuples()):
            l = lm_tuple[0]
            m = lm_tuple[1]
            self.arrayOfSpheriHarm[lm_tuple_idx,:,:] = vectorized_spheriHarm(m, l, gridOfPhi, gridOfTheta)

    def reconstructInRealSpace(self, arrayOfThetaValue=[], arrayOfPhiValue=[]):

        if len(arrayOfThetaValue) != 0:
            self.grid.thetaValues = arrayOfThetaValue
            #arrayOfThetaValue = np.linspace(0,np.pi,defaultNumOfThetaPoints)

        if len(arrayOfPhiValue) != 0:
            self.grid.phiValues = arrayOfPhiValue
            #arrayOfPhiValue = np.linspace(0,2*np.pi,defaultNumOfPhiPoints)

        #self._generateSphericalHarmonicsOnGrid(arrayOfThetaValue, arrayOfPhiValue)
        self._generateSphericalHarmonicsOnGrid(self.grid.thetaValues, self.grid.phiValues)
        # Summation on (l,m) indice
        data_2d_index_label = None
        if self.dimension_order == ['ell-m','rho']: data_2d_index_label = [0,1]
        elif self.dimension_order == ['rho','ell-m']: data_2d_index_label = [1,0]
        else: raise Exception("Unexpected value for dimension order: {0}".format(dimension_order))
        self.atRealspace = np.einsum(self.data_2d, data_2d_index_label, self.arrayOfSpheriHarm, [0,2,3])
        radial_grid = self.grid.getArrayOfRadialValue()
        for idx, r_val in enumerate(radial_grid):
            self.atRealspace[idx,:,:] = self.atRealspace[idx,:,:] * 1.0 / r_val

        #self.have_been_reconstructed_in_real_space = True

    def plot_wf_cross_section_at_constant_theta(self, const_theta = np.pi / 2.0, arrayOfThetaValue=[], arrayOfPhiValue=[],
            log_scale = True, vmin = None, vmax = None, fig = None, ax = None, fresh_reconstruction=True):

        if not self.wavefucntion_have_been_loaded:
            raise Exception("Wavefunction should be loaded first")

        if fresh_reconstruction:
            do_reconstruction = True

        thetaValuesAreDifferent = False
        if len(arrayOfThetaValue) != 0:
            if len(self.grid.thetaValues) != len(arrayOfThetaValue):
                thetaValuesAreDifferent = True
            elif not (self.grid.thetaValues == arrayOfThetaValue).all():
                thetaValuesAreDifferent = True
        if thetaValuesAreDifferent:
            self.grid.thetaValues = arrayOfThetaValue
            do_reconstruction = True
            #self.have_been_reconstructed_in_real_space = False

        phiValuesAreDifferent = False
        if len(arrayOfPhiValue) != 0:
            if len(self.grid.phiValues) != len(arrayOfPhiValue):
                phiValuesAreDifferent = True
            elif not (self.grid.phiValues == arrayOfPhiValue).all():
                phiValuesAreDifferent
        if phiValuesAreDifferent:
            self.grid.phiValues = arrayOfPhiValue
            do_reconstruction = True
            #self.have_been_reconstructed_in_real_space = False

        #self.reconstructInRealSpace(self.grid.thetaValues, self.grid.phiValues)
        #if not self.have_been_reconstructed_in_real_space:
            #self.reconstructInRealSpace(arrayOfThetaValue, arrayOfPhiValue)
        if do_reconstruction:
            self.reconstructInRealSpace(arrayOfThetaValue, arrayOfPhiValue)

        # [NOTE] The self.grid.phiValues and self.grid.thetaValues should be set in self.reconstructInRealSpace()
        assert (len(self.grid.phiValues) != 0) and (len(self.grid.thetaValues) != 0)

        r = self.grid.getArrayOfRadialValue()
        phi = self.grid.phiValues

        mesh_X, mesh_Y = construct_polar_mesh_for_colormesh(r, phi)

        # R, Phi = np.meshgrid(r,phi,indexing='ij')

        # ## Augmentation
        # R_augmented = np.empty((R.shape[0]+1,R.shape[1]))
        # R_augmented[0,:] = 0
        # R_augmented[1:,:] = R

        # Phi_augmented = np.empty((Phi.shape[0]+1, Phi.shape[1]))
        # Phi_augmented[1:,:] = Phi
        # Phi_augmented[0,:] = Phi[0,:]

        ## Get wavefunction at constant theta
        const_theta_index = get_index_of_nearest_element(self.grid.thetaValues, const_theta)
        Z = np.absolute(self.atRealspace[:,:,const_theta_index])
        Z = np.square(Z)

        ## Determine Figure and Axes object
        if fig is None:
            fig = plt.figure(figsize=(10,10))
        if ax is None:
            #ax = fig.gca(projection='polar')
            ax = fig.gca()

        ## Prevent log(0) error by matplotlib.colors.LogNorm()
        Z[Z==0] = 1e-50

        if log_scale:
            normalizer = LogNorm(vmax=vmax, vmin=vmin)
            #pcm = ax.pcolormesh(Phi_augmented, R_augmented, Z, norm=colors.LogNorm(vmax=vmax, vmin=vmin))
        else:
            #pcm = ax.pcolormesh(Phi_augmented, R_augmented, Z, norm=colors.Normalize(vmax=vmax, vmin=vmin))
            normalizer = Normalize(vmax=vmax, vmin=vmin)

        pcm = ax.pcolormesh(mesh_X, mesh_Y, Z, norm=normalizer)

        ax.axis('square')

        radius = r.max()
        ax.set_xlim(-radius, radius)
        ax.set_ylim(-radius, radius)

        ax.set_xlabel(r'x coordinate / Bohr Radius')
        ax.set_ylabel(r'y coordinate / Bohr Radius')

        cb = fig.colorbar(pcm, ax=ax, extend='min')

        return fig, ax




class Timed_Wavefunction(object):
    """An array-like object whose element corresponds to wavefunction(state function) at each timestep."""
    def __init__(self, grid, home, timed_wf_file_name='', time_file_name='', binary=True, dimension_order=[]):
        ## Check input arguments
        args_to_check = [grid, home, timed_wf_file_name, time_file_name, binary]
        args_types = [Grid, str, str, str, bool]
        assert len(args_to_check) == len(args_types)
        for arg, arg_type in zip(args_to_check, args_types):
            if not isinstance(arg, arg_type):
                raise TypeError("{0} should be of type {1}".format(arg, arg_type))
        
        self.grid = grid
        self.binary = binary
        
        self.dimension_order = Wavefunction.check_or_set_valid_dimension_order(dimension_order)
        
        assert isdir(home)
        self.home = abspath(home)
        
        self.timed_wf_file_name = None
        if timed_wf_file_name == '':
            self.timed_wf_file_name = default_config ['timed_wf_file_name']
            
        self.time_file_name = None
        if time_file_name == '':
            self.time_file_name = default_config ['time_file_name']
        
        self.timed_wf_file_path = join(self.home, self.timed_wf_file_name)
        assert isfile(self.timed_wf_file_path)
        self.time_file_path = join(self.home, self.time_file_name)
        assert isfile(self.time_file_path)
        
        time_array = np.loadtxt(self.time_file_path)
        assert isinstance(time_array, np.ndarray)
        assert time_array.ndim == 1
        assert time_array.size >= 1
        self.t = time_array
        self.num_of_timesteps = self.t.size
        
        self.buffer_wf = Wavefunction(self.grid)
        
    def __getitem__(self, index_exp):
        assert int(index_exp) == index_exp
        assert index_exp < self.num_of_timesteps
        if index_exp < 0: index_exp = self.num_of_timesteps + index_exp
        self.buffer_wf.load(self.timed_wf_file_path, index_exp, binary=self.binary, dimension_order=self.dimension_order)
        return self.buffer_wf



