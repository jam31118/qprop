import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import matplotlib.colors as colors

from .util import get_index_of_nearest_element


### To implement
# I need a blank MomentumSpectrumPolar object when I need sum up several MomentumSpectrumPolar objects as a initialzed base.

def is_seems_like_qprop(obj):
    attrs = ['initialParam', 'paramFileList']
    seems_like_qprop = True
    for attr in attrs:
        seems_like_qprop &= hasattr(obj, attr)
    return seems_like_qprop

class MomentumSpectrumPolar(object):
    def __init__(self, q = None):
        self.haveReadPolarSpectrum = False
        self.qprop_dim = None

        self.q = None

        ## Process input arguments
#        if type(q) is Qprop20:
        if is_seems_like_qprop(q):
            # If the Qprop object enters,
            # .. it tries to read qprop-dimension and polar spectrum data from file
            self.q = q
            self.qprop_dim = q.initialParam['qprop-dim']
            #try: self.readPolarSpectrumData(q)
            #except Exception: raise Warning("Failed to read polar spectrum data automatically")
        # If no Qprop object enters, do nothing
        elif q is None: pass
        # If the q is not Qprop object nor None type, meaning other type have been entered
        # .. a unknown type error is raised
        else: raise TypeError("Unsupported type: %s" % type(q))

    def integrate_angle_energy_34(self, spectra_k, k, delta_theta, sin_theta_array):
        f = sin_theta_array * spectra_k
        sum_k = 2.0 * np.pi * delta_theta * (f.sum() - 0.5 * (f[0] + f[-1]))
        return sum_k

    def integrate_angle_momentum_34(self, spectra_k, k, delta_theta, sin_theta_array):
        f = sin_theta_array * spectra_k
        sum_k = 2.0 * np.pi * delta_theta * k * (f.sum() - 0.5 * (f[0] + f[-1]))
        return sum_k

    def readPolarSpectrumData(self, q, dataFileName=''):
        """
        q: Qprop object
        dataFileName: the name of a file where polar spectrum data is contained
        """

        if 'tsurff.param' in q.paramFileList:
            algorithm_name = 'tsurff'
        elif 'winop.param' in q.paramFileList:
            algorithm_name = 'winop'
        else:
            raise Exception("Unrecognized algorithm. Use either winop or tsurff method.")


        ## Check data file name
        if dataFileName == '':
            # Check if this calculation uses tsurff or winop
            #if 'tsurff.param' in q.paramFileList:
            if algorithm_name == 'tsurff':
                dataFileNameGuessed = os.path.join(q.home,'tsurff-polar.dat')
            elif algorithm_name == 'winop':
                dataFileNameGuessed = os.path.join(q.home, 'spectrum_polar0.dat')
            else:
                raise Exception("Give an explicit dataFileName if you use custom data file name")
            dataFileName = dataFileNameGuessed

        ## Check whether the specified parameter file name exists
        if not os.path.exists(dataFileName):
            raise Exception("No file is found with name: %s" % dataFileName)

        # Get qprop-dimension for convenience from now on (kind of alias)
        self.qprop_dim = q.initialParam['qprop-dim']

        ## Import and set column names of DataFrame
        # Set column names
        if self.qprop_dim == 34:
            colNames = ['energy','k','theta_k','total_prob']
        elif self.qprop_dim == 44:
            colNames = ['energy','k','theta_k','phi_k','total_prob']
        else:
            raise ValueError("Input qprop-dim(=%d) is neither 34 nor 44." % (initialParam['qprop-dim']))

        ## Import data
        self.polarSpectrum = pd.read_table(dataFileName, sep='\s+', header=None, skip_blank_lines=True, dtype=np.double)

        ## Check whether the number of imported columns and expected number of columns are same
        num_dataCols = self.polarSpectrum.shape[1]
        num_expectedCols = len(colNames)
        if num_dataCols != num_expectedCols:
            raise Exception("Input data isn't consistent with parameter files")

        ## If the number of columns are consistent, assign list of column names to the DataFrame
        self.polarSpectrum.columns = colNames

        ## change num-theta if it's not odd
        # .. as defined in src/base/tsurffSpectrum.hh
        if algorithm_name == 'tsurff':
            actual_num_theta_surff = self._compensate_num_theta_surff(q.tsurffParam['num-theta-surff'])

            ## Check consistency with respect to the number of rows of DataFrame
            if self.qprop_dim == 34:
                expected_length = q.tsurffParam['num-k-surff'] * actual_num_theta_surff
            elif self.qprop_dim == 44:
                expected_length = q.tsurffParam['num-k-surff'] * actual_num_theta_surff * q.tsurffParam['num-phi-surff']

        elif algorithm_name == 'winop':
            actual_num_theta_winop = self._compensate_num_theta_winop(q.winopParam['num-theta'])

            if self.qprop_dim == 34:
                expected_length = q.winopParam['num-energy'] * actual_num_theta_winop
            elif self.qprop_dim == 44:
                expected_length = q.winopParam['num-energy'] * actual_num_theta_winop * q.winopParam['num-phi']

        if self.polarSpectrum.shape[0] != expected_length:
            raise Exception("The imported data's number of rows(=%d) isn't consistent with expected number of rows(=%d)"
                % (self.polarSpectrum.shape[0], expected_length))


        ## Sort the imported data according to the qprop dimension
        if self.qprop_dim == 34:
            sorting_criteria = ['k','theta_k']
        elif self.qprop_dim == 44:
            sorting_criteria = ['k','theta_k','phi_k']
        else:
            raise Exception("Input qprop-dim(=%d) is neither 34 nor 44." % (initialParam['qprop-dim']))
        self.polarSpectrum.sort_values(sorting_criteria, inplace=True)

        ## Set 3D polar k grid
        self.set_polar_k_grid(q)

    def _compensate_num_theta_winop(self, num_theta_winop):
        # if num_theta_winop is zero then convert it to 1 for preventing error.
        # .. This behavior was defined in winop.cc of qprop 2.0 version
        actual_num_theta_winop = max([num_theta_winop, 1])
        return actual_num_theta_winop

    def _compensate_num_theta_surff(self, num_theta_surff):
        """
        Change the num_theta_surff value defined in src/base/tsurffSpectrum.hh
        """
        #actual_num_theta_surff = num_theta_surff
        if (num_theta_surff < 3):
            actual_num_theta_surff = 3
        elif (num_theta_surff % 2) == 1:
            actual_num_theta_surff = num_theta_surff
        elif (num_theta_surff % 2) == 0:
            actual_num_theta_surff = num_theta_surff + 1
        else:
            raise Exception("Unexpected value for num_theta_surff(=%d)" % num_theta_surff)

        return actual_num_theta_surff

    # def get_kProbGrid_view(self, q = None, qprop_dim = None):
    #     if type(q) is Qprop20:
    #         num_k_surff = q.tsurffParam['num-k-surff']
    #         num_theta_surff = self._compensate_num_theta_surff(q.tsurffParam['num-theta-surff'])
    #         num_phi_surff = q.tsurffParam['num-phi-surff']
    #         qprop_dim = q.initialParam['qprop-dim']
    #     elif q is None:
    #         if self.qprop_dim is None:
    #             assert qprop_dim is not None
    #         elif qprop_dim is not None:
    #             assert qprop_dim == self.qprop_dim
    #         else:
    #             qprop_dim = self.qprop_dim
    #         kValues = self.polarSpectrum['k'].unique()
    #         num_k_surff = len(kValues)
    #         kThetaValues = self.polarSpectrum['theta_k'].unique()
    #         num_theta_surff = len(kThetaValues)
    #         if qprop_dim == 44:
    #             kPhiValues = self.polarSpectrum['phi_k'].unique()
    #             num_phi_surff = len(kPhiValues)
    #     else: raise TypeError("Unsupported type")

    #     assert self.qprop_dim is not None
    #     if self.qprop_dim == 34:
    #         self.kGridShape = (num_k_surff, num_theta_surff)
    #         self.kProbGrid = self.polarSpectrum['total_prob'].values.view().reshape(self.kGridShape)
    #     elif self.qprop_dim == 44:
    #         self.kGridShape = (num_k_surff, num_theta_surff, num_phi_surff)
    #         self.kProbGrid = self.polarSpectrum['total_prob'].values.view().reshape(self.kGridShape)
    #     else:
    #         raise IOError("Input qprop-dim(=%d) neither 34 nor 44." % (qprop_dim))

    #     return self.kProbGrid


    def set_polar_k_grid(self, q = None, qprop_dim = None):
        """
        Construct 3D polar momentum(denoted by k) grid

        q: qprop object
        """

#        if type(q) is Qprop20:
        if is_seems_like_qprop(q):

            if 'tsurff.param' in q.paramFileList:
                algorithm_name = 'tsurff'
            elif 'winop.param' in q.paramFileList:
                algorithm_name = 'winop'
            else:
                raise Exception("Unrecognized algorithm. Use either winop or tsurff method.")

            qprop_dim = q.initialParam['qprop-dim']

            if algorithm_name == 'tsurff':
                num_k = q.tsurffParam['num-k-surff']
                num_theta = self._compensate_num_theta_surff(q.tsurffParam['num-theta-surff'])
                num_phi = q.tsurffParam['num-phi-surff']
            elif algorithm_name == 'winop':
                num_k = q.winopParam['num-energy']  # number of energy and k are same.
                num_theta = self._compensate_num_theta_winop(q.winopParam['num-theta'])
                num_phi = self._compensate_num_theta_winop(q.winopParam['num-phi'])

        elif q is None:
            if self.qprop_dim is None:
                assert qprop_dim is not None
            elif qprop_dim is not None:
                assert qprop_dim == self.qprop_dim
            else:
                qprop_dim = self.qprop_dim
            kValues = self.polarSpectrum['k'].unique()
            num_k = len(kValues)
            kThetaValues = self.polarSpectrum['theta_k'].unique()
            num_theta = len(kThetaValues)
            if qprop_dim == 44:
                kPhiValues = self.polarSpectrum['phi_k'].unique()
                num_phi = len(kPhiValues)
        else: raise TypeError("Unsupported type")

        ## Setup 3D grid in spherical coordinate in case of qprop_dim == 44
        if qprop_dim == 34:
            self.kGridShape = (num_k, num_theta)
            #self.k_values = self.polarSpectrum['k'].values
            #self.kGrid = self.k_values.view().reshape(self.kGridShape)
            #self.kThetaGrid = self.polarSpectrum['theta_k'].values.view().reshape(self.kGridShape)
            #self.kProbGrid = self.polarSpectrum['total_prob'].values.view().reshape(self.kGridShape)
        elif qprop_dim == 44:
            self.kGridShape = (num_k, num_theta, num_phi)
            self.phi_values = self.polarSpectrum['phi_k'].unique()
            self.kPhiGrid = self.polarSpectrum['phi_k'].values.view().reshape(self.kGridShape)
            #self.kGrid = self.polarSpectrum['k'].values.view().reshape(self.kGridShape)
            #self.kThetaGrid = self.polarSpectrum['theta_k'].values.view().reshape(self.kGridShape)
            #self.kProbGrid = self.polarSpectrum['total_prob'].values.view().reshape(self.kGridShape)
        else:
            raise IOError("Input qprop-dim(=%d) neither 34 nor 44." % (qprop_dim))

        ## Set common k grids: kGrid, kThetaGrid (along with k_values and theta_values)
        self.k_values = self.polarSpectrum['k'].unique()
        self.kGrid = self.polarSpectrum['k'].values.view().reshape(self.kGridShape)
        self.theta_values = self.polarSpectrum['theta_k'].unique()
        self.kThetaGrid = self.polarSpectrum['theta_k'].values.view().reshape(self.kGridShape)
        self.kProbGrid = self.polarSpectrum['total_prob'].values.view().reshape(self.kGridShape)

        ## Turn on the flag for indicating the completion of reading
        self.haveReadPolarSpectrum = True


    def plot(self, *args, **kwargs):
        """Aliased plotting function"""
        if self.qprop_dim == 34:
            self.plot_under_linear_polarized_list(*args, **kwargs)
        elif self.qprop_dim == 44:
            self.plot_at_constant_theta(*args, **kwargs)
        else: raise Exception("'qprop-dim' should either be 34 or 44, not %s" % self.qprop_dim)


    def plot_under_linear_polarized_list(self, fig=None, ax=None, log_scale=False,
        vmin=None, vmax=None, xmin=None, xmax=None, ymin=None, ymax=None, 
        cmap='jet', decalcomanie=False, order_of_mag=3):
        ## Check prerequisites
        self.check_and_or_load()
        if self.qprop_dim != 34:
            raise Exception("Should have a linearly polarized light.")
        if ((fig is None) and (ax is not None)) or ((fig is not None) and (ax is None)):
            raise Exception("'fig' and 'ax' should be both None or not None.")

        assert int(order_of_mag) == order_of_mag
        for arg in [vmin, vmax, xmin, xmax, ymin, ymax]:
            if arg is not None:
                assert float(arg) == arg  # check if is is real-number

        #k_arr = self.kGrid[:,0]
        #theta_arr = self.kThetaGrid[0,:]

        N_k = len(self.k_values)
        N_theta = len(self.theta_values)

        #delta_k = (k_arr[-1] - k_arr[0]) / (N_k - 1)
        delta_theta = (self.theta_values[-1] - self.theta_values[0]) / (N_theta - 1)

        ## [180223 NOTE] the k_values may not be equally-distanced grid of k.
        #grid_k_values = (np.arange(N_k + 1) - 0.5) * delta_k
        grid_k_values = np.zeros((N_k + 1,))
        grid_k_values[1:-1] = 0.5 * (self.k_values[1:] + self.k_values[:-1])
        #print("N_k",N_k)
        #print("len(self.k_values[1:])",len(self.k_values[1:]))
        #print("len(grid_k_values[1:-1])", len(grid_k_values[1:-1]))
        #print("grid_k_values",grid_k_values)
        grid_k_values[0] = 0
        assert len(grid_k_values) >= 3
        grid_k_values[-1] = grid_k_values[-2] + (grid_k_values[-2] - grid_k_values[-3])
        grid_theta_values = (np.arange(N_theta + 1) - 0.5) * delta_theta
        #print("grid_theta_values",grid_theta_values)
        #print("len(grid_theta_values)",len(grid_theta_values))

        K_grid, Theta_grid = np.meshgrid(grid_k_values, grid_theta_values, indexing='ij')

        X_grid = K_grid * np.cos(Theta_grid)
        Y_grid = K_grid * np.sin(Theta_grid)
        C = self.kProbGrid

        if (fig is None) and (ax is None):
            fig, ax = plt.subplots()

        ## Font configuration
        fontsize = 'large'

        ## Set axes properties
        if xmin is None: xmin = -1.0 * self.k_values[-1]
        if xmax is None: xmax = self.k_values[-1]
        if ymin is None:
            if decalcomanie: ymin = -1.0 * self.k_values[-1]
            else: ymin = 0
        if ymax is None: ymax = self.k_values[-1]

        ax.axis('square')

        ax.set_xlim(xmin,xmax)
        ax.set_ylim(ymin,ymax)

        ax.set_xlabel(r"$p_x\,\,[a.u.]$",fontsize=fontsize)
        ax.set_ylabel(r"$p_y\,\,[a.u.]$",fontsize=fontsize)

        ## Set vmin, vmax (color data range)
        if (vmin is None) and (vmax is None):
            vmax = C.max()
            vmin = vmax * 10 ** (-order_of_mag)
        elif (vmin is None) and (vmax is not None):
            vmin = vmax * 10 ** (-order_of_mag)
        elif (vmin is not None) and (vmax is None):
            vmax = vmin * 10 ** (order_of_mag)

        ## Draw pcolormesh
        if log_scale: norm = colors.LogNorm(vmin, vmax)
        else: norm = colors.Normalize(vmin, vmax)

        pcm = ax.pcolormesh(X_grid,Y_grid,C, norm=norm, cmap=cmap)
        if decalcomanie: # For decalcomanie
            pcm_down = ax.pcolormesh(X_grid,-Y_grid,C, norm=norm, cmap=cmap)

        # if log_scale:
        #     pcm = ax.pcolormesh(X_grid,Y_grid,C, norm=colors.LogNorm(vmin, vmax), cmap =cmap)
        #     if decalcomanie: # For decalcomanie
        #         pcm_down = ax.pcolormesh(X_grid,-Y_grid,C, norm=colors.LogNorm(vmin, vmax), cmap =cmap)
        # else:
        #     pcm = ax.pcolormesh(X_grid,Y_grid,C, norm=colors.Normalize(vmin, vmax), cmap =cmap)
        #     # For decalcomanie
        #     if decalcomanie:
        #         pcm_down = ax.pcolormesh(X_grid,Y_grid,C, norm=colors.Normalize(vmin, vmax), cmap =cmap)


#        if fontsize is None: fontsize = 20 #fig.get_size_inches().mean() * 2

        cb = fig.colorbar(pcm, ax=ax)
        cb.ax.get_yaxis().labelpad = 20
        cb.set_label('differential probability', rotation=270, fontsize=fontsize)

        return fig, ax


    def check_and_or_load(self):
        if not self.haveReadPolarSpectrum:
            if self.q is None: raise Exception("Cannot read polar spectrum from file automatically")
            try:
                self.readPolarSpectrumData(self.q)
            except:
                raise Exception("No data to plot. Please read polar spectrum data first")

    def plot_at_constant_theta(self, theta = np.pi / 2 , saveFigureName='', threshold = 1e-8,
        vmin = None, vmax = None, log_scale=False, log_min=None, log_max=None,
        fig = None, ax = None, **kwargs):
        """
        Plot PhotoElectronSpectra with constant theta value
        kwargs is passed to pcolormesh()
        """
        ## Check prerequisites
        if not self.haveReadPolarSpectrum:
            if self.q is None: raise Exception("Cannot read polar spectrum from file automatically")
            try:
                self.readPolarSpectrumData(self.q)
            except:
                raise Exception("No data to plot. Please read polar spectrum data first")

        ## Identify theta index corresponding input value 'theta'
        # .. If 'theta' is similar enough with one of the existing theta values in data,
        # .. closer than threshold, then that theta value is considered as intended theta
        possibleThetaValues = self.polarSpectrum['theta_k'].unique()

        theta_index = get_index_of_nearest_element(possibleThetaValues, theta, threshold)


        ## Construct data based on the extracted theta index
        kGrid2D = self.kGrid[:,theta_index,:]
        kPhiGrid2D = self.kPhiGrid[:,theta_index,:]
        kProbGrid2D = self.kProbGrid[:,theta_index,:]
        # Shape of 2D grid array, associated with the theta value
        gridShape2D = kGrid2D.shape


        ## Augmentation of grid array for pcolormesh() plotting
        # Shape of augmented grid array
        augmentedGridShape = (gridShape2D[0]+1, gridShape2D[1]+1)
        # Augment Phi grid array
        kPhiAug = np.empty(augmentedGridShape)
        kPhiAug[slice(gridShape2D[0]),slice(gridShape2D[1])] = kPhiGrid2D
        kPhiAug[-1,:] = kPhiAug[-2,:]
        kPhiAug[:,-1] = 2*np.pi
        # Augment k grid array
        kAug = np.empty(augmentedGridShape)
        kAug[1:augmentedGridShape[0], 1:augmentedGridShape[1]] = kGrid2D
        kAug[0,:] = 0
        kAug[1:,0] = kGrid2D[:,0]


        ## Plot PhotoElectronSpectra using pcolormesh()
        cmap = 'jet'
        figsize = (10,10)
        if 'figsize' in kwargs.keys():
            figsize = kwargs.pop('figsize')
        if 'cmap' in kwargs.keys():
            cmap = kwargs.pop('cmap')

        # [NOTE 171211] Should be chanced
        plt.rcParams['xtick.labelsize'] = 13
        plt.rcParams['ytick.labelsize'] = 13
        plt.rcParams['axes.labelsize'] = 20

        if fig is None: fig = plt.figure(figsize=figsize)
        if ax is None: ax = fig.gca()


        # [NOTE 171102] TODO, let radius be determined automatically by referencing k-max-surff
        radius = 2

        ax.set_xlim(-radius, radius)
        ax.set_ylim(-radius, radius)

        if log_scale:
            C = kProbGrid2D.copy()
            zero_prevention = 1e-50
            C[C == 0] += zero_prevention
            #if log_min==None: log_min=C.min()
            #if log_max==None: log_max=C.max()
            if (vmin is None) and (log_min is not None): vmin = log_min
            if (vmax is None) and (log_max is not None): vmax = log_max

            mesh = ax.pcolormesh(kAug*np.cos(kPhiAug), kAug*np.sin(kPhiAug), C,
                cmap=cmap, norm=colors.LogNorm(vmin=vmin, vmax=vmax), **kwargs)

        else:
            mesh = ax.pcolormesh(kAug*np.cos(kPhiAug), kAug*np.sin(kPhiAug), kProbGrid2D,
                norm = colors.Normalize(vmin=vmin, vmax=vmax), cmap=cmap, **kwargs)

        # Make it to be seen as equal tick spacing
        ax.axis('square')

        ## Set axis labels
        ax.set_xlabel(r'$p_x$ (a.u.)')
        ax.set_ylabel(r'$p_y$ (a.u.)')

        ## Set colorbar
        cb = fig.colorbar(mesh, ax=ax)
        cb.ax.get_yaxis().labelpad = 20
        cb.set_label('differential probability', rotation=270)

        ## Save if specified
        if saveFigureName != '':
            fig.savefig(saveFigureName)

        ## Return ax object in case of further manipulation
        return fig, ax


    def __mul__(self, multiplier):
        ## Check prerequisites
        assert self.haveReadPolarSpectrum

        ## Instantiate new object which is going to be the result
        new_obj = self.copy()

        ## Classify the case of multiplier
        if type(multiplier) is MomentumSpectrumPolar:
            assert multiplier.haveReadPolarSpectrum
            new_obj.polarSpectrum['total_prob'] *= multiplier.polarSpectrum['total_prob']
        else:
            try: scalar = float(multiplier)
            except: raise ValueError("multiplier should be of numeric type")
            new_obj.polarSpectrum['total_prob'] *= scalar

        ## Turn up the flag
        new_obj.haveReadPolarSpectrum = True

        ## Return the added object
        return new_obj

    def __rmul__(self, multiplier):
        return self.__mul__(multiplier)

    def __truediv__(self, divider):
        ## Check prerequisites
        assert self.haveReadPolarSpectrum

        ## Instantiate new object which is going to be the result
        new_obj = self.copy()

        ## Classify the case of divider
        if type(divider) is MomentumSpectrumPolar:
            assert divider.haveReadPolarSpectrum
            new_obj.polarSpectrum['total_prob'] /= divider.polarSpectrum['total_prob']
        else:
            try: scalar = float(divider)
            except: raise ValueError("a divider should be of numeric type")
            new_obj.polarSpectrum['total_prob'] /= scalar

        ## Turn up the flag
        new_obj.haveReadPolarSpectrum = True

        ## Return the added object
        return new_obj

    def __rtruediv__(self, divider):
        ## Check prerequisites
        assert self.haveReadPolarSpectrum

        ## Instantiate new object which is going to be the result
        new_obj = self.copy()

        ## Classify the case of divider
        if type(divider) is MomentumSpectrumPolar:
            assert divider.haveReadPolarSpectrum
            new_obj.polarSpectrum['total_prob'] = \
                divider.polarSpectrum['total_prob'] / self.polarSpectrum['total_prob']
        else:
            try: scalar = float(divider)
            except: raise ValueError("a divider should be of numeric type")
            new_obj.polarSpectrum['total_prob'] = scalar / self.polarSpectrum['total_prob']

        ## Turn up the flag
        new_obj.haveReadPolarSpectrum = True

        ## Return the added object
        return new_obj



    def __add__(self, another_polar_spectrum):
        ## Check prerequisites
        assert type(another_polar_spectrum) is MomentumSpectrumPolar
        assert self.haveReadPolarSpectrum
        assert another_polar_spectrum.haveReadPolarSpectrum

        ## Add polar spectrum
        added_polar_spectrum = self.copy()
        added_polar_spectrum.polarSpectrum['total_prob'] += another_polar_spectrum.polarSpectrum['total_prob']

        ## Turn up the flag
        added_polar_spectrum.haveReadPolarSpectrum = True

        ## Return the added object
        return added_polar_spectrum

    def __sub__(self, another_polar_spectrum):
        return self + ( -1 * another_polar_spectrum )

    def abs(self):
        abs_spectrum = self.copy()
        abs_spectrum.polarSpectrum['total_prob'] = self.polarSpectrum['total_prob'].abs()
        return abs_spectrum

    def copy(self):
        new_obj = MomentumSpectrumPolar()

        new_obj.polarSpectrum = self.polarSpectrum.copy()
        new_obj.haveReadPolarSpectrum = True
        new_obj.qprop_dim = self.qprop_dim
        if new_obj.qprop_dim is not None:
            new_obj.set_polar_k_grid()

        return new_obj


    def get_reduced_k_arr_for_pz_average(self, z0):
#        if not self.haveReadPolarSpectrum: self.readPolarSpectrumData(self.q)
        self.check_and_or_load()
        _max_k = np.sqrt(self.k_values[-1]**2 - z0**2)
        _reduced_k_arr = self.k_values[self.k_values < _max_k]
        return _reduced_k_arr

    def get_pcolor_grid_for_pz_average_map(self, z0):
        self.check_and_or_load()
        _reduced_k_arr = self.get_reduced_k_arr_for_pz_average(z0)
        return self.construct_polar_pcolor_grid_arr(_reduced_k_arr, self.phi_values)
        

    def evaluate_pz_averaged_map(self, z0):
        """
        Evaluate the momentum map, averaged along pz

        # Argument
        - `z0` : the range of average for momentum in the direction of z
        """

        self.check_and_or_load()
        
        from scipy.interpolate import RegularGridInterpolator
        _P_func_interp = RegularGridInterpolator((self.k_values, self.theta_values, self.phi_values), self.kProbGrid)
        
        _reduced_k_arr = self.get_reduced_k_arr_for_pz_average(z0)
        _reduced_K_arr, _Phi_arr = np.meshgrid(_reduced_k_arr, self.phi_values, indexing='ij')
        
        from tdse.coordinate import P_bar_vec
        _P_bar_arr = P_bar_vec(_reduced_K_arr, _Phi_arr, _P_func_interp, z0=z0)	

        return _P_bar_arr 


    @staticmethod
    def is_the_proper_phi_arr(phi_arr):
        _is_equidistanced = np.std(np.diff(phi_arr)) < 1e-14
        _start_from_zero = phi_arr[0] == 0
        _is_periodic = np.abs(phi_arr[-1] + phi_arr[1] - 2.0*np.pi) < 1e-14
        return _is_equidistanced and _start_from_zero and _is_periodic

    @staticmethod
    def construct_polar_pcolor_grid_arr(k_arr, phi_arr):

        assert k_arr[0] == 0.0
        assert MomentumSpectrumPolar.is_the_proper_phi_arr(phi_arr)

        _k_grid_arr = np.empty((k_arr.size + 1,), dtype=float)
        _k_grid_arr[0] = 0.0
        _k_grid_arr[1:-1] = 0.5 * (k_arr[:-1] + k_arr[1:])
        _k_grid_arr[-1] = k_arr[-1] + 0.5 * (k_arr[-1] - k_arr[-2])
        
        _phi_grid_arr = np.empty((phi_arr.size + 1,), dtype=float)
        _delta_phi = phi_arr[1] - phi_arr[0]
        _phi_grid_arr[:-1] = phi_arr - 0.5 * _delta_phi
        _phi_grid_arr[-1] = phi_arr[-1] + 0.5 * _delta_phi
        
        _K_grid_arr, _Phi_grid_arr = np.meshgrid(_k_grid_arr, _phi_grid_arr, indexing='ij')
        _X_grid_arr = _K_grid_arr * np.cos(_Phi_grid_arr)
        _Y_grid_arr = _K_grid_arr * np.sin(_Phi_grid_arr)

        return _X_grid_arr, _Y_grid_arr

    def get_my_pcolor_grid_arrays(self):
        self.check_and_or_load()
        return self.construct_polar_pcolor_grid_arr(self.k_values, self.phi_values)


    def get_prob_arr_at_polarization_plane(self):
        self.check_and_or_load()
        _half_pi_theta_index = self.theta_values.size // 2
        _prob_arr_at_pol_plane = self.kProbGrid[:,_half_pi_theta_index,:]
        return _prob_arr_at_pol_plane


