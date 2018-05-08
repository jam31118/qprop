
# coding: utf-8

#### TODO ####
## [180409] Modulize this huge module into several smaller modules and leave core objects
## such as Qprop20 in this core.py module
## [180414] Try not to use pandas since it is good to keep
## the module to be lightweight.
## [180424] Make a unified, global config dictionary variable for default values
## e.g. 'default_data_file_name' etc.
## - Also, write in docstring that the default values of an argument of the method it describes 
## .. can be found at this global dictionary. It is especially useful when the default value
## .. of the argument isn't exposed directly.



# ## TODO
# - Should compare imported data's shape with expected shape inferred from parameter files
# - reduce the name length of polarSpectrum etc.
# - save file names(I mean, the input path to files) of param and data
# - initialize by just entering folders where parameter files resides
# - After reading parameter files, infer the grid size of the wavefunctions
# .. and also the vector potential etc. for reading in wavefuntion file etc.
# .. the grid size for imaginary and real propagation are different
# .. thus, please consult with the source code
# - Include grid object, resembled with qprop::grid object
# - Grid object should be instantiated and initialized when qprop object
# .. has been made and initialized with all required parameter files
# - Also, include einstein summation and related visulzation tool
# - Implement Tensor class, inheriting numpy.ndarray

# - Replace "subplot_index" argument in e.g. 'plotPESofLine..' method to get axes object
# - Interpolation of PES spectra to get much desner and smooth image out of finite data points

## TODO in implementing object
# - Make 'Param' object, representing parameter files
# - As child class of 'Param', implement 'CommonParam' that can be used in ModumQprop etc.
# .. It gets from list of Param objects and get its common paratmeters as its own member variables
# .. and figure out the varying parameters such as CEP etc.

## TODO in Visualzation
# - Fix bugs on logscale plots in which labels of colorbar disappears when specified xlim and ylim etc.

# This module is based on QPROP 2.0 format of output files such as:
# .. tsurff-parital.dat, tsurff-polar.dat in src/ati-tsurff/
# .. spectrum_0.dat, spectrum_polar_0.dat in src/ati-winop/
# Thus, it should be fixed for other version of QPROP

import os
from os.path import isfile, isdir, abspath, join, getsize
from sys import stderr

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.figure import Figure

from vis.plot import plot_1D
#import unit
from nunit import au as unit

from .wavefunction import Wavefunction
from .vecpot import Vecpot, Loaded_Vecpot
from .grid import Grid
from .util import get_index_of_nearest_element
from .default import default_config
from .momentum import MomentumSpectrumPolar


class parser:

    # Convert types in qprop's parameter files
    # .. into python's corresponding types
    type2castFunction = {
        'double' : lambda x: float(x),
        'long' : lambda x: int(x),
        'bool' : lambda x: bool(x),
    }

    @staticmethod
    # Convert qprop's parameter files
    # .. into pandas.Series object
    def paramfile2series(param_file_name):
        # Read paramter file according to QPROP 2.0 version
        df = pd.read_table(param_file_name,
                           sep='\s+', header=None, index_col=0,
                           names=['name','type','value'], dtype='str',
                           skip_blank_lines=True, comment='#'
                          )

        # Convert values in string type into values with proper numeric types
        # .. e.g. '1.0' -> 1.0 (i.e. string to float if specified as 'double' in parameter file)
        for idx in df.index:
            df.loc[idx, 'value'] = parser.type2castFunction[df.loc[idx,'type']](df.loc[idx, 'value'])

        return df['value']



### Define Base class for QPROP calculation
class Qprop(object):
    pass



### Define QPROP version 2.0 (i.e. qprop-with-tsurff) calculation object
# The name of class 'Qprop20' stands for QPORP version 2.0
# For other version, Qprop$(major)$(minor)
# .. e.g. for QPROP ver 1.7, the class name would be qprop17
class Qprop20(Qprop):

    def __init__(self, paramFileDir='.', home=None, guess_line_num_of_wf_file=False,
        vecpot_filename='', wf_filename=''):
        """
        If Default 'paramFileDir' is current directory (designated by '.')
        If 'home' isn't given, 'paramFileDir' becomes the home directory of this calculation object 'qprop20'
        """
        ## Check input parameters
        assert type(vecpot_filename) is str
        assert type(wf_filename) is str

        ## Try to change the input paramFileDir into python string type
        try:
            paramFileDir = str(paramFileDir)
        except:
            raise TypeError("Parameter file directory should be given as %s" % str)

        ## Construct list of filepaths (either relative or absolute path)
        #if type(str(paramFileDir)) == str:
        if (not os.path.exists(paramFileDir)) or (not os.path.isdir(paramFileDir)):
            raise IOError("Please give proper directory name for 'paramFileDir'")
        fileListIncludingParamFiles = []
        for fileName in os.listdir(paramFileDir):
            fileListIncludingParamFiles.append(os.path.join(paramFileDir,fileName))
        #else:
        #    raise TypeError("Parameter file directory should be given as %s" % str)

        ## Setting home directory of this home file
        if home is not None: self.home = home
        else: self.home = paramFileDir
        self.home = abspath(self.home)
        assert isdir(self.home)


        ## Support both list and dict type
        supportedDataType = [list, dict]
        if type(fileListIncludingParamFiles) not in supportedDataType:
            raise IOError("Please give me data one of a type in %s" % supportedDataType)

        ## Search and parse parameter files
        self.paramFileList = []
        for filePath in fileListIncludingParamFiles:
            fileName = os.path.split(filePath)[1]
            if fileName == 'initial.param':
                self.initialParam = parser.paramfile2series(filePath)
                self.initialParamFilePath = filePath
                self.paramFileList.append(fileName)
                #print("Reading %s ..." % fileName)
            elif fileName == 'propagate.param':
                self.propagateParam = parser.paramfile2series(filePath)
                self.propagateParamFilePath = filePath
                self.paramFileList.append(fileName)
                #print("Reading %s ..." % fileName)
            elif fileName == 'tsurff.param':
                self.tsurffParam = parser.paramfile2series(filePath)
                self.tsurffParamFilePath = filePath
                self.paramFileList.append(fileName)
                #print("Reading %s ..." % fileName)
            elif fileName == 'winop.param':
                self.winopParam = parser.paramfile2series(filePath)
                self.winopParamFilePath = filePath
                self.paramFileList.append(fileName)
                #print("Reading %s ..." % fileName)

        # Check whether all required parameter files have been read
        if 'initial.param' not in self.paramFileList:
            #raise IOError("No initial parameter file")
            raise NoParamFileError('initial.param')
        if 'propagate.param' not in self.paramFileList:
            #raise IOError("No propagate parameter file")
            raise NoParamFileError('propagate.param')
        if ('tsurff.param' not in self.paramFileList) and ('winop.param' not in self.paramFileList):
            #raise IOError("No tsurff paramter file or winop file")
            raise NoParamFileError(['tsurff.param', 'winop.param'], mode='or')

        ## Set several important parameters as member variables
        self.dimension = self.initialParam['qprop-dim']

        ## Instantiate Grid object
        ## [180213 NOTE] 'r_grid_size' and related variables are misleading
        ## .. since the 'r_grid_size' can vary from time to time
        ## .. e.g. in tsurff-mode, 'r_grid_size' may be either imag-width + R-tsurff or imag-width + R-tsurff + quiver-amplitude
        if guess_line_num_of_wf_file:
            if 'tsurff.param' in self.paramFileList:
                r_grid_size = self.propagateParam['imag-width'] + self.tsurffParam['R-tsurff']
            elif 'winop.param' in self.paramFileList:
                r_grid_size = self.propagateParam['imag-width'] + self.propagateParam['R-max']

            numOfRadialGrid = int(r_grid_size / self.initialParam['delta-r'])
            print("The number of radial grid poinst has been estimated and thus may not be accurate.",
                    file=stderr)

        else:
            ## Determine name of wavefunction data file to be used for estimating radial grid number.
            if wf_filename == '':
                contents_in_home_dir = os.listdir(self.home)
                for wf_file_name in default_config['wf_file_name']:
                    if wf_file_name in contents_in_home_dir:
                        wf_filename = join(self.home, wf_file_name)
                        break
            else: assert isfile(wf_filename)
            
            # [180219 NOTE] This should be fixed to more reliable method!
            # .. since the number of lines in 'hydrogen_re-wf.dat' changes during the calculation.
            # .. Thus, if user want to analyze some intermediate data (such as initial wavefunction),
            # .. misleading number of entry (and thus 'numOfRadialGrid') is obtained
            # .. which would spoil the analysis.
#            wf_filename = self.home + '/hydrogen_re-wf.dat'
#            wf_filename = os.path.abspath(wf_filename)

            #wf_raw = np.genfromtxt(wf_filename, delimiter=' ', dtype=np.double)
            #wf_data_1d_array = np.apply_along_axis(lambda arr: complex(*arr), 1, wf_raw)
            #del wf_raw
            #num_of_entry = wf_data_1d_array.shape[0]

#            with open(wf_filename) as f:
#                for i, l in enumerate(f): pass
#            num_of_entry = i + 1
#            assert num_of_entry == int(num_of_entry)
#            num_of_entry = int(num_of_entry)

            num_of_entry_per_wavefunction = self._get_num_of_complex_number_in_file(wf_filename)
#            num_of_entry_per_wavefunction = num_of_entry / 2
            if self.initialParam['qprop-dim'] == 34:
                numOfRadialGrid = num_of_entry_per_wavefunction / self.propagateParam['ell-grid-size']
            if self.initialParam['qprop-dim'] == 44:
                numOfRadialGrid = num_of_entry_per_wavefunction / (self.propagateParam['ell-grid-size'] ** 2)

        self.grid = Grid(numOfRadialGrid, self.propagateParam['ell-grid-size'],1, self.initialParam['qprop-dim'])
        self.grid.set_delta_r(self.initialParam['delta-r'])
        self.grid.set_initial_m(self.initialParam['initial-m'])


        ## Instantiate Grid for imaginary propagation
        numOfRadialGrid_initial = int(self.initialParam['radial-grid-size'] / self.initialParam['delta-r'])
        self.grid_initial = Grid(numOfRadialGrid_initial, self.initialParam['ell-grid-size'], 1, self.initialParam['qprop-dim'])
        self.grid_initial.set_delta_r(self.initialParam['delta-r'])
        self.grid_initial.set_initial_m(self.initialParam['initial-m'])


        ## Instantiate Vecpot(vector potential) object
        firstVecpotParamNames = ['omega','num-cycles','max-electric-field']
        secondVecpotParamNames = ['omega2','num-cycles2','max-electric-field2']

        haveFirstVecpotParam = all([paramName in self.propagateParam.index for paramName in firstVecpotParamNames])
        haveSecondVecpotParam = all([paramName in self.propagateParam.index for paramName in secondVecpotParamNames])

        firstVecpotParam = None
        secondVecpotParam = None
        if (haveFirstVecpotParam):
            firstVecpotParam = [self.propagateParam[paramName] for paramName in firstVecpotParamNames]
            if 'cep' in self.propagateParam.index:
                firstVecpotParam.append(self.propagateParam['cep'])
            else:
                firstVecpotParam.append(None) # for putting phi_cep as None
        if (haveSecondVecpotParam):
            secondVecpotParam = [self.propagateParam[paramName] for paramName in secondVecpotParamNames]
            if 'cep2' in self.propagateParam.index:
                secondVecpotParam.append(self.propagateParam['cep2'])
            else:
                secondVecpotParam.append(None)
        else:
            secondVecpotParam = [None, None, None, None]

        if 'delay' in self.propagateParam.index:
            delay = self.propagateParam['delay']
        else:
            delay = 0.0
        
        self.vecpot = None
#        if firstVecpotParam is None:
        if vecpot_filename == '':
            vecpot_filename = default_config['vecpot_file_name']
        else:
            if not isfile(vecpot_filename):
                vecpot_prefixed_filename = join(self.home, vecpot_filename)
                if isfile(vecpot_prefixed_filename):
                    vecpot_filename = vecpot_prefixed_filename
                else:
                    raise IOError("Couldn't find vector potential data file: {0}".format(vecpot_filename))
        

        if isfile(vecpot_filename):
            try: self.vecpot = Loaded_Vecpot(self.dimension, vecpot_filename)
            except: pass
        else:
            print("Couldn't find vector potential data file: {0}".format(vecpot_filename), file=stderr)

        if self.vecpot is None:
            print("[ERROR] Couldn't get vecter potential parameter. "
                  + "The vecpot object will not be available.", file=stderr)
            print("[ERROR] Please try supplying `vecpot_filename` as an argument of {0}".format(type(self)),
                  file=stderr)
#            print("Generic, multiple vector potential case should be implemented",
#                 file=stderr)

#        else:
#            self.vecpot = Vecpot(self.initialParam['qprop-dim'], self.home,
#                                 *firstVecpotParam,
#                                 *secondVecpotParam,
#                                 delay=delay)

        ## Define flags for information to methods
        #self.haveReadPolarSpectrum = False
        self.haveReadPartialSpectrum = False

        ## Define Momentum polar spectrum
        self.momentum_spectrum_polar = MomentumSpectrumPolar(self)

    def _get_num_of_complex_number_in_file(self, wf_filename, size_of_complex_num=None):
        assert isfile(wf_filename)
        num_of_complex_num = None
        if self._seems_binary_wf_file(wf_filename):
            size_of_complex_num = self.get_size_of_complex_number(size_of_complex_num)
            file_size = getsize(wf_filename)
            num_of_complex_num = int(file_size // size_of_complex_num)
            assert (file_size % size_of_complex_num) == 0
        else:
            with open(wf_filename) as f:
                for i, l in enumerate(f): pass
            num_of_complex_num = i + 1
            assert num_of_entry == int(num_of_complex_num)
            num_of_complex_num = int(num_of_complex_num)
        return num_of_complex_num

    def _seems_binary_wf_file(self, filepath):
        file_size = None
        if isfile(filepath): file_size = getsize(filepath)
        else: raise IOError("Cannot find file: {0}".format(filepath))

        size_of_file_can_be_factored_into_grid_dimension = \
                self._can_be_factored_into_grid_dimensions(file_size)
        seems_binary_wf_file = size_of_file_can_be_factored_into_grid_dimension
        return seems_binary_wf_file
    
    def _can_be_factored_into_grid_dimensions(self, size_in_bytes, size_of_complex_num=None):
        """Check whether the given size is dividable by grid sizes.
        The grid sizes include the number of radial grid points
        and ell_m_grid_size in qprop."""
        size_of_complex_num = self.get_size_of_complex_number(size_of_complex_num)
        ell_m_grid_size = self._get_ell_m_grid_size()

#        can_be_factored = True  # initialization
#
#        num_of_complex_number = size_in_bytes // size_per_complex_num
#        num_of_raidal_grid_point = num_of_complex_number // ell_m_grid_size
#
#        can_be_factored &= (size_in_bytes % size_per_complex_num) == 0
#        can_be_factored &= (num_of_complex_number % ell_m_grid_size) == 0
        remainder = size_in_bytes % (ell_m_grid_size * size_of_complex_num)
        can_be_factored = remainder == 0
        return can_be_factored

    @staticmethod
    def get_size_of_complex_number(size_of_complex_num=None):
        if size_of_complex_num is None:
            size_of_complex_num = default_config['size_of_complex_number']
        else:
            size_of_complex_num_int = int(size_of_complex_num)
            assert size_of_complex_num == size_of_complex_num_int
            size_of_complex_num = size_of_complex_num_int
        return size_of_complex_num

    def _get_ell_m_grid_size(self):
        """Return the number of basis (spherical harmonics) used in wavefunction expansion."""
        qprop_dim = self._get_qprop_dim()
        ell_grid_size = self.get_parameter('propagate.param', 'ell-grid-size')
        ell_m_grid_size = None
        if qprop_dim == 34:
            ell_m_grid_size = ell_grid_size
        elif qprop_dim == 44:
            ell_m_grid_size = ell_grid_size ** 2
        else: raise Exception("Unexpected qprop dimension: {0}".format(qprop_dim))
        return ell_m_grid_size

    def _get_qprop_dim(self):
        """Return grid dimension used in qprop"""
        return self.get_parameter('initial.param', 'qprop-dim')

    def get_parameter(self, param_file_name, param_name):
        param_series = None
        if param_file_name == "initial.param":
            param_series = self.initialParam
        elif param_file_name == "propagate.param":
            param_series = self.propagateParam
        elif param_file_name == "tsurff.param":
            param_series = self.tsurffParam
        elif param_file_name == "winop.param":
            param_series = self.winopParam

        assert param_name in param_series.index
        param = param_series[param_name]
        return param

    def _getPartialProbSubColNames(self, ell_grid_size_source='propagate.param'):
        if ell_grid_size_source == 'propagate.param':
            ell_grid_size = self.propagateParam['ell-grid-size']
        elif ell_grid_size_source == 'initial.param':
            ell_grid_size = self.initialParam['ell-grid-size']
        else:
            raise Exception("unsupported ell_grid_size_source")

        partial_prob_subColName_format = "l=%d,m=%d"
        partial_prob_subColNames = []

        if self.initialParam['qprop-dim'] == 34:
            m0 = self.initialParam['initial-m']
            for l in range(ell_grid_size):
                subColName = partial_prob_subColName_format % (l,m0)
                partial_prob_subColNames.append(subColName)

        elif self.initialParam['qprop-dim'] == 44:
            for l in range(ell_grid_size):
                for m in range(-l,l+1): # from m = -l, -l+1, ... , l-1, l
                    subColName = partial_prob_subColName_format % (l,m)
                    partial_prob_subColNames.append(subColName)

        else:
            raise IOError("Input dimension(=%d) is neither 34 nor 44." % (initialParam['qprop-dim']))

        return partial_prob_subColNames


    def readPartialSpectrumData(self, dataFileName='', ell_grid_size_source='propagate.param'):
        if dataFileName == '':
            # Check if this calculation uses tsurff
            if 'tsurff.param' in self.paramFileList:
                dataFileNameGuessed = os.path.join(self.home,'tsurff-partial.dat')
            elif 'winop.param' in self.paramFileList:
                dataFileNameGuessed = os.path.join(self.home, 'winop-partial.dat')
            else:
                raise IOError("Please Give an explicit dataFileName if you don't use tsurff method")

            if os.path.exists(dataFileNameGuessed):
                dataFileName = dataFileNameGuessed
            else:
                raise IOError("No file is found with name: %s" % dataFileNameGuessed)

        # Check whether the specified parameter file name exists
        if not os.path.exists(dataFileName):
            raise IOError("No file is found with name: %s" % dataFileName)

        data = pd.read_table(dataFileName, sep='\s+', header=None, skip_blank_lines=True, dtype=np.float64)

        primaryCol2subCol = {
            "energy" : [""],
            "k" : [""],
            "partial_prob" : self._getPartialProbSubColNames(ell_grid_size_source),
            "total_prob" : [""]
        }

        primaryColsInOrder = ["energy","k","partial_prob","total_prob"]

        for primaryCol in primaryColsInOrder:
            if primaryCol not in primaryCol2subCol.keys():
                #print(primaryCol)
                raise ValueError("Check whether primaryCol2subCol.keys() and primaryColsInOrder identical.")

        columns = []
        for primaryCol in primaryColsInOrder:
            for subCol in primaryCol2subCol[primaryCol]:
                columns.append((primaryCol,subCol))

        #print(columns)

        data.columns = pd.MultiIndex.from_tuples(columns)
        data.sort_values(by='k', inplace=True)

        self.partialSpectrum = data
        #return data

        ## Turn on the flag for completion of reading
        self.haveReadPartialSpectrum = True



    def get_ell_spectrum(self, wf_filename=None, wf_index=1):
        """

        ## Arguments ##
        # wf_index
        : an index of the target wavefunction in the wavefunction data file.
        - default is '1', which generally corresponds to final wavefunction.
        """
        ## Process input arguments
        if wf_filename is not None:
            assert type(wf_filename) is str
        else:
            abs_path_home = os.path.abspath(self.home)
            default_data_file_name = 'hydrogen_re-wf.dat'
            default_wf_file_abs_path = os.path.join(abs_path_home, default_data_file_name)
            #default_wf_filename = self.home + '/' + 'hydrogen_re-wf.dat'
            wf_filename = default_wf_file_abs_path
        assert os.path.isfile(wf_filename)

        assert int(wf_index) == wf_index
        wf_index = int(wf_index)

        wf = Wavefunction(self.grid)
        try: wf.load(wf_filename, wf_index)
        except: raise Exception("The wavefunction couldn't be loaded.")

        wf_sq = np.abs(wf.data_2d)
        wf_sq = np.square(wf_sq)
        wf_sq_spectrum = np.sum(wf_sq, axis=1)

        return wf_sq_spectrum


    def plot_ell_spectrum(self, cache_spectrum=True, wf_filename=None, wf_index=None, **plot_kwargs):
        """Plot norm-squared amplitude of each basis.

        ## Arguments
        # `wf_filename` - None or str
        : A path to the wavefunction data file from which each basis amplitude is read.
        
        """
        ## Check input arguments
        get_ell_spectrum_kwargs = {}
        if wf_filename is not None: get_ell_spectrum_kwargs['wf_filename'] = wf_filename
        if wf_index is not None: get_ell_spectrum_kwargs['wf_index'] = wf_index

        assert type(cache_spectrum) is bool


        wf_sq_spectrum = None
        fresh_load = True
        if cache_spectrum:
            fresh_load = False
            if hasattr(self, 'ell_spectrum_is_cached'):
                if self.ell_spectrum_is_cached:
                    wf_sq_spectrum = self.cached_ell_spectrum
                else: fresh_load = True
            else: fresh_load = True
        else: assert fresh_load is True

        if fresh_load:
            print("Reading wavefunction data . . . ", end='', flush=True)
            wf_sq_spectrum = self.get_ell_spectrum(**get_ell_spectrum_kwargs)
            print("done", flush=True)
            if cache_spectrum:
                self.cached_ell_spectrum = wf_sq_spectrum
                self.ell_spectrum_is_cached = True
            else: self.ell_spectrum_is_cached = False

        #wf_sq_spectrum = self.get_ell_spectrum(**get_ell_spectrum_kwargs)
        N = wf_sq_spectrum.size
        ell_indices = np.arange(N)

        xlabel = r'$ell\,index$'
        ylabel = ''
        if 'log_scale' in plot_kwargs.keys():
            log_scale = plot_kwargs.pop('log_scale')
            if log_scale:
                non_zero_min = wf_sq_spectrum[wf_sq_spectrum != 0].min()
                wf_sq_spectrum[wf_sq_spectrum == 0] = non_zero_min
                wf_sq_spectrum = np.log10(wf_sq_spectrum)
                ylabel = r"$log_{10}(amplitude)$"
            else:
                ylabel = r"$amplitude$"

        return plot_1D(ell_indices, wf_sq_spectrum, xlabel=xlabel, ylabel=ylabel, **plot_kwargs)


    def plotEnergySpectrum(self, ax=None, fig=None, logScale=True, x_unit='au', **kwargs):

        ## Check whether partial spectrum data has been read
        if not self.haveReadPartialSpectrum:
            try: self.readPartialSpectrumData()
            except: raise IOError("No data to plot. Please read partial spectrum data first")

        ## Default values
        default_figsize = (15,5)

        ## Determine figure object on which data is plotted
        if fig is None:
            # Determine figsize (size of figure)
            if 'figsize' in kwargs:
                figsize = kwargs['figsize']
            else: figsize = default_figsize
            # Instantiate figure object
            fig = plt.figure(figsize=figsize)

        elif type(fig) is Figure:
            pass
            #fig = figure

        else: raise TypeError("Unsupported type of figure: %s" % (type(fig)))

        ## Determine axis object on which data is plotted
        if ax is None:
            ax = fig.gca()

        ## Prepare data
        # Set label and data according to unit
        if x_unit.lower() == 'au'.lower():
            ax.set_xlabel('energy (a.u.)')
            x = self.partialSpectrum['energy']
        elif x_unit.lower() == 'eV'.lower():
            ax.set_xlabel('energy (eV)')
            x = self.partialSpectrum['energy'] * unit.au2si['energy'] / unit.e

        y = self.partialSpectrum['total_prob']
        ax.set_ylabel('Differential Probability')


        ## Actual plotting
        if logScale:
            ax.semilogy(x,y)
        else:
            ax.plot(x,y)

        ## Return plotting objects
        return fig, ax


    def plotPartialSpectrums(self, saveFigure=False, saveFigureName='', logScale=False, maxNumOfPlotToDisplay=10, xlim=(None,None), ylim=(None,None)):
        ## Check prerequisites
        # Check whether there is partial spectrum by checking spectrum-related parameters
        if 'tsurff.param' in self.paramFileList:
            if 'expansion-scheme' in self.tsurffParam.index:
                if self.tsurffParam['expansion-scheme'] == 1:
                    raise Warning("expansion-scheme in tsurff.param is set to 1, which implies no partial spectrum data had been generated")
            else:
                raise KeyError("No key with name %s in %s" % ("expansion-scheme", "tsurff.param"))

        # Check whether partial spectrum data has been read
        if not self.haveReadPartialSpectrum:
            try:
                self.readPartialSpectrumData()
            except:
                raise IOError("No data to plot. Please read partial spectrum data first")

        ## Set configurations
        #maxNumOfPlotToDisplay = 10
        totalNumOf_ell_m_unified_indice = self.grid.sizeOf_ell_m_unified_grid()
        listOf_ell_m_unified_indice = [int(idx) for idx in np.linspace(0,totalNumOf_ell_m_unified_indice - 1,maxNumOfPlotToDisplay)]

        ## Set shape of grid of subplots
        numOfRowInfigGrid = int(np.ceil(maxNumOfPlotToDisplay // 2))
        figGridShape = (numOfRowInfigGrid, 2)

        ## Set figure size of whole grid
        figSize = (figGridShape[1] * 8, figGridShape[0] * 5)

        ## Plot selected partial spectrum
        x = self.partialSpectrum.loc[:,'energy']
        fig, ax = plt.subplots(*figGridShape, figsize=figSize)
        fig.subplots_adjust(hspace=0.3)
        for rowIdx in range(figGridShape[0]):
            for colIdx in range(figGridShape[1]):
                totalIdx = rowIdx * figGridShape[1] + colIdx
                indexOfFigure = listOf_ell_m_unified_indice[totalIdx]
                multiIndex = ('partial_prob', self.partialSpectrum['partial_prob'].columns[indexOfFigure])
                y = self.partialSpectrum.loc[:,multiIndex]
                current_ax = ax[rowIdx, colIdx]

                if logScale:
                    current_ax.semilogy(x,y)
                    #current_ax.set_ylim(log_min, log_max)
                else:
                    current_ax.plot(x,y)

                current_ax.set_xlim(*xlim)
                current_ax.set_ylim(*ylim)

                current_ax.set_xlabel('electron energy (a.u.)')
                current_ax.set_ylabel('partial probability')
                text = r'$' + y.name[1] + r'$'
                current_ax.text(0.70, 0.85, text, fontsize=15, transform=current_ax.transAxes)

        ## Save figure
        if saveFigure:
            if saveFigureName == '':
                if logScale: saveFigureName = 'partial_spectrums_inLogScale.png'
                else: saveFigureName = 'partial_spectrums.png'
            plt.savefig(saveFigureName)


    ## [180115-2032 NOTE] Unusable functionality! Implement this method in momentum_spectrum_polar object!
    def plotPESofLinearlyPolarizedLight(self, saveFigureName='', rcParams={}, fig=None, subplot_index=(),
        inLogScale=True, log_min=None, log_max=None, **kwargs):

        ## Check prerequisites
        if not self.momentum_spectrum_polar.haveReadPolarSpectrum:
            try: self.momentum_spectrum_polar.readPolarSpectrumData(self)
            except: raise IOError("No data to plot. Please read polar spectrum data first, possibly by using readlPolarSpectrumData()")

        ## Data Preparation
        X = self.kGrid * np.cos(self.kThetaGrid)
        Y = self.kGrid * np.sin(self.kThetaGrid)
        C = self.kProbGrid

        ## Process input arguments for plotting
        # Process figure size 'figsize'
        if 'figsize' in kwargs.keys():
            figsize = kwargs.pop('figsize')
        elif 'figsize' in rcParams.keys():
            figsize = rcParams.pop('figsize')
        else:
            # Set default figure size
            # [NOTE] In case of PES with linearly polarized light, 2:1 ratio is advisable
            figsize = (16,8)
        # Process coloarmap ('cmap')
        if 'cmap' in kwargs.keys():
            cmap = kwargs.pop('cmap')
        else:
            # Set default color map
            cmap = 'jet'

        # Update rcParams according to the 'rcParams' argument
        plt.rcParams.update(rcParams)

        ## Set figure object
        if fig == None:
            fig = plt.figure(figsize=figsize)
        elif type(fig) == Figure:
            # Check if 'fig' is of type 'matplotlib.figure.Figure'
            fig = fig

        ## Set projection mode
        if self.initialParam['qprop-dim'] == 34:
            projection = None
        else:
            raise IOError("Unsupported qprop_dimension for this method: %d\n" % (self.qprop_dim))

        ## Set Axes object
        if (type(subplot_index) != tuple):
            # In case where subplot_index is given in a form of scalar
            # .. such as '111' which is equivalent to '1,1,1'
            ax = fig.add_subplot(subplot_index, projection=projection)
        elif (type(subplot_index) == tuple) and (len(subplot_index) != 0):
            # In case where subplot_index is given in a form of tuple
            # .. such as '(1,1,1)' which will be unpacked when passing to fig.add_subplot()
            ax = fig.add_subplot(*subplot_index, projection=projection)
        else:
            # If no subplot_index is given, a new instance of axes object is generated
            ax = fig.gca(projection=projection)

        ## Plotting
        if inLogScale:
            # 1e-20 is added for preventing division by zero
            if log_min == None: log_min = C.min() + 1e-20
            if log_max == None: log_max = C.max()
            pcm = ax.pcolormesh(X,Y,C, cmap=cmap, **kwargs, norm=colors.LogNorm(vmin=log_min, vmax=log_max))
        else:
            pcm = ax.pcolormesh(X,Y,C, cmap=cmap, **kwargs)

        # Set colorbar
        cb = fig.colorbar(pcm, ax=ax, extend='min')
        cb.ax.get_yaxis().labelpad = 20
        cb.set_label('probability', rotation=270)

        # Set x,y limits
        plt.axis('square')
        r = self.tsurffParam['k-max-surff']
        ax.set_xlim(-r,r)
        ax.set_ylim(0,r)

        # Labeling
        ax.set_xlabel('parellel momentum (a.u.)')
        ax.set_ylabel('perpendicular momentum (a.u.)')

        ## Save figure
        if saveFigureName != '':
            plt.savefig(saveFigureName)

        ## Return ax object in case of further manipulation
        return fig, ax, pcm, cb


    ### Calaulting radial mean asymmetric coefficient (or parameter)
    def getRadialAsymmetricParam(self, halfThetaOfInterest = np.pi / 6):
        """
        'q' implies Qprop object from which radial asymmetric parameters is calculated

        'halfThetaOfInterest': float, default numpy.pi / 6
        When calculating asymmetric parameters,
        only the spectra with theta range [0,'halfThetaOFInterest']
        and [pi - 'halfThetaOfInterest', pi] is included

        [NOTE 171018] The exact value of halfThetaOfInterest should be deteremined
        # .. by consulting related literature more carefully

        [NOTE 171019] Asymmetric paramter for not-linearly-polarized light isn't implemented yet
        """

        if self.initialParam['qprop-dim'] != 34:
            raise NotImplementedError("Asymmetric paramter for not-linearly-polarized light hasn't been implemented yet")

        ## Check prerequisites
        if not self.haveReadPolarSpectrum:
            try:
                self.readPolarSpectrumData()
            except:
                raise IOError("No data to plot. Please read polar spectrum data first")

        ## Construct boolean mask for selecting region(range of data) of interest
        thetaValues = np.linspace(0, np.pi, self.tsurffParam['num-theta-surff'])
        thetaGridMaskOfRightRegionOfInterest = thetaValues <= halfThetaOfInterest
        thetaGridMaskOfLeftRegionOfInterest = thetaValues >= (np.pi - halfThetaOfInterest)

        ## Select region of interest
        spectrumOnRightRegionOfInterest = self.kProbGrid[:,thetaGridMaskOfRightRegionOfInterest]
        spectrumOnLeftRegionOfInterest = self.kProbGrid[:,thetaGridMaskOfLeftRegionOfInterest]

        ## Sum in theta direction (wavefunction probability on same radius and different angular values)
        radialSpectrumOnRightRegionOfInterest = np.apply_along_axis(sum, 1, spectrumOnRightRegionOfInterest)
        radialSpectrumOnLeftRegionOfInterest = np.apply_along_axis(sum, 1, spectrumOnLeftRegionOfInterest)

        ## Radial spectrum of interest whose first column for left, and second column for right region
        radialSpectrum = np.empty((self.tsurffParam['num-k-surff'], 2), dtype = np.float64)
        radialSpectrum[:,0] = radialSpectrumOnLeftRegionOfInterest
        radialSpectrum[:,1] = radialSpectrumOnRightRegionOfInterest

        ## Calculate asymmetric parameters
        self.radialAsymmetry = np.apply_along_axis(lambda x: asymmetricParameter(*x), 1, radialSpectrum)

        ## Check consistency
        if self.radialAsymmetry.shape != (self.tsurffParam['num-k-surff'],):
            raise InconsistencyError(self.radialAsymmetry.shape, (self.tsurffParam['num-k-surff'],))

        ## Return asymmetric parameters
        return self.radialAsymmetry

    @staticmethod
    def has_complete_param_files(calc_home_path):
        assert os.path.isdir(calc_home_path)
        common_param_files = ["initial.param","propagate.param"]
        xor_param_files = ["tsurff.param","winop.param"]
        home_contents = os.listdir(calc_home_path)
        num_of_common_param_files = 0
        for common_param_file in common_param_files:
            if common_param_file in home_contents:
                num_of_common_param_files += 1
        num_of_xor_param_files = 0
        for xor_param_file in xor_param_files:
            if xor_param_file in home_contents:
                num_of_xor_param_files += 1

        all_common_files_exists = num_of_common_param_files == len(common_param_files)
        only_one_xor_file_exist = num_of_xor_param_files == 1
        complete_set_of_param_files_exist = all_common_files_exists and only_one_xor_file_exist
        return complete_set_of_param_files_exist


    @staticmethod
    def has_files_with_extensions(calc_home_path, required_extensions=[]):
        assert os.path.isdir(calc_home_path)
        home_contents = os.listdir(calc_home_path)
        extensions = [os.path.splitext(content)[1] for content in home_contents]
        #required_extension_list = [".raw"]
        assert type(required_extensions) in [list, tuple]
        assert len(required_extensions) > 0
        #required_extension_list = required_extensions
        #print(extensions)
        has_all_required_ext = True
        for required_extension in required_extensions:
            has_this_required_ext = required_extension in extensions
            has_all_required_ext &= has_this_required_ext

        return has_all_required_ext


    @staticmethod
    def has_raw_dat_etc(calc_home_path):
        return has_files_with_extensions(calc_home_path, [".raw", ".dat"])

    @classmethod
    def seems_to_be_a_calc_home(cls, calc_home_path):
        #return cls.has_complete_param_files(calc_home_path) and cls.has_raw_dat_etc(calc_home_path)
        return cls.has_complete_param_files(calc_home_path) #and cls.has_files_with_extensions(calc_home_path, required_extensions=[".raw"])

    @classmethod
    def get_list_of_calc_homes(cls, dir_path_in, verbose=False):
        """
        Return a list of calculation home directory path

        dir_path_in : string or list of string
        """
        ## Check and process input arguments
        if type(dir_path_in) is str:
            dir_list = [dir_path_in]
        elif type(dir_path_in) is list:
            for dir_path in dir_path_in:
                assert type(dir_path) is str
            dir_list = dir_path_in

        subdir_list = []
        for dir_path in dir_list:
            if verbose: print("Finding calculation home directory from: %s" % dir_path)
            assert os.path.isdir(dir_path)
            sub_contents = os.listdir(dir_path)
            for sub_content in sub_contents:
                sub_content_path = os.path.join(dir_path,sub_content)
                if os.path.isdir(sub_content_path):
                    if cls.seems_to_be_a_calc_home(sub_content_path):
                        subdir_list.append(sub_content_path)
                    else:
                        if verbose: print("'%s' doesn't seem to be a calculation home" % (sub_content_path))
                else:
                    if verbose: print("'%s' doesn't seem to be a directory" % (sub_content_path))

        return subdir_list



fontdict_black = {
    'fontsize' : 20,
    'color' : 'white',
    'backgroundcolor' : (0,0,0,0.7),
    'family' : 'monospace',
    'va' : 'center',
    'ha' : 'center',
    'weight' : 'bold'
}



# Develop Qprop20_Grid1D object for 1-parameter analysis
## Also self.plot() method

# Develop Qprop20_GridND object for N-parameter analysis
## Plotting function may be hard to be developed, but the still, by getting 1-D or 2-D partial slices, plotting is still possible

# Develop central, base class for Qprop20_Grid objects and then implement other Qprop20_Grid1D, Qprop20_Grid2D ... by inheriting it.

class Qprop20_Grid2D(object):
    def __init__(self, param_lists, dir_list=None, param_names=None, param_filenames=None, read_polar_spectrum_file=True,
        get_param_0=None, get_param_1=None, constraints=[], verbose=False):

        ## Set the dimension of this Grid, which is defined as the number of parameters
        ## [NOTE] Here, the notion of dimension is somewhat misleading; it is not the same concept as that of abstract vector space, but just an analogy of the original concept.
        self.ndim = 2

        ### Process Input Arguments
        #
        ## Check least set of argumetns
        no_callable = (get_param_0 is None) and (get_param_1 is None) and (param_names is not None) and (param_filenames is not None)
        no_names = (get_param_0 is not None) and (get_param_1 is not None) and (param_names is None) and (param_filenames is None)
        assert no_callable or no_names
        #
        ## Check input arguments and assign input arguments into class member variables
        if param_names is not None:
            assert (len(param_names) == self.ndim) and (type(param_names) in [list,tuple])

        if param_filenames is not None:
            assert (len(param_filenames) == self.ndim) and (type(param_filenames) in [list,tuple])

        assert (len(param_lists) == self.ndim) and (type(param_lists) in [list,tuple])

        for argument in [get_param_0, get_param_1]:
            if argument is not None:
                # Check if they are callable
                if not callable(argument): raise TypeError("'%s' is not 'callable'" % argument)

        self.param_names = param_names
        self.param_filenames = param_filenames
        self.param_lists = param_lists
        self.get_param_0 = get_param_0
        self.get_param_1 = get_param_1


        assert type(constraints) in [tuple, list]
        for constraint in constraints:
            assert type(constraint) in [tuple, list]
            assert len(constraint) == 3
        self.constraints = constraints


        ## parameter grid shape is a tuple of numbers of each parameter values
        ## .. e.g. let there are two parameters: first one is 'ell-grid-size' with values [30,50,70]
        ## .. and 'initial-m' with values [0,1] is the second one.
        ## .. then the self.param_grid_shape == (3,2)
        self.param_grid_shape = tuple(map(len,param_lists))

        self.q_arr = np.empty(self.param_grid_shape, dtype=object)

        if type(dir_list) in [list,str]:
            self._construct_qprop_grid(dir_list, verbose=verbose)
            if read_polar_spectrum_file:
                try: self.read_all_polar_momentum_spectrum_data_file(verbose=verbose)
                except:
                    print("Not all polar momentum spectrum file could be read in properly.")
                    #raise Warning("Not all polar momentum spectrum file could be read in properly.")
        else:
            print("No input 'dir_list' was given. An empty qprop grid was created.")



    def _construct_qprop_grid(self, dir_list, verbose=False):

        ## Filter out only qprop calculation directories
        calc_homes = Qprop20.get_list_of_calc_homes(dir_list, verbose=verbose)

        ## Construct a list of Qprop object
        q_list = [ Qprop20(home) for home in calc_homes ]

        ## Construct Qprop 2D grid
        ## [180305 NOTE] Optimize this code, there are some repetitions
        for q in q_list:
            matching_q_exist = True
            #if q.momentum_spectrum_polar.haveReadPolarSpectrum:
            # Find an index of the first parameter
            idx0, idx1 = None, None
            if callable(self.get_param_0):
                try:
                    idx0 = get_index_of_nearest_element(self.param_lists[0], self.get_param_0(q))
                except ValueError:
                    print("The parameter couldn't be found from the array")
                    print("self.get_param_0(q): ",self.get_param_0(q))
                    print("calculation home: %s" % q.home)
                    matching_q_exist = False
                except: raise Exception("Unexpected case")
            elif self.get_param_0 is None:
                assert type(self.param_lists[0]) is list
                param0 = q.get_parameter(self.param_filenames[0], self.param_names[0])
                try:
                    if verbose: print("self.param_lists[0], param0",self.param_lists[0], param0)
                    idx0 = get_index_of_nearest_element(self.param_lists[0], param0)
                except ValueError:
                    print("The parameter couldn't be found from the array")
                    print("calculation home: %s" % q.home)
                    print("param0: ",param0)
                    matching_q_exist = False
                except: raise Exception("Unexpected case")
                #idx0 = self.param_lists[0].index(q.get_parameter(self.param_filenames[0], self.param_names[0]))
            else: raise Exception("Unsupported type for 'get_param_0'")

            # Find an index of the second parameter
            if callable(self.get_param_1):
                try:
                    idx1 = get_index_of_nearest_element(self.param_lists[1], self.get_param_1(q))
                except ValueError:
                    print("The parameter couldn't be found from the array")
                    print("calculation home: %s" % q.home)
                    print("self.get_param_1(q): ",self.get_param_1(q))
                    matching_q_exist = False
                except: raise Exception("Unexpected case")
            elif self.get_param_1 is None:
                assert type(self.param_lists[1]) is list
                param1 = q.get_parameter(self.param_filenames[1], self.param_names[1])
                try:
                    if verbose: print("self.param_lists[1], param1",self.param_lists[1], param1)
                    idx1 = get_index_of_nearest_element(self.param_lists[1], param1)
                except ValueError:
                    print("The parameter couldn't be found from the array")
                    print("calculation home: %s" % q.home)
                    print("param1: ",param1)
                    matching_q_exist = False
                except: raise Exception("Unexpected case")
                #idx1 = self.param_lists[1].index()
            else: raise Exception("Unsupported type for 'get_param_1'")

            for constraint in self.constraints:
                param_file_name = constraint[0]
                param_name = constraint[1]
                param_value = constraint[2]
                param = q.get_parameter(param_file_name, param_name)
                if param != param_value:
                    if verbose: print("failed to be matched to constraint: %s" % str(constraint))
                    if verbose: print("given value != q's value == %s != %s" % (str(param_value), str(param)))
                    matching_q_exist = False

            # If all indices were found, assign this qprop object into the grid.
            if matching_q_exist:
                assert (idx0 is not None) and (idx1 is not None)
                self.q_arr[idx0,idx1] = q
            else:
                if verbose: print("[ LOG ] no matching Qprop object with home: %s" % q.home)

            #else:
                # If there's no matching object, let the user know.
                #print("Qprop object with the following calculation directory" \
                #    + " doesn't have 'momentum_spectrum_polar' that have been read : %s" % q.home)


        # for idx0 in range(self.param_grid_shape[0]):
        #     for idx1 in range(self.param_grid_shape[1]):
        #         param0 = self.param_lists[0][idx0]
        #         param1 = self.param_lists[1][idx1]
        #         no_matching_q_exist = True
        #         for q in q_list:
        #             param0_q = q.get_parameter(self.param_filenames[0], self.param_names[0])
        #             param1_q = q.get_parameter(self.param_filenames[1], self.param_names[1])
        #             if (param0 == param0_q) and (param1 == param1_q):
        #                 self.q_arr[idx0,idx1] = q
        #                 no_matching_q_exist = False

        #         if no_matching_q_exist:
        #             #raise Warning("No matching Qprop calculation directory for %s = %s" \
        #             #              % (self.param_names, (param0,param1)))
        #             print("No matching Qprop calculation directory for %s = %s" \
        #                           % (self.param_names, (param0,param1)))

    def read_all_polar_momentum_spectrum_data_file(self, verbose=False):
        ## Read polar momentum spectrum
        for idx0 in range(self.param_grid_shape[0]):
            for idx1 in range(self.param_grid_shape[1]):
                q = self.q_arr[idx0,idx1]
                if type(q) is not Qprop20: continue
                if verbose: print("in reading: ",idx0,idx1)
                try: q.momentum_spectrum_polar.readPolarSpectrumData(q)
                except:
                    print("Couldn't read polar spectrum from %s" % q.home)

    def __getitem__(self,slices):
        return self.q_arr[slices]

    def _get_list_of_only_qprop_obj(self):
        q_list = []
        for q in self.q_arr.flatten():
            if type(q) is Qprop20: q_list.append(q)
        return q_list

    def _check_qprop_dim_consistency(self):
        """Check whether all the qprop objects' qprop-dimension values are same."""
        q_list = self._get_list_of_only_qprop_obj()

        assert len(q_list) > 0
        if len(q_list) == 1:
            qprop_dim_is_consistent = True
        elif len(q_list) > 1:
            qprop_dim_is_consistent = True
            qprop_dim_0 = q_list[0]._get_qprop_dim() #q_list[0].get_parameter('initial.param','qprop-dim')
            for q in q_list:
                qprop_dim = q._get_qprop_dim() #get_parameter('propagate.param','qprop-dim')
                qprop_dim_is_consistent &= (qprop_dim_0 == qprop_dim)

        return qprop_dim_is_consistent

    def _get_qprop_dim(self):
        try: assert self._check_qprop_dim_consistency()
        except AssertionError as e:
            raise Exception("Qprop dimension values of each qprop objects are not consistent.")

        q_list = self._get_list_of_only_qprop_obj()
        # If consistency has been checked, any qprop object's dimension can be used.
        qprop_dim = q_list[0]._get_qprop_dim() #q_list[0].get_parameter('propagate.param','qprop-dim')

        return qprop_dim

    def get_global_color_range(self, prevent_zero_prob = True):
        assert type(prevent_zero_prob) is bool

        #prob_list = [q.momentum_spectrum_polar.polarSpectrum['total_prob'] for q in self.q_arr.flatten()]
        prob_list = []
        for q in self.q_arr.flatten():
            if type(q) is not Qprop20:
                continue
            if q.momentum_spectrum_polar.haveReadPolarSpectrum:
                prob_list.append(q.momentum_spectrum_polar.polarSpectrum['total_prob'])
        max_list = [max(prob) for prob in prob_list]
        min_list = [min(prob) for prob in prob_list]

        vmin = max(min_list)
        vmax = min(max_list)

        if (vmin == 0) and prevent_zero_prob: vmin = 1e-7

        return vmin, vmax


    def plot(self, log_scale=False, annotate=True, fontdict=None, use_same_range=False,
        get_param0_text=None, get_param1_text=None, xlim=None, ylim=None,
        order_of_mag=None,**kwargs):
        ## The fontdict is a dictionary object whose contents are font properties
        ## .. if the fontdict is not given by the input argument,
        ## .. default fontdict ('fontdict_black') is applied
        if fontdict is None: fontdict = fontdict_black
        assert type(fontdict) is dict

        for argument in [get_param0_text, get_param1_text]:
            if argument is not None:
                # Check if they are callable
                if not callable(argument): raise TypeError("'%s' is not 'callable'" % argument)

        for arg in [xlim, ylim]: assert type(arg) in [type(None), tuple, list]

        assert int(order_of_mag) == order_of_mag


        ## [180201 NOTE] The figure size should be determined in a rather proper way.
        ## -> [180228 NOTE] Fixed.
        #assert self._check_qprop_dim_consistency()
        qprop_dim = self._get_qprop_dim()
        #if self[0,0].momentum_spectrum_polar.qprop_dim == 34:
        if qprop_dim == 34:
            figsize = (self.param_grid_shape[1] * 12, self.param_grid_shape[0] * 5)
        #elif self[0,0].momentum_spectrum_polar.qprop_dim == 44:
        elif qprop_dim == 44:
            figsize = (self.param_grid_shape[1] * 12, self.param_grid_shape[0] * 9)
        fig, axes = plt.subplots(*self.param_grid_shape, figsize=figsize)

        # Make 'axes' a 2D array
        # [180312 NOTE] Implement for n-dimensional, generalized case
        if (axes.ndim == 1) and (1 in self.param_grid_shape):
            index_of_one = list(self.param_grid_shape).index(1)
            slices = ()
            for idx in range(self.ndim):
                if idx == index_of_one: slices += (np.newaxis,)
                else: slices += np.index_exp[:]
            axes = axes[slices]
        elif (axes.ndim == self.ndim): pass
        else: raise Exception("Unexpected axes.ndim == %s" % str(axes.ndim))


        vmin, vmax = None, None
        if use_same_range:
            if log_scale is True:
                vmin, vmax = self.get_global_color_range(prevent_zero_prob = True)
            else:
                vmin, vmax = self.get_global_color_range(prevent_zero_prob = False)
        else:
            # If user don't want to use same color range (global color range) for each plot,
            # .. don't use it
            vmin, vmax = None, None


        for idx0 in range(self.param_grid_shape[0]):
            for idx1 in range(self.param_grid_shape[1]):
                q = self[idx0,idx1]
                if type(q) is not Qprop20:
                    print("q[%d,%d] isn't Qprop20" % (idx0,idx1))
                    continue
                if not q.momentum_spectrum_polar.haveReadPolarSpectrum:
                    print("q[%d,%d] haven't read polar spectrum" % (idx0,idx1))
                    continue

                ax = axes[idx0,idx1]

                if order_of_mag is not None:
                    if order_of_mag >= 0:
                        vmax = q.momentum_spectrum_polar.kProbGrid.max()
                        vmin = vmax * 10 ** (-order_of_mag)
                    elif order_of_mag < 0:
                        vmin = q.momentum_spectrum_polar.kProbGrid.min()
                        vmax = vmin * 10 ** (-order_of_mag)

                q.momentum_spectrum_polar.plot(fig=fig, ax=ax, vmin=vmin, vmax=vmax, log_scale = log_scale, **kwargs)

                # if q.momentum_spectrum_polar.qprop_dim == 34:
                #     q.momentum_spectrum_polar.plot_under_linear_polarized_list(
                #         fig=fig, ax=ax, vmin=vmin, vmax=vmax, log_scale = log_scale, **kwargs)
                # elif q.momentum_spectrum_polar.qprop_dim == 44:
                #     q.momentum_spectrum_polar.plot_at_constant_theta(
                #         fig=fig, ax=ax, vmin=vmin, vmax=vmax, log_scale = log_scale, **kwargs)
                # else:
                #     raise Exception("Unsupported qprop_dim :",q.momentum_spectrum_polar.qprop_dim)

                # Set x, y limits
                if type(xlim) in [tuple, list]:
                    if len(xlim) == 2: ax.set_xlim(*xlim)
                elif xlim is None:
                    pass
                if type(ylim) in [tuple, list]:
                    if len(ylim) == 2: ax.set_ylim(*ylim)
                elif ylim is None:
                    pass

                if annotate:
                    ## Add annotation
                    # Determine text positions
                    y_min, y_max = ax.get_ylim()
                    x_min, x_max = ax.get_xlim()

                    y_height = y_max - y_min

                    pos_x = (x_max + x_min) * 0.5
                    pos_y = y_min + y_height * 0.85
                    if callable(get_param0_text):
                        text = get_param0_text(q)
                    elif get_param0_text is None:
                        param_name = 'first_param'
                        if self.param_names is not None: param_name = self.param_names[0]
                        text = "%s = %s" % (param_name, str(self.param_lists[0][idx0]))
                    ax.text(pos_x, pos_y, text, fontdict=fontdict)

                    pos_x = (x_max + x_min) * 0.5
                    pos_y = y_min + y_height * 0.75
                    if callable(get_param1_text):
                        text = get_param1_text(q)
                    elif get_param1_text is None:
                        param_name = 'second_param'
                        if self.param_names is not None: param_name = self.param_names[1]
                        text = "%s = %s" % (param_name, str(self.param_lists[1][idx1]))
                    ax.text(pos_x, pos_y, text, fontdict=fontdict)

        return fig, axes



### Object that can handles a collection of qprop calculations
class ModumQprop(object):
    """
    A collection of Qprop objects ('Modum' means collection in Korean)
    ModumQprop is initialized by a list of Qprop calculation home directories

    When 'parentDirOfQpropHomeDirs' is provided,
    """
    def __init__(self, parentDirOfQpropHomeDirs = '', listOfQpropHomeDir = []):
        ## Get Qprop object list
        if parentDirOfQpropHomeDirs != '':
            self.listOfQpropHomeDir = []
            self.listOfQpropObject = []
            for subdir in os.listdir(parentDirOfQpropHomeDirs):
                fullpath = os.path.join(parentDirOfQpropHomeDirs, subdir)
                # Select only directory, excluding files
                if os.path.isdir(fullpath):
                    try:
                        q = Qprop20(fullpath)
                    except NoParamFileError:
                        # Ignore directory that cannot be qualified as Qprop calculation directory
                        # .. which can be judged by checking the presence of parameter files
                        pass
                    else:
                        self.listOfQpropObject.append(q)
                        self.listOfQpropHomeDir.append(fullpath)

            ## If 'listOfQpropHomeDir' is given, check consistency
            if len(listOfQpropHomeDir) > 0:
                if set(self.listOfQpropObject) == set(listOfQpropHomeDir):
                    raise InconsistencyError(set(self.listOfQpropObject), set(listOfQpropHomeDir))

        elif len(listOfQpropHomeDir) > 0:
            self.listOfQpropHomeDir = listOfQpropHomeDir
            self.listOfQpropObject = list(map(Qprop20, self.listOfQpropHomeDir))

        else:
            Warning("No appropriate Qprop object(s) has been generated(instantiated)")

        ## Normalize home directory path string
        ## .. such as removing trailing '/' etc.
        ## .. for the sake of proper comparison
        self.listOfQpropHomeDir = list(map(os.path.normpath, self.listOfQpropHomeDir))

        ## Define several info related to Qprop objects 'modum'(collection)
        self.numOfQpropObj = len(self.listOfQpropObject)


        ## Check consistency in types of parameter files present
        arrayOfParamFileListLen = np.array(list(map(lambda x: len(x.paramFileList), self.listOfQpropObject)))
        allParamFileListHaveSameLength = ( arrayOfParamFileListLen.std() == 0.0 )
        if allParamFileListHaveSameLength:
            # any index is okay since all element of 'arrayOfParamFileListLen' are same
            numOfParamFile = arrayOfParamFileListLen[0]
        else:
            raise InconsistencyError('number of parameter files per Qprop calculation')

        ## Check if all paramFileList's contents are same
        ## .. for preventing case of same number but has different types of parameter files
        listOfParamFileListAsSet = list(map(lambda x: set(x.paramFileList) ,self.listOfQpropObject))
        commonParamFileList = set.intersection(*listOfParamFileListAsSet)
        if len(commonParamFileList) != numOfParamFile:
            raise InconsistencyError('types of parameter files per Qprop calculation')

        ## Construct common and uncommon parameters among colloection of Qprop objects
        self.commonParamDict = {}
        self.uncommonParamDict = {}
        if 'initial.param' in commonParamFileList:
            initialParams = list(map(lambda x: x.initialParam ,self.listOfQpropObject))
            self.common_initialParam = commonParam(initialParams)
            self.commonParamDict['initial.param'], self.uncommonParamDict['initial.param'] = commonInBothIndexAndValue(initialParams)

        if 'propagate.param' in commonParamFileList:
            propagateParams = list(map(lambda x: x.propagateParam ,self.listOfQpropObject))
            self.common_propagateParam = commonParam(propagateParams)
            self.commonParamDict['propagate.param'], self.uncommonParamDict['propagate.param'] = commonInBothIndexAndValue(propagateParams)

        if 'tsurff.param' in commonParamFileList:
            tsurffParams = list(map(lambda x: x.tsurffParam ,self.listOfQpropObject))
            self.common_tsurffParam = commonParam(tsurffParams)
            self.commonParamDict['tsurff.param'], self.uncommonParamDict['tsurff.param'] = commonInBothIndexAndValue(tsurffParams)

        if 'winop.param' in commonParamFileList:
            winopParams = list(map(lambda x: x.winopParam ,self.listOfQpropObject))
            self.common_winopParam = commonParam(winopParams)
            self.commonParamDict['winop.param'], self.uncommonParamDict['winop.param'] = commonInBothIndexAndValue(winopParams)

        ## Define miscelleneous attributes
        self.numOfUncommonParam = sum(list(map(len, self.uncommonParamDict.values())))
        self.numOfcommonParam = sum(list(map(len, self.commonParamDict.values())))
        # for key in self.uncommonParamDict.keys():
        #     self.numOfUncommonParam += len()

        ## Construct conversion from homedirectory to Qprop object
        self.home2obj = dict(zip(self.listOfQpropHomeDir, self.listOfQpropObject))

    def _getObjectByHomeDir(self, homeDir):
        homeDirNormalized = os.path.normpath(homeDir)
        return self.home2obj[homeDirNormalized]


    def getParam(self, homeDir, paramFileName, paramName=''):
        """
        It returns parameter list specified by 'paramFileName'
        of Qprop object specified by 'homeDir'.
        If 'paramName' is given, it returns corresponding paramter from the parameter list
        """
        #qObj = self.home2obj[homeDir]
        qObj = self._getObjectByHomeDir(homeDir)
        if paramFileName == 'initial.param':
            paramList = qObj.initialParam
        elif paramFileName == 'propagate.param':
            paramList = qObj.propagateParam
        elif paramFileName == 'tsurff.param':
            paramList = qObj.tsurffParam
        elif paramFileName == 'winop.param':
            paramList = qObj.winopParam
        else:
            raise Exception("No paramFileName as %s" % paramFileName)

        if paramName != '':
            if paramName in paramList.index:
                return paramList[paramName]
            else:
                raise Exception("No parameter name as %s" % paramName)
        else:
            return paramList





class ParametrizedModumQprop(ModumQprop):
    def __init__(self, parentDirOfQpropHomeDirs = '', parameters = '', listOfQpropHomeDir = []):
        """
        A Modum(collection) of Qprop calculations which have same parameter files except one or more parameter(s).
        These uncommon parameter is called either uncommon parameter or varying paramter.

        'parameters' is name (or a number of names) of uncommon parameter.
        There is a case where single parameter name can reside on multiple parameter files in which an ambiguity may arise.
        (e.g. 'ell-grid-size' is present in both 'initial.param' and 'propagate.param' in several examples in QPROP 2.0)
        Thus, it is advisable to specify both parameter name and a name of param file the name reside in
        to remove the ambiguity.
        If 'parameters' is string(str), it is perceived as single uncommon(varying) parameter with ambiguity on parameter name.
        If 'parameters' is list of string(s), it is perceived as one or more uncommon(varying) parameter(s) with ambiguity on parameter names.
        If 'parameters' is list of tuple(s), it is perceived as one or more uncommon(varying) paramter(s) without ambiguity
        whose each tuple consists of parameter name and its correponding parameter filename.

        If there's ambiguity on parameter name, inferrence is tried by comparing parameter files
        [NOTE: Should be written more]
        """

        super().__init__(parentDirOfQpropHomeDirs = parentDirOfQpropHomeDirs, listOfQpropHomeDir = listOfQpropHomeDir)

        ### Process input argument
        ## Process 'paramteters'
        # [NOTE on 'self.varyingParamCategory']
        # The keys of 'self.varyingParamCategory' are varying parameter
        # .. and its values are their keys(parameters)' parameter file name
        # e.g. if key is 'delta-r' then its value is 'initial'
        # .. since the parameter 'delta-r' belongs to 'initial.param' file in QPROP 2.0
        self.numOfVaryingParam = 0
        self.varyingParams = []
        self.varyingParamTuples = []
        #self.varyingParamCategory = {}
        if type(parameters) is str:
            if parameters == '':
                ## Infer parametrizing parameter if the number of uncommon parameter is one
                ## .. It should also consider, all other parameter are common(=same) ==> NOPE, it shouldn't.
                if self.numOfUncommonParam == 0:
                    raise Exception("There is no uncommon parameter, thus unable to infer parameter")

                for paramFile in self.uncommonParamDict.keys():
                    for param in self.uncommonParamDict[paramFile]:
                        self.varyingParams.append(param)
                        self.varyingParamTuples.append( ( param, paramFile ) )

                self.numOfVaryingParam = len(self.varyingParamTuples)

                if self.numOfVaryingParam != 1:
                    raise Exception('Failed to infer uncommon parameter')

                print("Inferred uncommon(varying) parameter(s): ")
                for varyingParamTuple in self.varyingParamTuples:
                    print(varyingParamTuple)

            else:
                parameter = parameters
                if self._isAmbiguousParam(parameter):
                    raise Exception('Failed to infer uncommon parameter because of ambiguity')
                else:
                    paramIsInParamFile = False
                    for key in self.uncommonParamDict:
                        paramIsInParamFile |= parameter in self.uncommonParamDict[key]
                        if paramIsInParamFile:
                            self.numOfVaryingParam = 1
                            self.varyingParams.append(parameter)
                            self.varyingParamTuples.append( ( parameter, key ) )
                            break

        elif type(parameters) is list:
            allParamsAreInParamFile = True
            for parameter in parameters:
                paramIsInParamFile = False
                for key in self.uncommonParamDict:
                    paramIsInParamFile |= parameter in self.uncommonParamDict[key]
                    if paramIsInParamFile:
                        self.numOfVaryingParam += 1
                        self.varyingParams.append(parameter)
                        self.varyingParamTuples.append( ( parameter, key ) )
                    else:
                        raise KeyError("Given parameter(=%s) is not registered in any list of uncommon parameter" % (parameter))
                allParamsAreInParamFile &= paramIsInParamFile

            if not allParamsAreInParamFile:
                raise KeyError("Given parameters(=%s) is not registered in any list of uncommon parameter" % (str(parameters)))

        self.setVariableTable()


    def setVariableTable(self):
        ## Construct a table of varying parameter(s) (called variable(s))
        # Construct index and columns
        self.variableTable = pd.DataFrame(index = self.listOfQpropHomeDir, columns = self.varyingParamTuples)
        # Fill in the contents
        for homeDir in self.variableTable.index:
            for varyingParamTuple in self.variableTable.columns:
                paramName = varyingParamTuple[0]
                paramFileName = varyingParamTuple[1]
                self.variableTable.loc[homeDir, varyingParamTuple] = self.getParam(homeDir, paramFileName, paramName)
        # Let the table have multi-level column for parameter name and its residing parameter file name
        self.variableTable.columns = pd.MultiIndex.from_tuples(self.variableTable.columns, names=['paramName', 'paramFileName'])


    def plotRadialAsymmetryMap(self, targetParam='', saveFigure=False, figName='', cep_repeat=None, x_tick_unit='', y_tick_unit=''):
        ## Check all Qprop objects in ModumQprop has same parameters (e.g. num-k-surff, delta-r)
        ## .. except some parameter that should vary (e.g. cep)
        # ...

        ## Make and save the common parameters into member variable as 'self.commonInitialParam' etc.

        ## Try automatic detection of targetParam
        if targetParam == '':
            if (self.numOfVaryingParam == 1) and (len(self.varyingParams) == 1):
                targetParam = self.varyingParams[0]
            else:
                raise Exception("Cannot abviously select target parameter to be plotted")

        ## If not ambiguous, find its parameter file name from tuples
        # Check if quried 'targetParam' is in columns of 'variableTable'
        # .. which is a list of varyingParamTuple == (parameter name, parameter file name)
        if targetParam in self.variableTable.columns:
            for colTuple in self.variableTable.columns:
                # Find varyingParamTuple that has matching parameter name with quried targetParam
                # .. and when found, assign the coupled parameter file name
                if colTuple[0] == targetParam:
                    targetParamFileName = colTuple[1]
        else:
            raise KeyError("No column name as %s" % targetParam)

        ## Construct column name of variable table for extracting quried data
        targetParamTuple = (targetParam, targetParamFileName)

        ## Sort by target values for plotting
        self.variableTable.sort_values(by=targetParamTuple, axis=0, ascending=True, inplace=True)

        ## Construct Asymmetry Map

        # Construct shape of Asymmetry Map's data
        asymMapShape = ( self.common_tsurffParam['num-k-surff'], self.numOfQpropObj )

        # Construct empty Asymmetry Map
        asymMap = np.empty(asymMapShape, dtype=np.float64)

        # Fill in Asymmetry Map with asymmetry coefficients
        for idx, home in enumerate(self.variableTable.index):
            qObj = self._getObjectByHomeDir(home)
            paramValue = self.variableTable.loc[home, targetParamTuple]
            asymMap[:,idx] = qObj.getRadialAsymmetricParam()

        ## Construct X and Y meshgrid of Axes.pcolormesh

        # Construct values for each axis
        targetParamValues = self.variableTable.loc[:,targetParamTuple].values
        #kValues = np.linspace(0, self.common_tsurffParam['k-max-surff'], self.common_tsurffParam['num-k-surff'] + 1)[1:]
        kValues = np.linspace(0, self.common_tsurffParam['k-max-surff'], self.common_tsurffParam['num-k-surff'])

        # Combine x and y axis to yield 2D meshgrid
        X_mesh, Y_mesh = np.meshgrid(targetParamValues, kValues, indexing='xy')

        #X_mesh2 = X_mesh + np.pi
        #Y_mesh2 = Y_mesh
        #asymMap2 = asymMap * (-1)**(2-1)

        #X = np.concatenate((X_mesh,X_mesh2), axis=0)
        #Y = np.concatenate((Y_mesh,Y_mesh2), axis=0)
        #C = np.concatenate((asymMap,asymMap2), axis=0)

        if cep_repeat == None:
            X = X_mesh
            Y = Y_mesh
            C = asymMap
        else:
            #print("entering cep-repeat mode")
            ## If 'cep_repeat' is not None, it implies this asymmetry plot depends on CEP values
            if targetParam.lower() != 'cep':
                raise Warning("The target paremter isn't 'cep'")

            ## The following assertion prevents non-numeric object
            assert type(int(cep_repeat)) is int
            ## The following assertion assures consistency of X_mesh, Y_mesh, and asymMap's shapes
            assert (X_mesh.shape == Y_mesh.shape) and (X_mesh.shape == asymMap.shape)

            ## Constructs repeated asymmetry map for CEP dependence
            repeatedArrayShape = (X_mesh.shape[0], X_mesh.shape[1] * cep_repeat)

            X = np.empty(repeatedArrayShape)
            Y = np.empty(repeatedArrayShape)
            C = np.empty(repeatedArrayShape)

            numOfParameterValues = X_mesh.shape[1]
            for i in range(cep_repeat):
                X[:,slice(numOfParameterValues*i, numOfParameterValues*(i+1))] = X_mesh + (np.pi * i)
                Y[:,slice(numOfParameterValues*i, numOfParameterValues*(i+1))] = Y_mesh
                C[:,slice(numOfParameterValues*i, numOfParameterValues*(i+1))] = asymMap * (-1)**(i)

            #print("shape of X,Y,C:")
            #print(X.shape, Y.shape, C.shape)


        fig = plt.figure(figsize=(10,10))
        ax = fig.gca()

        ## Unit conversion
        if x_tick_unit == '':
            X = X
            ax.set_xlabel(targetParamTuple[0])
        elif x_tick_unit.lower() == 'pi':
            X = X / np.pi
            xlabel_string = "%s [%s]" % (targetParamTuple[0],x_tick_unit)
            ax.set_xlabel(xlabel_string)

        if y_tick_unit == '':
            Y = Y
            ax.set_ylabel('mean radial momentum (a.u.)')
        elif y_tick_unit.lower() == 'eV'.lower():
            Y = (Y ** 2) / 2 * unit.au2si['energy'] / unit.e
            ax.set_ylabel('electron energy [eV]')


        ## Actual plotting
        pcm = ax.pcolormesh(X, Y, C, cmap='jet')


        cb = fig.colorbar(pcm)
        cb.set_label('Asymmetry Parameter', rotation=270)
        cb.ax.get_yaxis().labelpad = 20

        #ylim = ax.set_ylim(0,0.4)

        if saveFigure:
            if figName == '':
                figName = 'Asymmetry_Map.png'
            fig.savefig(figName)

        ## [NOTE 171019] Consider implementing in a separate object such as .. Asymmetry etc.
        ## .. and this plotAsymmetryMap() is replaced by Asymmetry.plot etc.
        ## CEP depence or anything else.. can be specified etc.

        ## Autodect variable to plot (e.g. CEP) from commonInBothIndexAndValue etc.
        return ax, fig, pcm, cb


    def _isAmbiguousParam(self, paramName):
        """Check if single parameter name resides in multiple parameter files"""
        numOfParamFileThatHasParam = 0
        for key in self.uncommonParamDict.keys():
            if paramName in self.uncommonParamDict[key]:
                numOfParamFileThatHasParam += 1
        if numOfParamFileThatHasParam > 1:
            return True
        elif numOfParamFileThatHasParam == 1:
            return False
        else:
            raise Exception("Given parameter(=%s) is not registered in any list of uncommon parameter" % (parameter))



### Function for calculating radial asymmetric parameter
# Returns 1 if completely aymmetric to right side
# Returns -1 if completely aymmetric to left side
def asymmetricParameter(left, right):
    denominator = right + left
    if denominator == 0:
        # Should be 0.0, not just 0 (i.e. integer zero) for preventing unexpected bug regarding type
        return 0.0
    else:
        return (right - left) / denominator



### Used in finding common parameter
def commonIndex(paramList):
    cupped = paramList[0].index.copy()
    for idx in range(len(paramList) - 1):
        cupped = cupped[cupped.isin(paramList[idx+1].index)]

    return cupped


def commonInBothIndexAndValue(paramList):
    common_index = commonIndex(paramList)
    A = paramList[0][common_index]
    common_param = A.isin(A)
    for idx in range(len(paramList) - 1):
        common_param &= (A == paramList[idx+1])

    return common_index[common_param].values, common_index[~common_param].values


def commonParam(paramList):
    commonIndex, notCommonIndex = commonInBothIndexAndValue(paramList)
    return paramList[0][commonIndex]


def commonList(listOfList):
    if type(listOfList) is not list:
        raise TypeError("Please give me type 'list'")

    intersected = set(listOfList[0])
    for idx in range(len(listOfList) - 1):
        intersected = intersected.intersection(listOfList[idx + 1])

    return list(intersected)


#class PartialSpectrum(object):
#    def __init__(self):



### Define several types of Error classes for clear debugging etc. ###

## Define base class for Error
#class Error(Exception):
#    """Base class for exceptions in this module"""
#    pass

## When inconsistency happens
class InconsistencyError(Exception):
    """Error raised when two objects which should be consistent each other are not consistent"""
    def __init__(self, *inconsistentObjects):
        self.message = "Following objects are inconsistent: %s" % ', '.join(map(str,inconsistentObjects))
        super().__init__(self.message)


class NoParamFileError(Exception):
    """Raised when one or more parameter file(s) is(are) absent in a specified directory"""
    def __init__(self, *nameOfAbsentParamFiles):
        self.message = "One or more of the following parameter file(s) is(are) required but absent: %s" % ', '.join(nameOfAbsentParamFiles)
        super().__init__(self.message)
