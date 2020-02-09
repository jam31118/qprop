from numbers import Real

import numpy as np
from numpy import pi
import pandas as pd
import matplotlib.pyplot as plt

supported_vecpot_shape = ['sin-square']
supported_vecpot_direction = ['x','y','z']
supported_dimension = [34, 44]

def is_supported_vecpot_shape(shape):
    assert type(shape) is str
    is_in_support_list = shape.lower() in supported_vecpot_shape
    return is_in_support_list

def is_supported_vecpot_direction(direction):
    assert type(direction) is str
    is_in_support_list = direction.lower() in supported_vecpot_direction
    return is_in_support_list

def is_supported_dimension(dimension):
    assert isinstance(dimension, Real)
    dimension = int(dimension)
    return dimension in supported_dimension

def make_str_list_to_lowercase(str_list):
    for string in str_list: assert type(string) is str
    return [string.lower() for string in str_list]

def directions_are_consistent(directions):
    for direction in directions:
        assert is_supported_vecpot_direction(direction)
    directions_lowercase = [direction.lower() for direction in directions]
    has_x_direction = 'x' in directions_lowercase
    has_y_direction = 'y' in directions_lowercase
    has_z_direction = 'z' in directions_lowercase
    xor = has_z_direction ^ (has_x_direction or has_y_direction)
    return xor

def get_dimensIon_from_directions(directions):
    assert directions_are_consistent(directions)
    directions_lowercase = [direction.lower() for direction in directions]
    dimension = None
    dimen_is_34 = 'z' in directions_lowercase
    dimen_is_44 = ('x' in directions_lowercase) or ('y' in directions_lowercase)
    assert dimen_is_34 ^ dimen_is_44
    if dimen_is_34: return 34
    if dimen_is_44: return 44
        
def get_phase_diff_of_ellipticity(e):
    cos_beta = (1 - (1-e)**2) / (1 + (1-e)**2)
    beta = np.arccos(cos_beta)
    return beta


from matplotlib.axes import Axes
from mpl_toolkits.mplot3d import Axes3D
def plot_vecpot(ax, dimension, t, Ax=None, Ay=None, Az=None, **plot_kwargs):
    assert isinstance(ax, Axes)
    assert is_supported_dimension(dimension)
#    Az_given = Az is not None
#    Ax_or_Ay_given = (Ax is not None) or (Ay is not None)
#
#    inferred_dimension = None
#    assert Az_given ^ Ax_or_Ay_given
#    if Az_given: inferred_dimension = 34
#    elif Ax_or_Ay_given: inferred_dimension = 44
#    else: raise Exception("Unexpected case.")
#    
#    if dimension is None: dimension = inferred_dimension
#    else:
#        is_supported_dimension(dimension) 
#        assert inferred_dimension == dimension
    
    if dimension == 34:
        assert Az is not None
        ax.plot(t, Az, **plot_kwargs)
    elif dimension == 44:
        assert (Ax is not None) and (Ay is not None)
        assert isinstance(ax, Axes3D)
        ax.plot(t, Ax, Ay, **plot_kwargs)


from os.path import isfile
def load_qprop_vecpot_from_file(filename, dimension=None):
    if not isfile(filename):
        err_mesg = "Vector potential data file doesn't exist: {0}"
        raise IOError(err_mesg.format(filename))
    if dimension is not None:
        assert is_supported_dimension(dimension)
    
    loaded_array = None
    try: loaded_array = np.loadtxt(filename)
    except: 
        err_mesg = "Failed to load vector potential array from file: {0}"
        raise IOError(err_mesg.format(filename))

    num_of_loaded_columns = loaded_array.shape[1]
    loaded_dimension = None
    if num_of_loaded_columns == 2:
        loaded_dimension = 34
    elif num_of_loaded_columns == 3:
        loaded_dimension = 44
    else: 
        err_mesg = "Unexpected number of columns ({0}) from file: {1}"
        raise Exception(err_mesg.format(num_of_loaded_columns, filename))
    
    assert loaded_dimension is not None
    if dimension is None: dimension = loaded_dimension
    else: assert loaded_dimension == dimension
    
    return loaded_array



class Vecpot(object):
#    def __init__(start_time, end_time):
#        self.start_time, self.end_time = start_time, end_time

    def get_duration(): pass

    def get_start_time(self): pass
        
    def get_end_time(self): pass

    def __call__(self, t): pass



class Loaded_Vecpot(Vecpot):
    def __init__(self, dimension, filename):
        assert is_supported_dimension(dimension)
        self.dimension = int(dimension)
        self.t, self.Ax, self.Ay, self.Az = [None] * 4
        zero_array = np.zeros_like(self.t)
        loaded_array = load_qprop_vecpot_from_file(filename, dimension=dimension)
        if self.dimension == 34:
            self.t, self.Az = loaded_array.transpose()
            self.Ax, self.Ay = zero_array.copy(), zero_array.copy()
        elif self.dimension == 44:
            self.t, self.Ax, self.Ay = loaded_array.transpose()
            self.Az = zero_array.copy()
        else: raise Exception("Unexpected dimension: {0}".format(self.dimension))

    def plot(self, ax=None):
        projection = None
        if self.dimension == 44: projection = '3d'
        if ax is None:
            fig = plt.figure()
            ax = fig.gca(projection=projection)

        plot_vecpot(ax, t=self.t, Ax=self.Ax, Ay=self.Ay, Az=self.Az, 
                dimension=self.dimension)

        return ax


class Single_Vecpot(Vecpot):
    def __init__(self, omega, num_cycles, E0, phase, direction, 
            start_time=0.0, shape='sin-square'):
        """
        
        Parameters
        ----------
        omega : float
            angular frequency in atomic unit
        num_cycles : int
            number of cycles        
        E0 : float
            max amplitude electric field along the given direction, 
            in atomic unit.
        phase : float
            offset phase of the carrier wave
        direction : one of 'x', 'y', 'z'
            oscillating direction of the vector potential
        start_time : float
            starting time at which the vector potential began to be non-zero
        """

        ## Check input arguments
        for arg in [omega, num_cycles, E0, phase, start_time]:
            assert isinstance(arg, Real)

        for positive_arg in [omega, num_cycles, E0, start_time]:
            assert positive_arg >= 0

        for str_arg in [direction, shape]:
            assert type(str_arg) is str

        assert is_supported_vecpot_shape(shape)
        assert is_supported_vecpot_direction(direction)
        
        ## Variables which should be initialized in __init__()
        self.omega = omega
        self.num_cycles = num_cycles
        self.E0 = E0
        self.phase = phase
        self.start_time = start_time
        self.duration = self.get_duration()
        self.end_time = self.get_end_time()
        self.shape = shape.lower()
        self.direction = direction.lower()
        self.period = 2.0 * pi / omega
    
    def get_duration(self):
        return self.num_cycles * 2 * pi / self.omega
    
    def get_start_time(self):
        return self.start_time
        
    def get_end_time(self):
        return self.start_time + self.get_duration()
    
    def __call__(self, t):
        ww = self.omega / (2.0 * self.num_cycles)
        carrier = np.sin(self.omega * (t-self.start_time) + self.phase)
        envelope = self.E0/self.omega*np.square(np.sin(ww*(t-self.start_time)))
        vecpot_value = envelope * carrier
        if isinstance(vecpot_value, np.ndarray):
            out_of_pulse_mask = (t < self.start_time) | (self.end_time < t)
            vecpot_value[out_of_pulse_mask] = 0.0
        elif isinstance(vecpot_value, Real):
            is_out_of_pulse = (t < self.start_time) or (self.end_time < t)
            if is_out_of_pulse:
                vecpot_value = 0.0
        return vecpot_value

    def __add__(self, vecpot):
        assert type(vecpot) is type(self)
        return Superposed_Vecpot([self, vecpot])


class Superposed_Vecpot(Vecpot):
    def __init__(self, vecpots):
        assert len(vecpots) >= 1
        for vecpot in vecpots: assert isinstance(vecpot, Vecpot)
        assert self.vecpots_directions_are_consistent(vecpots)

        self.vecpots = vecpots

        self.start_time = self.get_start_time()
        self.end_time = self.get_end_time()
        assert self.start_time < self.end_time
        
        self.duration = self.get_duration()
        
        self.dimension = self.get_dimensIon_from_vecpots(vecpots)


    def get_start_time(self):
        min_start_time = min(
                vp.get_start_time() for vp in self.vecpots)
        assert min_start_time >= 0
        return min_start_time

    def get_end_time(self):
        max_end_time = max(
                vp.get_end_time() for vp in self.vecpots)
        return max_end_time

    def get_duration(self):
        start_time = self.get_start_time()
        end_time = self.get_end_time()
        assert start_time < end_time
        return start_time - end_time

    def __call__(self, t):
        superposed_value = 0.0
        for vecpot in self.vecpots:
            superposed_value += vecpot(t)
        return superposed_value

    @staticmethod
    def get_directions_from_vecpots(vecpots):
        for vecpot in vecpots:
            assert isinstance(vecpot, Single_Vecpot)
            hasattr(vecpot, 'direction')
        directions = [vecpot.direction for vecpot in vecpots]
        return directions

    @classmethod
    def vecpots_directions_are_consistent(cls, vecpots):
        directions = cls.get_directions_from_vecpots(vecpots)
        return directions_are_consistent(directions)

    @classmethod
    def get_dimensIon_from_vecpots(cls, vecpots):
        directions = cls.get_directions_from_vecpots(vecpots)
        return get_dimensIon_from_directions(directions)

    def __add__(self, vecpot):
        assert type(vecpot) in [Single_Vecpot, Superposed_Vecpot]
        vecpots = list(self.vecpots)
        if isinstance(vecpot, Single_Vecpot):
            vecpots.append(vecpot)
        elif isinstance(vecpot, Superposed_Vecpot):
            vecpots += list(vecpot.vecpots)
        else: raise Exception("Unexpected type of vecpot: {0}".format(vecpot))
        result_vecpot = None
        try: result_vecpot = Superposed_Vecpot(vecpots)
        except Exception as e:
            err_megs = "Failed to construct superposed vector potential object. "
            err_megs += "Error log: {0}".format(e)
            raise Exception(err_megs)
        return result_vecpot

#    @staticmethod
#    def from_parameters(para):
#        assert hasattr(para, '__getitem__')
#
#        if param_names_are_in_original_form(para):
#            pass


class Param_to_Vecpot(object):
    ## Legacy settings (for backward compatibility)
    must_have_if_original = [
            'omega', 'num-cycles', 'max-electric-field' ]
    optional_if_original = ['phase-pi', 'start_time']
    # [TODO] this default should be put into default in qprop.default
    default_for_optional = {
            'phase-pi':0.0, 'start_time':0.0}

    ## Settings for new form
    suffix_form = '-{0}-{1}'


    def __init__(self, param):
        """Initialize
        
        Parameters
        ----------
        param : dict
            parameter key-value set
        """
        if not isinstance(param, dict):
            raise ValueError("`param` should be of type {}".format(dict))
        self.param = param


    def param_names_are_in_original_form(self):
    
        ## Check if para object has vector potential
        ## .. parameters in original form.
        param_name_is_in_original_form = True
        for must_have_key in self.must_have_if_original:
            have_key = must_have_key in self.param.keys()
            param_name_is_in_original_form &= have_key
        
        return param_name_is_in_original_form
    

    def param_to_single_vecpot(self):

        assert self.param_names_are_in_original_form(self.param)

        must_have_key_values = []
        for key in self.must_have_if_original:
            if key in self.param.keys():
                must_have_key_values.append(self.param[key])
            else:
                err_mesg = "key {0} should be in param.keys() {1}"
                raise KeyError(
                    err_mesg.format(key, self.param.keys()))

        optional_values = []
        for opt_key in self.optional_if_original:
            if opt_key in self.param.keys():
                optional_values.append(self.param[opt_key])
            else:
                default = cls.default_for_optional[opt_key]
                optional_values.append(default)
        all_key_values = must_have_key_values + optional_values
        
        vp = None
        try: vp = Single_Vecpot(*all_key_values)
        except: 
            raise Exception(
                    "Failed to construct Single_Vecpot"
                    + " from given param: {0}".format(self.param))
        return vp
    


import re

tail_pattern_in_new_form = '-([xyz])-([0-9]+)'


class ParameterList2Vecpot(object):
    
    mandatory_keys_in_old_form = ['max-electric-field','omega','num-cycles']
    
    mandatory_keys_in_new_form = [name + tail_pattern_in_new_form
                                  for name in mandatory_keys_in_old_form]
    
    def __init__(self):
        pass
#             optional_keys_in_new_form = [name+tail_pattern for name in ['phase-pi','start-time']]
    
    @staticmethod
    def is_valid_param_list(param_list):
        return hasattr(param_list, 'keys') and callable(getattr(param_list, 'keys'))
    
    @classmethod
    def has_new_param_form(cls, param_list):
        assert cls.is_valid_param_list(param_list)
        _has_new_form = any([re.match(_pattern, _key) 
                             for _pattern in cls.mandatory_keys_in_new_form 
                             for _key in param_list.keys()])
        return _has_new_form

    @classmethod
    def has_old_param_form(cls, param_list):
        assert cls.is_valid_param_list(param_list)
        _has_old_form = any([_pattern == _key 
                             for _pattern in cls.mandatory_keys_in_old_form 
                             for _key in param_list.keys()])
        return _has_old_form






#
#class Vecpot(object):
#    def __init__(self, qprop_dim, home_dir, omega, numOfCycle, E_max, phi_cep,
#                omega2=None, numOfCycle2=None, E_max2=None, phi_cep2=None, delay=None):
#
#        self.qprop_dim = qprop_dim
#        self.home_dir = home_dir
#        self.omega, self.numOfCycle, self.E_max, self.phi_cep = omega, numOfCycle, E_max, phi_cep
#        self.duration = self.numOfCycle * 2 * np.pi / self.omega
#        self.ww = self.omega / (2.0 * self.numOfCycle)
#
#        ## Check whether second wave is present
#        ## .. by checking 'omega, number of cycles, max electric field and delay' is present
#        self.haveSecondVecpotParam = all([omega2, numOfCycle2, E_max2])
#        self.haveDelay = delay != None
#        self.haveSecondVecpot = self.haveSecondVecpotParam and self.haveDelay
#
#        ## When second wave is included,
#        if (self.haveSecondVecpot):
#            self.omega2, self.numOfCycle2, self.E_max2, self.phi_cep2 = omega2, numOfCycle2, E_max2, phi_cep2
#            self.delay = delay
#            self.duration2 = self.numOfCycle2 * 2 * np.pi / self.omega2
#            self.ww2 = self.omega2 / (2.0 * self.numOfCycle2)
#
#        self.fileLoadingCompleted = False
#
#    def get_duration(self):
#        timeAtWhichFirstVecpotEnds = self.duration
#        if (self.haveSecondVecpot):
#            timeAtWhichSecondVecpotEnds = self.delay + self.duration2
#        else:
#            timeAtWhichSecondVecpotEnds = 0.0
#
#        if (timeAtWhichFirstVecpotEnds > timeAtWhichSecondVecpotEnds):
#            totalDuration = timeAtWhichFirstVecpotEnds
#        else:
#            totalDuration = timeAtWhichSecondVecpotEnds
#
#        if (self.E_max == 0.0) and (self.E_max2 == 0.0):
#            totalDuration = 0
#
#        return totalDuration
#
#    def load(self, vpot_filename=''):
#        ## Get vector potential data file name
#        if vpot_filename == '':
#            # Set filename automatically according to qprop dimension
#            if self.qprop_dim == 34:
#                filename_without_fullpath = 'hydrogen_re-vpot_z.dat'
#            elif self.qprop_dim == 44:
#                filename_without_fullpath = 'hydrogen_re-vpot_xy.dat'
#            else:
#                raise IOError("Unsupported qprop_dimension: %d\n" % (self.qprop_dim))
#            # Construct full file path
#            vpot_filename = os.path.join(self.home_dir,filename_without_fullpath)
#
#        # Check existence of the vecpot data file
#        if not os.path.exists(vpot_filename):
#            raise IOError("No vecpot data file with name: %s" % (vpot_filename))
#
#        ## Set column names
#        if (self.qprop_dim == 34):
#            column_names = ['time', 'vpot_z']
#        elif (self.qprop_dim == 44):
#            column_names = ['time', 'vpot_x', 'vpot_y']
#        else:
#            raise IOError("Unsupported qprop_dimension: %d\n" % (self.qprop_dim))
#        #self.data = pd.read_table(vpot_filename, header=None, sep='\s+', names=['time','vpot_x', 'vpot_y'])
#        self.data = pd.read_table(vpot_filename, header=None, sep='\s+', names=column_names)
#        self.fileLoadingCompleted = True
#
#
#
#    def plot(self, saveFigureName='', rcParams={}, fig=None, subplot_index=(), **kwargs):
#        """
#        'kwargs' is delivered to plotter such as matplotlib.pyplot.plot() etc.
#        """
#
#        if not self.fileLoadingCompleted:
#            try:
#                self.load()
#            except:
#                raise IOError("No data have been loaded from file")
#
#        ## Parse 'kwargs' for sophisticated plotting
#        if 'figsize' in kwargs.keys(): self.figsize = kwargs.pop('figsize')
#        else: self.figsize = (13,7)
#
#
#        ## Set figure object
#        if fig == None:
#            fig = plt.figure(figsize=self.figsize)
#        else:
#            # [NOTE] it would be good to check if 'fig' is of type 'matplotlib.figure.Figure'
#            fig = fig
#
#
#        ## Set projection mode
#        if self.qprop_dim == 34:
#            projection = None
#        elif self.qprop_dim == 44:
#            projection = '3d'
#        else:
#            raise IOError("Unsupported qprop_dimension: %d\n" % (self.qprop_dim))
#
#
#        ## Set Axes object
#        if (type(subplot_index) != tuple):
#            # In case where subplot_index is given in a form of scalar
#            # .. such as '111' which is equivalent to '1,1,1'
#            ax = fig.add_subplot(subplot_index, projection=projection)
#        elif (type(subplot_index) == tuple) and (len(subplot_index) != 0):
#            # In case where subplot_index is given in a form of tuple
#            # .. such as '(1,1,1)' which will be unpacked when passing to fig.add_subplot()
#            ax = fig.add_subplot(*subplot_index, projection=projection)
#        else:
#            ax = fig.gca(projection=projection)
#
#
#        ## Actual plotting happens here
#        if (self.qprop_dim == 34):
#            # Data Generation
#            timeMask = self.data['time'] < self.get_duration()
#            x = self.data.loc[timeMask,'time']
#            y = self.data.loc[timeMask,'vpot_z']
#
#            # Plot
#            ax.plot(x,y, **kwargs)
#
#            # Labeling
#            ax.set_xlabel('time (a.u.)')
#            ax.set_ylabel(r'$A_z$(a.u.)')
#
#        elif (self.qprop_dim == 44):
#            # Data Generation
#            timeMask = self.data['time'] < self.get_duration()
#            x = self.data.loc[timeMask,'time']
#            y = self.data.loc[timeMask,'vpot_x']
#            z = self.data.loc[timeMask,'vpot_y']
#
#            # Plot
#            #ax = fig.gca(projection='3d')
#            ax.plot(x,y,z, **kwargs)
#
#            # Labeling
#            ax.set_xlabel('time (a.u.)')
#            ax.set_ylabel(r'$A_x$(a.u.)')
#            ax.set_zlabel(r'$A_y$(a.u.)')
#
#        else:
#            raise IOError("Unsupported qprop_dimension: %d\n" % (self.qprop_dim))
#
#        ## Save figure if specified
#        if (saveFigureName != ''):
#            plt.savefig(saveFigureName)
#
#
#        ## Return handles
#        return fig, ax
#

