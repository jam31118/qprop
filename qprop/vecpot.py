import numpy as np
import pandas as pd
import matplotlib.pyplot as plt




class Vecpot(object):
    def __init__(self, qprop_dim, home_dir, omega, numOfCycle, E_max, phi_cep,
                omega2=None, numOfCycle2=None, E_max2=None, phi_cep2=None, delay=None):

        self.qprop_dim = qprop_dim
        self.home_dir = home_dir
        self.omega, self.numOfCycle, self.E_max, self.phi_cep = omega, numOfCycle, E_max, phi_cep
        self.duration = self.numOfCycle * 2 * np.pi / self.omega
        self.ww = self.omega / (2.0 * self.numOfCycle)

        ## Check whether second wave is present
        ## .. by checking 'omega, number of cycles, max electric field and delay' is present
        self.haveSecondVecpotParam = all([omega2, numOfCycle2, E_max2])
        self.haveDelay = delay != None
        self.haveSecondVecpot = self.haveSecondVecpotParam and self.haveDelay

        ## When second wave is included,
        if (self.haveSecondVecpot):
            self.omega2, self.numOfCycle2, self.E_max2, self.phi_cep2 = omega2, numOfCycle2, E_max2, phi_cep2
            self.delay = delay
            self.duration2 = self.numOfCycle2 * 2 * np.pi / self.omega2
            self.ww2 = self.omega2 / (2.0 * self.numOfCycle2)

        self.fileLoadingCompleted = False

    def get_duration(self):
        timeAtWhichFirstVecpotEnds = self.duration
        if (self.haveSecondVecpot):
            timeAtWhichSecondVecpotEnds = self.delay + self.duration2
        else:
            timeAtWhichSecondVecpotEnds = 0.0

        if (timeAtWhichFirstVecpotEnds > timeAtWhichSecondVecpotEnds):
            totalDuration = timeAtWhichFirstVecpotEnds
        else:
            totalDuration = timeAtWhichSecondVecpotEnds

        if (self.E_max == 0.0) and (self.E_max2 == 0.0):
            totalDuration = 0

        return totalDuration

    def load(self, vpot_filename=''):
        ## Get vector potential data file name
        if vpot_filename == '':
            # Set filename automatically according to qprop dimension
            if self.qprop_dim == 34:
                filename_without_fullpath = 'hydrogen_re-vpot_z.dat'
            elif self.qprop_dim == 44:
                filename_without_fullpath = 'hydrogen_re-vpot_xy.dat'
            else:
                raise IOError("Unsupported qprop_dimension: %d\n" % (self.qprop_dim))
            # Construct full file path
            vpot_filename = os.path.join(self.home_dir,filename_without_fullpath)

        # Check existence of the vecpot data file
        if not os.path.exists(vpot_filename):
            raise IOError("No vecpot data file with name: %s" % (vpot_filename))

        ## Set column names
        if (self.qprop_dim == 34):
            column_names = ['time', 'vpot_z']
        elif (self.qprop_dim == 44):
            column_names = ['time', 'vpot_x', 'vpot_y']
        else:
            raise IOError("Unsupported qprop_dimension: %d\n" % (self.qprop_dim))
        #self.data = pd.read_table(vpot_filename, header=None, sep='\s+', names=['time','vpot_x', 'vpot_y'])
        self.data = pd.read_table(vpot_filename, header=None, sep='\s+', names=column_names)
        self.fileLoadingCompleted = True



    def plot(self, saveFigureName='', rcParams={}, fig=None, subplot_index=(), **kwargs):
        """
        'kwargs' is delivered to plotter such as matplotlib.pyplot.plot() etc.
        """

        if not self.fileLoadingCompleted:
            try:
                self.load()
            except:
                raise IOError("No data have been loaded from file")

        ## Parse 'kwargs' for sophisticated plotting
        if 'figsize' in kwargs.keys(): self.figsize = kwargs.pop('figsize')
        else: self.figsize = (13,7)


        ## Set figure object
        if fig == None:
            fig = plt.figure(figsize=self.figsize)
        else:
            # [NOTE] it would be good to check if 'fig' is of type 'matplotlib.figure.Figure'
            fig = fig


        ## Set projection mode
        if self.qprop_dim == 34:
            projection = None
        elif self.qprop_dim == 44:
            projection = '3d'
        else:
            raise IOError("Unsupported qprop_dimension: %d\n" % (self.qprop_dim))


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
            ax = fig.gca(projection=projection)


        ## Actual plotting happens here
        if (self.qprop_dim == 34):
            # Data Generation
            timeMask = self.data['time'] < self.get_duration()
            x = self.data.loc[timeMask,'time']
            y = self.data.loc[timeMask,'vpot_z']

            # Plot
            ax.plot(x,y, **kwargs)

            # Labeling
            ax.set_xlabel('time (a.u.)')
            ax.set_ylabel(r'$A_z$(a.u.)')

        elif (self.qprop_dim == 44):
            # Data Generation
            timeMask = self.data['time'] < self.get_duration()
            x = self.data.loc[timeMask,'time']
            y = self.data.loc[timeMask,'vpot_x']
            z = self.data.loc[timeMask,'vpot_y']

            # Plot
            #ax = fig.gca(projection='3d')
            ax.plot(x,y,z, **kwargs)

            # Labeling
            ax.set_xlabel('time (a.u.)')
            ax.set_ylabel(r'$A_x$(a.u.)')
            ax.set_zlabel(r'$A_y$(a.u.)')

        else:
            raise IOError("Unsupported qprop_dimension: %d\n" % (self.qprop_dim))

        ## Save figure if specified
        if (saveFigureName != ''):
            plt.savefig(saveFigureName)


        ## Return handles
        return fig, ax


