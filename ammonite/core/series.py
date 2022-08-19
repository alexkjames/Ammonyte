import pyleoclim as pyleo
import numpy as np

from ..core.time_embedded_series import TimeEmbeddedSeries
from ..utils.parameters import tau_search

class Series(pyleo.Series):
    '''Ammonite series object, launching point for most ammonite analysis.

    Child of pyleoclim.Series, so shares all methods with pyleoclim.Series plus those
    defined here.
    '''

    def embed(self,m,tau=None,):
        '''Function to time delay a time series'''

        value_name = self.value_name
        value_unit = self.value_unit
        time_name = self.time_name
        time_unit = self.time_unit
        label = self.label

        if tau is None:
            tau = tau_search(self)

        values = self.value
        time_axis = self.time[:(-m*tau)]
        
        manifold = np.ndarray(shape = (len(values)-(m*tau),m))

        for idx, i in enumerate(values):
            if idx < (len(values)-(m*tau)):
                manifold[idx] = values[idx:idx+(m*tau):tau]

        embedded_data = manifold
        embedded_time = time_axis

        return TimeEmbeddedSeries(self,m,tau,embedded_data,embedded_time,value_name,value_unit,
                                  time_name,time_unit,label)