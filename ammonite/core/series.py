import pyleoclim as pyleo
import numpy as np

from tqdm import tqdm
from pyrqa.time_series import TimeSeries
from pyrqa.settings import Settings
from pyrqa.analysis_type import Classic
from pyrqa.neighbourhood import FixedRadius
from pyrqa.metric import EuclideanMetric
from pyrqa.computation import RQAComputation

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

        return TimeEmbeddedSeries(
            self,m,tau,embedded_data,embedded_time,value_name,value_unit,time_name,
            time_unit,label)

    def determinism(self,window_size,overlap,m,tau,radius):
        '''Calculate determinism of a series

        Note that series must be evenly spaced for this method.
        See interp, bin, and gkernel methods in parent class pyleoclim.Series for details.
        
        Parameters
        ----------
        
        window_size : int
            Size of window to use when calculating recurrence plots for determinism statistic.
            Note this is in units of the time axis.
        
        overlap : int
            Amount of overlap to allow between windows.
            Note this is in units of the time axis.

        m : int
            Embedding dimension to use when performing time delay embedding,
            
        tau : int
            Time delay to use when performing time delay embedding
            
        radius : float
            Size of radius to use to calculate recurrence matrix

        Returns
        -------

        det_series : ammonite.Series
            Ammonite.Series object containing time series of the determinism statistic
        '''
       
        series = self
        windows = np.arange(int(min(series.time)),int(max(series.time)),int(overlap/2))

        cutoff_index = -int(window_size/(overlap/2))

        res = []
        window_time = []

        for window in tqdm(windows[:cutoff_index]):
            
            series_slice = series.slice((window,window+window_size))

            window_values = series_slice.value
            time = series_slice.time[int((len(series_slice.time)-1)/2)]

            ts = TimeSeries(window_values,
                            embedding_dimension = m,
                            time_delay=tau)

            settings = Settings(ts,
                                analysis_type=Classic,
                                neighbourhood=FixedRadius(radius),
                                similarity_measure=EuclideanMetric)

            computation = RQAComputation.create(settings,
                                                verbose=False)
            
            result = computation.run()

            window_time.append(time)

            res.append(result.determinism)

        det_series = pyleo.Series(window_time,res,
                            time_name=series.time_name,
                            time_unit=series.time_unit,
                            value_name='DET',
                            label=series.label)

        return det_series

    def laminarity(self,window_size,overlap,m,tau,radius):
        '''Calculate laminarity of a series

        Note that series must be evenly spaced for this method.
        See interp, bin, and gkernel methods in parent class pyleoclim.Series for details.
        
        Parameters
        ----------
        
        window_size : int
            Size of window to use when calculating recurrence plots for determinism statistic.
            Note this is in units of the time axis.
        
        overlap : int
            Amount of overlap to allow between windows
            Note this is in units of the time axis.

        m : int
            Embedding dimension to use when performing time delay embedding,
            
        tau : int
            Time delay to use when performing time delay embedding
            
        radius : float
            Size of radius to use to calculate recurrence matrix

        Returns
        -------

        lam_series : ammonite.Series
            Ammonite.Series object containing time series of the laminarity statistic
        '''

        series = self
        windows = np.arange(int(min(series.time)),int(max(series.time)),int(overlap/2))

        cutoff_index = -int(window_size/(overlap/2))

        res = []
        window_time = []

        for window in tqdm(windows[:cutoff_index]):
            
            series_slice = series.slice((window,window+window_size))

            window_values = series_slice.value
            time = series_slice.time[int((len(series_slice.time)-1)/2)]

            ts = TimeSeries(window_values,
                            embedding_dimension = m,
                            time_delay=tau)

            settings = Settings(ts,
                                analysis_type=Classic,
                                neighbourhood=FixedRadius(radius),
                                similarity_measure=EuclideanMetric)

            computation = RQAComputation.create(settings,
                                                verbose=False)
            
            result = computation.run()

            window_time.append(time)

            res.append(result.laminarity)

        lam_series = pyleo.Series(window_time,res,
                            time_name=series.time_name,
                            time_unit=series.time_unit,
                            value_name='LAM',
                            label=series.label)
        
        return lam_series

