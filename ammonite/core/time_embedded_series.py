#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pyleoclim as pyleo
import numpy as np
import pandas as pd
import multiprocessing as mp

from pyrqa.time_series import EmbeddedSeries
from pyrqa.settings import Settings
from pyrqa.analysis_type import Classic
from pyrqa.neighbourhood import FixedRadius
from pyrqa.metric import EuclideanMetric
from pyrqa.computation import RPComputation

from .recurrence_matrix import RecurrenceMatrix
from .recurrence_network import RecurrenceNetwork
from ..utils.rm_search import rm_search
from ..utils.plotting import get_labels


class TimeEmbeddedSeries:
    '''Time embedded time series object. Precursor to recurrence matrix and recurrence network.

    series : pyleo.Series object or pandas.Series object
        Time series to be embedded
    
    m : int
        Embedding dimension
    
    tau : int
        Embedding delay

    embedded_data : array
        Time delay embedded data. If not passed will be calculated from series, m, and tau

    embedded_time : array
        Time axis corresponding to embedded_data. If embedded_data is passed without embedded_time, an error will be raised

    value_name : str
        Value name

    value_unit : str
        Value units

    time_name : str
        Time name

    time_unit : str
        Time units

    label : str
        Label for embedding
    '''

    def __init__(self,series,m,tau,embedded_data=None,embedded_time=None,value_name=None,value_unit=None,time_name=None,time_unit=None,label=None):
        self.series = series
        self.m = m
        self.tau = tau
        self.embedded_data = embedded_data
        self.embedded_time = embedded_time
        self.value_name = value_name
        self.value_unit = value_unit
        self.time_name = time_name
        self.time_unit = time_unit
        self.label = label

        if self.embedded_data is not None and self.embedded_time is None:
            raise ValueError('Embedded data was passed without associated time axis. Please pass neither or both')

        if self.embedded_data is None:

            if isinstance(series,pyleo.core.Series) or isinstance(series,pyleo.core.LipdSeries):
                values = series.value
                time_axis = series.time[:(-m*tau)]

            elif isinstance(series,pd.Series):
                values = series.values
                time_axis = list(series.index)[:(-m*tau)]

            else:
                raise ValueError('Unrecognized data type. Please pass a pyleoclim Series or pandas Series type object')
            
            manifold = np.ndarray(shape = (len(values)-(m*tau),m))

            for idx, i in enumerate(values):
                if idx < (len(values)-(m*tau)):
                    manifold[idx] = values[idx:idx+(m*tau):tau]

            self.embedded_data = manifold
            self.embedded_time = time_axis

        if self.value_name is None:
            self.value_name = self.series.value_name
        
        if self.value_unit is None:
            self.value_unit = self.series.value_unit

        if self.time_name is None:
            self.time_name = self.series.time_name
        
        if self.time_unit is None:
            self.time_unit = self.series.time_unit
        
        if self.label is None:
            self.label = self.series.label

    def find_epsilon(self,target_density,tolerance,num_processes=None,search_kwargs=None):
        '''Function to find epsilon value given target recurrence matrix density
        
        Parameters
        ----------

        target_density : float
            Target density of recurrent points in recurrence matrix. Should be a value between zero and one.

        tolerance : float
            Allowable difference between desired density and calculated density.
        
        num_processes : int
            Number of parallel processes to run. Check number of available processes on your machine with multiprocessing.cpu_count().

        search_kwargs : dict
            Key word arguments for ammonite.utils.rm_search.rm_search.

        Returns
        -------

        epsilon : float
            Epsilon value that produces desired recurrence density within tolerance value provided.

        See also
        --------

        ammonite.utils.rm_search
        '''

        search_kwargs={} if search_kwargs is None else search_kwargs.copy()

        if num_processes is None:
            num_processes = int(mp.cpu_count()/2)
        elif num_processes > mp.cpu_count():
            raise ValueError('num_processes is greater than the number of available cpus.')

        epsilon = rm_search(series=self.series,eps=1,m=self.m,tau=self.tau,target_density=target_density,tolerance=tolerance,**search_kwargs)

        return epsilon


    def create_recurrence_matrix(self,epsilon):
        '''Function to create Recurrence Matrix object
        
        Parameters
        ----------
        
        epsilon : float
            Fixed radius used to calculate whether two points are recurrent
            
        Returns
        -------
        
        RecurrenceMatrix : ammonite.RecurrenceMatrix object'''

        ts = EmbeddedSeries(self.embedded_data)

        settings = Settings(ts,
                            analysis_type=Classic,
                            neighbourhood=FixedRadius(epsilon),
                            similarity_measure=EuclideanMetric)

        computation = RPComputation.create(settings,
                                        verbose=False)

        result = computation.run()

        matrix = result.recurrence_matrix

        return RecurrenceMatrix(matrix=matrix,time=self.embedded_time,epsilon=epsilon,series=self.series,
                                value_name=self.value_name,value_unit=self.value_unit,time_name=self.time_name,
                                time_unit=self.time_unit,label=self.label)

    def create_recurrence_network(self,epsilon):
        '''Function to create Recurrence Network object
        
        Parameters
        ----------
        
        epsilon : float
            Fixed radius used to calculate whether two points are recurrent.
            
        Returns
        -------
        
        RecurrenceNetwork : ammonite.RecurrenceNetwork object'''
        ts = EmbeddedSeries(self.embedded_data)

        settings = Settings(ts,
                            analysis_type=Classic,
                            neighbourhood=FixedRadius(epsilon),
                            similarity_measure=EuclideanMetric)

        computation = RPComputation.create(settings,
                                        verbose=False)

        result = computation.run()

        matrix = result.recurrence_matrix

        return RecurrenceNetwork(matrix=matrix,time=self.embedded_time,epsilon=epsilon,series=self.series,
                                value_name=self.value_name,value_unit=self.value_unit,time_name=self.time_name,
                                time_unit=self.time_unit,label=self.label)