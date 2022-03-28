import pyleoclim as pyleo
import pandas as pd
import numpy as np

class TimeEmbeddedSeries:
    '''Time embedded time series object. Precursor to recurrence matrix and recurrence network.
    '''
    def __init__(self,embedded_series,time,embedding_dimension,embedding_delay,series= None, value_name=None,value_unit=None,time_name=None,time_unit=None,label=None):
        self.embedded_series = embedded_series
        self.time = time
        self.embedding_dimension = embedding_dimension
        self.embedding_delay = embedding_delay
        self.series = series
        self.value_name = value_name
        self.value_unit = value_unit
        self.time_name = time_name
        self.time_unit = time_unit
        self.label = label

    # def create_RecurrenceMatrix(self,epsilon):
    #     return RecurrenceMatrix()

    # def create_RecurrenceNetwork(self,epsilon):
    #     return RecurrenceNetwork()

class RecurrenceMatrix:
    '''Recurrence matrix object. Used for Recurrence Quantification Analysis (RQA).
    '''
    def __init__(self,matrix,time,epsilon,series = None,value_name=None,value_unit=None,time_name=None,time_unit=None,label=None):
        self.series = series
        self.matrix = matrix
        self.time = time
        self.epsilon = epsilon
        self.series = series
        self.value_name = value_name
        self.value_unit = value_unit
        self.time_name = time_name
        self.time_unit = time_unit
        self.label = label

class RecurrenceNetwork:
    '''Recurrence network object. Used for Recurrence Network Analysis (RNA).
    '''
    def __init__(self,matrix,time,epsilon,series=None,value_name=None,value_unit=None,time_name=None,time_unit=None,label=None):
        self.matrix = matrix
        self.time = time
        self.epsilon = epsilon
        self.series = series
        self.value_name = value_name
        self.value_unit = value_unit
        self.time_name = time_name
        self.time_unit = time_unit
        self.label = label

def create_TimeEmbeddedSeries(series,embedding_dimension,embedding_delay,cut='beginning'):
    '''Function to create time embedded time series object. returns ammonite.TimeEmbeddedSeries type object.

    series : pyleo.Series object or pandas.Series object
        Time series to be embedded
    
    embedding_dimension : int
        Embedding dimension
    
    embedding delay : int
        Embedding delay

    cut : str
        Where to cut data from series, either 'beginning' or 'end'.
    '''

    if isinstance(series,pyleo.core.ui.Series):
        values = series.value
        time = series.time[embedding_dimension*embedding_delay:]
        value_name = series.value_name
        value_unit = series.value_unit
        time_name = series.time_name
        time_unit = series.time_unit
        label = series.label
    
    elif isinstance(series,pyleo.core.ui.LipdSeries):
        values = series.value
        time = series.time[embedding_dimension*embedding_delay:]
        value_name = series.value_name
        value_unit = series.value_unit
        time_name = series.time_name
        time_unit = series.time_unit
        label = series.label

    elif isinstance(series,pd.core.series.Series):
        values = series.values
        time = list(series.index)[embedding_dimension*embedding_delay:]
        value_name = None
        value_unit = None
        time_name = None
        time_unit = None
        label = None

    else:
        print(type(series))
        raise ValueError('Object is not pyleoclim Series or LipdSeries or pandas Series object. Please convert into one of these object types.')

    manifold = np.ndarray(shape = (len(values)-(embedding_dimension*embedding_delay),embedding_dimension))

    if cut == 'beginning':
        time = time[embedding_dimension*embedding_delay:]
        for idx, i in enumerate(values):
            if idx >= (embedding_dimension*embedding_delay):
                embedding = []
                for j in range(embedding_dimension):
                    embedding.append(values[idx - (j*embedding_delay)])
                manifold[idx - (embedding_dimension*embedding_delay)] = (np.array(embedding))
    elif cut == 'end':
        time = time[:embedding_dimension*embedding_delay]
        for idx, i in enumerate(values):
            if idx <= (embedding_dimension*embedding_delay):
                embedding = []
                for j in range(embedding_dimension):
                    embedding.append(values[idx + (j*embedding_delay)])
                manifold[idx ] = (np.array(embedding))

    return TimeEmbeddedSeries(
        manifold=manifold,
        time = time,
        embedding_dimension = embedding_dimension,
        embedding_delay=embedding_delay,
        series=series,
        value_name=value_name,
        value_unit=value_unit,
        time_name=time_name,
        time_unit=time_unit,
        label=label
    )