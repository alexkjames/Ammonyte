
import pyleoclim as pyleo
import pandas as pd

class TimeEmbeddedSeries:
    '''Time embedded time series object. Precursor to recurrence matrix and recurrence network.
    '''
    def __init__(self,manifold,time,embedding_dimension,embedding_delay,series= None, value_name=None,value_unit=None,time_name=None,time_unit=None,label=None):
        self.manifold = manifold
        self.time = time
        self.embedding_dimension = embedding_dimension
        self.embedding_delay = embedding_delay
        self.series = series
        self.value_name = value_name
        self.value_unit = value_unit
        self.time_name = time_name
        self.time_unit = time_unit
        self.label = label

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

    elif isinstance(series,pandas.core.series.Series):
        values = series.values
        time = list(series.index)[embedding_dimension*embedding_delay:]
        value_name = None
        value_unit = None
        time_name = None
        time_unit = None
        label = None

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

def create_RecurrenceMatrix(manifold,epsilon):

def create_RecurrenceNetwork(manifold,epsilon):