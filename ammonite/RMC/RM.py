import pyleoclim as pyleo
import numpy as np
    
def RM(series, eps, m, delay, invert_time_axis = False):
    '''Function to calculate recurrence matrix from pyleoclim or pandas series
    
    Parameters
    ----------
    
    series : pyleoclim.Series or pandas.Series
        Timeseries from which the recurrence matrix will be calculated
    eps : int
        Epsilon value for calculation of the matrix
    m : int
        Embedding parameter
    delay : int
        Delay parameter for time embedding
    invert_time_axis : bool
        Whether or not to invert the time axis of your input
    '''
    
    if type(series) == 'pyleoclim.core.ui.Series':
        values = series.value
        time_axis = series.time[m*delay:]
    elif type(series) == 'pandas.core.series.Series':
        values = series.values
        time_axis = list(series.index)[m*delay:]
    #time delay is currently assumed as 1
    if invert_time_axis == True:
        values = np.flip(values) #Flip to have time axis moving forward (B.P. units)
        time_axis = np.flip(time_axis])
        
    embedded_series = []

    for idx, i in enumerate(values):
        if idx >= (m*delay):
            embedding = []
            for j in range(m):
                embedding.append(values[idx - (j*delay)])
            embedded_series.append(np.array(embedding))
        
    matrix = np.zeros((len(embedded_series), len(embedded_series)))

    for idj, j in enumerate(embedded_series[1:]):
        for idk, k in enumerate(embedded_series[1:idj]):
            x = eps - np.linalg.norm(embedded_series[idj]-embedded_series[idk])
            matrix[idj][idk] = np.heaviside(x,np.nan)

    matrix = matrix + matrix.T
    np.fill_diagonal(matrix, 1)

    return {'RM':matrix, 'time_axis':time_axis}