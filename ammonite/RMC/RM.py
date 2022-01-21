import pyleoclim as pyleo
import numpy as np
    
def RM(series, eps, m, delay, invert_time_axis = False):

    #time delay is currently assumed as 1
    if invert_time_axis == True:
        values = np.flip(series.value) #Flip to have time axis moving forward (B.P. units)
        time_axis = np.flip(series.time[m*delay:])
    elif invert_time_axis == False:
        values = series.value
        time_axis = series.time[m*delay:]
        
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