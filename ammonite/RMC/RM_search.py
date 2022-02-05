import pyleoclim as pyleo
import multiprocessing as mp
import numpy as np

from ammonite.RMC.RM import RM
from ammonite.utils.range_finder import range_finder

def RM_search(series, eps, m, delay ,target_hitrate, tolerance,amp = 15,invert_time_axis = False, initial_hitrate = None, 
              num_processes = None):
    '''Tool to find epsilon value tuned for specific target hitrate in recurrence matrix
    
    Parameters
    ----------
    
    series : pyleoclim.series object (pandas.series support incoming)
        Timeseries used to create recurrence matrix
    eps : float
        Starting epsilon value (best guess)
    m : int
        Embedding parameter for recurrence matrix
    delay : int
        Delay parameter for time embedding
    target_hitrate : float
        Desired recurrence matrix hitrate
    tolerance : float
        Amount of allowable difference between target hitrate and actual hitrate
    invert_time_axis : bool
        Whether or not to invert the time axis of the series being passed
    initial_hitrate : float
        If you've already calculated the initial hitrate for your settings you can pass it here to save computation time
    num_process : int
        Number of processes to run, automatically set to your cpu count
    amp : int
        The amplitude of the range of epsilon value search. Higher values cover ground quickly but converge slowly, the opposite is true for lower values
    '''
    
    if num_processes == None:
        num_processes = mp.cpu_count()
    
    if initial_hitrate == None:
        print(f'Finding initial hitrate from given epsilon value: {eps}')
        initial_result = RM(series, eps, m, delay, invert_time_axis)
        initial_hitrate = np.sum(initial_result['RM'])/np.size(initial_result['RM'])
        print(f'Initial hitrate is {initial_hitrate:.4f}')
    
    if np.abs(initial_hitrate - target_hitrate) <= tolerance:
        print('Initial hitrate is within the tolerance window!')
        return initial_result
    else:
        print('Initial hitrate is not within the tolerance window, searching...')
        hitrate = initial_hitrate
        flag = True

    while flag:

        with mp.Pool(num_processes) as pool:
            
            eps_range, flag = range_finder(eps,hitrate,target_hitrate,tolerance,num_processes,amp)
            
            if flag == False:
                
                eps = eps_range
                results = RM(series, eps, m, delay, invert_time_axis)
                return results
            
            results = {}
            for val in eps_range:
                r = pool.apply_async(RM, args = (series, val, m, delay, invert_time_axis))
                results[val] = r
            
            pool.close()
            pool.join()

        if flag == True:
            for i in results:
                matrix = results[i].get()['RM']
                new_hitrate = np.sum(matrix)/np.size(matrix)

                if np.abs(new_hitrate - .05) < np.abs(hitrate -.05):
                    hitrate = new_hitrate
                    eps = i

        else:
            continue
    
    return results