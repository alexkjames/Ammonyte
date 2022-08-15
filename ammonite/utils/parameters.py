import pyinform

import pyleoclim as pyleo
import multiprocessing as mp
import numpy as np

from scipy.signal import argrelextrema
from itertools import repeat

from ammonite.utils.rm import rm
from ammonite.utils.range_finder import range_finder

__all__ = [
    'tau_search',
    'eps_search'
]

def tau_search(series,num_lags=30,return_MI = False):
    '''Find optimal tau value for time delay embedding.
    
    First minimum of mutual information between series and time lagged copies of itself
    is "optimal" in this case in accordance with Abarnabel's "Analysis of Observed Chaotic Data"

    Parameters
    ----------

    series : pyleo.Series
        Series for which we'd like to find the optimal tau value

    num_lags : int
        Number of time delays to consider. Default is 30

    return_MI : bool, {True,False}
        Whether or not to return the list of mutual information values. 
        Useful if the first minimum seems spurious and you'd like to inspect the results.

    Returns
    -------

    tau : int
        Optimal time delay parameter according to first minimum of mutual information

    MI : list
        List of mutual information values.
        Indices + 1 correspond to amount of lag (index 0 is lag 1, index 1 is lag 2, etc.).
        Only returned if return_MI is set to True.
    
    Citations
    ---------
    
    I., Abarbanel Henry D. Analysis of Observed Chaotic Data. Springer, 1997. 
    '''
    lags = np.arange(1,num_lags)
    MI = []

    for lag in lags:
        values = series.value[:-lag] - min(series.value[:-lag])
        lagged_values = series.value[lag:] - min(series.value[lag:])
        MI.append(pyinform.mutualinfo.mutual_info(values, lagged_values, local=False))

    best_tau = argrelextrema(np.array(MI),np.less)[0][0] + 1

    if return_MI is True:
        return best_tau,MI
    else:
        return best_tau

def eps_search(series, m, tau ,target_density, tolerance, eps=1, amp = 15, initial_hitrate = None, num_processes = None):
    '''Tool to find epsilon value tuned for specific target density in recurrence matrix
    
    Parameters
    ----------
    
    series : pyleoclim.series object (pandas.series support incoming)
        Timeseries used to create recurrence matrix
    eps : float
        Starting epsilon value (best guess)
    m : int
        Embedding parameter for time delay embedding
    tau : int
        Delay parameter for time delay embedding
    target_density : float
        Desired recurrence matrix hitrate
    tolerance : float
        Amount of allowable difference between target hitrate and actual hitrate
    initial_hitrate : float
        If you've already calculated the initial hitrate for your settings you can pass it here to save computation time
    num_processes : int
        Number of processes to run, automatically set to your cpu count
    amp : int
        The amplitude of the range of epsilon value search. Higher values cover ground quickly but converge slowly, the opposite is true for lower values
    '''
    
    if num_processes is None:
        if mp.cpu_count() > 2:
            num_processes = mp.cpu_count() - 2
        else:
            num_processes = 1
    
    if initial_hitrate == None:
        print(f'Finding initial hitrate from given epsilon value: {eps}')
        initial_result = rm(series, eps, m, tau)
        initial_hitrate = np.sum(initial_result['rm'])/np.size(initial_result['rm'])
        print(f'Initial hitrate is {initial_hitrate:.4f}')
    
    if np.abs(initial_hitrate - target_density) <= tolerance:
        print('Initial hitrate is within the tolerance window!')
        results = {'Epsilon':eps,'Output':initial_result}
        return results
    else:
        print('Initial hitrate is not within the tolerance window, searching...')
        hitrate = initial_hitrate
        flag = True

    while flag:

        with mp.Pool(num_processes) as pool:
            
            eps_range, flag = range_finder(eps,hitrate,target_density,tolerance,num_processes,amp)
            
            if flag == False:
                
                eps = eps_range
                results = {'Epsilon':eps,'Output':rm(series, eps, m, tau)}
                return results
            
            r = pool.starmap(rm, zip(repeat(series), eps_range, repeat(m), repeat(tau)))
            
            pool.close()
            pool.join()

        if flag is True:
            for item in r:
                matrix = item['rm']
                new_eps = item['eps']
                new_hitrate = np.sum(matrix)/np.size(matrix)

                if np.abs(new_hitrate - .05) < np.abs(hitrate -.05):
                    hitrate = new_hitrate
                    eps = new_eps

        else:
            continue
    
    return results