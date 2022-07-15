import pyleoclim as pyleo
import multiprocessing as mp
import numpy as np

from ammonite.utils.rm import rm
from ammonite.utils.range_finder import range_finder

__all__ = [
    'rm_search'
]

def rm_search(series, eps, m, tau ,target_density, tolerance,amp = 15, initial_hitrate = None, num_processes = None):
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
    num_process : int
        Number of processes to run, automatically set to your cpu count
    amp : int
        The amplitude of the range of epsilon value search. Higher values cover ground quickly but converge slowly, the opposite is true for lower values
    '''
    
    if num_processes == None:
        num_processes = mp.cpu_count() - 2
    
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
            
            results = {}
            for val in eps_range:
                r = pool.apply_async(rm, args = (series, val, m, tau))
                results[val] = r
            
            pool.close()
            pool.join()

        if flag is True:
            for i in results:
                matrix = results[i].get()['rm']
                new_hitrate = np.sum(matrix)/np.size(matrix)

                if np.abs(new_hitrate - .05) < np.abs(hitrate -.05):
                    hitrate = new_hitrate
                    eps = i

        else:
            continue
    
    return results