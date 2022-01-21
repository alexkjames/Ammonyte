import pyleoclim as pyleo
from RM import RM
from utils import eps_rangefinder
import multiprocessing as mp
import numpy as np

def RM_search(series, eps, m, delay ,target_hitrate, tolerance,invert_time_axis = False, initial_hitrate = None, 
              num_processes = mp.cpu_count()):
    
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
            
            eps_range, flag = eps_rangefinder(eps,hitrate,target_hitrate,tolerance,num_processes)
            
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