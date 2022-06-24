import numpy as np

def range_finder(eps, hitrate, target_hitrate, tolerance, num_processes, amp):
    
    if hitrate < (target_hitrate - tolerance):
        
        miss = target_hitrate - hitrate

        eps_bounds = (eps, eps + np.sqrt(miss*amp))
        
        eps_range = np.linspace(min(eps_bounds), max(eps_bounds), num_processes)
        
        print(f'Epsilon value is: {eps:.4f}')
        print(f'Hitrate is {hitrate:.4f}. This is outside of the tolerance window.')
        
        flag = True
        
        return eps_range, flag
    
    elif hitrate > (target_hitrate + tolerance):
        
        miss = hitrate - target_hitrate

        eps_bounds = (eps, eps - np.sqrt(miss*amp))
        
        eps_range = np.linspace(min(eps_bounds), max(eps_bounds), num_processes)
        
        print(f'Epsilon value is: {eps:.4f}')
        print(f'Hitrate is {hitrate:.4f}. This is outside of the tolerance window.')
        
        flag = True
        
        return eps_range, flag
        
    else:
        
        print(f'Epsilon value is: {eps:.4f}')
        print(f'Hitrate is {hitrate:.4f} which is within the tolerance window')
        
        flag = False
        
        return eps, flag