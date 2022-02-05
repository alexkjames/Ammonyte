import numpy as np

def range_finder(eps, hitrate, target_hitrate, tolerance, num_processes, amp):
    
    if hitrate < (target_hitrate - tolerance):
        
        miss = target_hitrate - hitrate
        
        eps_range = np.linspace(eps, eps + np.sqrt(miss*amp), num_processes)
        
        print(f'Epsilon value is: {eps:.4f}')
        print(f'Hitrate is {hitrate:.4f}. This is outside of the tolerance window.')
        
        flag = True
        
        return eps_range, flag
    
    elif hitrate > (target_hitrate + tolerance):
        
        miss = hitrate - target_hitrate
        
        eps_range = np.linspace(eps, eps - np.sqrt(miss*amp)**2, num_processes)
        
        print(f'Epsilon value is: {eps:.4f}')
        print(f'Hitrate is {hitrate:.4f}. This is outside of the tolerance window.')
        
        flag = True
        
        return eps_range, flag
        
    else:
        
        print(f'Epsilon value is: {eps:.4f}')
        print(f'Hitrate is {hitrate:.4f} which is within the tolerance window')
        
        flag = False
        
        return eps, flag