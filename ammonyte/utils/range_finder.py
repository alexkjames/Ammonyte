import numpy as np

def range_finder(eps, density, target_density, tolerance, num_processes, amp):
    
    if density < (target_density - tolerance):
        
        miss = target_density - density
        eps_bounds = (eps, eps + miss*amp)
        eps_range = np.linspace(min(eps_bounds), max(eps_bounds), num_processes)
        flag = False

        return eps_range, flag
    
    elif density > (target_density + tolerance):
        
        miss = density - target_density
        eps_bounds = (eps, eps - miss*amp)    
        eps_range = np.linspace(min(eps_bounds), max(eps_bounds), num_processes)      
        flag = False
        
        return eps_range, flag
        
    else:
            
        flag = True
        
        return eps, flag