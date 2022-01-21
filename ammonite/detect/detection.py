import numpy as np
import pyleoclim as pyleo
import scipy as sp
from ammonite.RMC.RM import RM
from ammonite.fisher.fisher import FI

def detect_change(series, eps, m, delay, w_size, w_incre, invert_time_axis = False):
    '''Function to run regime change detection workflow
    
    Parameters
    ----------
    
    series : Pyleoclim series object
        See https://pyleoclim-util.readthedocs.io/en/latest/core/ui.html#series-pyleoclim-series
    
    eps : float
        Epsilon value for recurrence matrix calculation
    
    m : int
        Embedding dimension
        
    delay : int
        Embedding delay
        
    invert_time_axis : bool
        Whether or not to invert the time axis when generating the recurrence matrix
        
    w_size : int
        Window size for the fisher information 
    
    w_incre : int 
        Window increment for the fisher information 
    '''
    
    RM_res = RM(series,eps,m,delay,invert_time_axis)
    
    rm = RM_res['RM']
    time_axis = RM_res['time_axis']
    
    W = rm + 1

    D = np.zeros(W.shape)

    for i in range(len(W)):
        D[i,i] = np.sum(W[:,i])
        
    L = D - W

    eigval, eigvec = sp.linalg.eigh(L,D)
    
    eig_data = []
    for idx, i in enumerate(time_axis):
        eig_data.append([i,eigvec[idx,1],eigvec[idx,2],eigvec[idx,3],eigvec[idx,4]])
        
    fisher_info = FI(eig_data,w_size,w_incre)
    
    return fisher_info