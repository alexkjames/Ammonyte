import numpy as np
import pandas as pd
from sklearn.utils import resample
from scipy.stats import scoreatpercentile
    
def confidence_interval(series,upper=95,lower=5,w=50,n_samples=10000,random_state = 42):
    '''Function to calculate upper and lower values for passed confidence interval on series object via bootstrapping
       Designed to be used to conduct bootstrap testing on fisher information series

    fisher_information : pyleoclim.Series
        Series to be evaluated

    upper : int
        Upper bound on confidence interval

    w : int
        Size of random sample
    
    n_samples : int
        Number of random samples
    
    random_state : int
        Random state for sampling
    '''
    
    sub_arrays = []
    values = list(series.value)
    
    for i in range(n_samples):
        subset = np.random.choice(values,size = w)
        sub_arrays.append(subset)
    
    means = []

    for sequence in sub_arrays:
        means.append(np.mean(sequence))
    
    upper_val = scoreatpercentile(means, upper)
    lower_val = scoreatpercentile(means, lower)
    
    return upper_val, lower_val