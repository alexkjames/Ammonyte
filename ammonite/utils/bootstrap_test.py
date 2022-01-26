import numpy as np
import pandas as pd
from sklearn.utils import resample
from scipy.stats import scoreatpercentile
    
def confidence_interval(fisher_information,w,n_samples,random_state = 42):
    
    sub_arrays = []
    FI = fisher_information['FI'].copy()
    
    for i in range(n_samples):
        subset = np.random.choice(FI,size = w)
        sub_arrays.append(subset)
    
    means = []

    for sequence in sub_arrays:
        means.append(np.mean(sequence))
    
    ninety_five = scoreatpercentile(means, 95)
    five = scoreatpercentile(means, 5)
    
    return ninety_five, five