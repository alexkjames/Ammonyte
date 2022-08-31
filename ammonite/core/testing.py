import ammonyte as amt
import numpy as np
import matplotlib.pyplot as plt

def gen_normal(loc=0, scale=1, nt=100):
    ''' Generate random data with a Gaussian distribution
    '''
    t = np.arange(nt)
    np.random.seed(42)
    v = np.random.normal(loc=loc, scale=scale, size=nt)
    ts = amt.Series(t,v)
    return ts

ts_normal = gen_normal(nt=100)

ts_normal.time_name = 'time'
ts_normal.time_unit = 'yr'
ts_normal.value_name = 'value'
ts_normal.value_unit = 'permil'

amt_td = amt.TimeEmbeddedSeries(ts_normal,3,1)

eps = amt_td.find_epsilon(1,parallelize=False)

rm = amt_td.create_recurrence_matrix(1)

lp_series = rm.laplacian_eigenmaps(5,3,smooth=False)

lp_series.confidence_scatter_plot(marker='o',color='red',label='Scatter',legend=True,background_series=ts_normal,background_kwargs={'alpha':1})

plt.show()