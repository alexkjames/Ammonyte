''' Tests for ammonite.core.series
Naming rules:
1. class: Test{filename}{Class}{method} with appropriate camel case
2. function: test_{method}_t{test_id}

Notes on how to test:
0. Make sure [pytest](https://docs.pytest.org) has been installed: `pip install pytest`
1. execute `pytest {directory_path}` in terminal to perform all tests in all testing files inside the specified directory
    (certain tests will only work when run from the tests directory, so make sure to run from there!)
2. execute `pytest {file_path}` in terminal to perform all tests in the specified file
3. execute `pytest {file_path}::{TestClass}::{test_method}` in terminal to perform a specific test class/method inside the specified file
4. after `pip install pytest-xdist`, one may execute "pytest -n 4" to test in parallel with number of workers specified by `-n`
5. for more details, see https://docs.pytest.org/en/stable/usage.html
'''

import pytest
import pyleoclim as pyleo
import ammonite as amt
import numpy as np

def gen_normal(loc=0, scale=1, nt=100):
    ''' Generate random data with a Gaussian distribution
    '''
    t = np.arange(nt)
    np.random.seed(42)
    v = np.random.normal(loc=loc, scale=scale, size=nt)
    ts = amt.Series(t,v)
    return ts

class TestCoreSeriesEmbed:
    '''Tests for embed function
    '''

    @pytest.mark.parametrize('m,tau',[(10,5),(10,None)])
    def test_embed_t0(self,m,tau):
        '''Test embed function with and without a tau value'''

        ts = gen_normal()

        td = ts.embed(m,tau)

class TestCoreSeriesDeterminism:
    '''Tests for determinism function
    '''

    @pytest.mark.parametrize('window_size,overlap,radius,m,tau',[(10,5,1,5,2),(12,4,.1,8,4)])
    def test_determinism_t0(self,window_size,overlap,m,tau,radius):

        ts = gen_normal()

        det = ts.determinism(window_size,overlap,m,tau,radius)

class TestCoreSeriesLaminarity:
    '''Tests for laminarity function'''

    @pytest.mark.parametrize('window_size,overlap,radius,m,tau',[(10,5,1,5,2),(12,4,.1,8,4)])
    def test_laminarity_t0(self,window_size,overlap,m,tau,radius):

        ts = gen_normal()

        lam = ts.laminarity(window_size,overlap,m,tau,radius)

class TestCoreSeriesConfidenceScatterPlot:
    '''Tests for confidence scatter plot function'''

    @pytest.mark.parametrize('marker,size,color,legend,label,xlabel,ylabel,title,scatter_kwargs,lgd_kwargs',
                            [('o',42,'red',True,'label_test','xlabel_test','ylabel_test','title_test',{'alpha':.5},{'fontsize':42}),
                            (None,None,None,False,None,None,None,None,None,None)])
    def test_confidence_scatter_plot_t0(self,marker,size,color,legend,label,xlabel,ylabel,title,scatter_kwargs,lgd_kwargs):
        '''Testing different visual plot arguments'''
        #Parameter choices are completely arbitrary, just want to test if the plotting function works
        ts_normal = gen_normal(nt=100)
        amt_td = amt.TimeEmbeddedSeries(ts_normal,3,1)
        rm = amt_td.create_recurrence_matrix(1)
        lp_series = rm.laplacian_eigenmaps(5,3,smooth=False)
        lp_series.confidence_scatter_plot(marker=marker,size=size,color=color,legend=legend,label=label,xlabel=xlabel,ylabel=ylabel,title=title,
                                          scatter_kwargs=scatter_kwargs,lgd_kwargs=lgd_kwargs)

    @pytest.mark.parametrize('transition_interval,ci_kwargs',([(0,1),None],[(1,-1),None],[None,{'upper':75,'lower':15}]))
    def test_confidence_scatter_plot_t1(self,transition_interval,ci_kwargs):
        '''Testing different confidence interval calculations'''
        #Parameter choices are completely arbitrary, just want to test if the plotting function works
        ts_normal = gen_normal(nt=100)
        amt_td = amt.TimeEmbeddedSeries(ts_normal,3,1)
        rm = amt_td.create_recurrence_matrix(1)
        lp_series = rm.laplacian_eigenmaps(5,3,smooth=False)
        lp_series.confidence_scatter_plot(transition_interval=transition_interval,ci_kwargs=ci_kwargs)

    @pytest.mark.parametrize('background_kwargs',(None,{'alpha':1}))
    def test_confidence_scatter_plot_t2(self,background_kwargs):
        '''Testing with background plot'''
        #Parameter choices are completely arbitrary, just want to test if the plotting function works
        ts_normal = gen_normal(nt=100)
        amt_td = amt.TimeEmbeddedSeries(ts_normal,3,1)
        rm = amt_td.create_recurrence_matrix(1)
        lp_series = rm.laplacian_eigenmaps(5,3,smooth=False)
        lp_series.confidence_scatter_plot(background_series=ts_normal,background_kwargs=background_kwargs)

class TestCoreSeriesConfidenceFillPlot:
    '''Tests for confidence fill plot function'''

    @pytest.mark.parametrize('line_color,fill_color,fill_alpha,legend,label,xlabel,ylabel,title,plot_kwargs,lgd_kwargs',
                            (['green','purple',1,True,'label_test','xlabel_test','ylabel_test','title_test',{'alpha':.5},{'fontsize':42}],
                            [None,None,None,False,None,None,None,None,None,None]))
    def test_confidence_fill_plot_t0(self,line_color,fill_color,fill_alpha,legend,label,xlabel,ylabel,title,plot_kwargs,lgd_kwargs):
        '''Testing different visual plot arguments'''
        #Parameter choices are completely arbitrary, just want to test if the plotting function works
        ts_normal = gen_normal(nt=100)
        amt_td = amt.TimeEmbeddedSeries(ts_normal,3,1)
        rm = amt_td.create_recurrence_matrix(1)
        lp_series = rm.laplacian_eigenmaps(5,3,smooth=False)
        lp_series.confidence_fill_plot(line_color=line_color,fill_color=fill_color,fill_alpha=fill_alpha,
                                       legend=legend,label=label,xlabel=xlabel,ylabel=ylabel,title=title,plot_kwargs=plot_kwargs,
                                       lgd_kwargs=lgd_kwargs )
        
    @pytest.mark.parametrize('transition_interval,ci_kwargs',([(0,1),None],[(1,-1),None],[None,{'upper':75,'lower':15}]))
    def test_confidence_fill_plot_t1(self,transition_interval,ci_kwargs):
        '''Testing different confidence interval calculations'''
        #Parameter choices are completely arbitrary, just want to test if the plotting function works
        ts_normal = gen_normal(nt=100)
        amt_td = amt.TimeEmbeddedSeries(ts_normal,3,1)
        rm = amt_td.create_recurrence_matrix(1)
        lp_series = rm.laplacian_eigenmaps(5,3,smooth=False)
        lp_series.confidence_fill_plot(transition_interval=transition_interval,ci_kwargs=ci_kwargs)

    @pytest.mark.parametrize('background_kwargs',(None,{'alpha':1}))
    def test_confidence_fill_plot_t2(self,background_kwargs):
        '''Testing with background plot'''
        #Parameter choices are completely arbitrary, just want to test if the plotting function works
        ts_normal = gen_normal(nt=100)
        amt_td = amt.TimeEmbeddedSeries(ts_normal,3,1)
        rm = amt_td.create_recurrence_matrix(1)
        lp_series = rm.laplacian_eigenmaps(5,3,smooth=False)
        lp_series.confidence_fill_plot(background_series=ts_normal,background_kwargs=background_kwargs)