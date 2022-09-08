''' Tests for ammonyte.core.time_embedded_series
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
import ammonyte as amt
import numpy as np

def load_data():
    #Loads stott MD982176 record
    d = pyleo.Lipd('../example_data/MD982176.Stott.2004.lpd')
    return d

def gen_normal(loc=0, scale=1, nt=100):
    ''' Generate random data with a Gaussian distribution
    '''
    t = np.arange(nt)
    np.random.seed(42)
    v = np.random.normal(loc=loc, scale=scale, size=nt)
    ts = amt.Series(t,v)
    return ts

class TestCoreTimeEmbeddSeriesCreateRecurrenceMatrix:
    '''Tests for create_recurrence_matrix
    '''

    @pytest.mark.parametrize('m,tau',[(3,3)])
    def test_create_recurrence_matrix_t0(self,m,tau):
        ts_normal = gen_normal()

        td_sst = ts_normal.embed(m,tau)

        td_sst.create_recurrence_matrix(1)

class TestCoreTimeEmbeddSeriesCreateRecurrenceNetwork:
    '''Tests for create_recurrence_network
    '''

    @pytest.mark.parametrize('m,tau',[(3,3)])
    def test_create_recurrence_network_t0(self,m,tau):
        ts_normal = gen_normal()

        td_sst = ts_normal.embed(m,tau)

        td_sst.create_recurrence_network(1)

class TestCoreTimeEmbeddSeriesFindEpsiilon:
    '''Tests for find_epsilon
    '''
    def test_find_eps_t0(self):
        ts_normal = gen_normal()

        td_sst = ts_normal.embed(3,1)

        td_sst.find_epsilon(1)