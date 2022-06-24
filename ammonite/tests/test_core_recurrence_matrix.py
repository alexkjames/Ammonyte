''' Tests for ammonite.utils.recurrence_matrix
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

def load_data():
    #Loads stott MD982176 record
    d = pyleo.Lipd('../example_data/MD982176.Stott.2004.lpd')
    return d

class TestCoreRecurrenceMatrixLaplacianEigenmaps:
    '''Tests for laplacian eigenmaps
    '''

    #@pytest.mark.parametrize('smooth',['True','False'])
    def test_laplacian_eigenmaps_t0(self,smooth=True):
        d = load_data()
        sst = d.to_LipdSeries(number=5)

        td_sst = amt.TimeEmbeddedSeries(sst,m=3,tau=3)

        rm_sst = td_sst.create_recurrence_matrix(1) 

        lp_sst = rm_sst.laplacian_eigenmaps(w_size=50,w_incre=5,smooth=smooth)