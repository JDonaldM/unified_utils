import numpy as np
import pytest
from unified_utils import likelihood
import yaml

with open("default.yaml", "r") as stream:
    config = yaml.safe_load(stream)


class TestFixParams:
    '''
    Tests for fix_params.
    '''

    def test_def(self):
        '''
        Test default settings with theta as expected.
        '''

        # Generate some random samples with the expected shape
        # for the default settings.
        theta = np.random.randn(10,11)

        # Fix the parameters
        fixed_theta = likelihood.fix_params(theta)

        # Assert the relevant parameters have been set to zero.
        assert np.all(fixed_theta[:,[7,10,12]]==0.)
        # Assert the free parameters remain unchanged.
        assert np.all(theta==fixed_theta[:,[0,1,2,3,4,5,6,8,9,11,13]])

    def test_none(self):
        '''
        Test no fixed parameters with theta as expected.
        '''

        # Generate some random samples with no fixed parameters.
        theta = np.random.randn(10,14)

        # Define fixed parameter dictionary with no fixed parameters.
        fd = {'w_c':None, 'w_b':None, 'h':None, 'As':None, 'b1':None, 'c2':None,
        'b3':None, 'c4':None, 'cct':None, 'cr1':None, 'cr2':None, 'ce1':None,
        'cmono':None, 'cquad':None}

        # Call the function but fix no parameters.
        fixed_theta = likelihood.fix_params(theta, fixed_dict=fd)

        # Assert theta in equals theta out.
        assert np.all(theta==fixed_theta)

    def test_all(self):
        '''
        Test all fixed parameters with theta as expected.
        '''

        # Generate some random samples with no fixed parameters.
        # The shape of theta should be irrelevant here.
        theta = np.random.randn(10,14)

        # Define fixed parameter dictionary with all fixed parameters.
        fd = {'w_c':0., 'w_b':0., 'h':0., 'As':0., 'b1':0., 'c2':0.,
        'b3':0., 'c4':0., 'cct':0., 'cr1':0., 'cr2':0., 'ce1':0.,
        'cmono':0., 'cquad':0.}

        # Make sure error is raised.
        with pytest.raises(ValueError):
            likelihood.fix_params(theta, fixed_dict=fd)

    def test_wyaml(self):

        # Generate some random samples with the expected shape
        # for the default settings.
        theta = np.random.randn(10,11)

        # Define fixed parameter dictionary from yaml input.
        fd = {p: config['Parameters'][p]['fixed'] if 'fixed' in config['Parameters'][p] else None\
              for p in config['Parameters']}

        fixed_theta = likelihood.fix_params(theta)

        # Assert the relevant parameters have been set to zero.
        assert np.all(fixed_theta[:,[7,10,12]]==0.)
        # Assert the free parameters remain unchanged.
        assert np.all(theta==fixed_theta[:,[0,1,2,3,4,5,6,8,9,11,13]])


class TestKerns:

    P11 = np.random.randn(2,10,3,40)
    Ploop = np.random.randn(2,10,12,40)
    Pct = np.random.randn(2,10,6,40)
    ng = 1.
    km = 1.
    bias = np.random.randn(10,7)
    stoch = np.random.randn(10,3)
    kobs = np.arange(0,40)
    growth = np.random.randn(10,1)

    def test_PNG(self):

        PNG = likelihood.calc_P_NG(self.P11, self.Ploop, self.Pct,
                                   self.bias, self.growth, self.stoch,
                                   self.ng, self.kobs)

        assert PNG.shape==(10,80)