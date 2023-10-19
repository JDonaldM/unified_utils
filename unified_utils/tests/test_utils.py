from unified_utils import utils
import pytest
from scipy.stats import norm, uniform, truncnorm
from unified_utils import likelihood
import numpy as np
import glob
import yaml
import os
import inspect

class TestEmuBoundsCheck:

    # Define param. dict. in the form that it would be read from the
    # config file.
    param_dict={'w_c':{'prior':{'type':'uniform', 'min':'emu', 'max':'emu'}},
                'w_b':{'prior':{'type':'truncated normal', 'min':'emu', 'max':'emu', 'mean':0., 'stddev':1.}},
                'h':{'prior':{'type':'uniform', 'min':'emu', 'max':'emu'}},
                'As':{'prior':{'type':'uniform', 'min':'emu', 'max':'emu'}}}

    # Define emu. bound dict. with unique values.
    emu_bounds={'w_c':{'min':0., 'max':1.},
                'w_b':{'min':2., 'max':3.},
                'h':{'min':4., 'max':5.},
                'As':{'min':6., 'max':7.}}

    def test_basic_replace(self):
        '''
        Test for the case in which the function was intended to be used.
        Simply replacing ``'emu'``.   
        '''

        new_dict = utils.check_emu_bounds(self.param_dict.copy(), self.emu_bounds)

        assert (new_dict['w_c']['prior']['min']==0. and new_dict['w_c']['prior']['max']==1.)
        assert (new_dict['w_b']['prior']['min']==2. and new_dict['w_b']['prior']['max']==3.)
        assert (new_dict['h']['prior']['min']==4. and new_dict['h']['prior']['max']==5.)
        assert (new_dict['As']['prior']['min']==6. and new_dict['As']['prior']['max']==7.)

    def test_bounds_errors(self):

        # Test for param. max greater than emu bound.
        param_dict=self.param_dict.copy()
        param_dict['w_c']['prior']['max']=100.
        with pytest.raises(ValueError):
            utils.check_emu_bounds(param_dict, self.emu_bounds)

        # Test for param. min less than emu bound.
        param_dict=self.param_dict.copy()
        param_dict['w_c']['prior']['min']=-100.
        with pytest.raises(ValueError):
            utils.check_emu_bounds(param_dict, self.emu_bounds)

        # Test for fixed param. less than emu bound.
        param_dict=self.param_dict.copy()
        param_dict['w_c']={'fixed':-100.}
        with pytest.raises(ValueError):
            utils.check_emu_bounds(param_dict, self.emu_bounds)

        # Test for fixed param. greater than emu bound.
        param_dict=self.param_dict.copy()
        param_dict['w_c']={'fixed':100.}
        with pytest.raises(ValueError):
            utils.check_emu_bounds(param_dict, self.emu_bounds)

class TestPriorList:

    def test_uniform(self):
        '''Test uniform dist. works.'''

        # Specify prior in expected way.
        param_dict={'w_c':{'prior':{'type':'uniform', 'min':0., 'max':1.}}}

        # Generate prior from dict.
        prior_list = utils.make_prior_list(param_dict)

        # Make sure only one distribution.
        assert len(prior_list)==1

        # Make sure dist. is correct.
        # Here we use the moments of the distribution to check it is correct.
        # I'm sure this could be broken somehow but it works for now.
        assert prior_list[0].stats(moments='mvsk') == uniform(loc=0., scale=1.).stats(moments='mvsk')

    def test_norm(self):
        '''Test normal dist. works.'''

        # Specify prior in expected way.
        param_dict={'w_c':{'prior':{'type':'normal', 'mean':0., 'stddev':1.}}}

        # Generate prior from dict.
        prior_list = utils.make_prior_list(param_dict)

        # Make sure only one distribution.
        assert len(prior_list)==1

        # Make sure dist. is correct.
        assert prior_list[0].stats(moments='mvsk') == norm(loc=0., scale=1.).stats(moments='mvsk')

    def test_truncnorm(self):
        '''Test trunc. normal dist. works.'''

        # Specify prior in expected way.
        param_dict={'w_c':{'prior':{'type':'truncated normal',
                                    'min':1., 'max':5, 'mean':2., 'stddev':1.}}}

        # Generate prior from dict.
        prior_list = utils.make_prior_list(param_dict)

        # Make sure only one distribution.
        assert len(prior_list)==1

        # Calc. a and b
        a = (param_dict['w_c']['prior']['min']-param_dict['w_c']['prior']['mean'])/param_dict['w_c']['prior']['stddev']
        b = (param_dict['w_c']['prior']['max']-param_dict['w_c']['prior']['mean'])/param_dict['w_c']['prior']['stddev']

        # Make sure dist. is correct.
        assert prior_list[0].stats(moments='mvsk') == truncnorm(a, b, loc=2., scale=1.).stats(moments='mvsk')

    def test_both(self):
        '''Test that more than one dist. can be generated.'''

        # Specify prior in expected way.
        param_dict={'w_c':{'prior':{'type':'uniform', 'min':0., 'max':1.}},
                    'w_b':{'prior':{'type':'normal', 'mean':0., 'stddev':1.}}}

        # Generate prior from dict.
        prior_list = utils.make_prior_list(param_dict)

        # Make sure there are the right number of elements.
        assert len(prior_list)==2

        # Check the first dist. is correct.
        assert prior_list[0].stats(moments='mvsk') == uniform(loc=0., scale=1.).stats(moments='mvsk')

        # Check the second dist. is correct.
        assert prior_list[1].stats(moments='mvsk') == norm(loc=0., scale=1.).stats(moments='mvsk')

    def test_error(self):
        '''Make sure error is raised if anything other than normal, uniform, or truncated normal.'''

        # Specify a 'random' prior.
        param_dict={'w_c':{'prior':{'type':'random', 'a':0., 'b':1., 'c':100.}}}

        # Make sure error is raised.
        with pytest.raises(ValueError):
            utils.make_prior_list(param_dict)

class TestLoadLike:

    def test_bad_path(self):
        '''Make sure bad path error is raised.'''

        with pytest.raises(FileNotFoundError):
            utils.import_loglikelihood("random.py", "random")

    def test_bad_name(self):
        '''Make sure bad function name error is raised.'''

        with pytest.raises(KeyError):
            utils.import_loglikelihood(glob.glob("../likelihood.py")[0], "random")

    def test_works(self):
        '''Make sure function actually works.'''

        # Import a fix_params function.
        fix_from_file = utils.import_loglikelihood(glob.glob("../likelihood.py")[0], "fix_params")

        # Generate some data for the function.
        theta = np.random.randn(10,11)

        # Make sure the output agrees with the function imported normally.
        assert np.all(fix_from_file(theta)==likelihood.fix_params(theta))

class TestPriorEval:

    # Generate prior list with input as expected.
    param_dict={'w_c':{'prior':{'type':'uniform', 'min':0., 'max':1.}},
                'w_b':{'prior':{'type':'normal', 'mean':0., 'stddev':1.}}}
    prior_list = utils.make_prior_list(param_dict)

    def test_single(self):
        '''Make sure uniform works'''

        # Generate samples that are all within the prior.
        theta = np.random.uniform(0,1,size=(10,1))

        # Evaluate the prior.
        prior_eval = utils.evaluate_prior(theta, [self.prior_list[0]])

        # Make sure they all equal the expected value.
        assert np.all(prior_eval==0.)

    def test_multiple(self):
        '''Make sure function works with multiple dists.'''

        # Generate samples that are all within the prior.
        theta = np.random.uniform(0,1,size=(10,2))

        # Evaluate the prior.
        prior_eval = utils.evaluate_prior(theta, self.prior_list)

        # Make sure they all equal the expected value.
        # Here the expected value is only the norm logpdf
        # as uniform should be zero for all samps.
        assert np.all(prior_eval == norm(loc=0,scale=1.).logpdf(theta[:,1]))

class TestSamplePrior:

    # Generate prior list with input as expected.
    param_dict={'w_c':{'prior':{'type':'uniform', 'min':0., 'max':1.}},
                'w_b':{'prior':{'type':'normal', 'mean':0., 'stddev':1.}}}
    prior_list = utils.make_prior_list(param_dict)

    def test_shape(self):
        '''Make sure shape is correct.'''
        
        # Generate samples from multiple dists.
        samples = utils.sample_prior(self.prior_list, 10, random_state=0)

        # Check shape.
        assert samples.shape==(10,2)

        # Generate samples from single dist.
        samples = utils.sample_prior([self.prior_list[0]], 10, random_state=0)

        # Check shape.
        assert samples.shape==(10,1)

    def test_samples(self):
        '''Make sure samples make sense.'''

        # Generate samples with fixed seed.
        samples = utils.sample_prior(self.prior_list, 10, random_state=0)

        # Compare to those with scipy functions called explicitly.
        assert np.all(samples==np.vstack([uniform(loc=0.,scale=1.).rvs(10, random_state=0),
                                          norm(loc=0.,scale=1.).rvs(10, random_state=0)]).T)

class TestPocoBounds:

    # Define param. dict. in the form that it would be read from the
    # config file.
    param_dict={'w_c':{'prior':{'type':'uniform', 'min':0., 'max':1.}},
                'w_b':{'prior':{'type':'truncated normal', 'min':2., 'max':3., 'mean':1.5, 'stddev': 0.5}},
                'h':{'prior':{'type':'normal', 'mean':0., 'stddev':1.}},
                'As':{'prior':{'type':'uniform', 'min':0., 'max':1.}}}

    def test_expected_in(self):
        '''Tests with input as expected.'''

        # Produce bounds array.
        bounds_arr = utils.poco_bounds(self.param_dict)

        # Make sure shape is correct.
        assert bounds_arr.shape==(4,2)

        # Make sure hard bounds are correct.
        assert np.all(bounds_arr[[0,1,3]]==np.array([[0.,1.],
                                                     [2.,3.],
                                                     [0.,1.]]))
        # Make sure nans where needed.
        assert np.all(np.isnan(bounds_arr[2]))

    def test_bad_in(self):
        '''Test with a prior that isn't supported.'''

        # Define unsupported prior.
        self.param_dict['w_c']['prior']['type']='random'

        # Make sure error is raised.
        with pytest.raises(ValueError):
            utils.poco_bounds(self.param_dict)

class TestFname:

    with open("default.yaml", "r") as stream:
        config = yaml.safe_load(stream)

    def test_wo_tags(self):

        fname = utils.make_fname(self.config['Setup'], use_tags=False, overwrite=True)
        print(fname)
        assert fname=="/Users/jamie/Desktop/GitHubProjects/unified_analysis/unified_utils/tests/chain_from_test.npy"

    def test_wo_tags_overwrite(self):

        # Make file name without tags.
        fname = utils.make_fname(self.config['Setup'], use_tags=False, overwrite=True)

        # Save empty file.
        os.system(f"touch {fname}")

        # Make sure there is an error when overwrite=False.
        with pytest.raises(FileExistsError):
            utils.make_fname(self.config['Setup'], use_tags=False, overwrite=False)

        # Make sure there is no error when overwrite=True
        fname = utils.make_fname(self.config['Setup'], use_tags=False, overwrite=True)

        # Delete the file.
        os.system(f"rm {fname}")

    def test_tags(self):

        # Make file name with tags.
        fname = utils.make_fname(self.config['Setup'], use_tags=True, overwrite=True)
        #print(fname)
        assert fname=="/Users/jamie/Desktop/GitHubProjects/unified_analysis/unified_utils/tests/chain_from_test--0.61--NGC--0.01--0.2--pybird--1.0--likelihood.py--log_like--emulator--engine.py--emu_engine.npy"

class TestLoadEngine:

    def test_works(self):
        '''Make sure function actually works.'''

        # Import the toy engine.
        engine_from_file = utils.import_loglikelihood(
            glob.glob("toy_engine.py")[0],
            "toy_engine",
            engine_or_like='engine'
        )

        toy_engine = engine_from_file()

        # Make sure we have an object.
        assert engine_from_file.mro()[1]==object

        # Make sure this object has a predict function as expected
        assert 'predict' in dict(inspect.getmembers(engine_from_file, inspect.isfunction))
        assert toy_engine.predict() == 50
