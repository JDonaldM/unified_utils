import numpy as np
import glob
from . import pk_tools
from scipy.stats import norm, uniform, truncnorm, multivariate_normal
from importlib.util import spec_from_file_location, module_from_spec
import sys
import inspect
import os
from pathlib import Path
from .likelihood import calc_P_G, fix_params
from matryoshka import halo_model_funcs
from matryoshka import rsd
from scipy.interpolate import interp1d

def data_loader(redshift, split, use_mock=False, pole_selection=[True, False, True, False, False], norm_cov=1.,
                kmin=0.01, kmax=0.2, pybird_mock=False, path_to_repo="/mnt/lustre/jdonaldm/unified_analysis/",
                mock_type="P18"):
    '''
    Utility function for loading data.

    Args:
        redshift (float) : Redshift of data to load. Can be on of ``[0.096, 0.38, 0.61, 1.52]``.
        split (str) : Hemisphere. Can be either ``NGC`` or ``SGC``.
        use_mock (bool) : If ``True`` will load mock data. Default is ``False``.
         Note this cannot be ``True`` if ``pybird_mock`` is ``True``.
        pole_selection (list) : List of boolean elements that selects the desired multipoles.
         Default is ``[True, False, True, False, True]``, this selects the even multipoles 
         ``P0``, ``P2``, and ``P4``. Note that is ``pybird_mock=True`` only the even
         multipoles can be selected.
        path_to_repo (str) : Path to repo. Note that all data must be stored in a directory
         with name ``data``.
    Returns:
        kbins_fit (array) : The k-bins. Has shape (nki,).
        pk_data_fit (array) : The selected multipoles. Has shape (3*nki,).
        cov_fit (array): Covariance matrix. Has shape (3*nki, 3*nki).
        window (array) : Window function matrix. Has shape (200, 2000).
        M (array) : Wide angle matrix. Has shape (1200, 2000).
        range_selection (array): Array of boolean elements that cut the
         origonal data. Has shape (nkj,). 
    '''

    if use_mock and pybird_mock:
        raise ValueError("use_mock and pybird_mock cannot be both True.")

    BOSS_z_dict = {0.38:'z1', 0.61:'z3'}
    eBOSS_split_cov_dict = {'NGC':896, 'SGC':1000}
    eBOSS_split_data_dict = {'NGC':1270, 'SGC':1040}

    # TODO: Comporess this code! The structure is the same appart from file names.
    # Load the desired data
    if redshift == 0.38 or redshift ==0.61:
        if use_mock:
            # When using mock data we take the mean across all the mocks
            pk_data_fit = 0
            for file_i in glob.glob(f"{path_to_repo}data/Patchy_V6C_BOSS_DR12_{split}_{BOSS_z_dict[redshift]}_2/*"):
                pk_data_dict = pk_tools.read_power(file_i, combine_bins=10)
                kbins, pk_data_vector = pk_tools.dict_to_vec(pk_data_dict)
                kbins_fit = kbins[np.repeat(pole_selection , 40)]
                kbins_fit = kbins_fit[:40]
                pk_data_fit += pk_data_vector[np.repeat(pole_selection , 40)]
            pk_data_fit = pk_data_fit/len(glob.glob(f"{path_to_repo}data/Patchy_V6C_BOSS_DR12_{split}_{BOSS_z_dict[redshift]}_2/*"))
        else:
            pk_data_dict = pk_tools.read_power(f"{path_to_repo}data/ps1D_BOSS_DR12_{split}_{BOSS_z_dict[redshift]}_COMPnbar_TSC_700_700_700_400_renorm.dat",
                                            combine_bins=10)
            kbins, pk_data_vector = pk_tools.dict_to_vec(pk_data_dict)
            kbins_fit = kbins[np.repeat(pole_selection , 40)]
            kbins_fit = kbins_fit[:40]
            pk_data_fit = pk_data_vector[np.repeat(pole_selection , 40)]
        cov = pk_tools.read_matrix(f"{path_to_repo}data/C_2048_BOSS_DR12_{split}_{BOSS_z_dict[redshift]}_V6C_1_1_1_1_1_10_200_200_prerecon.matrix")
        cov = cov/norm_cov
        Nmocks = 2048
        window = pk_tools.read_matrix(f"{path_to_repo}data/W_BOSS_DR12_{split}_{BOSS_z_dict[redshift]}_V6C_1_1_1_1_1_10_200_2000_averaged_v1.matrix")
        M = pk_tools.read_matrix(f"{path_to_repo}data/M_BOSS_DR12_{split}_{BOSS_z_dict[redshift]}_V6C_1_1_1_1_1_2000_1200.matrix")
        cov_fit = cov[np.ix_(np.repeat(pole_selection , 40), np.repeat(pole_selection , 40))]
        window_fit = window[np.ix_(np.repeat(pole_selection , 40), np.ones(2000).astype(bool))]
    elif redshift == 1.52:
        if use_mock:
            pk_data_fit = 0
            for file_i in glob.glob(f"{path_to_repo}data/EZmocks_eBOSS_DR16_QSO_{split}/*"):
                pk_data_dict = pk_tools.read_power(file_i, combine_bins=10)
                kbins, pk_data_vector = pk_tools.dict_to_vec(pk_data_dict)
                kbins_fit = kbins[np.repeat(pole_selection , 40)]
                kbins_fit = kbins_fit[:40]
                pk_data_fit += pk_data_vector[np.repeat(pole_selection , 40)]
            pk_data_fit = pk_data_fit/len(glob.glob(f"{path_to_repo}data/EZmocks_eBOSS_DR16_QSO_{split}/*"))
        else:
            pk_data_dict = pk_tools.read_power(f"{path_to_repo}data/ps1D_eBOSS_DR16_QSO_{split}_TSC_0.8_2.2_{eBOSS_split_data_dict[split]}_{eBOSS_split_data_dict[split]}_{eBOSS_split_data_dict[split]}_400_renorm.dat",
                                            combine_bins=10)
            kbins, pk_data_vector = pk_tools.dict_to_vec(pk_data_dict)
            kbins_fit = kbins[np.repeat(pole_selection , 40)]
            kbins_fit = kbins_fit[:40]
            pk_data_fit = pk_data_vector[np.repeat(pole_selection , 40)]
        cov = pk_tools.read_matrix(f"{path_to_repo}data/C_{eBOSS_split_cov_dict[split]}_eBOSS_DR16_QSO_{split}_1_1_1_1_1_10_200_200_prerecon.matrix")
        cov = cov/norm_cov
        Nmocks = eBOSS_split_cov_dict[split]
        window = pk_tools.read_matrix(f"{path_to_repo}data/W_eBOSS_DR16_QSO_{split}_1_1_1_1_1_10_200_2000_averaged_v1.matrix")
        M = pk_tools.read_matrix(f"{path_to_repo}data/M_eBOSS_DR16_QSO_{split}_1_1_1_1_1_2000_1200.matrix")
        cov_fit = cov[np.ix_(np.repeat(pole_selection , 40), np.repeat(pole_selection , 40))]
        window_fit = window[np.ix_(np.repeat(pole_selection , 40), np.ones(2000).astype(bool))]
    else:
        if use_mock:
            pk_data_fit = 0
            for file_i in glob.glob(f"{path_to_repo}data/COLA_6dFGS_DR3/*"):
                pk_data_dict = pk_tools.read_power(file_i, combine_bins=10)
                kbins, pk_data_vector = pk_tools.dict_to_vec(pk_data_dict)
                kbins_fit = kbins[np.repeat(pole_selection , 40)]
                kbins_fit = kbins_fit[:40]
                pk_data_fit += pk_data_vector[np.repeat(pole_selection , 40)]
            pk_data_fit = pk_data_fit/len(glob.glob(f"{path_to_repo}data/COLA_6dFGS_DR3/*"))
        else:
            pk_data_dict = pk_tools.read_power(f"{path_to_repo}data/ps1D_6dFGS_DR3_TSC_600_600_600_400_renorm.dat",
                                            combine_bins=10)
            kbins, pk_data_vector = pk_tools.dict_to_vec(pk_data_dict)
            kbins_fit = kbins[np.repeat(pole_selection , 40)]
            kbins_fit = kbins_fit[:40]
            pk_data_fit = pk_data_vector[np.repeat(pole_selection , 40)]
        cov = pk_tools.read_matrix(f"{path_to_repo}data/C_600_6dFGS_DR3_1_1_1_1_1_10_200_200_prerecon.matrix")
        cov = cov/norm_cov
        Nmocks = 600
        window = pk_tools.read_matrix(f"{path_to_repo}data/W_6dFGS_DR3_1_1_1_1_1_10_200_2000_averaged_v1.matrix")
        M = pk_tools.read_matrix(f"{path_to_repo}data/M_6dFGS_DR3_1_1_1_1_1_2000_1200.matrix")
        cov_fit = cov[np.ix_(np.repeat(pole_selection , 40), np.repeat(pole_selection , 40))]
        window_fit = window[np.ix_(np.repeat(pole_selection , 40), np.ones(2000).astype(bool))]
    
    if pybird_mock:
        # Load the pybird mock data.
        # Matrix of shape (4,nk).
        kP024 = np.load(f"{path_to_repo}/data/pybird_mocks/poles/kP024--{mock_type}--z-{redshift}--split-{split}.npy")

        # Flatten multipole rows into single vector.
        pk_data_fit = kP024[1:][np.array(pole_selection)[[0,2,-1]]].flatten()

    range_selection = np.logical_and(kbins_fit>=kmin, kbins_fit<=kmax)

    kbins_fit = kbins_fit[range_selection].__array__()
    cov_fit = cov_fit[np.ix_(np.concatenate(sum(pole_selection)*[range_selection]), np.concatenate(sum(pole_selection)*[range_selection]))].__array__()
    window_fit = window_fit[np.ix_(np.concatenate(sum(pole_selection)*[range_selection]), np.ones(2000).astype(bool))].__array__()

    if pybird_mock:
        # kbins array for PyBird mocks is shorter.
        kbins_fit = kP024[0]
        # Define new range slection for shorter kbins array
        _range_selection = np.logical_and(kbins_fit>=kmin, kbins_fit<=kmax)
        # Convert pole selection to an array so it can be easily indexed.
        pole_selection = np.array(pole_selection)
        # Selected multipoles, and apply scale cut. 
        # ALL ODD MULTIPOLES WILL BE IGNORED AS THEY DON"T EXIST FOR THE PYBIRD MOCKS.
        pk_data_fit = pk_data_fit[np.concatenate(sum(pole_selection[[0,2,4]])*[_range_selection])].__array__()
        # Apply scale cut to kbins.
        kbins_fit = kbins_fit[_range_selection]
    else:
        pk_data_fit = pk_data_fit[np.concatenate(sum(pole_selection)*[range_selection])].__array__()


    return kbins_fit, pk_data_fit, cov_fit, window, M, range_selection, Nmocks
    
def check_emu_bounds(param_dict, emu_bounds):
    '''
    Function for checking ``Parameters`` nested dictionary as read from config. file.

    Args:
        param_dict (dict) : Nested dictionary for parameters of model. Each element should have the
         form ``{'pi': {'prior': {'type':'uniform', 'min':a, 'max':b}}}`` or they will be 
         ignored by the function. ``a`` and ``b`` can be either floats or ``'emu'``.
    Returns:
        ``param_dict`` with all appearances of ``'emu'`` replaced with floats and all extremes checked.
    '''

    # Check params in param_dict
    # Extract names of all params in the param_dict.
    # Define as set.
    all_params = set(param_dict.keys())

    # Define set of training parameters that need to be checked
    training_params = set(["w_c", "w_b", "h", "As"])

    # Find common parameters to check
    to_check = all_params.intersection(training_params)
    

    for p in to_check:
        if 'prior' not in param_dict[p]:
            if param_dict[p]['fixed']>emu_bounds[p]['max'] or param_dict[p]['fixed']<emu_bounds[p]['min']:
                raise ValueError(f"Fixed value for {p} outside emulator training space.")
        else:
            if param_dict[p]['prior']['type']=='uniform'\
                or param_dict[p]['prior']['type']=='truncated normal'\
                or param_dict[p]['prior']['type']=='truncated exponential':
                if param_dict[p]['prior']['min']=='emu':
                    param_dict[p]['prior']['min'] = emu_bounds[p]['min']
                elif param_dict[p]['prior']['min']<emu_bounds[p]['min']:
                    raise ValueError(f"Prior min outside training space for {p}")
                if param_dict[p]['prior']['max']=='emu':
                    param_dict[p]['prior']['max'] = emu_bounds[p]['max']
                elif param_dict[p]['prior']['max']>emu_bounds[p]['max']:
                    raise ValueError(f"Prior max outside training space for {p}")

    return param_dict

def make_prior_list(param_dict):
    '''
    Function for constructing prior from the config file.

    Args:
        param_dict (dict) : Nested dictionary for parameters of model.
    Returns:
        List of ``scipy.stats`` distributions corresponding to the prior
        each parameter. Will have length equal to the number of parameters
        in ``param_dict``.
    '''

    prior_list=[]
    for p in param_dict:
        if 'prior' not in param_dict[p]:
            continue
        else:
            if param_dict[p]['prior']['type']=='normal':
                prior_list.append(norm(loc=param_dict[p]['prior']['mean'],
                                    scale=param_dict[p]['prior']['stddev']))
            elif param_dict[p]['prior']['type']=='uniform' or param_dict[p]['prior']['type']=='mv':
                prior_list.append(uniform(loc=param_dict[p]['prior']['min'],
                                    scale=param_dict[p]['prior']['max']-param_dict[p]['prior']['min']))
            elif param_dict[p]['prior']['type']=='truncated normal':
                prior_list.append(truncnorm((param_dict[p]['prior']['min']-param_dict[p]['prior']['mean'])/param_dict[p]['prior']['stddev'],
                                            (param_dict[p]['prior']['max']-param_dict[p]['prior']['mean'])/param_dict[p]['prior']['stddev'],
                                            loc=param_dict[p]['prior']['mean'], scale=param_dict[p]['prior']['stddev']))
            elif param_dict[p]['prior']['type']=='truncated exponential':
                prior_list.append(trunc_expon(param_dict[p]['prior']['factor'], param_dict[p]['prior']['min'], param_dict[p]['prior']['max']))
            elif 'marg' in param_dict[p]['prior']['type']:
                continue
            elif param_dict[p]['prior']['type']=='jeff':
                if ('min' in param_dict[p]['prior']) and ('max' in param_dict[p]['prior']):
                    prior_list.append(uniform(loc=param_dict[p]['prior']['min'],
                                        scale=param_dict[p]['prior']['max']-param_dict[p]['prior']['min'])) 
                else:
                    prior_list.append(uniform(loc=-5*param_dict[p]['prior']['stddev'],
                                        scale=10*param_dict[p]['prior']['stddev'])) 
            else:
                raise ValueError("Can only use uniform, normal, or truncated normal priors.")

    return prior_list

def evaluate_prior(theta, prior_list):
    '''
    Evaluate the log prior given a set of parameters and list of prior 
    distributions for each parameter.

    Args:
        theta (array) : Parameter sets for which the prior should be
         evaluated. Should have shape ``(nsamp, nparam)``.
        prior_list (list) : List of prior distribtuions for each
         parameter. Each element should be a class with ``.logpdf()``
         method.
    Returns:
        Array with shape ``(nsamp,)`` containing the prior evaluation
        for each sample.
    '''

    # Initlaise at zero.
    prior = 0

    # Loop over each parameter.
    for i, pi in enumerate(prior_list):

        # Evaluate the logpdf of the parameter prior.
        prior += pi.logpdf(theta[:,i])

    return prior

def sample_prior(prior_list, Nsamp, random_state=None):
    '''
    Sample from specified prior.

    Args:
        prior_list (list) : List of prior distribtuions for each
         parameter. Each element should be a class with ``.rvs()``
         method.
        Nsamp (int) : The number of samples to generate.
        random_state (int) : Random state for sample generation.
         Default is ``None``.
    '''

    # Initalise as empty list
    samples = []

    # Loop over each parameter.
    for pi in prior_list:

        # Evaluate the logpdf of the parameter prior.
        samples.append(pi.rvs(Nsamp, random_state=random_state))

    return np.vstack(samples).T

def import_loglikelihood(path, fn_name, engine_or_like='like'):
    '''
    Modified version of ``cronus`` function. 
    See https://github.com/minaskar/cronus. 
    Extracts likelihood function from specified file.

    Args:
        path (str) : Path to ``.py`` file that contains likelihood function
         or engine class.
        fn_name (str) : The name of the function in the ``.py`` file.
        engine_or_like (str) : Are you loading a likelihood function or a
         predictione engine? Can be either ``'like'`` or ``'engine'``.
    Returns:
        Likelihood function or engine class.
    '''

    # Check path
    if not os.path.isfile(path):
        raise FileNotFoundError(f"File not found at specified location: {path}")

    # Load file as a module.
    spec = spec_from_file_location("likelihood_file", path)
    file_module = module_from_spec(spec)
    sys.modules[spec.name] = file_module
    spec.loader.exec_module(file_module)
    

    # Dictionary with all the functions/classes in the file.
    if engine_or_like == 'like':
        dict_of_functions = dict(inspect.getmembers(file_module, inspect.isfunction))
    elif engine_or_like == 'engine':
        dict_of_functions = dict(inspect.getmembers(file_module, inspect.isclass))

    # Check function name
    if fn_name not in dict_of_functions:
        raise KeyError("Function/class not found in specified file.")

    # Return the specified function/class only.
    return dict_of_functions[fn_name]

def poco_bounds(param_dict):
    '''
    Make a bounds array to be passed to poco. This needs to indicate any hard priors.

    Args:
        param_dict (dict) : Nested dictionary for parameters of model.
    Returns:
        Array with shape ``(nparam, 2)`` containing the upper and lower bounds of the
        priors. For priors that don't have hard bounds ``np.nan`` will be used.
    '''

    bound_list=[]
    for p in param_dict:
        if 'prior' not in param_dict[p]:
            continue
        else:
            if param_dict[p]['prior']['type']=='normal':
                bound_list.append(np.array([np.nan, np.nan]))
            elif (param_dict[p]['prior']['type']=='uniform') or\
                 (param_dict[p]['prior']['type']=='truncated normal') or\
                 (param_dict[p]['prior']['type']=='truncated exponential') or\
                 (param_dict[p]['prior']['type']=='mv'):
                bound_list.append(np.array([param_dict[p]['prior']['min'],
                                            param_dict[p]['prior']['max']]))
            elif 'marg' in param_dict[p]['prior']['type']:
                continue
            elif (param_dict[p]['prior']['type']=='jeff'):
                if ('min' in param_dict[p]['prior']) and ('max' in param_dict[p]['prior']):
                    bound_list.append(np.array([param_dict[p]['prior']['min'],
                                                param_dict[p]['prior']['max']]))
                else:
                    bound_list.append(np.array([-5*param_dict[p]['prior']['stddev'],
                                                5*param_dict[p]['prior']['stddev']]))
            else:
                raise ValueError("Can only use uniform or normal priors.")

    return np.vstack(bound_list)

def make_fname(setup_dict, use_tags=True, overwrite=False):
    '''
    Make file name from config file. File name will have tags that reflect the
    setup.

    Args:
        setup_dict (dict) : Nested dictionary as read from ``.yaml`` file.
        use_tags (bool) : Use tags that refelect the setup in the name.
         This can make file names quite long so set ``False`` to turn this
         off. Default is ``True``.
    Returns:
        The name for a ``.npy`` file.
    '''

    # Extract save directory and start of file name.
    dir_path = setup_dict['save']['path']
    fname = setup_dict['save']['fname']

    # Check if save directory exists.
    if not os.path.isdir(dir_path):
        raise FileNotFoundError("Directory not found at specified save location.")


    if dir_path[-1]!='/':
        dir_path+="/"

    # If the user wants no tags.
    if (not use_tags):
        # Check if file exists and the user has not wanted to overwrite.
        if os.path.isfile(f"{dir_path}{fname}.npy") and (not overwrite):
            raise FileExistsError("File already exists at specified save location. Set overwrite=True if you wish to save wish to overwrite this file.")
        else:
            return f"{dir_path}{fname}.npy"
    else:
        # Loop over the setup arguments and append them to the file name.
        for si in setup_dict:
            # Skip over save arguments.
            if si=='save':
                continue
            # If the arguments are not themselves a dictionary simply add them
            if type(setup_dict[si]) is not dict:
                # If the argument is boolean, add it only if it is True.
                if (type(setup_dict[si]) is bool) and setup_dict[si]:
                    fname+=f"--{str(si).split('/')[-1]}"
                elif (type(setup_dict[si]) is bool) and (not setup_dict[si]):
                    continue
                else:
                    fname+=f"--{str(setup_dict[si]).split('/')[-1]}"
            else:
                # If they are a dictionary loop over their items.
                for sj in setup_dict[si]:
                    # skip over engine_kwargs as there can be a lot of these!
                    if sj=='engine_kwargs':
                        continue
                    if (type(setup_dict[si][sj]) is bool) and setup_dict[si][sj]:
                        fname+=f"--{str(sj).split('/')[-1]}"
                    elif (type(setup_dict[si][sj]) is bool) and (not setup_dict[si][sj]):
                        continue
                    else:
                        fname+=f"--{str(setup_dict[si][sj]).split('/')[-1]}"

        # Remove junk characters from the file name 
        for chara in ["[", "]", ",", "'", " "]:
            fname = fname.replace(chara, "")

        # Chech if file name is too long.
        # Limit for .npy files is 251
        # Check for 251-5 as likelihood and diagnostic are 5 characters longer.
        if len(fname)>(251-5):
            raise OSError("File name is too long. Consider setting use_tags=False when defining the file name.")
        
        # Make sure name with tags doesn't already exist and cause an unwanted overwrite.
        if os.path.isfile(f"{dir_path}{fname}.npy") and (not overwrite):
            raise FileExistsError("File already exists at specified save location. Set overwrite=True if you wish to save wish to overwrite this file.")
        else:
            return f"{dir_path}{fname}.npy"
        
def special_treatment(param_dict):
    '''
    Determine if parameters need spcial treatment, like analytic marginalisation or Jeffreys prior.

    Args:
        param_dict (dict) : Dictionary defining the parameters and priors as expected in the ``.yaml`` config file.
    Returns:
        marg_cov (array) : Inverse prior matrix for marginalised parameters. Will have shape
         ``(len(marg_names), len(marg_names))``.
        marg_names (list) : List of names of marginalised parameters.
        jeff_cov (array) : Inverse prior matrix for marginalised parameters to be used in the Jeffreys prior. Will have
         shape ``(len(jeff_names), len(jeff_names))``. This shape may be different from ``marg_cov``.
        jeff_names (list) : List of names of parameters with Jeffreys prior.        
    '''

    # Define empty list to store names of parameters with Jeffreys prior. 
    jeff_names = []
    # And those that are analytically marginalised.
    marg_names = []

    # Define empty list that will store prior standard deviations for analytically marginlaised parameters.
    stddevs = []

    # Loop over parameters
    for p in param_dict:
        if 'fixed' in param_dict[p]:
            continue
        else:
            if 'marg' in param_dict[p]['prior']['type']:
                marg_names.append(p)
                stddevs.append(param_dict[p]['prior']['stddev'])
            if 'jeff' in param_dict[p]['prior']['type']:
                jeff_names.append(p)

    if (len(jeff_names)==len(marg_names)) and (len(jeff_names)>0):
        # If equal length then all marginalised parmaeters have a Jeffreys prior.
        marg_cov = np.diag(1./np.array(stddevs)**2)
        jeff_cov = marg_cov
        return marg_cov, marg_names, jeff_cov, jeff_names
    
    elif (len(jeff_names)!=len(marg_names)):
        if (len(marg_names)>0) and (len(jeff_names)==0):
            marg_cov = np.diag(1./np.array(stddevs)**2)
            return marg_cov, marg_names, None, None
        
        elif (len(marg_names)==0) and (len(jeff_names)>0):
            return None, None, None, jeff_names
        elif len(set(marg_names).intersection(set(jeff_names))) == 0:
            marg_cov = np.diag(1./np.array(stddevs)**2)
            return marg_cov, marg_names, None, jeff_names
        else:
            # Find indecies in all_names for each element of marg_names
            _idx1 = np.concatenate([np.where(p==np.array(jeff_names))[0] for p in marg_names])
            _idx2 = np.concatenate([np.where(p==np.array(marg_names))[0] for p in list(set(marg_names).intersection(jeff_names))])
            # Define a new array for the prior 1/stddevs^2 and set inf for non-marginalised params
            _stddevs = np.repeat(np.inf, len(jeff_names))
            _stddevs[_idx1] = np.array(stddevs)[_idx2]
            marg_cov = np.diag(1./np.array(stddevs)**2)
            jeff_cov = np.diag(1./_stddevs**2)
            return np.diag(1./np.array(stddevs)**2), marg_names, np.diag(1./_stddevs**2), jeff_names
        
    else:
        return None, None, None, None

def jeff_counter(theta, engine, kobs, cinv, fixed_vals, ng, km,
                 jeff_names, prior_list):
    theta = fix_params(np.atleast_2d(theta), fixed_vals)

    # Sperate cosmo and bias params.
    # Oc, Ob, h, As
    cosmo = theta[:,:4]
    # b1, c2, b3, c4, cct, cr1, cr2
    bias = theta[:,4:11]

    P11, Ploop, Pct = engine.predict_kernels(np.atleast_2d(cosmo), kobs)

    f = np.atleast_2d(halo_model_funcs.fN_vec((cosmo[:,0]+cosmo[:,1])/cosmo[:,2]**2,
                                            engine.redshift)).T

    derivs = calc_P_G(Ploop, Pct, np.atleast_2d(bias), f, ng, 0.7, kobs,
                      marg_names=jeff_names)

    db1  = 2*bias[:,0][None,:,None]*P11[:,:,0] + 2*f[:,0][None,:,None]*P11[:,:,1] + 2*bias[:,4][None,:,None]*Pct[:,:,0] + 2*bias[:,5][None,:,None]*Pct[:,:,1]\
        + 2*bias[:,6][None,:,None]*Pct[:,:,2] + Ploop[:,:,1] + 2*bias[:,0][None,:,None]*Ploop[:,:,5] + bias[:,1][None,:,None]*Ploop[:,:,6]\
        + bias[:,2][None,:,None]*Ploop[:,:,7] + bias[:,2][None,:,None]*Ploop[:,:,8]
    db1 = np.hstack([db1[0,:,:], db1[1,:,:]])

    db2 = Ploop[:,:,2] + bias[:,0][None,:,None]*Ploop[:,:,6] + 2*bias[:,1][None,:,None]*Ploop[:,:,9] + bias[:,3][None,:,None]*Ploop[:,:,10]
    db2 = np.hstack([db2[0,:,:], db2[1,:,:]])

    derivs = np.vstack([derivs, db1[np.newaxis,:,:], db2[np.newaxis,:,:]])

    pri_eval = np.zeros((cosmo.shape[0],))
    for s in range(cosmo.shape[0]):

        Fm = np.zeros((derivs.shape[0], derivs.shape[0]))
        for i in range(derivs.shape[0]):
            for j in range(derivs.shape[0]):
                Fm[i,j] = np.dot(np.dot(derivs[i][s], cinv), derivs[j][s])

        pri_eval[s] = np.log(np.sqrt(np.linalg.det(Fm)))

    return pri_eval+evaluate_prior(theta, prior_list)

def jeff_counter_sys(theta, engine, kobs, cinv, fixed_vals, ng, km, jeff_names, prior_list, Om_AP, window, M,
                     range_selection, additional_prior=True, gaussian_prior=None):
    theta = fix_params(np.atleast_2d(theta), fixed_vals)

    # Sperate cosmo and bias params.
    # Oc, Ob, h, As
    cosmo = theta[:,:4]
    # b1, c2, b3, c4, cct, cr1, cr2
    bias = theta[:,4:11]
    biasi = np.copy(bias)
    # Make copies of c2 and c4.
    c2 = np.copy(biasi[:,1])
    c4 = np.copy(biasi[:,3])
    # Use the copies to calculate b2 and b4
    biasi[:,1] = (c2+c4)/np.sqrt(2)
    biasi[:,3] = (c2-c4)/np.sqrt(2)

    P11, Ploop, Pct = engine.predict_kernels(np.atleast_2d(cosmo), engine.kbins)

    f = np.atleast_2d(halo_model_funcs.fN_vec((cosmo[:,0]+cosmo[:,1])/cosmo[:,2]**2,
                                            engine.redshift)).T

    derivs_list = []
    jeff_names_i = jeff_names.copy()
    if 'b1' in jeff_names_i:

        # Sum individual components of the b1 derivative
        # We use np.add.reduce() as it allows to sum multiple terms and include inline comments.
        db1 = np.add.reduce([
            2*biasi[:,0][None,:,None]*P11[:,:,0], # 2*b1
            2*f[:,0][None,:,None]*P11[:,:,1], # 2*f
            2*biasi[:,4][None,:,None]*Pct[:,:,0]/km**2, # 2*cct
            2*biasi[:,5][None,:,None]*Pct[:,:,1]/km**2, # 2*cr1
            2*biasi[:,6][None,:,None]*Pct[:,:,2]/km**2, # 2*cr2
            Ploop[:,:,1],
            2*biasi[:,0][None,:,None]*Ploop[:,:,5], # 2*b1
            biasi[:,1][None,:,None]*Ploop[:,:,6], # b2
            biasi[:,2][None,:,None]*Ploop[:,:,7], # b3
            biasi[:,3][None,:,None]*Ploop[:,:,8] # b4
        ])
        # reshape so we have (nl, nsamples, nk)
        db1 = np.hstack([db1[0,:,:], db1[1,:,:]])

        derivs_list.append(db1[np.newaxis,:,:])

        # Delete b1 from list of parameters with Jeff. prior.
        # We do this because a key error will be thrown when calculating the derivs w.r.t linear params if we don't.
        jeff_names_i.remove('b1')

    if 'c2' in jeff_names_i:

        dc2 = np.add.reduce([
            Ploop[:,:,2]/np.sqrt(2), # 1/sqrt(2)
            Ploop[:,:,4]/np.sqrt(2), # 1/sqrt(2)
            biasi[:,0][None,:,None]*Ploop[:,:,6]/np.sqrt(2), # b1/sqrt(2)
            biasi[:,0][None,:,None]*Ploop[:,:,8]/np.sqrt(2), # b1/sqrt(2)
            (c2+c4)[None,:,None]*Ploop[:,:,9], # c2+c4
            c2[None,:,None]*Ploop[:,:,10], # c2
            (c2-c4)[None,:,None]*Ploop[:,:,11], # c2-c4
        ])
        dc2 = np.hstack([dc2[0,:,:], dc2[1,:,:]])

        derivs_list.append(dc2[np.newaxis,:,:])

        jeff_names_i.remove('c2')

    if 'c4' in jeff_names_i:
        raise NotImplementedError
    
    if len(jeff_names_i) == 0:
        derivs = np.vstack(derivs_list)
    else:
        derivs = calc_P_G(Ploop, Pct, np.atleast_2d(biasi), f, ng, km, engine.kbins,
                        marg_names=jeff_names_i)
        derivs = np.vstack(derivs_list+[derivs])
    
    PG = derivs
    PG = np.split(derivs, 2, axis=-1)

    # Calculate DA and H0 for each cosmology.
    DA = rsd.DA_vec(cosmo[:,0]/cosmo[:,2]**2+cosmo[:,1]/cosmo[:,2]**2,
                    engine.redshift)
    Hubble = rsd.Hubble(cosmo[:,0]/cosmo[:,2]**2+cosmo[:,1]/cosmo[:,2]**2,
                        engine.redshift)

    # Compute DA and H0 for Om_AP
    DA_fid = rsd.DA_vec(Om_AP, engine.redshift)
    Hubble_fid = rsd.Hubble(Om_AP, engine.redshift)
    
    # Calculate AP parameters.
    qperp = DA/DA_fid
    qpar = Hubble_fid/Hubble

    # Inlcude AP effect.
    PG_sys = np.zeros((PG[0].shape[0], cosmo.shape[0], PG[0].shape[-1]*2))
    for i in range(cosmo.shape[0]):
        PG_i = rsd.AP(np.stack([PG[0][:,i,:], PG[1][:,i,:]]),
                        np.linspace(-1,1,301), engine.kbins, qperp[i],
                        qpar[i])
        PG_sys[:,i,:] = np.hstack(PG_i)

    # Interpolate predictions over kbins needed to use matrices.
    # Assume P4 = 0.
    PG_sys = np.dstack([interp1d(engine.kbins, PG_sys[:,:,:engine.kbins.shape[0]],
                                    kind='cubic', bounds_error=False,
                                    fill_value='extrapolate')(np.linspace(0,0.4,400)), # P0
                        interp1d(engine.kbins, PG_sys[:,:,engine.kbins.shape[0]:],
                                    kind='cubic', bounds_error=False,
                                    fill_value='extrapolate')(np.linspace(0,0.4,400)), #P2
                        np.zeros((PG_sys.shape[0],PG_sys.shape[1],400)) # P4
                        ])

    # Do wide angle convolution.
    PG_sys = np.einsum("ij, knj -> kni", M, PG_sys)

    # Do window convolution.
    PG_sys = np.einsum("ij, knj -> kni", window, PG_sys)

    # Only select P0 and P2
    pole_selection = [True, False, True, False, False]
    PG_sys = PG_sys[:,:,np.repeat(pole_selection , 40)][:,:,np.concatenate(sum(pole_selection)*[range_selection])]

    PG = PG_sys
    derivs = PG_sys

    pri_eval = np.zeros((cosmo.shape[0],))
    for s in range(cosmo.shape[0]):
        Fm = np.zeros((derivs.shape[0], derivs.shape[0]))
        for i in range(derivs.shape[0]):
            for j in range(derivs.shape[0]):
                Fm[i,j] = np.dot(np.dot(derivs[i][s], cinv), derivs[j][s])

        if gaussian_prior is not None:
            Fm += gaussian_prior

        pri_eval[s] = np.log(np.sqrt(np.linalg.det(Fm)))

    if additional_prior:
        return pri_eval+evaluate_prior(theta, prior_list)
    else:
        return pri_eval

class trunc_expon:

    def __init__(self, factor, xmin, xmax):

        self.factor = factor
        self.xmin = xmin
        self.xmax = xmax

    def rvs(self, N, random_state=None):
        np.random.seed(random_state)
        u = np.random.uniform(size=N)
        c = np.exp(self.factor*self.xmax) - np.exp(self.factor*self.xmin)
        return np.log(c*u + np.exp(self.factor*self.xmin))/self.factor

    def logpdf(self, x):
        return np.log(self.pdf(x))

    def pdf(self, x):
        norm = np.exp(self.xmax*self.factor)/self.factor\
               - np.exp(self.xmin*self.factor)/self.factor

        return np.exp(x*self.factor)/norm
    
def combo_jeff(theta, data, samples, ngs, redshifts, km, fixed_params, engine_dict, Om_AP, nshared, nindiv, jeff_names,
               prior_list, gaussian_prior=None):
    
    # Select parameters that are shared across all samples.
    shared_theta = theta[:, :nshared]

    if gaussian_prior is None:
        gaussian_prior = len(data)*[None]

    # Loop over all samples to and sum prior
    prior = 0
    for i in range(len(data)):

        # Define indecies to select parameters specific to the current sample.
        _idx1 = int(nshared+np.sum(nindiv[:i]))
        _idx2 = int(nshared+np.sum(nindiv[:i+1]))
        individual_theta = theta[:,_idx1:_idx2]

        # Combine the shared and individual parameters
        theta_i = np.hstack([shared_theta, individual_theta])

        prior += jeff_counter_sys(
            theta_i,
            engine_dict[redshifts[i]],
            data[samples[i]]['k'],
            data[samples[i]]['icov'],
            fixed_params[i],
            ngs[i],
            km,
            jeff_names[i],
            prior_list,
            Om_AP,
            data[samples[i]]['window'],
            data[samples[i]]['M'],
            data[samples[i]]['range'],
            additional_prior=False,
            gaussian_prior=gaussian_prior[i]
        )

    return prior+evaluate_prior(theta, prior_list)

def derivs_for_JP(theta, engine, fixed_vals, ng, km, jeff_names, Om_AP, window, M, range_selection):
    theta = fix_params(np.atleast_2d(theta), fixed_vals)

    # Sperate cosmo and bias params.
    # Oc, Ob, h, As
    cosmo = theta[:,:4]
    # b1, c2, b3, c4, cct, cr1, cr2
    bias = theta[:,4:11]
    biasi = np.copy(bias)
    # Make copies of c2 and c4.
    c2 = np.copy(biasi[:,1])
    c4 = np.copy(biasi[:,3])
    # Use the copies to calculate b2 and b4
    biasi[:,1] = (c2+c4)/np.sqrt(2)
    biasi[:,3] = (c2-c4)/np.sqrt(2)

    P11, Ploop, Pct = engine.predict_kernels(np.atleast_2d(cosmo), engine.kbins)

    f = np.atleast_2d(halo_model_funcs.fN_vec((cosmo[:,0]+cosmo[:,1])/cosmo[:,2]**2,
                                            engine.redshift)).T

    derivs_list = []
    if 'b1' in jeff_names:

        # Sum individual components of the b1 derivative
        # We use np.add.reduce() as it allows to sum multiple terms and include inline comments.
        db1 = np.add.reduce([
            2*biasi[:,0][None,:,None]*P11[:,:,0], # 2*b1
            2*f[:,0][None,:,None]*P11[:,:,1], # 2*f
            2*biasi[:,4][None,:,None]*Pct[:,:,0]/km**2, # 2*cct
            2*biasi[:,5][None,:,None]*Pct[:,:,1]/km**2, # 2*cr1
            2*biasi[:,6][None,:,None]*Pct[:,:,2]/km**2, # 2*cr2
            Ploop[:,:,1],
            2*biasi[:,0][None,:,None]*Ploop[:,:,5], # 2*b1
            biasi[:,1][None,:,None]*Ploop[:,:,6], # b2
            biasi[:,2][None,:,None]*Ploop[:,:,7], # b3
            biasi[:,3][None,:,None]*Ploop[:,:,8] # b4
        ])
        # reshape so we have (nl, nsamples, nk)
        db1 = np.hstack([db1[0,:,:], db1[1,:,:]])

        derivs_list.append(db1[np.newaxis,:,:])

        # Delete b1 from list of parameters with Jeff. prior.
        # We do this because a key error will be thrown when calculating the derivs w.r.t linear params if we don't.
        jeff_names.remove('b1')

    if 'c2' in jeff_names:

        dc2 = np.add.reduce([
            Ploop[:,:,2]/np.sqrt(2), # 1/sqrt(2)
            Ploop[:,:,4]/np.sqrt(2), # 1/sqrt(2)
            biasi[:,0][None,:,None]*Ploop[:,:,6]/np.sqrt(2), # b1/sqrt(2)
            biasi[:,0][None,:,None]*Ploop[:,:,8]/np.sqrt(2), # b1/sqrt(2)
            (c2+c4)[None,:,None]*Ploop[:,:,9], # c2+c4
            c2[None,:,None]*Ploop[:,:,10], # c2
            (c2-c4)[None,:,None]*Ploop[:,:,11], # c2-c4
        ])
        dc2 = np.hstack([dc2[0,:,:], dc2[1,:,:]])

        derivs_list.append(dc2[np.newaxis,:,:])

        jeff_names.remove('c2')

    if 'c4' in jeff_names:
        raise NotImplementedError
    
    derivs = calc_P_G(Ploop, Pct, np.atleast_2d(biasi), f, ng, km, engine.kbins,
                      marg_names=jeff_names)
    
    derivs = np.vstack([derivs]+derivs_list)
    PG = derivs
    PG = np.split(derivs, 2, axis=-1)

    # Calculate DA and H0 for each cosmology.
    DA = rsd.DA_vec(cosmo[:,0]/cosmo[:,2]**2+cosmo[:,1]/cosmo[:,2]**2,
                    engine.redshift)
    Hubble = rsd.Hubble(cosmo[:,0]/cosmo[:,2]**2+cosmo[:,1]/cosmo[:,2]**2,
                        engine.redshift)

    # Compute DA and H0 for Om_AP
    DA_fid = rsd.DA_vec(Om_AP, engine.redshift)
    Hubble_fid = rsd.Hubble(Om_AP, engine.redshift)
    
    # Calculate AP parameters.
    qperp = DA/DA_fid
    qpar = Hubble_fid/Hubble

    # Inlcude AP effect.
    PG_sys = np.zeros((PG[0].shape[0], cosmo.shape[0], PG[0].shape[-1]*2))
    for i in range(cosmo.shape[0]):
        PG_i = rsd.AP(np.stack([PG[0][:,i,:], PG[1][:,i,:]]),
                        np.linspace(-1,1,301), engine.kbins, qperp[i],
                        qpar[i])
        PG_sys[:,i,:] = np.hstack(PG_i)

    # Interpolate predictions over kbins needed to use matrices.
    # Assume P4 = 0.
    PG_sys = np.dstack([interp1d(engine.kbins, PG_sys[:,:,:engine.kbins.shape[0]],
                                    kind='cubic', bounds_error=False,
                                    fill_value='extrapolate')(np.linspace(0,0.4,400)), # P0
                        interp1d(engine.kbins, PG_sys[:,:,engine.kbins.shape[0]:],
                                    kind='cubic', bounds_error=False,
                                    fill_value='extrapolate')(np.linspace(0,0.4,400)), #P2
                        np.zeros((PG_sys.shape[0],PG_sys.shape[1],400)) # P4
                        ])

    # Do wide angle convolution.
    PG_sys = np.einsum("ij, knj -> kni", M, PG_sys)

    # Do window convolution.
    PG_sys = np.einsum("ij, knj -> kni", window, PG_sys)

    # Only select P0 and P2
    pole_selection = [True, False, True, False, False]
    PG_sys = PG_sys[:,:,np.repeat(pole_selection , 40)][:,:,np.concatenate(sum(pole_selection)*[range_selection])]

    PG = PG_sys
    derivs = PG_sys

    return derivs

def make_joint_matrix(data_dict, key='cov'):
    num_samps = len(data_dict)
    mat_shape = data_dict[list(data_dict)[0]][key].shape
    padded_mats = []
    for i, sample_i in enumerate(data_dict):
        padded_mats.append(
            np.pad(
            data_dict[sample_i][key],
            [
                (i*mat_shape[0], (num_samps-1-i)*mat_shape[0]),
                (0,0)
            ]
            )
        )
    return np.hstack(padded_mats)

def make_joint_data_vector(data_dict):
    return np.concatenate(
        [data_dict[sample_i]['data'] for sample_i in data_dict]
    )