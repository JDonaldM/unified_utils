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
        norm_cov (float) : Normalisation factor for the covariance matrix. Default
         is ``1.``.
        kmin (float) : Set minimum scale. Default is ``0.01``.
        kmin (float) : Set maximum scale. Default is ``0.2``.
        pybird_mock (bool) : If ``True`` will load the appropriate ``PyBird`` mock
         produced in arXiv:2307.07475.
        path_to_repo (str) : Path to repo. Note that all data must be stored in a directory
         with name ``data``.
        mock_type (str) : Tag for the ``PyBird`` mocks. The mocks produced in
         arXiv:2307.07475 all have the tag ``P18``.
    Returns:
        kbins_fit (array) : The k-bins. Has shape (nki,).
        pk_data_fit (array) : The selected multipoles. Has shape (3*nki,).
        cov_fit (array): Covariance matrix. Has shape (3*nki, 3*nki).
        window (array) : Window function matrix. Has shape (200, 2000).
        M (array) : Wide angle matrix. Has shape (1200, 2000).
        range_selection (array): Array of boolean elements that cut the
         origonal data. Has shape (nkj,).
        Nmocks (int) : The number of mocks used to calculate the covariance.
    '''

    # Make sure the user is explicit about what type of mock to load.
    if use_mock and pybird_mock:
        raise ValueError("use_mock and pybird_mock cannot be both True.")

    # Define dictionaries that have mappings between input variables and tags in
    # the file names.
    BOSS_z_dict = {0.38:'z1', 0.61:'z3'}
    eBOSS_split_cov_dict = {'NGC':896, 'SGC':1000}
    eBOSS_split_data_dict = {'NGC':1270, 'SGC':1040}

    if redshift == 0.38 or redshift ==0.61:
        # Must be BOSS data.
        # Define appropriate file names.
        cov_mock_files = glob.glob(f"{path_to_repo}data/Patchy_V6C_BOSS_DR12_{split}_{BOSS_z_dict[redshift]}_2/*")
        data_file = f"{path_to_repo}data/ps1D_BOSS_DR12_{split}_{BOSS_z_dict[redshift]}_COMPnbar_TSC_700_700_700_400_renorm.dat"
        cov_file = f"{path_to_repo}data/C_2048_BOSS_DR12_{split}_{BOSS_z_dict[redshift]}_V6C_1_1_1_1_1_10_200_200_prerecon.matrix"
        window_file = f"{path_to_repo}data/W_BOSS_DR12_{split}_{BOSS_z_dict[redshift]}_V6C_1_1_1_1_1_10_200_2000_averaged_v1.matrix"
        M_file = f"{path_to_repo}data/M_BOSS_DR12_{split}_{BOSS_z_dict[redshift]}_V6C_1_1_1_1_1_2000_1200.matrix"

        Nmocks = 2048
    elif redshift == 1.52:
        # Must be eBOSS data.
        # Define appropriate file names.
        cov_mock_files = glob.glob(f"{path_to_repo}data/EZmocks_eBOSS_DR16_QSO_{split}/*")
        data_file = f"{path_to_repo}data/ps1D_eBOSS_DR16_QSO_{split}_TSC_0.8_2.2_{eBOSS_split_data_dict[split]}_{eBOSS_split_data_dict[split]}_{eBOSS_split_data_dict[split]}_400_renorm.dat"
        cov_file = f"{path_to_repo}data/C_{eBOSS_split_cov_dict[split]}_eBOSS_DR16_QSO_{split}_1_1_1_1_1_10_200_200_prerecon.matrix"
        window_file = f"{path_to_repo}data/W_eBOSS_DR16_QSO_{split}_1_1_1_1_1_10_200_2000_averaged_v1.matrix"
        M_file = f"{path_to_repo}data/M_eBOSS_DR16_QSO_{split}_1_1_1_1_1_2000_1200.matrix"

        Nmocks = eBOSS_split_cov_dict[split]
    elif redshift == 0.096:
        # Must be 6dFGS data.
        # Define appropriate file names.
        cov_mock_files = glob.glob(f"{path_to_repo}data/COLA_6dFGS_DR3/*")
        data_file = f"{path_to_repo}data/ps1D_6dFGS_DR3_TSC_600_600_600_400_renorm.dat"
        cov_file = f"{path_to_repo}data/C_600_6dFGS_DR3_1_1_1_1_1_10_200_200_prerecon.matrix"
        window_file = f"{path_to_repo}data/W_6dFGS_DR3_1_1_1_1_1_10_200_2000_averaged_v1.matrix"
        M_file = f"{path_to_repo}data/M_6dFGS_DR3_1_1_1_1_1_2000_1200.matrix"

        Nmocks = 600
    else:
        raise ValueError(f"No unified multipoles for z={redshift}. Possible redshifts are [0.096, 0.38, 0.61, 1.52].")

    # Load the desired data
    if use_mock:
        # When using mock data we take the mean across all the mocks
        #Start with zero sum.
        pk_data_fit = 0
        # Loop over all files.
        for file_i in cov_mock_files:
            # Use pk_tools to load data
            pk_data_dict = pk_tools.read_power(file_i, combine_bins=10)
            kbins, pk_data_vector = pk_tools.dict_to_vec(pk_data_dict)

            # Select desired multipoles in data vector and kbins.
            # Also add data vector to sum.
            kbins_fit = kbins[np.repeat(pole_selection , 40)]
            pk_data_fit += pk_data_vector[np.repeat(pole_selection , 40)]

            # The kbins are repeadted for each selected multipole.
            # Keep only the first set.
            kbins_fit = kbins_fit[:40]

        #Divide by the number of files summed to get the mean.   
        pk_data_fit = pk_data_fit/len(cov_mock_files)
    else:
        # When using ral data simply load the file with pk_tools and repeat the 
        # same multipole selection as above.
        pk_data_dict = pk_tools.read_power(data_file, combine_bins=10)
        kbins, pk_data_vector = pk_tools.dict_to_vec(pk_data_dict)
        kbins_fit = kbins[np.repeat(pole_selection , 40)]
        kbins_fit = kbins_fit[:40]
        pk_data_fit = pk_data_vector[np.repeat(pole_selection , 40)]

    # Load covaraince, window, and wide-angle matrices.
    cov = pk_tools.read_matrix(cov_file)
    window = pk_tools.read_matrix(window_file)
    M = pk_tools.read_matrix(M_file)

    # Normalise the matirx.
    cov = cov/norm_cov

    # Select desired multipoles in covaraince matrix.
    cov_fit = cov[np.ix_(np.repeat(pole_selection , 40), np.repeat(pole_selection , 40))]
    
    # Define boolean array that will impose scale cuts.
    range_selection = np.logical_and(kbins_fit>=kmin, kbins_fit<=kmax)

    # Impose scale cuts to covaraince matrix.
    cov_fit = cov_fit[
        np.ix_(np.concatenate(sum(pole_selection)*[range_selection]),
        np.concatenate(sum(pole_selection)*[range_selection]))
    ].__array__() # .__array__() to convert to normal numpy array.

    if pybird_mock:
        # Check that multipole selection only includes even multipoles
        if pole_selection[1] or pole_selection[3]:
            raise ValueError("Odd multipoles don't exist for the PyBird mocks. Check pole selection.")

        # Load the pybird mock data.
        # Matrix of shape (4,nk).
        kP024 = np.load(f"{path_to_repo}/data/pybird_mocks/poles/kP024--{mock_type}--z-{redshift}--split-{split}.npy")

        # Flatten multipole rows into single vector.
        pk_data_fit = kP024[1:][np.array(pole_selection)[[0,2,-1]]].flatten()

        # kbins array for PyBird mocks is shorter.
        kbins_fit = kP024[0]

        # Define new range slection for shorter kbins array
        _range_selection = np.logical_and(kbins_fit>=kmin, kbins_fit<=kmax)

        # Convert pole selection to an array so it can be easily indexed.
        pole_selection = np.array(pole_selection)

        # Selected multipoles, and apply scale cut. 
        # ALL ODD MULTIPOLES WILL BE IGNORED AS THEY DON'T EXIST FOR THE PYBIRD MOCKS.
        pk_data_fit = pk_data_fit[np.concatenate(sum(pole_selection[[0,2,4]])*[_range_selection])].__array__()

        # Apply scale cut to kbins.
        kbins_fit = kbins_fit[_range_selection]
    else:
        pk_data_fit = pk_data_fit[np.concatenate(sum(pole_selection)*[range_selection])].__array__()
        kbins_fit = kbins_fit[range_selection].__array__()

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
    
    # Loop over parameters.
    for p in to_check:
        if ('prior' not in param_dict[p]) and ('fixed' in param_dict[p]):
            # If parameter is defined as fixed, check that the fixed value lies
            # in the training space.
            if param_dict[p]['fixed']>emu_bounds[p]['max'] or param_dict[p]['fixed']<emu_bounds[p]['min']:
                raise ValueError(f"Fixed value for {p} outside emulator training space.")
        elif 'prior' in param_dict[p]:
            # If the parameter has a prior, check that it is an allowed type.
            if param_dict[p]['prior']['type']=='uniform'\
                or param_dict[p]['prior']['type']=='truncated normal':
                # If the type is allowed check that the hard bounds are within
                # the training space or set them equal to the training space if 
                # desired. 
                if param_dict[p]['prior']['min']=='emu':
                    param_dict[p]['prior']['min'] = emu_bounds[p]['min']
                elif param_dict[p]['prior']['min']<emu_bounds[p]['min']:
                    raise ValueError(f"Prior min outside training space for {p}")
                if param_dict[p]['prior']['max']=='emu':
                    param_dict[p]['prior']['max'] = emu_bounds[p]['max']
                elif param_dict[p]['prior']['max']>emu_bounds[p]['max']:
                    raise ValueError(f"Prior max outside training space for {p}")
            else:
                raise ValueError(f"Specified prior type for {p} not allowed. Must be wother 'unifrom' or 'truncated normal'.")
        else:
            raise ValueError(f"Parameters must be defined as 'fixed' or have a prior distribution defined. Check definition of {p}.")

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
