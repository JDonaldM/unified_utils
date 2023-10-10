import numpy as np
import argparse
import pocomc as pc
from unified_utils import utils
import yaml
import os
import logging

# Adjust the Python logging level so that more than warning and/or errors are shown.
logging.basicConfig(level=logging.INFO)

# Define path to repo on sciama
path_to_repo = "/mnt/lustre/jdonaldm/unified_analysis/"

# Read configuration
#################################################################################################

# Read in path to .yaml file.
parser = argparse.ArgumentParser()
parser.add_argument('--config', help="Path to .yaml configuration file.",
                    required=True)            
args = parser.parse_args()

# Make sure config file exists.
if not os.path.isfile(args.config):
    raise ValueError("No config file found at specified location.")

# Read conifg file.
logging.info(f"Reading configuraation file: {args.config}")
with open(args.config, "r") as stream:
    config_from_file = yaml.safe_load(stream)

#################################################################################################

# Data setup
#################################################################################################

# Extract all the setup variables.
logging.info("Extracting setup variables.")
redshifts = config_from_file['Setup']['redshifts']
logging.info(f"redshifts: {redshifts}")
samples = config_from_file['Setup']['samples']
logging.info(f"samples: {samples}")
kmins = config_from_file['Setup']['kmins']
logging.info(f"kmin: {kmins}")
kmaxs = config_from_file['Setup']['kmaxs']
logging.info(f"kmax: {kmaxs}")
data_type = config_from_file['Setup']['data_type']
logging.info(f"data type: {data_type}")
norm_cov = config_from_file['Setup']['norm_cov']
logging.info(f"covariance normalisation factor: {norm_cov}")

# Deine the number density.
# This is almost certainly wrong for 6dF.
ngs = []
for zi in redshifts:
    if zi == 0.096 or zi == 0.38 or zi == 0.61:
        ngs.append(4e-4)
    elif zi == 1.52:
        ngs.append(1.5e-5)
logging.info(f"number density: {ngs}")


# Define km. This scales the stcohastic counterterms
km=0.7

# Define the fiucial value of Om used to measure the power spectra.
Om_fid = 0.31

# Define the true cosmology.
# Found in table 2.9 at this link: https://wiki.cosmos.esa.int/planck-legacy-archive/images/9/9b/Grid_limit68.pdf
# Should match the cosmology of the mocks pretty closely but not exactly.
# This *is* the true cosmology for the PyBird mocks.
cosmo_true = np.array([0.11889, 0.022161, 0.6766, 3.0973, 0.9611])

#################################################################################################

# Load the engine
#################################################################################################

logging.info("Loading prediction engines.")
engine_dict = {}
path_to_model = config_from_file['Setup']['engine']['engine_kwargs']['path_to_model']
for zi in np.unique(redshifts):
    engine_from_file = utils.import_loglikelihood(config_from_file['Setup']['engine']['path'],
                                                                config_from_file['Setup']['engine']['function'],
                                                                engine_or_like='engine')
    path_to_model_i = path_to_model + f'z{zi}/'
    config_from_file['Setup']['engine']['engine_kwargs']['path_to_model'] = path_to_model_i
    engine = engine_from_file(zi, **config_from_file['Setup']['engine']['engine_kwargs'])

    engine_dict[zi] = engine

#################################################################################################

# Load the data
#################################################################################################

# Define the multipole selection.
# This will select only P0 and P2 from Florian's data.
pole_selection = [True, False, True, False, False]

# Define boolean variables for selecting data type.
# TODO: Should probably change the data_loader function so this isn't required.
if data_type=='obs':
    use_mock=False
    pybird_mock=False
    mock_type = None
elif data_type=='mock':
    use_mock=True
    pybird_mock=False
    mock_type=None
else:
    use_mock=False
    pybird_mock=True
    if "_" in data_type:
        mock_type = data_type.split("_")[-1]
    else:
        mock_type = "P18"

data_dict = {}
data_dict_template = {
    'data': None,
    'k': None,
    'cov': None,
    'icov': None,
    'window': None,
    'M': None,
    'range': None
}
for i, sample_i in enumerate(samples):
    data_dict[sample_i] = data_dict_template.copy()
    split = sample_i.split(' ')[-1]
    # Limit range to that covered by the emulator or that specified by the user.
    kbins_emu = engine_dict[redshifts[i]].kbins

    # Check kmin and kmax are compatible with the emulator.
    if kmaxs[i]>=kbins_emu.max():
        raise ValueError(f"Specified kmax greater than emulator kmax. Emulator kmax at z={redshifts[i]} is {kbins_emu.max()}")
    if kmins[i]<=kbins_emu.min():
        raise ValueError(f"Specified kmin less than emulator kmin. Emulator kmin at z={redshifts[i]} is {kbins_emu.min()}")

    logging.info(f"Loading {sample_i} data.")
    kbins_fit, pk_data_fit, cov_fit, window, M, range_selection, Nmocks = utils.data_loader(
        redshifts[i],
        split,
        use_mock=use_mock,
        kmin=kmins[i],
        kmax=kmaxs[i], 
        path_to_repo=path_to_repo,
        norm_cov=norm_cov,
        pole_selection=pole_selection,
        pybird_mock=pybird_mock,
        mock_type=mock_type
    )
    data_dict[sample_i]['data'] = pk_data_fit
    data_dict[sample_i]['k'] = kbins_fit
    data_dict[sample_i]['cov'] = cov_fit
    data_dict[sample_i]['icov'] = np.linalg.inv(cov_fit)
    data_dict[sample_i]['window'] = window
    data_dict[sample_i]['M'] = M
    data_dict[sample_i]['range'] = range_selection

#################################################################################################

# Infer. setup
#################################################################################################

# Extract the extremes of the training sapce for the emulator.
# Can be used as a prior on cosmo. params.
if config_from_file['Setup']['engine']['type']=='emulator':
    emu_bounds = np.stack([engine_dict[redshifts[0]].P0_emu.P11.scalers[0].min_val,
                           engine_dict[redshifts[0]].P0_emu.P11.scalers[0].min_val+engine_dict[redshifts[0]].P0_emu.P11.scalers[0].diff]).T
else:
    raise NotImplementedError

# Convert emu. bounds array into nested dictionary.
emu_bounds_dict = {p:{'min':emu_bounds[i,0], 'max':emu_bounds[i,1]} for i,p in enumerate(['w_c', 'w_b', 'h', 'As'])}

# Check that the user specified extremes for the cosmo. priors lie within the emulator training space.
# All min or max values specified as 'emu' with the corresponding emu. bounds value. 
config_from_file['Parameters']['Shared'] = utils.check_emu_bounds(
    config_from_file['Parameters']['Shared'],
    emu_bounds_dict
)

# Define order that the likelihood functions expect the parameters.
expected = ['w_c', 'w_b', 'h', 'As', 'b1', 'c2', 'b3', 'c4', 'cct', 'cr1', 'cr2', 'ce1', 'cmono', 'cquad']

# Find these parameters in either the shared or individual parameter dicts.
expected_shared = []
expected_individual = []
# Loop over all expected parameters
for pi in expected:
    if pi in config_from_file['Parameters']['Shared']:
        # Parameter is shared
        expected_shared.append(pi)
    if pi in config_from_file['Parameters'][samples[0]]:
        # Parameter is unique for each sample
        expected_individual.append(pi)

# Check that all individual param dicts have the same params
for sample_i in samples:
    for pi in expected_individual:
        try:
            config_from_file['Parameters'][sample_i][pi]
        except:
            raise KeyError(f"{pi} not found in {sample_i} dict.")

# Make sure all parameters are define.
if len(expected)!=(len(expected_shared)+len(expected_individual)):
    raise ValueError("Not all parameters are accounted for! Check config file.")

# Reorder shared parameters in config file so they are in the expected order.
config_from_file['Parameters']['Shared'] = {p: config_from_file['Parameters']['Shared'][p] for p in expected_shared}
# Construct prior for shared params
prior_list = utils.make_prior_list(config_from_file['Parameters']['Shared'])

# Loop over parameter dicts for all samples 
for i in config_from_file['Parameters']:
    if i=='Fixed' or i=='Shared':
        continue
    # Reorder
    config_from_file['Parameters'][i] = {p: config_from_file['Parameters'][i][p] for p in expected_individual}
    # Add to the prior list.
    prior_list += utils.make_prior_list(config_from_file['Parameters'][i])

logging.info("Check for parameters that need special treatment.")
marg_cov = []
marg_names = []
jeff_cov = []
jeff_names = []
for i in range(len(samples)):
    marg_cov_i, marg_names_i, jeff_cov_i, jeff_names_i = utils.special_treatment(
        config_from_file['Parameters'][samples[i]]
    )
    
    marg_cov.append(marg_cov_i)
    jeff_cov.append(jeff_cov_i)
    marg_names.append(marg_names_i)
    jeff_names.append(jeff_names_i)

marg_params = False
if not (marg_names[0] is None):
    logging.info(f"Parameters analytically marginalised")
    marg_params = True
jeff_params = False
if not (jeff_names[0] is None):
    logging.info(f"Parameters have Jeffreys prior")
    jeff_params = True

logging.info("Extracting fixed values.")
fixed_shared = dict()
nshared = 0
for pi in config_from_file['Parameters']['Shared']:
    if 'fixed' in config_from_file['Parameters']['Shared'][pi]:
        fixed_shared[pi] = config_from_file['Parameters']['Shared'][pi]['fixed']
    else:
        fixed_shared[pi] = None
        nshared += 1
logging.info(f"Shared fixed parameters: {fixed_shared}")
fixed_values = []
nindiv = []
for sample_i in samples:
    fixed_indiv_i = dict()
    nindiv_i = 0
    for pi in config_from_file['Parameters'][sample_i]:
        if 'fixed' in config_from_file['Parameters'][sample_i][pi]:
            fixed_shared[pi] = config_from_file['Parameters'][sample_i][pi]['fixed']
        else:
            fixed_shared[pi] = None
            nindiv_i += 1
    logging.info(f"Fixed values {sample_i}: {fixed_indiv_i}")
    fixed_values.append({**fixed_shared, **fixed_indiv_i})
    nindiv.append(nindiv_i)

# If there are parameters to be marginalised over fix their values on the
# fixed_values dict.
if marg_params:
    for sample_i in range(len(samples)):
        for p in marg_names[sample_i]:
                fixed_values[sample_i][p]=0.
                nindiv[sample_i] -= 1

# Load the likelihood function
logging.info("Loading log likelihood function.")
loglike = utils.import_loglikelihood(config_from_file['Setup']['likelihood']['path'],
                                             config_from_file['Setup']['likelihood']['function'])

save_fname = utils.make_fname(config_from_file['Setup'],
                                            use_tags=config_from_file['Setup']['save']['fname_tags'],
                                            overwrite=config_from_file['Setup']['save']['overwrite'])

# Extract sampler settings (if there are any)
try:
    logging.info("Found sampler settings.")
    sampler_settings = config_from_file['Setup']['sampler']
    nwalk = sampler_settings['nwalk']
    logging.info(f"number of particles: {nwalk}")
    nsamples = sampler_settings['nsamples']
    logging.info(f"total number of samples to generate: {nsamples}")
except:
    logging.info("Sampler settings not found. Using Default.")
    nwalk = 1000
    nsamples = 10000


#################################################################################################

# Infer. with poco
#################################################################################################

# Generate 1000 random samples from the prior.
# These are the starting positions for the particles.
# Speaking with Minas 1000 is generally a good number.
logging.info("Sampling from prior.")
prior_samples = utils.sample_prior(prior_list, nwalk)

# Produce a bounds array for all the hard priors.
bounds_arr = []
for key_i in config_from_file['Parameters']:
    if key_i=='Fixed':
        continue
    bounds_arr.append(utils.poco_bounds(config_from_file['Parameters'][key_i]))
bounds_arr = np.vstack(bounds_arr)
logging.info(f"Hard bounds defined: {bounds_arr}")

logging.info("Defining sampler.")
if config_from_file['Setup']['likelihood']['function']=='combo_like' and not jeff_params and not marg_params:
    likelihood_args = [
        data_dict,
        samples,
        ngs,
        redshifts,
        km,
        fixed_values,
        engine_dict,
        Om_fid,
        nshared,
        nindiv
    ]

    # Define the sampler
    sampler = pc.Sampler(
        nwalk,
        prior_samples.shape[1],
        loglike,
        utils.evaluate_prior,
        vectorize_likelihood=True,
        vectorize_prior=True,
        bounds=bounds_arr,
        log_likelihood_args=likelihood_args,
        log_prior_args=[prior_list],
        infer_vectorization=False
    )
elif config_from_file['Setup']['likelihood']['function']=='combo_like' and jeff_params and not marg_params:
    likelihood_args = [
        data_dict,
        samples,
        ngs,
        redshifts,
        km,
        fixed_values,
        engine_dict,
        Om_fid,
        nshared,
        nindiv
    ]
    prior_args = [
        data_dict,
        samples,
        ngs,
        redshifts,
        km,
        fixed_values,
        engine_dict,
        Om_fid,
        nshared,
        nindiv,
        jeff_names,
        prior_list
    ]

    # Define the sampler
    sampler = pc.Sampler(
        nwalk,
        prior_samples.shape[1],
        loglike,
        utils.combo_jeff,
        vectorize_likelihood=True,
        vectorize_prior=True,
        bounds=bounds_arr,
        log_likelihood_args=likelihood_args,
        log_prior_args=prior_args,
        infer_vectorization=False
    )
elif config_from_file['Setup']['likelihood']['function']=='combo_marg_like' and not jeff_params and marg_params:
    likelihood_args = [
        data_dict,
        samples,
        ngs,
        redshifts,
        km,
        fixed_values,
        engine_dict,
        Om_fid,
        nshared,
        nindiv,
        marg_names,
        marg_cov
    ]

    sampler = pc.Sampler(
        nwalk,
        prior_samples.shape[1],
        loglike,
        utils.evaluate_prior,
        vectorize_likelihood=True,
        vectorize_prior=True,
        bounds=bounds_arr,
        log_likelihood_args=likelihood_args,
        log_prior_args=[prior_list],
        infer_vectorization=False
    )
elif config_from_file['Setup']['likelihood']['function']=='combo_marg_like' and jeff_params and marg_params:
    likelihood_args = [
        data_dict,
        samples,
        ngs,
        redshifts,
        km,
        fixed_values,
        engine_dict,
        Om_fid,
        nshared,
        nindiv,
        marg_names,
        marg_cov
    ]

    prior_args = [
        data_dict,
        samples,
        ngs,
        redshifts,
        km,
        fixed_values,
        engine_dict,
        Om_fid,
        nshared,
        nindiv,
        jeff_names,
        prior_list,
        jeff_cov
    ]

    sampler = pc.Sampler(
        nwalk,
        prior_samples.shape[1],
        loglike,
        utils.combo_jeff,
        vectorize_likelihood=True,
        vectorize_prior=True,
        bounds=bounds_arr,
        log_likelihood_args=likelihood_args,
        log_prior_args=prior_args,
        infer_vectorization=False
    )
else:
    raise NotImplementedError

# Run the MCMC
logging.info("Starting run.")
sampler.run(prior_samples, progress=False)

# Generate more samples.
# By default poco only generates n_particles number of samples.
logging.info("Generating additional samples.")
sampler.add_samples(nsamples-nwalk, progress=False)

# Extract the samples from the burn-in step.
samples = sampler.results['samples']

# Save samples
logging.info(f"Saving samples with name: {save_fname}")
np.save(save_fname, samples)

if 'diagnostics' not in config_from_file['Setup']['save']:
    config_from_file['Setup']['save']['diagnostics'] = False

if 'bridgez' not in config_from_file['Setup']:
    config_from_file['Setup']['bridgez'] = False

# If specified save some quantities that summarise how the chain progressed.
if config_from_file['Setup']['save']['diagnostics']:

    # Extract the desired quantities from the main results dict.
    new_dict = {k: sampler.results[k].tolist() for k in ['logz', 'ncall', 'accept', 'scale', 'steps']}

    # If specified do bridge sampling to estimate the model evidence.
    if config_from_file['Setup']['bridgez']:
        logging.info(f"Starting bridge sampling.")
        logz_bs, logz_bs_error = sampler.bridge_sampling()

        # Add the bridge sampling evidence and it's error to the dict.
        new_dict['logz_bs'] = {'value': float(logz_bs), 'error': float(logz_bs_error)}

    # Replace the start of the chain name, and make yaml file.
    diag_name = save_fname.replace('chain', 'diagnostic').replace('.npy', '.yaml')
    
    # Save dict. to yaml.
    logging.info(f"Saving diagnostic file with name: {diag_name}")
    with open(diag_name, 'w') as yaml_file:
        yaml.dump(new_dict, yaml_file, default_flow_style=False)

# Save likelihood for samples.

if sampler_settings['save_like']:
    like_fname = save_fname.replace('chain', 'likelihood')
    logging.info(f"Saving sample likelihoods with name: {like_fname}")
    np.save(like_fname, sampler.results['loglikelihood'])