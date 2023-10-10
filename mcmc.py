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
redshift = config_from_file['Setup']['redshift']
logging.info(f"redshift: {redshift}")
split = config_from_file['Setup']['split']
logging.info(f"split: {split}")
kmin = config_from_file['Setup']['kmin']
logging.info(f"kmin: {kmin}")
kmax = config_from_file['Setup']['kmax']
logging.info(f"kmax: {kmax}")
data_type = config_from_file['Setup']['data_type']
logging.info(f"data type: {data_type}")
norm_cov = config_from_file['Setup']['norm_cov']
logging.info(f"covariance normalisation factor: {norm_cov}")

# Deine the number density.
# This is almost certainly wrong for 6dF.
if redshift == 0.096 or redshift == 0.38 or redshift == 0.61:
    ng=4e-4
elif redshift == 1.52:
    ng=1.5e-5
logging.info(f"number density: {ng}")


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

logging.info("Loading prediction engine.")
engine_from_file = utils.import_loglikelihood(config_from_file['Setup']['engine']['path'],
                                                            config_from_file['Setup']['engine']['function'],
                                                            engine_or_like='engine')
engine = engine_from_file(redshift, **config_from_file['Setup']['engine']['engine_kwargs'])

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

# Limit range to that covered by the emulator or that specified by the user.
kbins_emu = engine.kbins

# Check kmin and kmax are compatible with the emulator.
if kmax>=kbins_emu.max():
    raise ValueError(f"Specified kmax greater than emulator kmax. Emulator kmax at z={redshift} is {kbins_emu.max()}")
if kmin<=kbins_emu.min():
    raise ValueError(f"Specified kmin less than emulator kmin. Emulator kmin at z={redshift} is {kbins_emu.min()}")

logging.info("Loading data.")
kbins_fit, pk_data_fit, cov_fit, window, M, range_selection, Nmocks = utils.data_loader(
    redshift,
    split,
    use_mock=use_mock,
    kmin=kmin,
    kmax=kmax, 
    path_to_repo=path_to_repo,
    norm_cov=norm_cov,
    pole_selection=pole_selection,
    pybird_mock=pybird_mock,
    mock_type=mock_type
)

#################################################################################################

# Infer. setup
#################################################################################################

# Invert the covariance matrix
icov = np.linalg.inv(cov_fit)
try:
    config_from_file['Setup']['hartlap']
    if config_from_file['Setup']['hartlap']:
        icov *= (Nmocks - len(pk_data_fit) - 2)/(Nmocks - 1)
        logging.info(f"Applying Hartlap factor. Nmocks={Nmocks}. H={(Nmocks - len(pk_data_fit) - 2)/(Nmocks - 1)}")
except:
    logging.info("Not specified if Hartlap factor should be used. Default behaviour is not to use it.")

# Make sure the order of the parameters is as expected.
# Define the expected order
expected_order = ['w_c', 'w_b', 'h', 'As', 'b1', 'c2', 'b3', 'c4',
                  'cct', 'cr1', 'cr2', 'ce1', 'cmono', 'cquad']
# If not in the expected order, reorder.
if not np.all(np.array(list(config_from_file['Parameters'].keys()))==np.array(expected_order)):
    config_from_file['Parameters'] = {p: config_from_file['Parameters'][p] for p in expected_order}

# Extract the fixed parameter values from the config file.
logging.info("Extracting fixed values.")
fixed_values = {p: config_from_file['Parameters'][p]['fixed'] if 'fixed' in config_from_file['Parameters'][p] else None\
               for p in config_from_file['Parameters']}
logging.info(f"Fixed values: {fixed_values}")

# Extract the extremes of the training sapce for the emulator.
# Can be used as a prior on cosmo. params.
if config_from_file['Setup']['engine']['type']=='emulator':
    emu_bounds = np.stack([engine.P0_emu.P11.scalers[0].min_val,
                           engine.P0_emu.P11.scalers[0].min_val+engine.P0_emu.P11.scalers[0].diff]).T
else:
    if 'path_to_model' in config_from_file['Setup']['engine']['engine_kwargs']:
        xscaler_min_diff = np.load(f"{config_from_file['Setup']['engine']['engine_kwargs']}scalers/xscaler_min_diff.npy")
        emu_bounds = np.stack([xscaler_min_diff[0], xscaler_min_diff[0]+xscaler_min_diff[1]])

# Convert emu. bounds array into nested dictionary.
emu_bounds_dict = {p:{'min':emu_bounds[i,0], 'max':emu_bounds[i,1]} for i,p in enumerate(['w_c', 'w_b', 'h', 'As'])}

# Check that the user specified extremes for the cosmo. priors lie within the emulator training space.
# All min or max values specified as 'emu' with the corresponding emu. bounds value. 
config_from_file['Parameters'] = utils.check_emu_bounds(config_from_file['Parameters'], emu_bounds_dict)

logging.info("Check for parameters that need special treatment.")
marg_cov, marg_names, jeff_cov, jeff_names = utils.special_treatment(config_from_file['Parameters'])
marg_params = False
if not (marg_names is None):
    logging.info(f"Parameters analytically marginalised: {marg_names}")
    logging.info(f"Prior covariance: {marg_cov}")
    marg_params = True
jeff_params = False
if not (jeff_names is None):
    logging.info(f"Parameters to have Jeffreys prior: {jeff_names}")
    logging.info(f"Prior covariance: {jeff_cov}")
    jeff_params = True

# If there are parameters to be marginalised over fix their values on the
# fixed_values dict.
if marg_params:
    for p in marg_names:
            fixed_values[p]=0.

# Construct prior from config.
prior_list = utils.make_prior_list(config_from_file['Parameters'])

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

try:
    config_from_file['Setup']['perturb_cond']
    if config_from_file['Setup']['perturb_cond']:
        perturb_cond = True
        logging.info(f"Using perturbative condition when sampling.")
    else:
        perturb_cond = False
except:
    logging.info("Not specified if perturbative condition should be used. Default is Fasle.")
    perturb_cond = False


#################################################################################################

# Infer. with poco
#################################################################################################

# Generate 1000 random samples from the prior.
# These are the starting positions for the particles.
# Speaking with Minas 1000 is generally a good number.
logging.info("Sampling from prior.")
prior_samples = utils.sample_prior(prior_list, nwalk)

# Produce a bounds array for all the hard priors.
bounds_arr = utils.poco_bounds(config_from_file['Parameters'])
logging.info(f"Hard bounds defined: {bounds_arr}")

logging.info("Defining sampler.")
if config_from_file['Setup']['likelihood']['function']=='marg_like' and not jeff_params:
    likelihood_args = [kbins_fit, pk_data_fit, icov, np.dot(icov,pk_data_fit),
                       ng, km, marg_cov, fixed_values, engine, marg_names,
                       Om_fid, window, M, range_selection]

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
elif config_from_file['Setup']['likelihood']['function']=='full_like' and jeff_params:
    likelihood_args = [kbins_fit, pk_data_fit, icov, ng, km, fixed_values,
                       engine, Om_fid, window, M, range_selection, perturb_cond]
    prior_args = [engine, kbins_fit, icov, fixed_values, ng, km, jeff_names,
                  prior_list, Om_fid, window, M, range_selection, True, jeff_cov]

    # Define the sampler
    sampler = pc.Sampler(
        nwalk,
        prior_samples.shape[1],
        loglike,
        utils.jeff_counter_sys,
        vectorize_likelihood=True,
        vectorize_prior=True,
        bounds=bounds_arr,
        log_likelihood_args=likelihood_args,
        log_prior_args=prior_args,
        infer_vectorization=False
    )
elif config_from_file['Setup']['likelihood']['function']=='marg_like' and jeff_params and marg_params:
    likelihood_args = [kbins_fit, pk_data_fit, icov, np.dot(icov,pk_data_fit),
                       ng, km, marg_cov, fixed_values, engine, marg_names,
                       Om_fid, window, M, range_selection]
    prior_args = [engine, kbins_fit, icov, fixed_values, ng, km, jeff_names,
                  prior_list, Om_fid, window, M, range_selection, True, jeff_cov]

    # Define the sampler
    sampler = pc.Sampler(
        nwalk,
        prior_samples.shape[1],
        loglike,
        utils.jeff_counter_sys,
        vectorize_likelihood=True,
        vectorize_prior=True,
        bounds=bounds_arr,
        log_likelihood_args=likelihood_args,
        log_prior_args=prior_args,
        infer_vectorization=False
    )
elif config_from_file['Setup']['likelihood']['function']=='full_like':
    likelihood_args = [kbins_fit, pk_data_fit, icov, ng, km, fixed_values,
                       engine, Om_fid, window, M, range_selection, perturb_cond]

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