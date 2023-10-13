import numpy as np
from matryoshka import eft_funcs
from matryoshka import halo_model_funcs
from matryoshka import rsd
from scipy.interpolate import interp1d
from unified_utils.utils import fix_params

def marg_like(theta, kobs, y, icov, icovd, ng, km, bg_prior, fixed_vals,
              engine, marg_names, Om_AP=None, window=None, M=None, 
              range_selection=None):
    '''
    Marginalised likelihood function. Marginalisation is done over parameters
    that appear linearly in the model.

    Args:
        theta (array) : Array containing the parameters that appear nonlinearly.
         Should have shape ``(nsamp, npar)`` or ``(npar,)``.
        kobs (array) : k-bins of data being fit. Should have shape ``(nk,)``.
        y (array) : Multipoles being fit. Should have shape ``(nl*nk,)``.
        icov (array) : Covaraince matrix for the multipoles. Should have shape
         ``(nl*nk, nl*nk)``.
        icovd (array) : The dot product of ``icov`` and ``y``. Should have
         shape ``(nl*nk, )``.
        ng (float) : Number density of data being fit.
        km (float) : Value of km.
        bg_prior (array) : Variance matrix for the parameters that appear
         linearly. Should have shape ``(nbg, nbg)``.
        fixed_vals (dict) : The values of any fixed parameters. Should have
         shape ``(nfix, )``.
        engine (class) : Prediction engine class. Should have a
         ``.predict_kernels()`` method.
        marg_names (list) : Names of parameters to be analytically marginalised.
         Should have length ``nbg``. Each element can be from the following 
         ``['b3', 'cct', 'cr1', 'cr2', 'ce1', 'cmono', 'cquad']``.
        Om_AP (float) : The fiducial value of Om. Default is ``None``. If a 
         value is passed AP will be included in prediction.
        window (array) : Window function matrix. If ``None`` predictions will
         not be convolved with the window function. Default is ``None``. 
         ``Om_AP`` and ``M`` must also be passed.
        M (array) : Wide angle matrix. If ``None`` wide angle effects will not
         be included. If not ``None``, ``Om_AP`` and ``window`` must also be
         passed. Default is ``None``.
        range_selection (array) : Array of boolean elements for imposing scale
         cuts. Only used if ``window`` and ``M`` are not ``None. Default is
         ``None``.
    Returns:
        Array with shape ``(nsamp, )`` with each element being the marginal
        likelihood for ``theta``.
    '''

    if theta.shape[0] == 0:
        return np.array([])

    # Fix parameters
    # wc, wb, h, As, b1, c2, b3, c4, cct, cr1, cr2, ce1, cmono, cquad
    # TODO: This line works with an old version of fixed params. Update for new
    # version.
    theta = fix_params(np.atleast_2d(theta), fixed_vals)

    # Sperate cosmo and bias params.
    # Oc, Ob, h, As
    cosmo = theta[:,:4]
    # b1, c2, b3, c4, cct, cr1, cr2
    bias = theta[:,4:11]
    # ce1, cmono, cquad
    stoch = theta[:,11:]
    
    # Make copies of c2 and c4.
    c2 = np.copy(bias[:,1])
    c4 = np.copy(bias[:,3])
    # Use the copies to calculate b2 and b4
    bias[:,1] = (c2+c4)/np.sqrt(2)
    bias[:,3] = (c2-c4)/np.sqrt(2)

    # Divide by km here to make things a bit cleaner calculating PG.
    bias[:,4:] = bias[:,4:]/km**2
    stoch[:,1:] = stoch[:,1:]/km**2
    
    # Predict growth for each cosmology.
    # Force shape to be (nsamp,1).
    f = np.atleast_2d(halo_model_funcs.fN_vec((cosmo[:,0]+cosmo[:,1])/cosmo[:,2]**2,
                                              engine.redshift)).T

    # If the window and M matrix are not none the appropriate
    # convolutions will be done.
    if (window is not None) and (M is not None):
        # Produce kernel predictions for each cosmology over full theory
        # k-range.
        # All have shape (nl, nsamp, nkern, nk)
        P11_pred, Ploop_pred, Pct_pred = engine.predict_kernels(cosmo,
                                                                engine.kbins)

        # Calculate the contributions from nonlinear parameters.
        # poles=True, so P_NG is list of len l. Elements have shape (nsamp, nk).
        P_NG = engine.calc_P_NG(P11_pred, Ploop_pred, Pct_pred, bias, f,
                         stoch=stoch, ng=ng, kobs=engine.kbins, poles=True)
        
        # Calculate the conributions from the linear parameters.
        # poles=True, so PG is list of len l. Elements have shape
        # (nkern, nsamp, nk).
        PG = engine.calc_P_G(Ploop_pred, Pct_pred, bias, f, ng, km, engine.kbins,
                      marg_names, poles=True)

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
        PNG_sys = np.zeros((cosmo.shape[0], P_NG[0].shape[1]*2))
        PG_sys = np.zeros((PG[0].shape[0], cosmo.shape[0], PG[0].shape[-1]*2))
        for i in range(cosmo.shape[0]):
            PNG_i = rsd.AP(np.stack([P_NG[0][i,:], P_NG[1][i,:]]),
                           np.linspace(-1,1,301), engine.kbins, qperp[i],
                           qpar[i])
            PNG_sys[i] = np.concatenate(PNG_i)

            PG_i = rsd.AP(np.stack([PG[0][:,i,:], PG[1][:,i,:]]),
                          np.linspace(-1,1,301), engine.kbins, qperp[i],
                          qpar[i])
            PG_sys[:,i,:] = np.hstack(PG_i)

        # Interpolate predictions over kbins needed to use matrices.
        # Assume P4 = 0.
        PNG_sys = np.hstack([interp1d(engine.kbins,
                                      PNG_sys[:,:engine.kbins.shape[0]],
                                      kind='cubic', bounds_error=False,
                                      fill_value='extrapolate')(np.linspace(0,0.4,400)), # P0
                             interp1d(engine.kbins, PNG_sys[:,engine.kbins.shape[0]:],
                                      kind='cubic', bounds_error=False,
                                      fill_value='extrapolate')(np.linspace(0,0.4,400)), #P2
                             np.zeros((PNG_sys.shape[0],400)) # P4
                            ])
        PG_sys = np.dstack([interp1d(engine.kbins, PG_sys[:,:,:engine.kbins.shape[0]],
                                     kind='cubic', bounds_error=False,
                                     fill_value='extrapolate')(np.linspace(0,0.4,400)), # P0
                            interp1d(engine.kbins, PG_sys[:,:,engine.kbins.shape[0]:],
                                     kind='cubic', bounds_error=False,
                                     fill_value='extrapolate')(np.linspace(0,0.4,400)), #P2
                            np.zeros((PG_sys.shape[0],PG_sys.shape[1],400)) # P4
                            ])

        # Do wide angle convolution.
        PNG_sys = np.einsum("ij, nj -> ni", M, PNG_sys)
        PG_sys = np.einsum("ij, knj -> kni", M, PG_sys)

        # Do window convolution.
        PNG_sys = np.einsum("ij, nj -> ni", window, PNG_sys)
        PG_sys = np.einsum("ij, knj -> kni", window, PG_sys)

        # Only select P0 and P2
        pole_selection = [True, False, True, False, False]
        PNG_sys = PNG_sys[:,np.repeat(pole_selection , 40)][:,np.concatenate(sum(pole_selection)*[range_selection])]
        PG_sys = PG_sys[:,:,np.repeat(pole_selection , 40)][:,:,np.concatenate(sum(pole_selection)*[range_selection])]

        PG = PG_sys
        P_NG = PNG_sys

    # If Om_AP is passed without window and M, only include AP.
    elif Om_AP is not None:
        # Produce kernel predictions for each cosmology at kobs.
        # All have shape (nl, nsamp, nkern, nk)
        P11_pred, Ploop_pred, Pct_pred = engine.predict_kernels(cosmo, kobs)

        # Calculate the contributions from nonlinear parameters.
        # poles=True, so P_NG is list of len l. Elements have shape (nsamp, nk).
        P_NG = engine.calc_P_NG(P11_pred, Ploop_pred, Pct_pred, bias, f, stoch=stoch,
                         ng=ng, kobs=kobs, poles=True)
        
        # Calculate the conributions from the linear parameters.
        # poles=True, so PG is list of len l. Elements have shape
        # (nkern, nsamp, nk).
        PG = engine.calc_P_G(Ploop_pred, Pct_pred, bias, f, ng, km, kobs, marg_names,
                      poles=True)

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
        PNG_sys = np.zeros((cosmo.shape[0], P_NG[0].shape[1]*2))
        PG_sys = np.zeros((PG[0].shape[0], cosmo.shape[0], PG[0].shape[-1]*2))
        for i in range(cosmo.shape[0]):
            PNG_i = rsd.AP(np.stack([P_NG[0][i,:], P_NG[1][i,:]]),
                           np.linspace(-1,1,301), kobs, qperp[i], qpar[i])
            PNG_sys[i] = np.concatenate(PNG_i)

            PG_i = rsd.AP(np.stack([PG[0][:,i,:], PG[1][:,i,:]]),
                          np.linspace(-1,1,301), kobs, qperp[i], qpar[i])
            PG_sys[:,i,:] = np.hstack(PG_i)
        
        PG = PG_sys
        P_NG = PNG_sys

    # If none of the systematic variables are passed simply predict the kernels.
    else:

        # Produce kernel predictions for each cosmology at kobs.
        # All have shape (nl, nsamp, nkern, nk)
        P11_pred, Ploop_pred, Pct_pred = engine.predict_kernels(cosmo, kobs)

        # Calculate the contributions from nonlinear parameters.
        # poles=False, so P_NG is array with shape (nsamp, 2*nk).
        P_NG = engine.calc_P_NG(P11_pred, Ploop_pred, Pct_pred, bias, f, stoch=stoch,
                         ng=ng, kobs=kobs, poles=False)
        
        # Calculate the contributions from nonlinear parameters.
        # poles=False, so PG is array with shape (nkern, nsamp, 2*nk).
        PG = engine.calc_P_G(Ploop_pred, Pct_pred, bias, f, ng, km, kobs, marg_names,
                      poles=False)
    
    # Calculate eq 26 from arXiv:2003.07956 for each cosmo.
    F2 = np.einsum("kni,ij,pnj->nkp", PG, icov, PG) + bg_prior[np.newaxis,:,:]
    F2_inv = np.linalg.inv(F2)
    
    # Caclulate eq 27 from arXiv:2003.07956 for each cosmo.
    # There is an extra minus sign in eq 27 that we do not include here.
    # It has no impact on the calculation of the marg. like. as it always
    # appears squared.
    F1 = np.einsum("ni,ij,knj->nk",P_NG,icov,PG) - np.einsum("i,kni->nk",icovd,PG)
    
    # Calculate eq 28 from arXiv:2003.07956 for each cosmo.
    # In eq 28 there is a term for the prior on the nonlinear params.
    # We exclude this as we want the marg. like. not the marg. posterior.
    F0 = np.einsum("nj,ij,in->n", (P_NG-y), icov, (P_NG-y).T)
    
    # Calculate eq 29 from arXiv:2003.07956 for each cosmo.
    chi2 = -np.einsum("ni,nij,nj->n",F1, F2_inv, F1) + F0\
           + np.log(np.linalg.det(F2/(2*np.pi)))
    
    return -0.5*chi2

def full_like(theta, kobs, y, icov, ng, km, fixed_vals, engine, Om_AP=None,
              window=None, M=None, range_selection=None, perturb_cond=False):
    '''
    Likelihood function.

    Args:
        theta (array) : Array containing the parameters that appear nonlinearly.
         Should have shape ``(nsamp, npar)`` or ``(npar,)``.
        kobs (array) : k-bins of data being fit. Should have shape ``(nk,)``.
        y (array) : Multipoles being fit. Should have shape ``(nl*nk,)``.
        icov (array) : Covaraince matrix for the multipoles. Should have shape
         ``(nl*nk, nl*nk)``.
        icovd (array) : The dot product of ``icov`` and ``y``. Should habe
         shape ``(nl*nk, )``.
        ng (float) : Number density of data being fit.
        km (float) : Value of km.
        fixed_vals (dict) : The values of any fixed parameters. Should have
         shape ``(nfix, )``.
        engine (class) : Prediction engine class. Should have a ``.predict()``
         method.
        Om_AP (float) : The fiducial value of Om. Default is ``None``. If a 
         value is passed AP will be included in prediction.
        window (array) : Window function matrix. If ``None`` predictions will
         not be convolved with the window function. Default is ``None``. 
         ``Om_AP`` and ``M`` must also be passed.
        M (array) : Wide angle matrix. If ``None`` wide angle effects will not
         be included. If not ``None``, ``Om_AP`` and ``window`` must also be
         passed. Default is ``None``.
        range_selection (array) : Array of boolean elements for imposing scale
         cuts. Only used if ``window`` and ``M`` are not ``None. Default is
         ``None``.
    Returns:
        Array with shape ``(nsamp, )`` with each element being the likelihood
         for ``theta``.
    '''

    if theta.shape[0] == 0:
        return np.array([])

    # Fix parameters
    # wc, wb, h, As, b1, c2, b3, c4, cct, cr1, cr2, ce1, cmono, cquad, Om_AP
    # TODO: This line works with an old version of fixed params. Update for new
    # version.
    theta = fix_params(np.atleast_2d(theta), fixed_vals)

    # Sperate cosmo and bias params.
    # Oc, Ob, h, As
    cosmo = theta[:,:4]
    # b1, c2, b3, c4, cct, cr1, cr2
    bias = theta[:,4:11]
    # ce1, cmono, cquad
    stoch = theta[:,11:]
    
    # Make copies of c2 and c4.
    c2 = np.copy(bias[:,1])
    c4 = np.copy(bias[:,3])
    # Use the copies to calculate b2 and b4
    bias[:,1] = (c2+c4)/np.sqrt(2)
    bias[:,3] = (c2-c4)/np.sqrt(2)

    # If the window and M matrix are not none the appropriate
    # convolutions will be done.
    if (window is not None) and (M is not None) and (Om_AP is not None):

        # If the perturbative condtion should be applied
        if perturb_cond:

            # Make a copy of the stochastic parameters, and include km.
            stochi = np.copy(stoch)
            stochi[:,1:] = stoch[:,1:]/km**2

            biasi = np.copy(bias)
            biasi[:,4:] = biasi[:,4:]/km**2

            # Make predictions for the kernels over the full emulator range.
            P11_pred, Ploop_pred, Pct_pred = engine.predict_kernels(cosmo, engine.kbins)

            # Make predictions for the growth rates.
            fs = halo_model_funcs.fN_vec((cosmo[:,0]+cosmo[:,1])/cosmo[:,2]**2, engine.redshift)

            # Calculate the individual contributions to the monopole.
            lin0, loop0, counter0, _ = eft_funcs.multipole_vec(
                [
                    P11_pred[0],
                    Ploop_pred[0],
                    Pct_pred[0]
                ],
                biasi,
                fs.reshape(-1,1),
                seperate=True
            )

            # Calculate the individual contributions to the quad.
            lin2, loop2, counter2, _ = eft_funcs.multipole_vec(
                [
                    P11_pred[1],
                    Ploop_pred[1],
                    Pct_pred[1]
                ],
                biasi,
                fs.reshape(-1,1),
                seperate=True
            )

            # Boolean array for samples that fail the condidtion.
            failed_cond = np.logical_or(
                perturb_cond_fn(lin0, loop0, counter0),
                perturb_cond_fn(lin2, loop2, counter2)
            )
            

            # Make prediction for P0 including stochastic terms.
            P0_pred = eft_funcs.multipole_vec(
                [
                    P11_pred[0],
                    Ploop_pred[0],
                    Pct_pred[0]
                ],
                biasi,
                fs.reshape(-1,1),
                stochastic=stochi,
                kbins=engine.kbins,
                ng=ng,
                multipole=0
            )

            # Make prediction for P2 including stochastic terms.
            P2_pred = eft_funcs.multipole_vec(
                [
                    P11_pred[1],
                    Ploop_pred[1],
                    Pct_pred[1]
                ],
                biasi,
                fs.reshape(-1,1),
                stochastic=stochi,
                kbins=engine.kbins,
                ng=ng,
                multipole=2
            )

            # Combine predictions into single array.
            pkl = np.hstack([P0_pred, P2_pred])

        else:
            # If no perturbative condidtion make predictions over the full emulator range as normal.
            pkl = engine.predict(cosmo, bias, stoch, ng, km)

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
        for i in range(cosmo.shape[0]):
            pred_i = rsd.AP(pkl[i].reshape(2,engine.kbins.shape[0]),
                            np.linspace(-1,1,301), engine.kbins, qperp[i],
                            qpar[i])

            pkl[i] = np.concatenate(pred_i)
        
        # Interpolate predictions over kbins needed to use matrices.
        # Assume P4 = 0.
        pkl = np.hstack([interp1d(engine.kbins, pkl[:,:engine.kbins.shape[0]],
                                  kind='cubic', bounds_error=False,
                                  fill_value='extrapolate')(np.linspace(0,0.4,400)), # P0
                         interp1d(engine.kbins, pkl[:,engine.kbins.shape[0]:],
                                  kind='cubic', bounds_error=False,
                                  fill_value='extrapolate')(np.linspace(0,0.4,400)), #P2
                         np.zeros((pkl.shape[0],400)) # P4
                         ])

        # Do wide angle convolution.
        pkl = np.einsum("ij, nj -> ni", M, pkl)

        # Do window convolution.
        pkl = np.einsum("ij, nj -> ni", window, pkl)

        # Only select P0 and P2
        pole_selection = [True, False, True, False, False]
        pkl = pkl[:,np.repeat(pole_selection , 40)][:,np.concatenate(sum(pole_selection)*[range_selection])]

    # If Om_AP is passed without window and M, only include AP.
    elif Om_AP is not None:

        if perturb_cond:
            raise NotImplementedError
        else:
            pkl = engine.predict(cosmo, bias, stoch, ng, km, kobs)

        DA = rsd.DA_vec(cosmo[:,0]/cosmo[:,2]**2+cosmo[:,1]/cosmo[:,2]**2,
                        engine.redshift)
        Hubble = rsd.Hubble(cosmo[:,0]/cosmo[:,2]**2+cosmo[:,1]/cosmo[:,2]**2,
                            engine.redshift)
        
        DA_fid = rsd.DA_vec(Om_AP, engine.redshift)
        Hubble_fid = rsd.Hubble(Om_AP, engine.redshift)
        
        qperp = DA/DA_fid
        qpar = Hubble_fid/Hubble
        
        for i in range(cosmo.shape[0]):
            pred_i = rsd.AP(pkl[i].reshape(2,kobs),
                                np.linspace(-1,1,301), kobs, qperp[i], qpar[i])

            pkl[i] = np.concatenate(pred_i)

    # If none of the systematic variables are passed simply make a theory
    # prediction with the engine.
    else:
        
        if perturb_cond:
            raise NotImplementedError
        else:
            pkl = engine.predict(cosmo, bias, stoch, ng, km, kobs)

    # Calculate the residuals between the predictions and the data.
    res = pkl-y
    
    if perturb_cond:
        l = -0.5*np.einsum("nj,ij,in->n", res, icov, res.T)
        l[failed_cond] -= 1e20
        return l
    else:
        return -0.5*np.einsum("nj,ij,in->n", res, icov, res.T)

def perturb_cond_fn(lin, loop, counterterm):
    '''
    Test if model remains perturbative based on two conditions: 1) Linear contribution is greater than loop contribution
     , 2) Linear contribution is greater than counterterm contribution.

     Args:
        lin (array) : Linear contributions.
        loop (array) : Loop contributions.
        counterterm (array) : Counterterm contributions.
    '''

    # Make sure arrays have shape (n_samp, nk), and take absolute values.
    lin = np.atleast_2d(lin)
    loop = np.atleast_2d(loop)
    counterterm = np.atleast_2d(counterterm)

    # Find any sample where loop > lin for any k considered.
    cond1 = np.any(np.abs(lin) < np.abs(loop), axis=1)
    cond1 = np.zeros_like(cond1).astype(bool)

    # Find any sample where counter > lin for any k considered.
    cond2 = np.any(np.abs(lin) < np.abs(counterterm+loop), axis=1)

    # Retun True for any samples that violate either cond1 or cond2
    return np.logical_or(cond1, cond2)

def combo_like(theta, data, samples, ngs, redshifts, km, fixed_params, engine_dict, Om_AP, nshared, nindiv):

    # Select parameters that are shared across all samples.
    shared_theta = theta[:, :nshared]

    # Loop over all samples to and sum likelihood
    like = 0
    for i in range(len(data)):

        # Define indecies to select parameters specific to the current sample.
        _idx1 = nshared+np.sum(nindiv[:i])
        _idx2 = nshared+np.sum(nindiv[:i+1])
        individual_theta = theta[:,_idx1:_idx2]

        # Combine the shared and individual parameters
        theta_i = np.hstack([shared_theta, individual_theta])

        like += full_like(
            theta_i,
            data[samples[i]]['k'],
            data[samples[i]]['data'],
            data[samples[i]]['icov'],
            ngs[i],
            km,
            fixed_params[i],
            engine_dict[redshifts[i]],
            Om_AP=Om_AP,
            window=data[samples[i]]['window'],
            M=data[samples[i]]['M'],
            range_selection=data[samples[i]]['range'],
            perturb_cond=False)

    return like

def combo_marg_like(theta, data, samples, ngs, redshifts, km, fixed_params, engine_dict, Om_AP, nshared, nindiv,
                    marg_names, marg_cov):

    # Select parameters that are shared across all samples.
    shared_theta = theta[:, :nshared]

    # Loop over all samples to and sum likelihood
    like = 0
    for i in range(len(data)):

        # Define indecies to select parameters specific to the current sample.
        _idx1 = int(nshared+np.sum(nindiv[:i]))
        _idx2 = int(nshared+np.sum(nindiv[:i+1]))
        individual_theta = theta[:,_idx1:_idx2]

        # Combine the shared and individual parameters
        theta_i = np.hstack([shared_theta, individual_theta])

        like += marg_like(
            theta_i,
            data[samples[i]]['k'],
            data[samples[i]]['data'],
            data[samples[i]]['icov'],
            np.dot(
                data[samples[i]]['icov'],
                data[samples[i]]['data']
            ),
            ngs[i],
            km,
            marg_cov[i],
            fixed_params[i],
            engine_dict[redshifts[i]],
            marg_names[i],
            Om_AP=Om_AP,
            window=data[samples[i]]['window'],
            M=data[samples[i]]['M'],
            range_selection=data[samples[i]]['range']
        )

    return like

def combo_marg_like_joint(theta, data, samples, ngs, redshifts, km, fixed_params, engine_dict, Om_AP, nshared, nindiv,
                          marg_names, joint_marg_cov, joint_icov, joint_data_vector, icovd):
    
    # Select parameters that are shared across all samples.
    shared_theta = theta[:, :nshared]

    # Loop over each sample and compute the Gaussian and non-Gaussian contributions for each sample prediction.
    PGs = []
    PNGs = []
    for i in range(len(data)):

        # Define indecies to select parameters specific to the current sample.
        _idx1 = nshared+np.sum(nindiv[:i])
        _idx2 = nshared+np.sum(nindiv[:i+1])
        individual_theta = theta[:,_idx1:_idx2]

        # Combine the shared and individual parameters
        theta_i = np.hstack([shared_theta, individual_theta])

        theta_i = fix_params(np.atleast_2d(theta_i), fixed_params[i])
        engine_i = engine_dict[redshifts[i]]
        ng_i = ngs[i]
        M_i = data[samples[i]]['M']
        window_i = data[samples[i]]['window']

        # Sperate cosmo and bias params.
        # Oc, Ob, h, As
        cosmo = theta_i[:,:4]
        # b1, c2, b3, c4, cct, cr1, cr2
        bias = theta_i[:,4:11]
        # ce1, cmono, cquad
        stoch = theta_i[:,11:]
        
        # Make copies of c2 and c4.
        c2 = np.copy(bias[:,1])
        c4 = np.copy(bias[:,3])
        # Use the copies to calculate b2 and b4
        bias[:,1] = (c2+c4)/np.sqrt(2)
        bias[:,3] = (c2-c4)/np.sqrt(2)

        # Divide by km here to make things a bit cleaner calculating PG.
        bias[:,4:] = bias[:,4:]/km**2
        stoch[:,1:] = stoch[:,1:]/km**2
        
        # Predict growth for each cosmology.
        # Force shape to be (nsamp,1).
        f = np.atleast_2d(halo_model_funcs.fN_vec((cosmo[:,0]+cosmo[:,1])/cosmo[:,2]**2,
                                                engine_i.redshift)).T

        P11_pred, Ploop_pred, Pct_pred = engine_i.predict_kernels(cosmo,
                                                                  engine_i.kbins)

        # Calculate the contributions from nonlinear parameters.
        # poles=True, so P_NG is list of len l. Elements have shape (nsamp, nk).
        P_NG = engine.calc_P_NG(P11_pred, Ploop_pred, Pct_pred, bias, f,
                         stoch=stoch, ng=ng_i, kobs=engine_i.kbins, poles=True)
        
        # Calculate the conributions from the linear parameters.
        # poles=True, so PG is list of len l. Elements have shape
        # (nkern, nsamp, nk).
        PG = engine.calc_P_G(Ploop_pred, Pct_pred, bias, f, ng_i, km, engine_i.kbins,
                      marg_names[i], poles=True)

        # Calculate DA and H0 for each cosmology.
        DA = rsd.DA_vec(cosmo[:,0]/cosmo[:,2]**2+cosmo[:,1]/cosmo[:,2]**2,
                        engine_i.redshift)
        Hubble = rsd.Hubble(cosmo[:,0]/cosmo[:,2]**2+cosmo[:,1]/cosmo[:,2]**2,
                            engine_i.redshift)
        
        # Compute DA and H0 for Om_AP
        DA_fid = rsd.DA_vec(Om_AP, engine_i.redshift)
        Hubble_fid = rsd.Hubble(Om_AP, engine_i.redshift)
        
        # Calculate AP parameters.
        qperp = DA/DA_fid
        qpar = Hubble_fid/Hubble

        # Inlcude AP effect.
        PNG_sys = np.zeros((cosmo.shape[0], P_NG[0].shape[1]*2))
        PG_sys = np.zeros((PG[0].shape[0], cosmo.shape[0], PG[0].shape[-1]*2))
        for j in range(cosmo.shape[0]):
            PNG_i = rsd.AP(np.stack([P_NG[0][j,:], P_NG[1][j,:]]),
                           np.linspace(-1,1,301), engine_i.kbins, qperp[j],
                           qpar[j])
            PNG_sys[j] = np.concatenate(PNG_i)

            PG_i = rsd.AP(np.stack([PG[0][:,j,:], PG[1][:,j,:]]),
                          np.linspace(-1,1,301), engine_i.kbins, qperp[j],
                          qpar[j])
            PG_sys[:,j,:] = np.hstack(PG_i)

        # Interpolate predictions over kbins needed to use matrices.
        # Assume P4 = 0.
        PNG_sys = np.hstack([interp1d(engine_i.kbins,
                                      PNG_sys[:,:engine_i.kbins.shape[0]],
                                      kind='cubic', bounds_error=False,
                                      fill_value='extrapolate')(np.linspace(0,0.4,400)), # P0
                             interp1d(engine_i.kbins, PNG_sys[:,engine_i.kbins.shape[0]:],
                                      kind='cubic', bounds_error=False,
                                      fill_value='extrapolate')(np.linspace(0,0.4,400)), #P2
                             np.zeros((PNG_sys.shape[0],400)) # P4
                            ])
        PG_sys = np.dstack([interp1d(engine_i.kbins, PG_sys[:,:,:engine_i.kbins.shape[0]],
                                     kind='cubic', bounds_error=False,
                                     fill_value='extrapolate')(np.linspace(0,0.4,400)), # P0
                            interp1d(engine_i.kbins, PG_sys[:,:,engine_i.kbins.shape[0]:],
                                     kind='cubic', bounds_error=False,
                                     fill_value='extrapolate')(np.linspace(0,0.4,400)), #P2
                            np.zeros((PG_sys.shape[0],PG_sys.shape[1],400)) # P4
                            ])

        # Do wide angle convolution.
        PNG_sys = np.einsum("ij, nj -> ni", M_i, PNG_sys)
        PG_sys = np.einsum("ij, knj -> kni", M_i, PG_sys)

        # Do window convolution.
        PNG_sys = np.einsum("ij, nj -> ni", window_i, PNG_sys)
        PG_sys = np.einsum("ij, knj -> kni", window_i, PG_sys)

        # Only select P0 and P2
        pole_selection = [True, False, True, False, False]
        PNG_sys = PNG_sys[:,np.repeat(pole_selection , 40)][:,np.concatenate(sum(pole_selection)*[data[samples[i]]['range']])]
        PG_sys = PG_sys[:,:,np.repeat(pole_selection , 40)][:,:,np.concatenate(sum(pole_selection)*[data[samples[i]]['range']])]

        # We need to pad the Gaussian contributions.
        # We do this as the for each sample do not affect other samples.
        PG_sys = np.pad(
            PG_sys,
            [
                (0,0),
                (0,0),
                (i*PG_sys.shape[-1], (len(samples)-1-i)*PG_sys.shape[-1]) # Only pad along k dimension.
            ]
        )
        PGs.append(PG_sys)
        PNGs.append(PNG_sys)

    PG = np.vstack(PGs)
    P_NG = np.hstack(PNGs)

    # Calculate eq 26 from arXiv:2003.07956 for each cosmo.
    F2 = np.einsum("kni,ij,pnj->nkp", PG, joint_icov, PG) + joint_marg_cov[np.newaxis,:,:]
    F2_inv = np.linalg.inv(F2)
    
    # Caclulate eq 27 from arXiv:2003.07956 for each cosmo.
    # There is an extra minus sign in eq 27 that we do not include here.
    # It has no impact on the calculation of the marg. like. as it always
    # appears squared.
    F1 = np.einsum("ni,ij,knj->nk",P_NG,joint_icov,PG) - np.einsum("i,kni->nk",icovd,PG)
    
    # Calculate eq 28 from arXiv:2003.07956 for each cosmo.
    # In eq 28 there is a term for the prior on the nonlinear params.
    # We exclude this as we want the marg. like. not the marg. posterior.
    F0 = np.einsum("nj,ij,in->n", (P_NG-joint_data_vector), joint_icov, (P_NG-joint_data_vector).T)
    
    # Calculate eq 29 from arXiv:2003.07956 for each cosmo.
    chi2 = -np.einsum("ni,nij,nj->n",F1, F2_inv, F1) + F0\
           + np.log(np.linalg.det(F2/(2*np.pi)))
    
    return -0.5*chi2

def template_like(theta, kobs, y, icov, ng, km, fixed_vals, kernels_fid, kbins_fid,
              window=None, M=None, range_selection=None):
    '''
    Likelihood function.

    Args:
        theta (array) : Array containing the parameters that appear nonlinearly.
         Should have shape ``(nsamp, npar)`` or ``(npar,)``.
        kobs (array) : k-bins of data being fit. Should have shape ``(nk,)``.
        y (array) : Multipoles being fit. Should have shape ``(nl*nk,)``.
        icov (array) : Covaraince matrix for the multipoles. Should have shape
         ``(nl*nk, nl*nk)``.
        icovd (array) : The dot product of ``icov`` and ``y``. Should habe
         shape ``(nl*nk, )``.
        ng (float) : Number density of data being fit.
        km (float) : Value of km.
        fixed_vals (dict) : The values of any fixed parameters. Should have
         shape ``(nfix, )``.
        kernels_fid (list) : Kernels computed at fiducial cosmology. Should have form ``[P11, Ploop, Pct]``, where
         ``P11`` is the a ``np.array`` containing the linear kernels.
        kbins_fid (array) : kbins corresponding to ``kernels_fid``.
        window (array) : Window function matrix. If ``None`` predictions will
         not be convolved with the window function. Default is ``None``. 
         ``Om_AP`` and ``M`` must also be passed.
        M (array) : Wide angle matrix. If ``None`` wide angle effects will not
         be included. If not ``None``, ``Om_AP`` and ``window`` must also be
         passed. Default is ``None``.
        range_selection (array) : Array of boolean elements for imposing scale
         cuts. Only used if ``window`` and ``M`` are not ``None. Default is
         ``None``.
    Returns:
        Array with shape ``(nsamp, )`` with each element being the likelihood
         for ``theta``.
    '''

    if theta.shape[0] == 0:
        return np.array([])

    # Fix parameters
    # qperp, qpar, f, b1, c2, b3, c4, cct, cr1, cr2, ce1, cmono, cquad, Om_AP
    theta = fix_params(np.atleast_2d(theta), fixed_vals)

    # Sperate cosmo and bias params.
    # qperp, qpar, f
    cosmo = theta[:,:3]
    # b1, c2, b3, c4, cct, cr1, cr2
    bias = theta[:,3:10]
    # ce1, cmono, cquad
    stoch = theta[:,10:]
    
    # Make copies of c2 and c4.
    c2 = np.copy(bias[:,1])
    c4 = np.copy(bias[:,3])
    # Use the copies to calculate b2 and b4
    bias[:,1] = (c2+c4)/np.sqrt(2)
    bias[:,3] = (c2-c4)/np.sqrt(2)

    stoch[:,1:] = stoch[:,1:]/km**2
    bias[:,4:] = bias[:,4:]/km**2

    # If the window and M matrix are not none the appropriate
    # convolutions will be done.
    if (window is not None) and (M is not None):

        # Make prediction for P0 including stochastic terms.
        P0_pred = eft_funcs.multipole_vec(
            [
                np.stack(cosmo.shape[0]*[kernels_fid[0][0]]),
                np.stack(cosmo.shape[0]*[kernels_fid[1][0]]),
                np.stack(cosmo.shape[0]*[kernels_fid[2][0]])
            ],
            bias,
            cosmo[:,-1].reshape(-1,1),
            stochastic=stoch,
            kbins=kbins_fid,
            ng=ng,
            multipole=0
        )

        # Make prediction for P2 including stochastic terms.
        P2_pred = eft_funcs.multipole_vec(
            [
                np.stack(cosmo.shape[0]*[kernels_fid[0][1]]),
                np.stack(cosmo.shape[0]*[kernels_fid[1][1]]),
                np.stack(cosmo.shape[0]*[kernels_fid[2][1]])
            ],
            bias,
            cosmo[:,-1].reshape(-1,1),
            stochastic=stoch,
            kbins=kbins_fid,
            ng=ng,
            multipole=2
        )

        # Combine predictions into single array.
        pkl = np.hstack([P0_pred, P2_pred])
        
        # Inlcude AP effect.
        for i in range(cosmo.shape[0]):
            pred_i = rsd.AP(pkl[i].reshape(2,kbins_fid.shape[0]),
                            np.linspace(-1,1,301), kbins_fid, cosmo[i][0],
                            cosmo[i][1])

            pkl[i] = np.concatenate(pred_i)
        
        # Interpolate predictions over kbins needed to use matrices.
        # Assume P4 = 0.
        pkl = np.hstack([interp1d(kbins_fid, pkl[:,:kbins_fid.shape[0]],
                                  kind='cubic', bounds_error=False,
                                  fill_value='extrapolate')(np.linspace(0,0.4,400)), # P0
                         interp1d(kbins_fid, pkl[:,kbins_fid.shape[0]:],
                                  kind='cubic', bounds_error=False,
                                  fill_value='extrapolate')(np.linspace(0,0.4,400)), #P2
                         np.zeros((pkl.shape[0],400)) # P4
                         ])

        # Do wide angle convolution.
        pkl = np.einsum("ij, nj -> ni", M, pkl)

        # Do window convolution.
        pkl = np.einsum("ij, nj -> ni", window, pkl)

        # Only select P0 and P2
        pole_selection = [True, False, True, False, False]
        pkl = pkl[:,np.repeat(pole_selection , 40)][:,np.concatenate(sum(pole_selection)*[range_selection])]

    # Calculate the residuals between the predictions and the data.
    res = pkl-y
    
    return -0.5*np.einsum("nj,ij,in->n", res, icov, res.T)