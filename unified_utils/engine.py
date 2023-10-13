import numpy as np
import matryoshka.emulator as MatEmu
from scipy.interpolate import interp1d
from matryoshka.training_funcs import LogScaler, UniformScaler
from matryoshka import halo_model_funcs
from matryoshka import rsd

scaler_fn_dict = {'log':LogScaler, 'uniform':UniformScaler}

class emu_engine:
    '''
    EFTEMU engine.

    Args:
        redshift (float) : Redshift at which to make predictions. A version of the EFTEMU
         needs to have been trained at this redshift.
        **engine_kwargs : Keyword arguments for ``matryoshka.emulator.EFT``.
    '''

    def __init__(self, redshift, **engine_kwargs):

        self.P0_emu = MatEmu.EFT(multipole=0, version=engine_kwargs['version'], redshift=redshift,
                                 path_to_model=engine_kwargs['path_to_model'],
                                 yscaler_Ploop=scaler_fn_dict[engine_kwargs['Ploop0_scaler']](),
                                 yscaler_P11=scaler_fn_dict[engine_kwargs['P110_scaler']](),
                                 yscaler_Pct=scaler_fn_dict[engine_kwargs['Pct0_scaler']]())
        self.P2_emu = MatEmu.EFT(multipole=2, version=engine_kwargs['version'], redshift=redshift,
                                 path_to_model=engine_kwargs['path_to_model'],
                                 yscaler_Ploop=scaler_fn_dict[engine_kwargs['Ploop2_scaler']](),
                                 yscaler_P11=scaler_fn_dict[engine_kwargs['P112_scaler']](),
                                 yscaler_Pct=scaler_fn_dict[engine_kwargs['Pct2_scaler']]())

        self.redshift = redshift
        self.kbins = self.P0_emu.P11.kbins

    def predict(self, cosmo, bias, stoch, ng, km, kobs=None):
        '''
        Make multipole predictions with the engine.

        Args:
            cosmo (array) : Cosmology for prediction. Should have shape 
             ``(nsamp, 4)`` or ``(4, )``.
            bias (array) : Bias parameters and counterterms for prediction.
             Should have shape ``(nsamp, 7)`` or ``(7, )``.
            stoch (array) : Stochastic counterterms for prediction. Should
             have shape ``(nsamp, 3)`` or ``(3, )``.
            ng (float) : Number density from prediction.
            km (float) : km value.
            kobs (array) : kbins at which to make prediction. Default is 
             ``None``, in which case predictions will be made at
             ``self.kbins``.
        Returns:
            Predictions for the power spectum multipoles. Will have shape ``(nsamp, nl*nk)``.
        '''

        P0_pred = self.P0_emu.emu_predict(cosmo, bias, stochastic=stoch, ng=ng,
                                          km=km, kvals=kobs)
        P2_pred = self.P2_emu.emu_predict(cosmo, bias, stochastic=stoch, ng=ng,
                                          km=km, kvals=kobs)

        return np.hstack([P0_pred, P2_pred])


    def predict_kernels(self, cosmo, kobs):
        '''
        Make kernel predictions with the engine.

        Args:
            cosmo (array) : Cosmology for prediction. Should have shape 
             ``(nsamp, 4)`` or ``(4, )``.
            kobs (array) : k-bins of data being fit. Should have shape ``(nk,)``.
        Returns:
            Array containing predictions for the cosmology-dependent kernels. 
            Will have shape ``(nl=2, nsamp, nkern=21, nk)``.
        '''

        P11_pred = np.array([self.P0_emu.P11.emu_predict(cosmo).reshape(cosmo.shape[0],3,self.P0_emu.P11.kbins.shape[0]),
                            self.P2_emu.P11.emu_predict(cosmo).reshape(cosmo.shape[0],3,self.P2_emu.P11.kbins.shape[0])])
        P11_pred = interp1d(self.P0_emu.P11.kbins, P11_pred, axis=-1)(kobs)

        Ploop_pred = np.array([self.P0_emu.Ploop.emu_predict(cosmo).reshape(cosmo.shape[0],12,self.P0_emu.Ploop.kbins.shape[0]),
                            self.P2_emu.Ploop.emu_predict(cosmo).reshape(cosmo.shape[0],12,self.P2_emu.Ploop.kbins.shape[0])])
        Ploop_pred = interp1d(self.P0_emu.Ploop.kbins, Ploop_pred, axis=-1)(kobs)

        Pct_pred = np.array([self.P0_emu.Pct.emu_predict(cosmo).reshape(cosmo.shape[0],6,self.P0_emu.Pct.kbins.shape[0]),
                            self.P2_emu.Pct.emu_predict(cosmo).reshape(cosmo.shape[0],6,self.P2_emu.Pct.kbins.shape[0])])
        Pct_pred = interp1d(self.P0_emu.Ploop.kbins, Pct_pred, axis=-1)(kobs)
        
        return P11_pred, Ploop_pred, Pct_pred

    def derivs_for_jeff(self, theta, kobs, ng, km, jeff_names, Om_AP=None,
                        window=None, M=None, range_selection=None):

        # Calculation of derivatives only supported for linearly appearing
        # nuiscance parameters.
        # Loop over all nonlinear model parameters and raise error if any of
        # them appear in jeff_names
        for pi in ['w_c', 'w_b', 'h', 'A_s', 'b_1', 'c_2', 'c_4']:
            if pi in jeff_names:
                raise NotImplementedError(f"Calculation of derivative wrt {pi} is not implemented. Check configuration for {pi}.")

        # Much of what follows is the same as in the marginalised likelihood 
        # function.
        #TODO: Work on inlcuding calculatio for marg like in engine to avoid
        # all of this repeated code.
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
        f = np.atleast_2d(halo_model_funcs.fN_vec(
            (cosmo[:,0]+cosmo[:,1])/cosmo[:,2]**2,
            self.redshift
        )).T

        # If the window and M matrix are not none the appropriate
        # convolutions will be done.
        if (window is not None) and (M is not None):
            # Produce kernel predictions for each cosmology over full theory
            # k-range.
            # All have shape (nl, nsamp, nkern, nk)
            P11_pred, Ploop_pred, Pct_pred = self.predict_kernels(
                cosmo,
                self.kbins
            )
            
            # Calculate the conributions from the linear parameters.
            # poles=True, so PG is list of len l. Elements have shape
            # (nkern, nsamp, nk).
            PG = self.calc_P_G(
                Ploop_pred,
                Pct_pred,
                bias,
                f,
                ng,
                km,
                self.kbins,
                jeff_names,
                poles=True
                )

            # Calculate DA and H0 for each cosmology.
            DA = rsd.DA_vec(
                cosmo[:,0]/cosmo[:,2]**2+cosmo[:,1]/cosmo[:,2]**2,
                self.redshift
            )
            Hubble = rsd.Hubble(
                cosmo[:,0]/cosmo[:,2]**2+cosmo[:,1]/cosmo[:,2]**2,
                self.redshift
            )
            
            # Compute DA and H0 for Om_AP
            DA_fid = rsd.DA_vec(Om_AP, self.redshift)
            Hubble_fid = rsd.Hubble(Om_AP, self.redshift)
            
            # Calculate AP parameters.
            qperp = DA/DA_fid
            qpar = Hubble_fid/Hubble

            # Inlcude AP effect.
            PG_sys = np.zeros((PG[0].shape[0], cosmo.shape[0], PG[0].shape[-1]*2))
            for i in range(cosmo.shape[0]):
                PG_i = rsd.AP(
                    np.stack(
                        [
                            PG[0][:,i,:],
                            PG[1][:,i,:]
                        ]
                    ),
                    np.linspace(-1,1,301),
                    self.kbins,
                    qperp[i],
                    qpar[i]
                )
                PG_sys[:,i,:] = np.hstack(PG_i)

            # Interpolate predictions over kbins needed to use matrices.
            # Assume P4 = 0.
            PG_sys = np.dstack([interp1d(self.kbins, PG_sys[:,:,:self.kbins.shape[0]],
                                        kind='cubic', bounds_error=False,
                                        fill_value='extrapolate')(np.linspace(0,0.4,400)), # P0
                                interp1d(self.kbins, PG_sys[:,:,self.kbins.shape[0]:],
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

        # If Om_AP is passed without window and M, only include AP.
        elif Om_AP is not None:
            # Produce kernel predictions for each cosmology at kobs.
            # All have shape (nl, nsamp, nkern, nk)
            P11_pred, Ploop_pred, Pct_pred = self.predict_kernels(cosmo, kobs)
            
            # Calculate the conributions from the linear parameters.
            # poles=True, so PG is list of len l. Elements have shape
            # (nkern, nsamp, nk).
            PG = self.calc_P_G(
                Ploop_pred,
                Pct_pred,
                bias,
                f,
                ng,
                km,
                kobs,
                jeff_names,
                poles=True
            )

            # Calculate DA and H0 for each cosmology.
            DA = rsd.DA_vec(
                cosmo[:,0]/cosmo[:,2]**2+cosmo[:,1]/cosmo[:,2]**2,
                self.redshift
            )
            Hubble = rsd.Hubble(
                cosmo[:,0]/cosmo[:,2]**2+cosmo[:,1]/cosmo[:,2]**2,
                self.redshift
            )
            
            # Compute DA and H0 for Om_AP
            DA_fid = rsd.DA_vec(Om_AP, self.redshift)
            Hubble_fid = rsd.Hubble(Om_AP, self.redshift)
            
            # Calculate AP parameters.
            qperp = DA/DA_fid
            qpar = Hubble_fid/Hubble

            # Inlcude AP effect.
            PG_sys = np.zeros((PG[0].shape[0], cosmo.shape[0], PG[0].shape[-1]*2))
            for i in range(cosmo.shape[0]):
                PG_i = rsd.AP(np.stack([PG[0][:,i,:], PG[1][:,i,:]]),
                            np.linspace(-1,1,301), kobs, qperp[i], qpar[i])
                PG_sys[:,i,:] = np.hstack(PG_i)
            
            PG = PG_sys

        # If none of the systematic variables are passed simply predict the kernels.
        else:

            # Produce kernel predictions for each cosmology at kobs.
            # All have shape (nl, nsamp, nkern, nk)
            P11_pred, Ploop_pred, Pct_pred = self.predict_kernels(cosmo, kobs)
            
            # Calculate the contributions from nonlinear parameters.
            # poles=False, so PG is array with shape (nkern, nsamp, 2*nk).
            PG = self.calc_P_G(
                Ploop_pred,
                Pct_pred,
                bias,
                f,
                ng,
                km,
                kobs,
                jeff_names,
                poles=False
            )

        return PG
    @staticmethod
    def calc_P_NG(P11, Ploop, Pct, bias, f, stoch, ng, kobs, poles=False):
        '''
        Calculate terms that contain no linearly appearing parameters.

        Args:
            P11 (array) : Array containing P11 kernels.
            Ploop (array) : Arary containing Ploop kernels.
            Pct (array) : Array containing Pct kernels.
            bias (array) : Bias paramters and counterterms.
            f (array) : Growth.
            stoch (array) : Stochastic counterterms.
            ng (float) : Number density of data.
            kobs (array) : k-bins of data being fit. Should have shape ``(nk,)``.
        '''
        
        # Make predictions for nonlinear contributions.
        P_NG_0 = eft_funcs.multipole_vec([P11[0], Ploop[0], Pct[0]], bias, f,
                                        stochastic=stoch, ng=ng, multipole=0,
                                        kbins=kobs)
        P_NG_2 = eft_funcs.multipole_vec([P11[1], Ploop[1], Pct[1]], bias, f,
                                        stochastic=stoch, ng=ng, multipole=2,
                                        kbins=kobs)
        
        # Return single array for both multipoles or seperate arrays for each
        # multipole.
        if poles:
            return P_NG_0, P_NG_2
        else:
            return np.hstack([P_NG_0, P_NG_2])
    @staticmethod
    def calc_P_G(Ploop, Pct, bias, f, ng, km, kobs, marg_names, poles=False):
        '''
        Calculate terms that contain linearly appearing parameters.

        Args:
            Ploop (array) : Arary containing Ploop kernels.
            Pct (array) : Array containing Pct kernels.
            bias (array) : Bias paramters and counterterms.
            f (array) : Growth.
            ng (float) : Number density of data.
            km (float) : Value of km.
            kobs (array) : k-bins of data being fit. Should have shape ``(nk,)``.
            marg_names (list) : Names of parameters to be analytically marginalised
        '''
        
        # Define 'kernels' for the stochastic terms.
        sudo_kernel_ce1 = np.array([np.ones((Ploop.shape[1], kobs.shape[0]))/ng,
                                    np.zeros((Ploop.shape[1], kobs.shape[0]))])
        sudo_kernel_cquad = np.array([np.zeros((Ploop.shape[1], kobs.shape[0])),
                                    np.stack(Ploop.shape[1]*[kobs**2/ng/km**2])])
        sudo_kernel_cmono = np.array([np.stack(Ploop.shape[1]*[kobs**2/ng/km**2]),
                                    np.zeros((Ploop.shape[1], kobs.shape[0]))])

        # Make single array with all possible linear parameter kernels.
        P_G = np.array([Ploop[:,:,3,:]+Ploop[:,:,7]*bias[:,0][None,:,None], #b3
                        (2*f[:,0][None,:,None]*Pct[:,:,3]+2*bias[:,0][None,:,None]*Pct[:,:,0])/km**2, #cct
                        (2*f[:,0][None,:,None]*Pct[:,:,4]+2*bias[:,0][None,:,None]*Pct[:,:,1])/km**2, #cr1
                        (2*f*Pct[:,:,5]+2*bias[:,0][None,:,None]*Pct[:,:,2])/km**2, #cr2
                        sudo_kernel_ce1, #ce1
                        sudo_kernel_cmono, #cmono
                        sudo_kernel_cquad, #cquad
                    ])

        # Find indexes into P_G for selected kernels.
        id_dict = {"b3":0, "cct":1, "cr1":2, "cr2":3, "ce1":4, "cmono":5, "cquad":6}
        kern_select = [id_dict[i] for i in marg_names]
        
        # Return single array for both multipoles or seperate arrays for each
        # multipole.
        if poles:
            return P_G[kern_select,0,:,:], P_G[kern_select,1,:,:]
        else:
            return np.dstack([P_G[kern_select,0,:,:], P_G[kern_select,1,:,:]])