import numpy as np
import matryoshka.emulator as MatEmu
from scipy.interpolate import interp1d
from matryoshka.training_funcs import LogScaler, UniformScaler
#from classy import Class
#from pybird_dev import pybird

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

class pybird_engine:
    '''
    PyBird engine.

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
        self.kk = np.logspace(-5, 0, 200)

    def predict_kernels(self, cosmo, kobs):
        # PyBird setup
        common = pybird.Common(optiresum=False, Nl=3)
        nonlinear = pybird.NonLinear(load=True, save=True, co=common)
        resum = pybird.Resum(co=common)

        # Class setup
        M = Class()
        M.set({'output': 'mPk',
            'P_k_max_1/Mpc': 1.0,
            'z_max_pk': self.redshift})


        M.set({'ln10^{10}A_s': cosmo[3],
            'n_s': 0.965,
            'h': cosmo[2],
            'omega_b': cosmo[1],
            'omega_cdm': cosmo[0],
            })
        M.compute()

        Pk = [M.pk(ki*M.h(), zi)*M.h()**3 for ki in kk]
            
        f = M.scale_independent_growth_factor_f(self.redshift)
        
        cosmobird = { "k11": kk, "P11": Pk, "f": f, "DA": None, "H": None }
        bird = pybird.Bird(cosmobird, co=common, with_bias=False)
        
        nonlinear.PsCf(bird)
        bird.setPsCfl()
        resum.Ps(bird)

        P11_pred = np.array([bird.P11l[0],
                             bird.P11l[1]])
        P11_pred = interp1d(self.kbins, P11_pred, axis=-1)(kobs)

        Ploop_pred = np.array([bird.Ploopl[0],
                             bird.Ploopl[1]])
        Ploop_pred = interp1d(self.kbins, Ploop_pred, axis=-1)(kobs)

        Pct_pred = np.array([bird.Pctl[0],
                             bird.Pctl[1]])
        Pct_pred = interp1d(self.kbins, Pct_pred, axis=-1)(kobs)
        
        return P11_pred, Ploop_pred, Pct_pred

    #def predict(self, cosmo, bias, stoch, ng, km, kobs=None):
     #   