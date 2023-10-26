===================
Configuration files
===================

The analysis setup for a given run is defined in a configuration ``.yaml`` file.
The form of the file is slightly different if more than one sample is being 
analysed. Examples fo both cases are given below.

-------------
Single sample
-------------

An example configuration file is given below::

    Parameters:
        As:
            prior:
                max: emu
                min: emu
                type: uniform
        b1:
            prior:
                max: 4.0
                min: 0.0
                type: uniform
        b3:
            prior:
                mean: 0.0
                stddev: 2.0
                type: marg
        c2:
            prior:
                max: 4.0
                min: -4.0
                type: uniform
        c4:
            fixed: 0.0
        cct:
            prior:
                mean: 0.0
                stddev: 2.0
                type: marg
        ce1:
            prior:
                mean: 0.0
                stddev: 0.16
                type: marg
        cmono:
            fixed: 0.0
        cquad:
            prior:
                mean: 0.0
                stddev: 2.0
                type: marg
        cr1:
            prior:
                mean: 0.0
                stddev: 8.0
                type: marg
        cr2:
            fixed: 0.0
        h:
            prior:
                max: emu
                min: emu
                type: uniform
        w_b:
            prior:
                max: emu
                mean: 0.02235
                min: emu
                stddev: 0.00049
                type: truncated normal
        w_c:
            prior:
                max: emu
                min: emu
                type: uniform
    Setup:
        bridgez: true
        data_type: pybird
        engine:
            engine_kwargs:
                P110_scaler: log
                P112_scaler: log
                Pct0_scaler: log
                Pct2_scaler: log
                Ploop0_scaler: log
                Ploop2_scaler: log
                path_to_model: /mnt/lustre/jdonaldm/unified_analysis/models/highAs/z1.52/
                version: custom
            function: emu_engine
            path: /users/jdonaldm/UnifiedUtils/unified_utils/engine.py
            type: emulator
        kmax: 0.15
        kmin: 0.01
        likelihood:
            function: marg_like
            path: /users/jdonaldm/UnifiedUtils/unified_utils/likelihood.py
        norm_cov: 50.0
        redshift: 1.52
        sampler:
            nsamples: 50000
            nwalk: 1000
            save_like: true
        save:
            diagnostics: true
            fname: chain.highAs.model1
            fname_tags: true
            overwrite: true
            path: /mnt/lustre/jdonaldm/unified_analysis/results/package_run/
        split: SGC

Reading the above configuration file will result in a nested dictionary.
At the first level we can see there are two keys ``"Parameters"`` and ``"Setup"``.
Each key of the ``"Parameters"`` dictionary should be a parameter of the model.
In this case the EFTofLSS model.
We can see that it is simple to fix parameters to constant values.
We simply use the ``"fixed"`` key.
If a parameter is free we define the type of prior by setting the ``"type"`` key to:

* ``"uniform"``: Results in a uniform prior with extremes defined in the ``"min"`` and ``"max"`` keys.
* ``"normal"``: Results in a normal (Gaussian) prior with standard deviation and mean defined in the ``"stddev"`` and ``"mean"`` keys, respectively.
* ``"truncated normal"``: Results in a truncated noraml prior with hard bounds defined in the ``"min"`` and ``"max"`` keys and standard deviation and mean defined in the ``"stddev"`` and ``"mean"`` keys, respectively.
* ``"marg"``: Results in a normal prior with zero mean and standard deviation defined in the ``"stddev"`` key. The parameter will be analytically marginalised and not sampled. Can only be used for linearly appearing parameters.
* ``"jeff"``: Results in a Jeffreys prior. The prediction engine should be able to produce predictions of the derivaive w.r.t. parameter. Hard bounds can be imposed with the ``"min"`` and ``"max"`` keys.
* ``"marg jeff"``: Results in Jeffreys prior on parameters that will be analytically marginalised. An additional Gaussian prior can be imposed by passing a finite ``"stddev"``.

Each of the keys in the setup dictionary controls a different aspect of the
analysis setup.
If ``conf["Setup"]["save"]["fname_tags"] == True``, the keys and values of the
``"Setup"`` dictionary will be used as tags in the file name for the saved 
posterior samples. We expand on the elements of the ``"Setup"`` dictionary below:

* ``"bridgez"``: If ``True`` then bridge sampling as implamented in ``pocoMC`` will be used to calculate the evidence.
* ``"data_type"``: Specifies what data should be loaded with the ``utils.data_loader function``. If ``obs``, the muipole measurements will be loaded. If ``mock``, the simulation mock multipoles will be loaded. If ``pybird``, the mock multipole produced with ``PyBird`` will be loaded.
* ``"norm_cov"``: Normalisation factor for the covaraince matrix.
* ``"redshift"``: Redshift of the data to load.
* ``"split"``: What hemisphere split to use. For 6dFGS data (``...["redshift"] == 0.096``) this is ignored.
* ``"kmin"`` and ``"kmax"``: Scale cuts to be used.
* ``"engine"``: Information about the prediction engine.

    - ``"engine_kwargs"``: Keyword arguments for the user defined prediction engine. These will *not* be used as file name tags as there can be a lot of them.
    - ``"path"``: Path to ``.py`` file that has the user defined prediction engine.
    - ``function``: Name of prediction engine in the ``.py`` file that has the engine.
    - ``type``: The type of prediction engine.

* ``"likelihood"``: Information about the likelihood function.

    - ``"path"``: Path to ``.py`` file that has the user defined likelihood function.
    - ``function``: Name of prediction engine in the ``.py`` file that has the likelihood function.

* ``"sampler"``: Arguments for the sampler.

    - ``"nsamples"``: The total number of posterior samples to generate.
    - ``"nwalk"``: The number of walker (particles) to use.
    - ``"save_like"``: If ``True``, the likelihood evalution for each posterior sample will be saved.

* ``"save"``: Specifics of saving.

    - ``"diagnostics"``: Saves information that can be useful for identifying problems with sampling.
    - ``"fname"``: The file name for the posterior samples. In the absence of tags this is all that will be used.
    - ``"fname_tags"``: If ``True``, keys and values from the ``"Setup"`` dictionary will be used as file name tags.
    - ``"overwrite"``: If ``True``, existing files will be overwritten.
    - ``"path"``: Path to directory for storing results.


----------------
Multiple samples
----------------

An example configuration file is given below::

    Parameters:
        Shared:
            As:
                prior:
                    max: emu
                    min: emu
                    type: uniform
            h:
                prior:
                    max: emu
                    min: emu
                    type: uniform
            w_b:
                prior:
                    max: emu
                    mean: 0.02235
                    min: emu
                    stddev: 0.00049
                    type: truncated normal
            w_c:
                prior:
                    max: emu
                    min: emu
                    type: uniform
        eBOSS NGC:
            b1:
                prior:
                    max: 4.0
                    min: 0.0
                    type: uniform
            b3:
                prior:
                    mean: 0.0
                    stddev: 2.0
                    type: marg
            c2:
                prior:
                    max: 4.0
                    min: -4.0
                    type: uniform
            cct:
                prior:
                    mean: 0.0
                    stddev: 2.0
                    type: marg
            ce1:
                prior:
                    mean: 0.0
                    stddev: 0.16
                    type: marg
            cquad:
                prior:
                    mean: 0.0
                    stddev: 2.0
                    type: marg
            cr1:
                prior:
                    mean: 0.0
                    stddev: 8.0
                    type: marg
            c4:
                fixed: 0.0
            cr2:
                fixed: 0.0
            cmono:
                fixed: 0.0
        eBOSS SGC:
            b1:
                prior:
                    max: 4.0
                    min: 0.0
                    type: uniform
            b3:
                prior:
                    mean: 0.0
                    stddev: 2.0
                    type: marg
            c2:
                prior:
                    max: 4.0
                    min: -4.0
                    type: uniform
            cct:
                prior:
                    mean: 0.0
                    stddev: 2.0
                    type: marg
            ce1:
                prior:
                    mean: 0.0
                    stddev: 0.16
                    type: marg
            cquad:
                prior:
                    mean: 0.0
                    stddev: 2.0
                    type: marg
            cr1:
                prior:
                    mean: 0.0
                    stddev: 8.0
                    type: marg
            c4:
                fixed: 0.0
            cr2:
                fixed: 0.0
            cmono:
                fixed: 0.0
    Setup:
        bridgez: true
        data_type: obs
        engine:
            engine_kwargs:
                P110_scaler: log
                P112_scaler: log
                Pct0_scaler: log
                Pct2_scaler: log
                Ploop0_scaler: log
                Ploop2_scaler: log
                path_to_model: /mnt/lustre/jdonaldm/unified_analysis/models/highAs/
                version: custom
            function: emu_engine
            path: /users/jdonaldm/UnifiedUtils/unified_utils/engine.py
            type: emulator
        kmaxs:
        - 0.2
        - 0.2
        kmins:
        - 0.01
        - 0.01
        likelihood:
            function: combo_marg_like
            path: /users/jdonaldm/UnifiedUtils/unified_utils/likelihood.py
        norm_cov: 1.0
        redshifts: 
        - 1.52
        - 1.52
        sampler:
            nsamples: 50000
            nwalk: 1000
            save_like: True
        samples: 
        - eBOSS NGC
        - eBOSS SGC
        save:
            diagnostics: true
            fname: chain.eBOSS.highAs.model1
            fname_tags: true
            overwrite: true
            path: /mnt/lustre/jdonaldm/unified_analysis/results/package_run/

Most of the elements of the configuration file are the same as in the single 
sample case. Rather than defining these all again we highligh the differences:

* The ``"Parameters"`` dictionary is now split into seperate dictionaries for the parameters that are shared for all samples and for parameters that are specific to each sample. In the example above we have two samples that we have named ``eBOSS NGC`` and ``eBOSS SGC`` that have unique nuisance parameters but shared cosmological parameters.
* In the ``"Setup"`` dictionary there is a new ``"samples"`` key that takes the form of a list of the user defined names for each sample.
* The scale cuts ``"kmax"`` and ``"kmin"`` for each sample are passed as lists. In the example above they are the same for each sample but that does not need to be the case.
* The redshifts for each sample are defined as a list in ``"redshifts"`` which replaces the single value ``"redshift"``.
