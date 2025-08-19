Generating SSP Grids File
=========================

Here is an example of how to generate an SSP grid using FSPS:

.. code-block:: python

    from galsyn.ssp_generator_fsps import generate_ssp_grid
    import numpy as np

    ssp_grid_fsps = 'ssp_fsps.hdf5'    # output file name

    # Logarithmic grid from 1 Myr to 13.8 Gyr (approx age of universe)
    ages_gyr = np.logspace(np.log10(0.001), np.log10(13.8), 300)

    # Linear grid for log10(Z/Z_sun)
    logzsol_grid = np.linspace(-2.0, 0.2, 300)

    imf_type = 1  # Chabrier (2003) IMF
    n_jobs = 5    # number of CPU cores to be used for the calculations
    generate_ssp_grid(output_filename=ssp_grid_fsps,
                          ages_gyr=ages_gyr,
                          logzsol_grid=logzsol_grid,
                          imf_type=imf_type,
                          overwrite=True,
                          n_jobs=n_jobs,
                          rest_wave_min=500,
                          rest_wave_max=30000)