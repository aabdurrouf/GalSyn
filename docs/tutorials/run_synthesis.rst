Generating Synthetic Spectrophotometric Data Cubes
==================================================

With all the preparatory steps completed, we can now initialize the ``GalaxySynthesizer`` class and run the synthesis process. 
We will generate synthetic imaging and IFU data cubes using the following script:

.. code-block:: python

    from galsyn import GalaxySynthesizer

    gs = GalaxySynthesizer(sim_file, z=z, filters=filters,
                           filter_transmission_path=filter_transmission_path)

    gs.ssp_filepath = ssp_grid_fsps
    gs.ssp_interpolation_method = 'nearest'

    gs.dim_kpc = 80
    gs.pix_arcsec = 0.03
    gs.flux_unit = 'erg/s/cm2/A'
    gs.polar_angle_deg = 45
    gs.azimuth_angle_deg = 60
    gs.dust_law = 0                 # modified Calzetti et al. (2000) with dynamic bump amplitude and slope
    gs.dust_eta = 1.0
    gs.dust_index_bc = -0.7

    gs.ncpu = 5                     # number of CPU cores to be used for the calculations
    gs.output_pixel_spectra = True  # output spatially resolved spectra

    # output file name
    gs.name_out_img = 'galsyn_'+str(snap_number)+'_'+str(subhalo_id)+'.fits'

    gs.run_synthesis()
