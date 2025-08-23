Simulating the Observational Effects
====================================


Adding observational effects on synthetic imaging data 
------------------------------------------------------

.. code-block:: python

    # select a set of filters whose images will be further processed by adding observational effects
    filters1 = ['hst_acs_f435w', 'hst_acs_f606w', 'hst_acs_f814w', 'hst_wfc3_ir_f125w', 
                'hst_wfc3_ir_f140w', 'hst_wfc3_ir_f160w', 'jwst_nircam_f090w', 'jwst_nircam_f115w', 
                'jwst_nircam_f150w', 'jwst_nircam_f200w', 'jwst_nircam_f277w', 'jwst_nircam_f356w', 
                'jwst_nircam_f410m', 'jwst_nircam_f444w']
    filter_transmission_path1 = make_filter_transmission_text_pixedfit(filters1, output_dir="filters")

    # pixel size of the PSF images
    psf_pixel_scales = {'hst_acs_f435w': 0.039999999999999584, 'hst_acs_f606w': 0.039999999999999584, 
                'hst_acs_f814w': 0.039999999999999584, 'hst_wfc3_ir_f125w': 0.039999999999999584, 
                'hst_wfc3_ir_f140w': 0.039999999999999584, 'hst_wfc3_ir_f160w': 0.039999999999999584, 
                'jwst_nircam_f090w': 0.019999999999999792, 'jwst_nircam_f115w': 0.019999999999999792, 
                'jwst_nircam_f150w': 0.019999999999999792, 'jwst_nircam_f200w': 0.019999999999999792,  
                'jwst_nircam_f277w': 0.039999999999999584, 'jwst_nircam_f356w': 0.039999999999999584, 
                'jwst_nircam_f410m': 0.039999999999999584, 'jwst_nircam_f444w': 0.039999999999999584}

    # Desired limiting magnitudes to be achieved
    limiting_magnitude = {'hst_acs_f435w': 29.26534652709961, 'hst_acs_f606w': 28.869548797607422,
                    'hst_acs_f814w': 29.109628677368164, 'hst_wfc3_ir_f125w': 28.493282318115234,
                    'hst_wfc3_ir_f140w': 28.17842388153076, 'hst_wfc3_ir_f160w': 28.057647705078125,
                    'jwst_nircam_f090w': 29.730396270751953, 'jwst_nircam_f115w': 30.158058166503906,
                    'jwst_nircam_f150w': 29.93120574951172, 'jwst_nircam_f182m': 29.351133346557617,
                    'jwst_nircam_f200w': 30.098361015319824, 'jwst_nircam_f210m': 29.0757417678833,
                    'jwst_nircam_f277w': 30.89869499206543, 'jwst_nircam_f335m': 30.47265338897705,
                    'jwst_nircam_f356w': 30.837305068969727, 'jwst_nircam_f410m': 30.089418411254883,
                    'jwst_nircam_f444w': 30.196468353271484}

    # Exposure time used in the observations. 
    exposure_time = {'hst_acs_f435w': 68473.340625, 'hst_acs_f606w': 11525.210546875,
                    'hst_acs_f814w': 61992.34609375, 'hst_wfc3_ir_f125w': 18281.091796875,
                    'hst_wfc3_ir_f140w': 6903.4927734375, 'hst_wfc3_ir_f160w': 19381.936328124997,
                    'jwst_nircam_f090w': 11338.0, 'jwst_nircam_f115w': 22676.0,
                    'jwst_nircam_f150w': 11338.0, 'jwst_nircam_f182m': 7171.604003906249,
                    'jwst_nircam_f200w': 11338.0, 'jwst_nircam_f210m': 5375.104101562494,
                    'jwst_nircam_f277w': 11330.7314453125, 'jwst_nircam_f335m': 8503.0,
                    'jwst_nircam_f356w': 11328.221875, 'jwst_nircam_f410m': 8503.0,
                    'jwst_nircam_f444w': 11319.9638671875}

    psf_paths = {} 
    mag_zp = {}
    snr_limit = {}
    aperture_radius_arcsec = {}
    desired_pixel_scales = {}
    for ff in filters1:
        psf_paths[ff] = "PSF_"+ff+".fits"    # path to PSF images
        mag_zp[ff] = 28.1
        snr_limit[ff] = 5.0
        aperture_radius_arcsec[ff] = 0.1
        desired_pixel_scales[ff] = 0.03

    from galsyn import GalSynMockObservation_imaging

    fits_file_path = 'galsyn_39_107965.fits'
    simg = GalSynMockObservation_imaging(fits_file_path, filters1, psf_paths, psf_pixel_scales, mag_zp, 
                                        limiting_magnitude, snr_limit, aperture_radius_arcsec, 
                                        exposure_time, filter_transmission_path1, 
                                        desired_pixel_scales)
    simg.process_images(apply_noise_to_image=True, dust_attenuation=True)

    output_fits_path = 'obsimg_galsyn_39_107965_30mas.fits'
    simg.save_results_to_fits(output_fits_path=output_fits_path)



Adding observational effects on synthetic IFU data 
--------------------------------------------------

In this example, we will simulate mock NIRSpec IFU high-resolution data G140H/F070LP. 
First we need to model the PSF cube for this specific disperser-filter combination, which you can do using the STPSF package. 
The procedures for doing this can be seen at https://stpsf.readthedocs.io/en/latest/jwst_ifu_datacubes.html#Simulating-IFU-mode-and-Datacubes 

.. code-block:: python

    # The PSF FITS file from STPSF package has multiple extensions. We can use DET_DIST extension for our analysis.
    # We need to store the data into a new FITS file that has only one extension.
    hdu = fits.open('psf_cube_G140H_F070LP.fits')
    psf_cube_data = hdu['DET_DIST'].data
    cube_psf_wave_um = np.zeros(psf_cube_data.shape[0])
    for i in range(psf_cube_data.shape[0]):
        cube_psf_wave_um[i] = hdu['det_dist'].header["WVLN%04d" % i]*1e+6
    hdu.close()

    # make a standardized input file for the PSF data cube
    hdul = fits.HDUList()
    hdul.append(fits.ImageHDU(data=psf_cube_data, name='psf_cube'))
    hdul.writeto('psf_G140H_F070LP_standard.fits', overwrite=True)

    from scipy.interpolate import interp1d
    from galsyn import GalSynMockObservation_ifu

    fits_file_path = 'galsyn_39_107965.fits'

    desired_wave_grid = cube_psf_wave_um * 1e+4    # we set the final wavelength grids to be the same as that of PSF data cube
    psf_cube_path = 'psf_G140H_F070LP_standard.fits'

    psf_pixel_scale = 0.1
    spectral_resolution_R = 2700
    mag_zp = 28.0

    # For a simplified example, we will set limiting magnitudes at the edges of the wavelength grid
    # with the shortest wavelength end being slightly less sensitive than the longest wavelegth end.
    limiting_magnitude_wave_func = interp1d([min(desired_wave_grid), max(desired_wave_grid)], 
                                            [28.5, 28.0], fill_value="extrapolate")

    snr_limit = 5.0
    final_pixel_scale_arcsec = 0.1
    exposure_time = 15000

    sifu = GalSynMockObservation_ifu(fits_file_path, desired_wave_grid, psf_cube_path, psf_pixel_scale,
                    spectral_resolution_R, mag_zp, limiting_magnitude_wave_func, snr_limit,
                    final_pixel_scale_arcsec, exposure_time)

    sifu.process_datacube(dust_attenuation=True, apply_noise_to_cube=True)

    output_fits_path = 'obsifu_nirspec_g140h_f070lp_galsyn_39_107965_100mas.fits'
    sifu.save_results_to_fits(output_fits_path)