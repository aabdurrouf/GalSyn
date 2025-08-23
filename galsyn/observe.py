import os
import numpy as np
from astropy.io import fits
from astropy.convolution import convolve_fft, Gaussian1DKernel
from photutils.psf.matching import resize_psf
from scipy.integrate import simpson
from scipy import stats
from scipy.interpolate import interp1d, RectBivariateSpline
from astropy.wcs import WCS
from .imgutils import convert_flux_map

class GalSynMockObservation_imaging:
    """
    A class to simulate observational effects on synthetic galaxy images,
    including PSF convolution, noise injection, RMS image generation,
    and resampling to a desired pixel scale.

    Parameters:
    -----------
    fits_file_path : str
        Path to the input FITS file.
    filters : list
        List of filter names (e.g., ['jwst_nircam_f115w', 'jwst_nircam_f200w']) for which images will be processed.
    psf_paths : dict
        Dictionary where keys are filter names and values are paths to PSF FITS images.
    psf_pixel_scales : dict
        Dictionary where keys are filter names and values are pixel scales of PSF images in arcsec.
    mag_zp : dict
        Dictionary of magnitude zero-points for each filter.
    limiting_magnitude : dict
        Dictionary of limiting magnitudes for each filter.
    snr_limit : dict
        Dictionary of signal-to-noise ratios at the limiting magnitude for each filter.
    aperture_radius_arcsec : dict
        Dictionary of radii for the circular aperture (in arcsec) used in measuring the magnitude limit for each filter.
    exposure_time : dict
        Dictionary of exposure times in seconds for each filter.
    filter_transmission_path : dict
        Dictionary of paths to text files containing the transmission function for filters.
    desired_pixel_scales : dict
        Dictionary where keys are filter names and values are the desired final pixel
        scales in arcsec for the resampled images.

    """

    def __init__(self, fits_file_path, filters, psf_paths, psf_pixel_scales, mag_zp, limiting_magnitude, snr_limit,
                 aperture_radius_arcsec, exposure_time, filter_transmission_path, desired_pixel_scales):
        """
        Initializes the GalSynMockObservation_imaging with input parameters.

        Parameters:
        -----------
        fits_file_path : str
            Path to the input FITS file.
        filters : list
            List of filter names (e.g., ['jwst_nircam_f115w', 'jwst_nircam_f200w']) for which images will be processed.
        psf_paths : dict
            Dictionary where keys are filter names and values are paths to PSF FITS images.
        psf_pixel_scales : dict
            Dictionary where keys are filter names and values are pixel scales of PSF images in arcsec.
        mag_zp : dict
            Dictionary of magnitude zero-points for each filter.
        limiting_magnitude : dict
            Dictionary of limiting magnitudes for each filter.
        snr_limit : dict
            Dictionary of signal-to-noise ratios at the limiting magnitude for each filter.
        aperture_radius_arcsec : dict
            Dictionary of radii for the circular aperture (in arcsec) used in measuring the magnitude limit for each filter.
        exposure_time : dict
            Dictionary of exposure times in seconds for each filter.
        filter_transmission_path : dict
            Dictionary of paths to text files containing the transmission function for filters.
        desired_pixel_scales : dict
            Dictionary where keys are filter names and values are the desired final pixel
            scales in arcsec for the resampled images.
        """
        self.fits_file_path = fits_file_path
        self.filters = filters
        self.psf_paths = psf_paths
        self.psf_pixel_scales = psf_pixel_scales
        self.mag_zp = mag_zp
        self.limiting_magnitude = limiting_magnitude
        self.snr_limit = snr_limit
        self.aperture_radius_arcsec = aperture_radius_arcsec
        self.exposure_time = exposure_time
        self.filter_transmission_path = filter_transmission_path
        self.desired_pixel_scales = desired_pixel_scales

        # --- Validate that all dictionary inputs have keys for all filters ---
        param_dicts = {
            'mag_zp': self.mag_zp,
            'limiting_magnitude': self.limiting_magnitude,
            'snr_limit': self.snr_limit,
            'aperture_radius_arcsec': self.aperture_radius_arcsec,
            'exposure_time': self.exposure_time
        }
        for name, p_dict in param_dicts.items():
            if not isinstance(p_dict, dict) or not all(f in p_dict for f in self.filters):
                raise ValueError(f"Parameter '{name}' must be a dictionary with keys for all specified filters.")

        self.hdul = fits.open(fits_file_path)
        self.image_header = self.hdul[0].header
        self.pixel_scale_kpc = self.image_header['PIX_KPC']
        self.initial_pixel_scale_arcsec = self.image_header['PIXSIZE']
        self.original_flux_unit = self.image_header['BUNIT']

        self.sci_images = {} # Stores final processed images
        self.rms_images = {}      # Stores final RMS images

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.hdul.close()

    def _get_flux_data(self, filter_name, dust_attenuation=True):
        """
        Retrieves flux data for a given filter from the FITS file.
        This function converts the stored flux (in its original `self.original_flux_unit`)
        back to `erg/s/cm^2/Angstrom` for internal calculations.

        IMPORTANT: If the original FITS unit is 'MJy/sr', this function returns
        spectral flux per pixel (i.e., erg/s/cm^2/Angstrom/pixel).
        If the original FITS unit is 'nJy' or 'AB magnitude', this function returns
        spectral flux density per unit wavelength (erg/s/cm^2/Angstrom).
        The interpretation as 'per pixel' or 'per unit area' is crucial for later steps.
        For resampling, the data needs to be explicitly converted to a surface brightness
        unit if it's not already.
        """
        ext_name = f"{'DUST_' if dust_attenuation else 'NODUST_'}{filter_name.upper()}"
        try:
            flux_data_in_fits_unit = self.hdul[ext_name].data

            # Get effective wavelength for this filter
            _, filter_wave_pivot_data = self._load_filter_transmission_from_paths_local(self.filters, self.filter_transmission_path)
            wave_eff = filter_wave_pivot_data[filter_name]

            c_angstrom_s = 2.99792458e18  # speed of light in Angstrom/s

            converted_flux_original = flux_data_in_fits_unit

            if self.original_flux_unit == 'erg/s/cm2/A':
                return converted_flux_original

            elif self.original_flux_unit == 'nJy':
                f_nu_erg_s_cm2_Hz = converted_flux_original / 1e9 * 1e-23
                return f_nu_erg_s_cm2_Hz * c_angstrom_s / wave_eff**2

            elif self.original_flux_unit == 'AB magnitude':
                f_nu_erg_s_cm2_Hz = 10**((converted_flux_original + 48.6)/(-2.5))
                return f_nu_erg_s_cm2_Hz * c_angstrom_s / wave_eff**2

            elif self.original_flux_unit == 'MJy/sr':
                pixel_area_sr = (self.initial_pixel_scale_arcsec * np.pi / (180.0 * 3600.0))**2
                f_nu_jy_per_pixel = converted_flux_original * pixel_area_sr * 1e6
                f_nu_erg_s_cm2_Hz_per_pixel = f_nu_jy_per_pixel * 1e-23
                return f_nu_erg_s_cm2_Hz_per_pixel * c_angstrom_s / wave_eff**2
            else:
                raise ValueError(f"Unsupported original_flux_unit for inverse conversion: {self.original_flux_unit}")

        except KeyError:
            raise ValueError(f"Filter {filter_name} with dust_attenuation={dust_attenuation} not found in FITS file. "
                             f"Available extensions: {[hdu.name for hdu in self.hdul]}.")


    def _get_psf(self, filter_name):
        """
        Loads and resamples the PSF image to match the synthetic image's initial pixel scale.
        """
        psf_path = self.psf_paths.get(filter_name)
        if psf_path is None:
            raise ValueError(f"PSF path not provided for filter: {filter_name}")

        if not os.path.exists(psf_path):
            raise FileNotFoundError(f"PSF file not found at: {psf_path}")

        psf_hdu = fits.open(psf_path)
        psf_data = psf_hdu[0].data
        psf_hdu.close()

        psf_pixel_scale_arcsec = self.psf_pixel_scales.get(filter_name)
        if psf_pixel_scale_arcsec is None:
            raise ValueError(f"PSF pixel scale not provided for filter: {filter_name}")

        if not np.isclose(psf_pixel_scale_arcsec, self.initial_pixel_scale_arcsec):
            print(f"Resampling PSF for {filter_name}: PSF pixel scale ({psf_pixel_scale_arcsec:.4f} arcsec) "
                  f"differs from initial image pixel scale ({self.initial_pixel_scale_arcsec:.4f} arcsec).")

            resampled_psf = resize_psf(psf_data, psf_pixel_scale_arcsec, self.initial_pixel_scale_arcsec)
            resampled_psf /= np.sum(resampled_psf)
            return resampled_psf
        else:
            psf_data /= np.sum(psf_data)
            return psf_data

    def _load_filter_transmission_from_paths_local(self, filters_list, filter_transmission_path_dict):
        """
        Helper function to load filter transmission data and calculate pivot wavelengths.
        """
        filter_transmission_data = {}
        filter_wave_pivot_data = {}

        for f_name in filters_list:
            file_path = filter_transmission_path_dict.get(f_name)
            if file_path is None:
                raise ValueError(f"Filter transmission path not found for {f_name}.")

            data = np.loadtxt(file_path)
            wave = data[:, 0]
            trans = data[:, 1]

            filter_transmission_data[f_name] = {'wave': wave, 'trans': trans}

            numerator = simpson(wave * trans, wave)
            denominator = simpson(trans / np.where(wave != 0, wave, 1e-10), wave)
            pivot_wavelength = np.sqrt(numerator / denominator) if denominator > 0 else np.nan
            filter_wave_pivot_data[f_name] = pivot_wavelength

        return filter_transmission_data, filter_wave_pivot_data

    def process_images(self, dust_attenuation=None, apply_noise_to_image=True):
        """
        Executes the full pipeline of observational effects for images,
        including PSF convolution, noise injection, and resampling using bicubic interpolation.
        """
        print("\nStarting full image processing pipeline...")
        
        process_types = [True, False] if dust_attenuation is None else [dust_attenuation]
        
        for current_dust_attenuation in process_types:
            ext_name_check = f"{'DUST_' if current_dust_attenuation else 'NODUST_'}{self.filters[0].upper()}"
            if ext_name_check not in self.hdul:
                print(f"Skipping processing for dust_attenuation={current_dust_attenuation} as data is not available.")
                continue

            for f_name in self.filters:
                print(f"\nProcessing filter: {f_name} with dust_attenuation={current_dust_attenuation}")

                # --- Get filter-specific parameters ---
                mag_zp = self.mag_zp[f_name]
                limiting_magnitude = self.limiting_magnitude[f_name]
                snr_limit = self.snr_limit[f_name]
                aperture_radius_arcsec = self.aperture_radius_arcsec[f_name]
                exposure_time = self.exposure_time[f_name]
                desired_pixel_scale = self.desired_pixel_scales[f_name]
                pixel_area_arcsec2_initial = self.initial_pixel_scale_arcsec**2

                # --- 1. Get Initial Image Data and convert to flux per pixel ---
                initial_flux_per_pixel = self._get_flux_data(f_name, dust_attenuation=current_dust_attenuation)  ## in units of erg/s/cm^2/A

                # --- 2. PSF Convolution ---
                print(f"  Convolving {f_name} image with PSF...")
                psf_data = self._get_psf(f_name)
                convolved_flux_per_pixel = convolve_fft(initial_flux_per_pixel, psf_data, boundary='fill', fill_value=0.0)
                print(f"  {f_name} image convolved.")

                # --- 3. Noise Simulation and Injection ---
                print(f"  Simulating noise for {f_name} image...")
                _, filter_wave_pivot_data = self._load_filter_transmission_from_paths_local(self.filters, self.filter_transmission_path)
                wave_eff = filter_wave_pivot_data[f_name]

                C_aperture = exposure_time * (10**(0.4 * (mag_zp - limiting_magnitude)))
                sigma_bg_aperture_sq = (C_aperture / snr_limit)**2 - C_aperture
                sigma_bg_aperture_sq = np.maximum(0, sigma_bg_aperture_sq)
                sigma_bg_aperture = np.sqrt(sigma_bg_aperture_sq)
                aperture_area_pix2_for_bg = np.pi * (aperture_radius_arcsec / self.initial_pixel_scale_arcsec)**2
                if aperture_area_pix2_for_bg <= 0:
                    raise ValueError("Aperture area per pixel must be positive.")
                sigma_bg_per_pixel = sigma_bg_aperture / np.sqrt(aperture_area_pix2_for_bg)

                c_angstrom_s = 2.99792458e18
                f_nu_erg_s_cm2_Hz_pixel = convolved_flux_per_pixel * wave_eff**2 / c_angstrom_s
                pixel_mag_AB = -2.5 * np.log10(np.clip(f_nu_erg_s_cm2_Hz_pixel, 1e-50, None)) - 48.6
                source_counts_per_pixel_expected = exposure_time * (10**(0.4 * (mag_zp - pixel_mag_AB)))
                source_counts_per_pixel_expected = np.maximum(0, source_counts_per_pixel_expected)

                mag_for_1_count = mag_zp - 2.5 * np.log10(1.0 / exposure_time)
                f_nu_erg_s_cm2_Hz_for_1_count = 10**((mag_for_1_count + 48.6)/(-2.5))
                flux_per_total_count_per_A_per_pixel = f_nu_erg_s_cm2_Hz_for_1_count * c_angstrom_s / wave_eff**2

                noisy_image_flux_per_initial_pixel = convolved_flux_per_pixel.copy()
                if apply_noise_to_image:
                    photon_shot_noise_sampled_counts = stats.poisson.rvs(source_counts_per_pixel_expected)
                    total_noisy_counts = photon_shot_noise_sampled_counts + np.random.normal(0, sigma_bg_per_pixel, size=convolved_flux_per_pixel.shape)
                    #total_noisy_counts = np.maximum(0, total_noisy_counts)
                    noisy_image_flux_per_initial_pixel = total_noisy_counts * flux_per_total_count_per_A_per_pixel
                    #noisy_image_flux_per_initial_pixel = np.maximum(1e-30, noisy_image_flux_per_initial_pixel)

                total_variance_counts = source_counts_per_pixel_expected + sigma_bg_per_pixel**2
                total_rms_counts_per_pixel = np.sqrt(total_variance_counts)
                rms_image_flux_per_initial_pixel = total_rms_counts_per_pixel * flux_per_total_count_per_A_per_pixel
                print(f"  Noise simulated and injected for {f_name}.")

                # --- 4. Resampling (Bicubic Interpolation) ---
                print(f"  Resampling {f_name} image to desired pixel scale...")
                if desired_pixel_scale is None:
                    raise ValueError(f"Desired pixel scale not provided for filter: {f_name}.")
                
                if np.isclose(desired_pixel_scale, self.initial_pixel_scale_arcsec):
                    print(f"  Desired pixel scale for {f_name} is already {desired_pixel_scale:.4f} arcsec. No resampling needed.")
                    final_processed_data_erg = noisy_image_flux_per_initial_pixel
                    final_rms_data_erg = rms_image_flux_per_initial_pixel
                else:
                    print(f"  Resampling from {self.initial_pixel_scale_arcsec:.4f} arcsec to {desired_pixel_scale:.4f} arcsec.")
                    
                    # Convert to surface brightness for resampling
                    initial_surface_brightness = noisy_image_flux_per_initial_pixel / pixel_area_arcsec2_initial
                    initial_rms_surface_brightness = rms_image_flux_per_initial_pixel / pixel_area_arcsec2_initial
                    
                    old_ny, old_nx = initial_surface_brightness.shape
                    new_ny = int(np.round(old_ny * self.initial_pixel_scale_arcsec / desired_pixel_scale))
                    new_nx = int(np.round(old_nx * self.initial_pixel_scale_arcsec / desired_pixel_scale))

                    y_old = np.linspace(0, old_ny-1, old_ny)
                    x_old = np.linspace(0, old_nx-1, old_nx)
                    y_new = np.linspace(0, old_ny-1, new_ny)
                    x_new = np.linspace(0, old_nx-1, new_nx)

                    # Bicubic interpolation on surface brightness data
                    interp_func_sci = RectBivariateSpline(y_old, x_old, initial_surface_brightness)
                    resampled_sci_sb = interp_func_sci(y_new, x_new)
                    
                    interp_func_rms = RectBivariateSpline(y_old, x_old, initial_rms_surface_brightness)
                    resampled_rms_sb = interp_func_rms(y_new, x_new)

                    # Convert resampled surface brightness back to flux per pixel
                    pixel_area_arcsec2_final = desired_pixel_scale**2
                    final_processed_data_erg = resampled_sci_sb * pixel_area_arcsec2_final
                    final_rms_data_erg = resampled_rms_sb * pixel_area_arcsec2_final

                # --- 5. Convert back to original units ---
                _, filter_wave_pivot_data = self._load_filter_transmission_from_paths_local(self.filters, self.filter_transmission_path)
                wave_eff = filter_wave_pivot_data[f_name]
                
                final_processed_data = convert_flux_map(
                    final_processed_data_erg, 
                    wave_eff, 
                    to_unit=self.original_flux_unit,
                    pixel_scale_arcsec=desired_pixel_scale
                )

                final_rms_data = convert_flux_map(
                    final_rms_data_erg, 
                    wave_eff, 
                    to_unit=self.original_flux_unit,
                    pixel_scale_arcsec=desired_pixel_scale
                )
                print(f"  {f_name} image converted back to original units ({self.original_flux_unit}).")


                key_processed = f"{f_name}_{'dust' if current_dust_attenuation else 'nodust'}"
                self.sci_images[key_processed] = final_processed_data
                key_rms = f"{f_name}_{'dust' if current_dust_attenuation else 'nodust'}_rms"
                self.rms_images[key_rms] = final_rms_data
                print(f"  {f_name} image resampled and stored.")

        print("\nFull image processing pipeline complete.")


    def save_results_to_fits(self, output_fits_path):
        """
        Saves all processed (noise-injected and resampled) images and RMS images
        to a new FITS file, maintaining the original flux units.
        """
        print(f"\nSaving results to FITS file: {output_fits_path}...")
        hdul_out = fits.HDUList()
        prihdr = fits.Header()
        prihdr['COMMENT'] = 'Mock Observation Results'
        prihdr['BUNIT'] = self.original_flux_unit
        hdul_out.append(fits.PrimaryHDU(header=prihdr))

        for key, img_data in self.sci_images.items():
            ext_hdr = fits.Header()
            parts = key.rsplit('_', 1)
            filter_name = parts[0]
            attenuation_type = parts[1]
            ext_name = f"SCI_{attenuation_type.upper()}_{filter_name.upper()}"
            ext_hdr['EXTNAME'] = ext_name
            ext_hdr['COMMENT'] = f'Convolved, noise-injected, and resampled image for: {key}'
            ext_hdr['BUNIT'] = self.original_flux_unit
            hdul_out.append(fits.ImageHDU(data=img_data, header=ext_hdr))

        for key, rms_data in self.rms_images.items():
            ext_hdr = fits.Header()
            base_key = key.rsplit('_', 1)[0]
            parts = base_key.rsplit('_', 1)
            filter_name = parts[0]
            attenuation_type = parts[1]
            ext_name = f"RMS_{attenuation_type.upper()}_{filter_name.upper()}"
            ext_hdr['EXTNAME'] = ext_name
            ext_hdr['COMMENT'] = f'RMS image for: {key}'
            ext_hdr['BUNIT'] = self.original_flux_unit
            hdul_out.append(fits.ImageHDU(data=rms_data, header=ext_hdr))

        output_dir = os.path.dirname(output_fits_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        hdul_out.writeto(output_fits_path, overwrite=True, output_verify='fix')
        hdul_out.close()
        print(f"Results saved to {output_fits_path}")


class GalSynMockObservation_ifu:
    """
    A class to simulate observational effects on synthetic IFU data cubes,
    including wavelength cutting, spectral smoothing, spatial PSF convolution,
    noise injection, and RMS data cube generation.

    Parameters:
    -----------
    fits_file_path : str
        Path to the FITS file output from galsyn_run_fsps (containing OBS_SPEC_NODUST/DUST and WAVELENGTH_GRID).
    desired_wave_grid : np.ndarray
        1D numpy array of the desired final wavelength grid (in Angstrom) for the data cube.
    psf_cube_path : str
        Path to the FITS file containing the 3D PSF data cube (wavelength, y, x).
        The wavelength axis of the PSF cube must match `desired_wave_grid`.
    psf_pixel_scale : float
        Pixel scale of the PSF cube in arcsec.
    spectral_resolution_R : float
        Desired constant spectral resolution (R = lambda / d_lambda) for the output cube.
    mag_zp : float or callable
        Magnitude zero-point. Can be a constant float or a function of wavelength (in Angstrom).
    limiting_magnitude_wave_func : callable
        A function that takes wavelength (in Angstrom) as input and returns
        the limiting magnitude at that wavelength. This magnitude is assumed
        to be per area of the `final_pixel_scale_arcsec`.
    snr_limit : float or callable
        Signal-to-noise ratio corresponding to the limiting magnitude. Can be a constant float or a function of wavelength.
    final_pixel_scale_arcsec : float
        Desired final spatial pixel size in arcsec for the output data cube.
    exposure_time : float or callable
        Exposure time in seconds. Can be a constant float or a function of wavelength.

    """

    def __init__(self, fits_file_path, desired_wave_grid, psf_cube_path, psf_pixel_scale,
                 spectral_resolution_R, mag_zp, limiting_magnitude_wave_func, snr_limit,
                 final_pixel_scale_arcsec, exposure_time):
        """
        Initializes the GalSynMockObservation_ifu with input parameters.

        Parameters:
        -----------
        fits_file_path : str
            Path to the FITS file output from galsyn_run_fsps (containing OBS_SPEC_NODUST/DUST and WAVELENGTH_GRID).
        desired_wave_grid : np.ndarray
            1D numpy array of the desired final wavelength grid (in Angstrom) for the data cube.
        psf_cube_path : str
            Path to the FITS file containing the 3D PSF data cube (wavelength, y, x).
            The wavelength axis of the PSF cube must match `desired_wave_grid`.
        psf_pixel_scale : float
            Pixel scale of the PSF cube in arcsec.
        spectral_resolution_R : float
            Desired constant spectral resolution (R = lambda / d_lambda) for the output cube.
        mag_zp : float or callable
            Magnitude zero-point. Can be a constant float or a function of wavelength (in Angstrom).
        limiting_magnitude_wave_func : callable
            A function that takes wavelength (in Angstrom) as input and returns
            the limiting magnitude at that wavelength. This magnitude is assumed
            to be per area of the `final_pixel_scale_arcsec`.
        snr_limit : float or callable
            Signal-to-noise ratio corresponding to the limiting magnitude. Can be a constant float or a function of wavelength.
        final_pixel_scale_arcsec : float
            Desired final spatial pixel size in arcsec for the output data cube.
        exposure_time : float or callable
            Exposure time in seconds. Can be a constant float or a function of wavelength.
        """
        self.fits_file_path = fits_file_path
        self.desired_wave_grid = desired_wave_grid
        self.psf_cube_path = psf_cube_path
        self.psf_pixel_scale = psf_pixel_scale
        self.spectral_resolution_R = spectral_resolution_R
        self.mag_zp = mag_zp
        self.limiting_magnitude_wave_func = limiting_magnitude_wave_func
        self.snr_limit = snr_limit
        self.final_pixel_scale_arcsec = final_pixel_scale_arcsec
        self.exposure_time = exposure_time

        self.hdul = fits.open(fits_file_path)
        self.image_header = self.hdul[0].header
        self.initial_pixel_scale_arcsec = self.image_header.get('PIXSIZE')
        if self.initial_pixel_scale_arcsec is None:
            raise ValueError("Input FITS file header must contain 'PIXSIZE' (initial pixel scale in arcsec).")
        self.original_flux_unit = self.image_header.get('BUNIT', 'erg/s/cm2/A')
        
        try:
            wavelength_hdu = self.hdul['WAVELENGTH_GRID']
            self.original_wave_grid = wavelength_hdu.data['WAVELENGTH']
        except KeyError:
            raise ValueError("Input FITS file must contain a 'WAVELENGTH_GRID' binary table extension.")

        self.initial_datacube_nodust = self.hdul['OBS_SPEC_NODUST'].data if 'OBS_SPEC_NODUST' in self.hdul else None
        self.initial_datacube_dust = self.hdul['OBS_SPEC_DUST'].data if 'OBS_SPEC_DUST' in self.hdul else None

        self.sci_datacubes = {}
        self.rms_datacubes = {}

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.hdul.close()

    def _get_wave_dependent_param(self, param, wavelength):
        """Helper to return value from a parameter that can be a float or a callable."""
        if callable(param):
            return param(wavelength)
        return param

    def _load_psf_cube(self):
        """
        Loads the 3D PSF cube and ensures its wavelength axis matches desired_wave_grid.
        Also resamples each 2D PSF slice to the initial image pixel scale.
        Returns a 3D PSF cube (wavelength, y, x) ready for convolution.
        """
        if not os.path.exists(self.psf_cube_path):
            raise FileNotFoundError(f"PSF cube file not found at: {self.psf_cube_path}")

        with fits.open(self.psf_cube_path) as psf_hdu:
            psf_cube_data = psf_hdu[0].data

        if psf_cube_data.ndim != 3:
            raise ValueError(f"PSF cube must be 3D (wavelength, y, x), but has {psf_cube_data.ndim} dimensions.")

        if psf_cube_data.shape[0] != len(self.desired_wave_grid):
            raise ValueError(f"PSF cube wavelength channels ({psf_cube_data.shape[0]}) "
                             f"must match desired_wave_grid length ({len(self.desired_wave_grid)}).")

        new_y = int(np.round(psf_cube_data.shape[1] * self.psf_pixel_scale / self.initial_pixel_scale_arcsec))
        new_x = int(np.round(psf_cube_data.shape[2] * self.psf_pixel_scale / self.initial_pixel_scale_arcsec))
        resampled_psf_cube = np.zeros((psf_cube_data.shape[0], new_y, new_x))

        for i_wave in range(psf_cube_data.shape[0]):
            psf_slice = psf_cube_data[i_wave, :, :]
            if not np.isclose(self.psf_pixel_scale, self.initial_pixel_scale_arcsec):
                resampled_psf = resize_psf(psf_slice, self.psf_pixel_scale, self.initial_pixel_scale_arcsec)
            else:
                resampled_psf = psf_slice
            
            resampled_psf /= np.sum(resampled_psf)
            resampled_psf_cube[i_wave, :, :] = resampled_psf

        return resampled_psf_cube

    def _get_flux_data_ifufunc(self, input_datacube, dust_attenuation=True):
        """
        Retrieves flux data from the IFU FITS file and converts it to
        erg/s/cm^2/A for internal calculations.

        Parameters:
        -----------
        input_datacube : np.ndarray
            The input data cube to process.
        dust_attenuation : bool
            Indicates whether to process the dust-attenuated or non-attenuated cube.

        Returns:
        --------
        np.ndarray
            The data cube in units of erg/s/cm^2/A.
        """
        if self.original_flux_unit == 'erg/s/cm2/A':
            return input_datacube

        elif self.original_flux_unit == 'nJy/sr':
            # Assuming nJy/sr is the unit for IFU cubes in galsyn
            # Convert nJy/sr to erg/s/cm2/A/sr
            c_angstrom_s = 2.99792458e18
            output_cube = np.zeros_like(input_datacube)
            for i_wave, wave_eff in enumerate(self.original_wave_grid):
                # Convert to f_nu in erg/s/cm^2/Hz/sr
                f_nu_erg_s_cm2_Hz_sr = input_datacube[i_wave, :, :] / 1e9 * 1e-23
                # Convert to f_lambda in erg/s/cm^2/A/sr
                output_cube[i_wave, :, :] = f_nu_erg_s_cm2_Hz_sr * c_angstrom_s / wave_eff**2
            return output_cube

        else:
            raise ValueError(f"Unsupported original_flux_unit for IFU cube: {self.original_flux_unit}")

    def process_datacube(self, dust_attenuation=None, apply_noise_to_cube=True):
        """
        Executes the full pipeline of observational effects for the IFU data cube,
        using bicubic interpolation for spatial resampling.
        """
        print("\nStarting full IFU data cube processing pipeline...")

        process_types = [True, False] if dust_attenuation is None else [dust_attenuation]
        
        for current_dust_attenuation in process_types:
            input_datacube_raw_units = self.initial_datacube_dust if current_dust_attenuation and self.initial_datacube_dust is not None else self.initial_datacube_nodust
            
            if input_datacube_raw_units is None:
                print(f"Skipping IFU processing for dust_attenuation={current_dust_attenuation} as data is not available.")
                continue

            print(f"Processing IFU data for dust_attenuation={current_dust_attenuation}")
            print(f"  Initial data cube shape: {input_datacube_raw_units.shape} (Angstrom, Y, X)")
            
            # --- Convert to erg/s/cm^2/A (internal unit) ---
            print("  Converting to internal units of erg/s/cm^2/A...")
            input_datacube_erg = self._get_flux_data_ifufunc(input_datacube_raw_units, dust_attenuation=current_dust_attenuation)


            # --- 1. Cut/Interpolate Wavelength Grid ---
            print("  Cutting/interpolating data cube to desired wavelength grid...")
            reshaped_cube = input_datacube_erg.reshape(input_datacube_erg.shape[0], -1).T
            interpolated_cube_reshaped = np.zeros((reshaped_cube.shape[0], len(self.desired_wave_grid)))
            for i in range(reshaped_cube.shape[0]):
                interp_func = interp1d(self.original_wave_grid, reshaped_cube[i, :], kind='linear', bounds_error=False, fill_value=0.0)
                interpolated_cube_reshaped[i, :] = interp_func(self.desired_wave_grid)
            processed_cube_wave_cut = interpolated_cube_reshaped.T.reshape(len(self.desired_wave_grid), input_datacube_erg.shape[1], input_datacube_erg.shape[2])
            print(f"  Data cube cut to wavelength grid. New shape: {processed_cube_wave_cut.shape}")

            # --- 2. Spectral Smoothing ---
            print(f"  Smoothing spectra to R={self.spectral_resolution_R}...")
            smoothed_cube_flux = np.zeros_like(processed_cube_wave_cut)
            if len(self.desired_wave_grid) > 1:
                delta_lambda_per_pixel = np.mean(np.diff(self.desired_wave_grid))
                sigma_lambda_per_channel = self.desired_wave_grid / self.spectral_resolution_R / (2 * np.sqrt(2 * np.log(2)))
                mean_sigma_pixels = np.mean(sigma_lambda_per_channel / delta_lambda_per_pixel)

                if mean_sigma_pixels > 0:
                    gauss_kernel = Gaussian1DKernel(stddev=mean_sigma_pixels)
                    for i_y in range(processed_cube_wave_cut.shape[1]):
                        for i_x in range(processed_cube_wave_cut.shape[2]):
                            smoothed_cube_flux[:, i_y, i_x] = convolve_fft(processed_cube_wave_cut[:, i_y, i_x], gauss_kernel, boundary='fill', fill_value=0.0)
                else:
                    smoothed_cube_flux = processed_cube_wave_cut
            else:
                 smoothed_cube_flux = processed_cube_wave_cut
            print("  Spectra smoothed.")

            # --- 3. Spatial PSF Convolution ---
            print("  Convolving each spatial slice with PSF...")
            psf_cube = self._load_psf_cube()
            convolved_cube_flux = np.zeros_like(smoothed_cube_flux)
            for i_wave in range(smoothed_cube_flux.shape[0]):
                convolved_cube_flux[i_wave, :, :] = convolve_fft(smoothed_cube_flux[i_wave, :, :], psf_cube[i_wave, :, :], boundary='fill', fill_value=0.0)
            print("  Spatial PSF convolution complete.")

            # --- 4. Noise Simulation and Injection ---
            print("  Simulating and injecting noise...")
            noisy_cube_flux = convolved_cube_flux.copy()
            rms_cube_flux = np.zeros_like(convolved_cube_flux)
            c_angstrom_s = 2.99792458e18
            pixel_area_arcsec2_initial = self.initial_pixel_scale_arcsec**2

            for i_wave in range(convolved_cube_flux.shape[0]):
                current_wave = self.desired_wave_grid[i_wave]
                mag_zp = self._get_wave_dependent_param(self.mag_zp, current_wave)
                lim_mag_at_wave = self.limiting_magnitude_wave_func(current_wave)
                snr_limit = self._get_wave_dependent_param(self.snr_limit, current_wave)
                exposure_time = self._get_wave_dependent_param(self.exposure_time, current_wave)
                
                current_slice_flux_per_initial_pixel = convolved_cube_flux[i_wave, :, :]
                
                # --- Convert to surface brightness before resampling for consistent calculations ---
                current_slice_surface_brightness = current_slice_flux_per_initial_pixel / pixel_area_arcsec2_initial
                
                # We need to calculate noise based on a "per-pixel" basis for the original pixel scale
                # to do this, we need to convert the surface brightness back to flux per initial pixel
                # to get the expected counts.
                f_nu_erg_s_cm2_Hz_pixel = current_slice_flux_per_initial_pixel * current_wave**2 / c_angstrom_s
                pixel_mag_AB = -2.5 * np.log10(np.clip(f_nu_erg_s_cm2_Hz_pixel, 1e-50, None)) - 48.6
                source_counts_per_pixel_expected = exposure_time * (10**(0.4 * (mag_zp - pixel_mag_AB)))
                source_counts_per_pixel_expected = np.maximum(0, source_counts_per_pixel_expected)

                C_aperture_at_wave = exposure_time * (10**(0.4 * (mag_zp - lim_mag_at_wave)))
                noise_scaling_factor = (self.final_pixel_scale_arcsec / self.initial_pixel_scale_arcsec)**2
                sigma_bg_counts_sq_per_final_pixel = (C_aperture_at_wave / snr_limit)**2 - C_aperture_at_wave
                sigma_bg_counts_sq_per_final_pixel = np.maximum(0, sigma_bg_counts_sq_per_final_pixel)
                sigma_bg_counts_sq_per_initial_pixel = sigma_bg_counts_sq_per_final_pixel / noise_scaling_factor
                sigma_bg_counts_per_initial_pixel = np.sqrt(sigma_bg_counts_sq_per_initial_pixel)

                mag_for_1_count = mag_zp - 2.5 * np.log10(1.0 / exposure_time)
                f_nu_erg_s_cm2_Hz_for_1_count = 10**((mag_for_1_count + 48.6)/(-2.5))
                flux_per_total_count_per_A_per_pixel = f_nu_erg_s_cm2_Hz_for_1_count * c_angstrom_s / current_wave**2

                if apply_noise_to_cube:
                    photon_shot_noise = stats.poisson.rvs(source_counts_per_pixel_expected)
                    background_noise = np.random.normal(0, sigma_bg_counts_per_initial_pixel, size=source_counts_per_pixel_expected.shape)
                    total_noisy_counts = photon_shot_noise + background_noise
                    #total_noisy_counts = np.maximum(0, photon_shot_noise + background_noise)
                    noisy_cube_flux[i_wave, :, :] = total_noisy_counts * flux_per_total_count_per_A_per_pixel
                    #noisy_cube_flux[i_wave, :, :] = np.maximum(1e-30, total_noisy_counts * flux_per_total_count_per_A_per_pixel)
                else:
                    noisy_cube_flux[i_wave, :, :] = current_slice_flux_per_initial_pixel

                total_variance_counts_slice = source_counts_per_pixel_expected + sigma_bg_counts_sq_per_initial_pixel
                total_rms_counts_per_pixel_slice = np.sqrt(total_variance_counts_slice)
                rms_cube_flux[i_wave, :, :] = np.maximum(1e-30, total_rms_counts_per_pixel_slice * flux_per_total_count_per_A_per_pixel)
            print("  Noise simulated and injected.")

            # --- 5. Spatial Resampling (Bicubic Interpolation) ---
            print(f"  Resampling data cube spatially to {self.final_pixel_scale_arcsec:.4f} arcsec...")
            
            pixel_area_arcsec2_initial = self.initial_pixel_scale_arcsec**2
            pixel_area_arcsec2_final = self.final_pixel_scale_arcsec**2

            if np.isclose(self.final_pixel_scale_arcsec, self.initial_pixel_scale_arcsec):
                resampled_processed_cube_flux_erg = noisy_cube_flux
                resampled_rms_cube_flux_erg = rms_cube_flux
            else:
                old_ny, old_nx = noisy_cube_flux.shape[1:]
                new_ny = int(np.round(old_ny * self.initial_pixel_scale_arcsec / self.final_pixel_scale_arcsec))
                new_nx = int(np.round(old_nx * self.initial_pixel_scale_arcsec / self.final_pixel_scale_arcsec))

                resampled_processed_cube_flux_erg = np.zeros((noisy_cube_flux.shape[0], new_ny, new_nx))
                resampled_rms_cube_flux_erg = np.zeros((rms_cube_flux.shape[0], new_ny, new_nx))
                
                y_old = np.linspace(0, old_ny-1, old_ny)
                x_old = np.linspace(0, old_nx-1, old_nx)
                y_new = np.linspace(0, old_ny-1, new_ny)
                x_new = np.linspace(0, old_nx-1, new_nx)

                # Resample each slice independently
                for i_wave in range(noisy_cube_flux.shape[0]):
                    # Convert to surface brightness for interpolation
                    initial_sb_slice = noisy_cube_flux[i_wave, :, :] / pixel_area_arcsec2_initial
                    initial_rms_sb_slice = rms_cube_flux[i_wave, :, :] / pixel_area_arcsec2_initial

                    interp_func_sci = RectBivariateSpline(y_old, x_old, initial_sb_slice)
                    resampled_sci_sb = interp_func_sci(y_new, x_new)
                    
                    interp_func_rms = RectBivariateSpline(y_old, x_old, initial_rms_sb_slice)
                    resampled_rms_sb = interp_func_rms(y_new, x_new)

                    # Convert back to flux per pixel
                    resampled_processed_cube_flux_erg[i_wave, :, :] = resampled_sci_sb * pixel_area_arcsec2_final
                    resampled_rms_cube_flux_erg[i_wave, :, :] = resampled_rms_sb * pixel_area_arcsec2_final
            
            print("  Spatial resampling complete.")
            
            # --- 6. Convert back to original units ---
            print(f"  Converting final data cubes back to original units ({self.original_flux_unit})...")
            final_processed_cube_raw_units = np.zeros_like(resampled_processed_cube_flux_erg)
            final_rms_cube_raw_units = np.zeros_like(resampled_rms_cube_flux_erg)

            for i_wave, wave in enumerate(self.desired_wave_grid):
                final_processed_cube_raw_units[i_wave, :, :] = convert_flux_map(
                    resampled_processed_cube_flux_erg[i_wave, :, :],
                    wave,
                    to_unit=self.original_flux_unit,
                    pixel_scale_arcsec=self.final_pixel_scale_arcsec
                )
                final_rms_cube_raw_units[i_wave, :, :] = convert_flux_map(
                    resampled_rms_cube_flux_erg[i_wave, :, :],
                    wave,
                    to_unit=self.original_flux_unit,
                    pixel_scale_arcsec=self.final_pixel_scale_arcsec
                )

            key = 'dust' if current_dust_attenuation else 'nodust'
            self.sci_datacubes[key] = final_processed_cube_raw_units
            self.rms_datacubes[key] = final_rms_cube_raw_units
        print("\nFull IFU data cube processing pipeline complete.")

    def save_results_to_fits(self, output_fits_path):
        """
        Saves all processed data cubes and the RMS data cubes
        from `self.sci_datacubes` and `self.rms_datacubes` to a new FITS file.
        The data is saved in flux per pixel units (erg/s/cm2/A/pixel).
        """
        print(f"\nSaving IFU results to FITS file: {output_fits_path}...")
        hdul_out = fits.HDUList()
        #prihdr = self.image_header.copy()
        prihdr = fits.Header()
        prihdr['BUNIT'] = self.original_flux_unit
        prihdr['COMMENT'] = 'Mock IFU Observation Results'
        prihdr['RES_R'] = self.spectral_resolution_R
        prihdr['PIXSIZE'] = self.final_pixel_scale_arcsec
        
        central_wave = self.desired_wave_grid[len(self.desired_wave_grid) // 2]
        prihdr['ZP_MAG'] = (self._get_wave_dependent_param(self.mag_zp, central_wave), 'ZP at central wavelength')
        prihdr['SNR_LIM'] = (self._get_wave_dependent_param(self.snr_limit, central_wave), 'SNR at central wavelength')
        prihdr['EXP_TIME'] = (self._get_wave_dependent_param(self.exposure_time, central_wave), 'Exposure time (s) at central wavelength')
        prihdr['BUNIT'] = self.original_flux_unit

        hdul_out.append(fits.PrimaryHDU(header=prihdr))

        for key, processed_cube in self.sci_datacubes.items():
            if processed_cube is not None:
                extname = f'SCI_{key.upper()}'
                ext_hdr_proc = self._create_ifu_header(extname, processed_cube.shape, prihdr['BUNIT'])
                hdul_out.append(fits.ImageHDU(data=processed_cube, header=ext_hdr_proc))

        for key, rms_cube in self.rms_datacubes.items():
            if rms_cube is not None:
                extname = f'RMS_{key.upper()}'
                ext_hdr_rms = self._create_ifu_header(extname, rms_cube.shape, prihdr['BUNIT'])
                hdul_out.append(fits.ImageHDU(data=rms_cube, header=ext_hdr_rms))

        if len(self.desired_wave_grid) > 0:
            col = fits.Column(name='WAVELENGTH', format='D', array=self.desired_wave_grid)
            wavelength_hdu = fits.BinTableHDU.from_columns([col], name='WAVELENGTH')
            wavelength_hdu.header['BUNIT'] = 'Angstrom'
            hdul_out.append(wavelength_hdu)

        output_dir = os.path.dirname(output_fits_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        hdul_out.writeto(output_fits_path, overwrite=True, output_verify='fix')
        hdul_out.close()
        print(f"IFU results saved to {output_fits_path}")

    def _create_ifu_header(self, extname, shape, flux_unit):
        """Helper to create a standard header for IFU cube extensions."""
        hdr = fits.Header()
        hdr['EXTNAME'] = extname
        hdr['BUNIT'] = flux_unit
        hdr['CTYPE1'] = 'WAVE'
        hdr['CRPIX1'] = 1.0 
        hdr['CRVAL1'] = self.desired_wave_grid[0]
        hdr['CDELT1'] = (self.desired_wave_grid[1] - self.desired_wave_grid[0]) if len(self.desired_wave_grid) > 1 else 0.0
        hdr['CUNIT1'] = 'Angstrom'
        hdr['CTYPE2'] = 'DEC--TAN'
        hdr['CRPIX2'] = shape[1] / 2.0 + 0.5 
        hdr['CDELT2'] = self.final_pixel_scale_arcsec
        hdr['CUNIT2'] = 'arcsec'
        hdr['CTYPE3'] = 'RA---TAN'
        hdr['CRPIX3'] = shape[2] / 2.0 + 0.5 
        hdr['CDELT3'] = self.final_pixel_scale_arcsec
        hdr['CUNIT3'] = 'arcsec'
        return hdr