import os
import numpy as np
from astropy.io import fits
from astropy.convolution import convolve_fft, Gaussian1DKernel
from photutils.psf.matching import resize_psf
from scipy.integrate import simpson
from scipy import stats
from scipy.interpolate import interp1d
from astropy.nddata import NDData
from reproject import reproject_adaptive
from .imgutils import convert_flux_map

class GalSynMockObservation_imaging:
    """
    A class to simulate observational effects on synthetic galaxy images,
    including PSF convolution, noise injection, RMS image generation,
    and resampling to a desired pixel scale.
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
        self.flux_unit = self.image_header['BUNIT']
        self.flux_scale = self.image_header['SCALE']

        self.processed_images = {} # Stores final processed (noisy, resampled) images
        self.rms_images = {}      # Stores final resampled RMS images

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.hdul.close()

    def _get_flux_data(self, filter_name, dust_attenuation=True):
        """
        Retrieves flux data for a given filter from the FITS file.
        This function converts the stored flux (in its original `self.flux_unit`)
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

            # The data in FITS is `converted_flux / self.flux_scale`.
            # So to get `converted_flux`: `data_in_fits_unit * self.flux_scale`.
            # Then, perform the inverse conversion from `converted_flux` (in `self.flux_unit`)
            # back to `erg/s/cm^2/Angstrom` (or erg/s/cm^2/Angstrom/pixel in case of MJy/sr input).

            converted_flux_original = flux_data_in_fits_unit * self.flux_scale

            if self.flux_unit == 'erg/s/cm2/A':
                # This is already a spectral flux density, not surface brightness.
                # If it's expected to be surface brightness for the input, it implies
                # the FITS data was already scaled by pixel area before saving.
                return converted_flux_original

            elif self.flux_unit == 'nJy':
                # f_nu [erg/s/cm^2/Hz] = nJy / 1e9 * 1e-23
                f_nu_erg_s_cm2_Hz = converted_flux_original / 1e9 * 1e-23
                # erg/s/cm^2/Angstrom = f_nu * c / wave_eff^2
                # This is a spectral flux density.
                return f_nu_erg_s_cm2_Hz * c_angstrom_s / wave_eff**2

            elif self.flux_unit == 'AB magnitude':
                # For AB magnitude, the `flux_scale` in galsyn_run_fsps is 1.0, so `converted_flux_original` is just the AB mag.
                # f_nu [erg/s/cm^2/Hz] = 10^((AB_mag + 48.6)/(-2.5))
                f_nu_erg_s_cm2_Hz = 10**((converted_flux_original + 48.6)/(-2.5))
                # This is a spectral flux density.
                return f_nu_erg_s_cm2_Hz * c_angstrom_s / wave_eff**2

            elif self.flux_unit == 'MJy/sr':
                # MJy/sr is a surface brightness unit.
                # Here, we convert MJy/sr (flux per solid angle) to flux per pixel (erg/s/cm2/A/pixel).
                # This is done by multiplying by the pixel area in steradians.
                pixel_area_sr = (self.initial_pixel_scale_arcsec * np.pi / (180.0 * 3600.0))**2
                f_nu_jy_per_pixel = converted_flux_original * pixel_area_sr * 1e6 # MJy/sr -> Jy/sr * sr/pixel = Jy/pixel
                f_nu_erg_s_cm2_Hz_per_pixel = f_nu_jy_per_pixel * 1e-23
                # Result is spectral flux per pixel (i.e., total flux in the pixel per unit wavelength)
                return f_nu_erg_s_cm2_Hz_per_pixel * c_angstrom_s / wave_eff**2
            else:
                raise ValueError(f"Unsupported flux_unit for inverse conversion: {self.flux_unit}")

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

            # Ensure the resampled PSF sums to 1 (normalization)
            resampled_psf /= np.sum(resampled_psf)
            return resampled_psf
        else:
            # Ensure the PSF sums to 1 (normalization) if no resampling
            psf_data /= np.sum(psf_data)
            return psf_data

    def _load_filter_transmission_from_paths_local(self, filters_list, filter_transmission_path_dict):
        """
        Helper function to load filter transmission data and calculate pivot wavelengths.
        This is a simplified version of the one in galsyn_run_fsps, for local use.
        """
        filter_transmission_data = {}
        filter_wave_pivot_data = {}

        for f_name in filters_list:
            file_path = filter_transmission_path_dict.get(f_name)
            if file_path is None:
                raise ValueError(f"Filter transmission path not found for {f_name}. Please provide `filter_transmission_path` to the constructor.")

            data = np.loadtxt(file_path)
            wave = data[:, 0]
            trans = data[:, 1]

            filter_transmission_data[f_name] = {'wave': wave, 'trans': trans}

            numerator = simpson(wave * trans, wave)
            denominator = simpson(trans / np.where(wave != 0, wave, 1e-10), wave) # Avoid div by zero

            pivot_wavelength = np.sqrt(numerator / denominator) if denominator > 0 else np.nan
            filter_wave_pivot_data[f_name] = pivot_wavelength

        return filter_transmission_data, filter_wave_pivot_data


    def process_images(self, dust_attenuation=True, apply_noise_to_image=True):
        """
        Executes the full pipeline of observational effects:
        1. Convolves images with PSF.
        2. Simulates and injects noise, and generates RMS images.
        3. Resamples the processed (noisy) images and RMS images to the desired pixel scale.

        Crucially, this pipeline ensures that data passed to `reproject_adaptive`
        is in surface brightness units (flux per unit solid angle) for proper flux conservation.

        Parameters:
        -----------
        dust_attenuation : bool
            Whether to process images with dust attenuation (True) or without (False).
        apply_noise_to_image : bool
            If True, noise is added to the convolved image. Otherwise, only RMS is calculated.
        """
        print("\nStarting full image processing pipeline...")
        for f_name in self.filters:
            print(f"\nProcessing filter: {f_name}")

            # --- Get filter-specific parameters from dictionaries ---
            mag_zp = self.mag_zp[f_name]
            limiting_magnitude = self.limiting_magnitude[f_name]
            snr_limit = self.snr_limit[f_name]
            aperture_radius_arcsec = self.aperture_radius_arcsec[f_name]
            exposure_time = self.exposure_time[f_name]
            desired_pixel_scale = self.desired_pixel_scales[f_name]

            # --- 1. Get Initial Image Data (Flux per pixel or flux density) ---
            image_data_initial_raw_units = self._get_flux_data(f_name, dust_attenuation)

            # --- Convert initial data to a common surface brightness unit (erg/s/cm^2/Angstrom/arcsec^2) ---
            pixel_area_arcsec2_initial = self.initial_pixel_scale_arcsec**2
            if pixel_area_arcsec2_initial <= 0:
                raise ValueError("Initial pixel area must be positive.")

            image_surface_brightness = image_data_initial_raw_units / pixel_area_arcsec2_initial
            print(f"  Initial {f_name} image converted to surface brightness (erg/s/cm^2/Angstrom/arcsec^2).")

            # --- 2. PSF Convolution ---
            print(f"  Convolving {f_name} image with PSF...")
            psf_data = self._get_psf(f_name)
            convolved_surface_brightness = convolve_fft(image_surface_brightness, psf_data, boundary='fill', fill_value=0.0)
            print(f"  {f_name} image convolved.")

            # --- 3. Noise Simulation and Injection ---
            print(f"  Simulating noise for {f_name} image...")
            image_flux_per_initial_pixel = convolved_surface_brightness * pixel_area_arcsec2_initial

            _, filter_wave_pivot_data = self._load_filter_transmission_from_paths_local(self.filters, self.filter_transmission_path)
            wave_eff = filter_wave_pivot_data[f_name]

            # 1. Estimate background RMS (sigma_bg)
            C_aperture = exposure_time * (10**(0.4 * (mag_zp - limiting_magnitude)))
            sigma_bg_aperture_sq = (C_aperture / snr_limit)**2 - C_aperture
            sigma_bg_aperture_sq = np.maximum(0, sigma_bg_aperture_sq)
            sigma_bg_aperture = np.sqrt(sigma_bg_aperture_sq)

            aperture_area_pix2_for_bg = np.pi * (aperture_radius_arcsec / self.initial_pixel_scale_arcsec)**2
            if aperture_area_pix2_for_bg <= 0:
                raise ValueError("Aperture area per pixel must be positive.")
            sigma_bg_per_pixel = sigma_bg_aperture / np.sqrt(aperture_area_pix2_for_bg)

            # 2. Convert pixel flux to counts
            c_angstrom_s = 2.99792458e18
            f_nu_erg_s_cm2_Hz_pixel = image_flux_per_initial_pixel * wave_eff**2 / c_angstrom_s
            pixel_mag_AB = -2.5 * np.log10(np.clip(f_nu_erg_s_cm2_Hz_pixel, 1e-50, None)) - 48.6
            source_counts_per_pixel_expected = exposure_time * (10**(0.4 * (mag_zp - pixel_mag_AB)))
            source_counts_per_pixel_expected = np.maximum(0, source_counts_per_pixel_expected)

            mag_for_1_count = mag_zp - 2.5 * np.log10(1.0 / exposure_time)
            f_nu_erg_s_cm2_Hz_for_1_count = 10**((mag_for_1_count + 48.6)/(-2.5))
            flux_per_total_count_per_A_per_pixel = f_nu_erg_s_cm2_Hz_for_1_count * c_angstrom_s / wave_eff**2

            noisy_image_flux_per_initial_pixel = image_flux_per_initial_pixel.copy()

            if apply_noise_to_image:
                photon_shot_noise_sampled_counts = stats.poisson.rvs(source_counts_per_pixel_expected)
                total_noisy_counts = photon_shot_noise_sampled_counts + np.random.normal(0, sigma_bg_per_pixel, size=image_flux_per_initial_pixel.shape)
                total_noisy_counts = np.maximum(0, total_noisy_counts)
                noisy_image_flux_per_initial_pixel = total_noisy_counts * flux_per_total_count_per_A_per_pixel
                noisy_image_flux_per_initial_pixel = np.maximum(1e-30, noisy_image_flux_per_initial_pixel)

            total_variance_counts = source_counts_per_pixel_expected + sigma_bg_per_pixel**2
            total_rms_counts_per_pixel = np.sqrt(total_variance_counts)
            rms_image_flux_per_initial_pixel = total_rms_counts_per_pixel * flux_per_total_count_per_A_per_pixel
            print(f"  Noise simulated and injected for {f_name}.")

            # --- Convert back to Surface Brightness for Resampling ---
            final_noisy_surface_brightness = noisy_image_flux_per_initial_pixel / pixel_area_arcsec2_initial
            final_rms_surface_brightness = rms_image_flux_per_initial_pixel / pixel_area_arcsec2_initial

            # --- 4. Resampling to Desired Pixel Scale (Flux Conserving) ---
            print(f"  Resampling {f_name} image to desired pixel scale...")
            if desired_pixel_scale is None:
                raise ValueError(f"Desired pixel scale not provided for filter: {f_name}. "
                                 "Please ensure 'desired_pixel_scales' dictionary is complete.")

            if np.isclose(desired_pixel_scale, self.initial_pixel_scale_arcsec):
                print(f"  Desired pixel scale for {f_name} is already {desired_pixel_scale:.4f} arcsec. No resampling needed.")
                resampled_noisy_image_sb = final_noisy_surface_brightness
                resampled_rms_image_sb = final_rms_surface_brightness
            else:
                print(f"  Resampling from {self.initial_pixel_scale_arcsec:.4f} arcsec to {desired_pixel_scale:.4f} arcsec.")
                old_ny, old_nx = final_noisy_surface_brightness.shape
                resampling_factor = self.initial_pixel_scale_arcsec / desired_pixel_scale
                new_shape = (int(np.round(old_ny * resampling_factor)), int(np.round(old_nx * resampling_factor)))

                resampled_noisy_image_sb = reproject_adaptive(
                    NDData(final_noisy_surface_brightness),
                    output_projection=None,
                    shape_out=new_shape,
                    fill_value=0.0,
                    flux_conserving=True
                )[0]

                resampled_rms_image_sb = reproject_adaptive(
                    NDData(final_rms_surface_brightness),
                    output_projection=None,
                    shape_out=new_shape,
                    fill_value=0.0,
                    flux_conserving=True
                )[0]

            key_processed = f"{'dust' if dust_attenuation else 'nodust'}_{f_name}_processed"
            self.processed_images[key_processed] = resampled_noisy_image_sb

            key_rms = f"{f_name}_{'dust' if dust_attenuation else 'nodust'}_rms"
            self.rms_images[key_rms] = resampled_rms_image_sb
            print(f"  {f_name} image resampled and stored.")

        print("\nFull image processing pipeline complete.")


    def save_results_to_fits(self, output_fits_path, dust_attenuation=True):
        """
        Saves the processed (noise-injected and resampled) images and RMS images to a new FITS file.
        The data stored will be converted to the `self.flux_unit` specified during initialization,
        using the `desired_pixel_scales` for the conversion of surface brightness units.
        """
        print(f"\nSaving results to FITS file: {output_fits_path}...")
        hdul_out = fits.HDUList()

        prihdr = self.hdul[0].header.copy()
        prihdr['COMMENT'] = 'Mock Observation Results'
        prihdr['NOISE_SIM'] = 'True'

        first_filter = self.filters[0] if self.filters else None
        if first_filter:
            prihdr['ZP_MAG'] = (self.mag_zp[first_filter], f'ZP for {first_filter}')
            prihdr['LIM_MAG'] = (self.limiting_magnitude[first_filter], f'Limiting mag for {first_filter}')
            prihdr['SNR_LIM'] = (self.snr_limit[first_filter], f'SNR at lim mag for {first_filter}')
            prihdr['APER_RAD'] = (self.aperture_radius_arcsec[first_filter], f'Aperture radius (arcsec) for {first_filter}')
            prihdr['EXP_TIME'] = (self.exposure_time[first_filter], f'Exposure time (s) for {first_filter}')
            prihdr['PIXSIZE'] = self.desired_pixel_scales.get(first_filter, self.initial_pixel_scale_arcsec)
        else:
            prihdr['PIXSIZE'] = self.initial_pixel_scale_arcsec

        if first_filter and f"dust_{first_filter}_processed" in self.processed_images:
            primary_data_sb_erg_s_cm2_A_arcsec2 = self.processed_images[f"dust_{first_filter}_processed"]
            _, filter_wave_pivot_data = self._load_filter_transmission_from_paths_local(self.filters, self.filter_transmission_path)
            wave_eff = filter_wave_pivot_data[first_filter]
            final_pixel_scale_for_conversion = self.desired_pixel_scales.get(first_filter, self.initial_pixel_scale_arcsec)
            primary_data_flux_per_pixel_erg_s_cm2_A = primary_data_sb_erg_s_cm2_A_arcsec2 * (final_pixel_scale_for_conversion**2)
            primary_data_final_unit = convert_flux_map(primary_data_flux_per_pixel_erg_s_cm2_A, wave_eff, to_unit=self.flux_unit, pixel_scale_arcsec=final_pixel_scale_for_conversion)
            final_scaled_primary_data = primary_data_final_unit / self.flux_scale
            prihdr['BUNIT'] = self.flux_unit
            prihdr['SCALE'] = self.flux_scale
            hdul_out.append(fits.PrimaryHDU(data=final_scaled_primary_data, header=prihdr))
        else:
            hdul_out.append(fits.PrimaryHDU(header=prihdr))

        for f_name in self.filters:
            key_processed = f"{'dust' if dust_attenuation else 'nodust'}_{f_name}_processed"
            if key_processed in self.processed_images:
                img_data_sb_erg_s_cm2_A_arcsec2 = self.processed_images[key_processed]
                _, filter_wave_pivot_data = self._load_filter_transmission_from_paths_local(self.filters, self.filter_transmission_path)
                wave_eff = filter_wave_pivot_data[f_name]
                final_pixel_scale_for_conversion = self.desired_pixel_scales.get(f_name, self.initial_pixel_scale_arcsec)
                img_data_flux_per_pixel_erg_s_cm2_A = img_data_sb_erg_s_cm2_A_arcsec2 * (final_pixel_scale_for_conversion**2)
                img_data_final_unit = convert_flux_map(img_data_flux_per_pixel_erg_s_cm2_A, wave_eff, to_unit=self.flux_unit, pixel_scale_arcsec=final_pixel_scale_for_conversion)
                final_scaled_data = img_data_final_unit / self.flux_scale
                ext_hdr = fits.Header()
                ext_hdr['EXTNAME'] = f"PROCESSED_IMG_{f_name.upper()}"
                ext_hdr['FILTER'] = f_name
                ext_hdr['COMMENT'] = f'Convolved, noise-injected, and resampled image for filter: {f_name}'
                ext_hdr['BUNIT'] = self.flux_unit
                ext_hdr['SCALE'] = self.flux_scale
                ext_hdr['PIXSIZE'] = final_pixel_scale_for_conversion
                ext_hdr['ZP_MAG'] = self.mag_zp[f_name]
                ext_hdr['LIM_MAG'] = self.limiting_magnitude[f_name]
                ext_hdr['SNR_LIM'] = self.snr_limit[f_name]
                ext_hdr['APER_RAD'] = self.aperture_radius_arcsec[f_name]
                ext_hdr['EXP_TIME'] = self.exposure_time[f_name]
                hdul_out.append(fits.ImageHDU(data=final_scaled_data, header=ext_hdr))

        for f_name in self.filters:
            key_rms = f"{f_name}_{'dust' if dust_attenuation else 'nodust'}_rms"
            if key_rms in self.rms_images:
                rms_data_sb_erg_s_cm2_A_arcsec2 = self.rms_images[key_rms]
                _, filter_wave_pivot_data = self._load_filter_transmission_from_paths_local(self.filters, self.filter_transmission_path)
                wave_eff = filter_wave_pivot_data[f_name]
                final_pixel_scale_for_conversion = self.desired_pixel_scales.get(f_name, self.initial_pixel_scale_arcsec)
                rms_data_flux_per_pixel_erg_s_cm2_A = rms_data_sb_erg_s_cm2_A_arcsec2 * (final_pixel_scale_for_conversion**2)
                rms_data_final_unit = convert_flux_map(rms_data_flux_per_pixel_erg_s_cm2_A, wave_eff, to_unit=self.flux_unit, pixel_scale_arcsec=final_pixel_scale_for_conversion)
                final_scaled_rms = rms_data_final_unit / self.flux_scale
                ext_hdr = fits.Header()
                ext_hdr['EXTNAME'] = f"RMS_IMG_{f_name.upper()}"
                ext_hdr['FILTER'] = f_name
                ext_hdr['COMMENT'] = f'RMS image for filter: {f_name}'
                ext_hdr['BUNIT'] = self.flux_unit
                ext_hdr['SCALE'] = self.flux_scale
                ext_hdr['PIXSIZE'] = final_pixel_scale_for_conversion
                hdul_out.append(fits.ImageHDU(data=final_scaled_rms, header=ext_hdr))

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

        try:
            wavelength_hdu = self.hdul['WAVELENGTH_GRID']
            self.original_wave_grid = wavelength_hdu.data['WAVELENGTH']
        except KeyError:
            raise ValueError("Input FITS file must contain a 'WAVELENGTH_GRID' binary table extension.")

        self.initial_datacube_nodust = self.hdul['OBS_SPEC_NODUST'].data if 'OBS_SPEC_NODUST' in self.hdul else None
        self.initial_datacube_dust = self.hdul['OBS_SPEC_DUST'].data if 'OBS_SPEC_DUST' in self.hdul else None

        self.processed_datacube = None
        self.rms_datacube = None

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

        # Pre-allocate memory for the resampled PSF cube
        new_y = int(np.round(psf_cube_data.shape[1] * self.psf_pixel_scale / self.initial_pixel_scale_arcsec))
        new_x = int(np.round(psf_cube_data.shape[2] * self.psf_pixel_scale / self.initial_pixel_scale_arcsec))
        resampled_psf_cube = np.zeros((psf_cube_data.shape[0], new_y, new_x))

        for i_wave in range(psf_cube_data.shape[0]):
            psf_slice = psf_cube_data[i_wave, :, :]
            if not np.isclose(self.psf_pixel_scale, self.initial_pixel_scale_arcsec):
                resampled_psf = resize_psf(psf_slice, self.psf_pixel_scale, self.initial_pixel_scale_arcsec)
            else:
                resampled_psf = psf_slice
            
            resampled_psf /= np.sum(resampled_psf) # Normalize
            resampled_psf_cube[i_wave, :, :] = resampled_psf

        return resampled_psf_cube


    def process_datacube(self, dust_attenuation=True, apply_noise_to_cube=True):
        """
        Executes the full pipeline of observational effects for the IFU data cube.
        """
        print("\nStarting full IFU data cube processing pipeline...")

        input_datacube = self.initial_datacube_dust if dust_attenuation and self.initial_datacube_dust is not None else self.initial_datacube_nodust
        if input_datacube is None:
            raise ValueError(f"Requested data cube (dust_attenuation={dust_attenuation}) not found in FITS file.")

        print(f"  Initial data cube shape: {input_datacube.shape} (Angstrom, Y, X)")

        # --- 1. Cut/Interpolate Wavelength Grid ---
        print("  Cutting/interpolating data cube to desired wavelength grid...")
        reshaped_cube = input_datacube.reshape(input_datacube.shape[0], -1).T
        interpolated_cube_reshaped = np.zeros((reshaped_cube.shape[0], len(self.desired_wave_grid)))
        for i in range(reshaped_cube.shape[0]):
            interp_func = interp1d(self.original_wave_grid, reshaped_cube[i, :], kind='linear', bounds_error=False, fill_value=0.0)
            interpolated_cube_reshaped[i, :] = interp_func(self.desired_wave_grid)
        processed_cube_wave_cut = interpolated_cube_reshaped.T.reshape(len(self.desired_wave_grid), input_datacube.shape[1], input_datacube.shape[2])
        print(f"  Data cube cut to wavelength grid. New shape: {processed_cube_wave_cut.shape}")

        # --- Convert to Surface Brightness ---
        pixel_area_arcsec2_initial = self.initial_pixel_scale_arcsec**2
        processed_cube_sb = processed_cube_wave_cut / pixel_area_arcsec2_initial
        print(f"  Data cube converted to spectral surface brightness (erg/s/cm^2/Angstrom/arcsec^2).")

        # --- 3. Spectral Smoothing ---
        print(f"  Smoothing spectra to R={self.spectral_resolution_R}...")
        smoothed_cube_sb = np.zeros_like(processed_cube_sb)
        if len(self.desired_wave_grid) > 1:
            delta_lambda_per_pixel = np.mean(np.diff(self.desired_wave_grid))
            sigma_lambda_per_channel = self.desired_wave_grid / self.spectral_resolution_R / (2 * np.sqrt(2 * np.log(2)))
            mean_sigma_pixels = np.mean(sigma_lambda_per_channel / delta_lambda_per_pixel)

            if mean_sigma_pixels > 0:
                gauss_kernel = Gaussian1DKernel(stddev=mean_sigma_pixels)
                for i_y in range(processed_cube_sb.shape[1]):
                    for i_x in range(processed_cube_sb.shape[2]):
                        smoothed_cube_sb[:, i_y, i_x] = convolve_fft(processed_cube_sb[:, i_y, i_x], gauss_kernel, boundary='fill', fill_value=0.0)
            else:
                smoothed_cube_sb = processed_cube_sb
        else:
             smoothed_cube_sb = processed_cube_sb # No smoothing for a single wavelength point
        print("  Spectra smoothed.")

        # --- 4. Spatial PSF Convolution ---
        print("  Convolving each spatial slice with PSF...")
        psf_cube = self._load_psf_cube()
        convolved_cube_sb = np.zeros_like(smoothed_cube_sb)
        for i_wave in range(smoothed_cube_sb.shape[0]):
            convolved_cube_sb[i_wave, :, :] = convolve_fft(smoothed_cube_sb[i_wave, :, :], psf_cube[i_wave, :, :], boundary='fill', fill_value=0.0)
        print("  Spatial PSF convolution complete.")

        # --- 5. Noise Simulation and Injection ---
        print("  Simulating and injecting noise...")
        noisy_cube_flux_per_initial_pixel = np.zeros_like(convolved_cube_sb)
        rms_cube_flux_per_initial_pixel = np.zeros_like(convolved_cube_sb)
        c_angstrom_s = 2.99792458e18
        
        for i_wave in range(convolved_cube_sb.shape[0]):
            current_wave = self.desired_wave_grid[i_wave]
            
            # Get wavelength-dependent parameters
            mag_zp = self._get_wave_dependent_param(self.mag_zp, current_wave)
            lim_mag_at_wave = self.limiting_magnitude_wave_func(current_wave)
            snr_limit = self._get_wave_dependent_param(self.snr_limit, current_wave)
            exposure_time = self._get_wave_dependent_param(self.exposure_time, current_wave)

            # Convert current slice SB to flux per initial pixel for noise calculation
            current_slice_flux_per_initial_pixel = convolved_cube_sb[i_wave, :, :] * pixel_area_arcsec2_initial
            
            # Source counts
            f_nu_erg_s_cm2_Hz_pixel = current_slice_flux_per_initial_pixel * current_wave**2 / c_angstrom_s
            pixel_mag_AB = -2.5 * np.log10(np.clip(f_nu_erg_s_cm2_Hz_pixel, 1e-50, None)) - 48.6
            source_counts_per_pixel_expected = exposure_time * (10**(0.4 * (mag_zp - pixel_mag_AB)))
            source_counts_per_pixel_expected = np.maximum(0, source_counts_per_pixel_expected)

            # Background counts variance
            C_aperture_at_wave = exposure_time * (10**(0.4 * (mag_zp - lim_mag_at_wave)))
            sigma_bg_counts_sq_per_pixel = (C_aperture_at_wave / snr_limit)**2 - C_aperture_at_wave
            sigma_bg_counts_sq_per_pixel = np.maximum(0, sigma_bg_counts_sq_per_pixel)
            sigma_bg_counts_per_pixel = np.sqrt(sigma_bg_counts_sq_per_pixel)

            # Conversion factor from counts back to flux
            mag_for_1_count = mag_zp - 2.5 * np.log10(1.0 / exposure_time)
            f_nu_erg_s_cm2_Hz_for_1_count = 10**((mag_for_1_count + 48.6)/(-2.5))
            flux_per_total_count_per_A_per_pixel = f_nu_erg_s_cm2_Hz_for_1_count * c_angstrom_s / current_wave**2

            if apply_noise_to_cube:
                photon_shot_noise = stats.poisson.rvs(source_counts_per_pixel_expected)
                background_noise = np.random.normal(0, sigma_bg_counts_per_pixel, size=source_counts_per_pixel_expected.shape)
                total_noisy_counts = np.maximum(0, photon_shot_noise + background_noise)
                noisy_cube_flux_per_initial_pixel[i_wave, :, :] = np.maximum(1e-30, total_noisy_counts * flux_per_total_count_per_A_per_pixel)
            else:
                noisy_cube_flux_per_initial_pixel[i_wave, :, :] = current_slice_flux_per_initial_pixel

            total_variance_counts_slice = source_counts_per_pixel_expected + sigma_bg_counts_sq_per_pixel
            total_rms_counts_per_pixel_slice = np.sqrt(total_variance_counts_slice)
            rms_cube_flux_per_initial_pixel[i_wave, :, :] = np.maximum(1e-30, total_rms_counts_per_pixel_slice * flux_per_total_count_per_A_per_pixel)
        print("  Noise simulated and injected.")

        # Convert back to surface brightness for resampling
        noisy_cube_sb = noisy_cube_flux_per_initial_pixel / pixel_area_arcsec2_initial
        rms_cube_sb = rms_cube_flux_per_initial_pixel / pixel_area_arcsec2_initial

        # --- 6. Spatial Resampling to Final Pixel Scale ---
        print(f"  Resampling data cube spatially to {self.final_pixel_scale_arcsec:.4f} arcsec...")
        if np.isclose(self.final_pixel_scale_arcsec, self.initial_pixel_scale_arcsec):
            resampled_processed_cube_sb = noisy_cube_sb
            resampled_rms_cube_sb = rms_cube_sb
        else:
            old_ny, old_nx = noisy_cube_sb.shape[1:]
            resampling_factor = self.initial_pixel_scale_arcsec / self.final_pixel_scale_arcsec
            new_ny, new_nx = int(np.round(old_ny * resampling_factor)), int(np.round(old_nx * resampling_factor))
            new_spatial_shape = (new_ny, new_nx)

            resampled_processed_cube_sb = np.zeros((noisy_cube_sb.shape[0], new_ny, new_nx))
            resampled_rms_cube_sb = np.zeros((rms_cube_sb.shape[0], new_ny, new_nx))

            for i_wave in range(noisy_cube_sb.shape[0]):
                resampled_processed_cube_sb[i_wave, :, :] = reproject_adaptive(
                    NDData(noisy_cube_sb[i_wave, :, :]), None, shape_out=new_spatial_shape, fill_value=0.0, flux_conserving=True)[0]
                resampled_rms_cube_sb[i_wave, :, :] = reproject_adaptive(
                    NDData(rms_cube_sb[i_wave, :, :]), None, shape_out=new_spatial_shape, fill_value=0.0, flux_conserving=True)[0]
        print("  Spatial resampling complete.")

        self.processed_datacube = resampled_processed_cube_sb
        self.rms_datacube = resampled_rms_cube_sb
        print("\nFull IFU data cube processing pipeline complete.")

    def save_results_to_fits(self, output_fits_path, flux_unit='erg/s/cm2/A'):
        """
        Saves the processed data cube and the RMS data cube to a new FITS file.
        """
        print(f"\nSaving IFU results to FITS file: {output_fits_path}...")
        hdul_out = fits.HDUList()

        prihdr = self.image_header.copy()
        prihdr['COMMENT'] = 'Mock IFU Observation Results'
        prihdr['NOISE_SIM'] = 'True'
        prihdr['RES_R'] = self.spectral_resolution_R
        prihdr['PIXSIZE'] = self.final_pixel_scale_arcsec
        
        # Save parameter values at the central wavelength for header reference
        central_wave = self.desired_wave_grid[len(self.desired_wave_grid) // 2]
        prihdr['ZP_MAG'] = (self._get_wave_dependent_param(self.mag_zp, central_wave), 'ZP at central wavelength')
        prihdr['SNR_LIM'] = (self._get_wave_dependent_param(self.snr_limit, central_wave), 'SNR at central wavelength')
        prihdr['EXP_TIME'] = (self._get_wave_dependent_param(self.exposure_time, central_wave), 'Exposure time (s) at central wavelength')

        hdul_out.append(fits.PrimaryHDU(header=prihdr))

        if self.processed_datacube is not None:
            pixel_area_arcsec2_final = self.final_pixel_scale_arcsec**2
            processed_cube_flux_per_pixel = self.processed_datacube * pixel_area_arcsec2_final
            final_processed_cube = np.zeros_like(processed_cube_flux_per_pixel)

            for i_wave in range(processed_cube_flux_per_pixel.shape[0]):
                final_processed_cube[i_wave, :, :] = convert_flux_map(
                    processed_cube_flux_per_pixel[i_wave, :, :],
                    self.desired_wave_grid[i_wave],
                    to_unit=flux_unit,
                    pixel_scale_arcsec=self.final_pixel_scale_arcsec)
            
            output_flux_scale = 1e-20 if flux_unit == 'erg/s/cm2/A' else 1.0
            final_processed_cube /= output_flux_scale

            ext_hdr_proc = self._create_ifu_header('PROCESSED_IFU_CUBE', final_processed_cube.shape, flux_unit, output_flux_scale)
            hdul_out.append(fits.ImageHDU(data=final_processed_cube, header=ext_hdr_proc))

        if self.rms_datacube is not None:
            pixel_area_arcsec2_final = self.final_pixel_scale_arcsec**2
            rms_cube_flux_per_pixel = self.rms_datacube * pixel_area_arcsec2_final
            final_rms_cube = np.zeros_like(rms_cube_flux_per_pixel)

            for i_wave in range(rms_cube_flux_per_pixel.shape[0]):
                final_rms_cube[i_wave, :, :] = convert_flux_map(
                    rms_cube_flux_per_pixel[i_wave, :, :],
                    self.desired_wave_grid[i_wave],
                    to_unit=flux_unit,
                    pixel_scale_arcsec=self.final_pixel_scale_arcsec)
            
            output_flux_scale = 1e-20 if flux_unit == 'erg/s/cm2/A' else 1.0
            final_rms_cube /= output_flux_scale
            
            ext_hdr_rms = self._create_ifu_header('RMS_IFU_CUBE', final_rms_cube.shape, flux_unit, output_flux_scale)
            hdul_out.append(fits.ImageHDU(data=final_rms_cube, header=ext_hdr_rms))

        if len(self.desired_wave_grid) > 0:
            col = fits.Column(name='WAVELENGTH', format='D', array=self.desired_wave_grid)
            wavelength_hdu = fits.BinTableHDU.from_columns([col], name='WAVELENGTH_GRID_FINAL')
            wavelength_hdu.header['BUNIT'] = 'Angstrom'
            hdul_out.append(wavelength_hdu)

        output_dir = os.path.dirname(output_fits_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        hdul_out.writeto(output_fits_path, overwrite=True, output_verify='fix')
        hdul_out.close()
        print(f"IFU results saved to {output_fits_path}")

    def _create_ifu_header(self, extname, shape, flux_unit, scale):
        """Helper to create a standard header for IFU cube extensions."""
        hdr = fits.Header()
        hdr['EXTNAME'] = extname
        hdr['BUNIT'] = flux_unit
        hdr['SCALE'] = scale
        # WCS for (wavelength, y, x)
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