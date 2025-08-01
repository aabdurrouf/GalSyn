import os
import numpy as np
from astropy.io import fits
from astropy.convolution import convolve_fft, Gaussian1DKernel
from photutils.psf.matching import resize_psf
from scipy.integrate import simpson
from scipy import stats
from scipy.interpolate import interp1d
from astropy.nddata import NDData

# Import reproject_adaptive from the reproject package
from reproject import reproject_adaptive

# Import convert_flux_map from the provided imgutils.py
from .imgutils import convert_flux_map

# Existing GalSynMockObservation_imaging class (from previous response)
class GalSynMockObservation_imaging:
    """
    A class to simulate observational effects on synthetic galaxy images,
    including PSF convolution, noise injection, RMS image generation,
    and resampling to a desired pixel scale.
    """

    def __init__(self, fits_file_path, filters, psf_paths, psf_pixel_scales,
                 mag_zp, limiting_magnitude, snr_limit, aperture_radius_arcsec, exposure_time,
                 filter_transmission_path, desired_pixel_scales):
        """
        Initializes the GalSynMockObservation_imaging with input parameters.

        Parameters:
        -----------
        fits_file_path : str
            Path to the FITS file output from galsyn_run_fsps.
        filters : list
            List of filter names (e.g., ['FUV', 'NUV']) for which images will be processed.
        psf_paths : dict
            Dictionary where keys are filter names and values are paths to PSF FITS images.
        psf_pixel_scales : dict
            Dictionary where keys are filter names and values are pixel scales of PSF images in arcsec.
        mag_zp : float
            Magnitude zero-point of the observation system.
        limiting_magnitude : float
            Limiting magnitude of the observation.
        snr_limit : float
            Signal-to-noise ratio at the limiting magnitude.
        aperture_radius_arcsec : float
            Radius of the circular aperture (in arcsec) used in measuring the magnitude limit.
        exposure_time : float
            Exposure time in seconds.
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

        self.hdul = fits.open(fits_file_path)
        self.image_header = self.hdul[0].header
        self.pixel_scale_kpc = self.image_header['PIX_KPC']
        # This is the *initial* pixel scale of the synthetic image
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

            # --- 1. Get Initial Image Data (Flux per pixel or flux density) ---
            # _get_flux_data returns:
            # - If initial_flux_unit is MJy/sr: flux_erg_s_cm2_A is actually Flux/pixel (erg/s/cm^2/A/pixel)
            # - If initial_flux_unit is nJy, ABmag, erg/s/cm2/A: flux_erg_s_cm2_A is Spectral Flux Density (erg/s/cm^2/A)
            image_data_initial_raw_units = self._get_flux_data(f_name, dust_attenuation)

            # --- Convert initial data to a common surface brightness unit (erg/s/cm^2/Angstrom/arcsec^2) ---
            # This is critical for PSF convolution and subsequent steps to be on a consistent
            # surface brightness basis before applying noise and resampling.
            # We assume initial_pixel_scale_arcsec applies to the "effective area" of the flux
            # represented by image_data_initial_raw_units, which is typical for synthetic images.

            # Convert to surface brightness for internal processing:
            # (Flux / pixel) / (pixel_area in arcsec^2) = Flux / arcsec^2
            # Or (Flux_density) / (pixel_area in arcsec^2) = Flux_density / arcsec^2 (if initial was just density)
            pixel_area_arcsec2_initial = self.initial_pixel_scale_arcsec**2
            if pixel_area_arcsec2_initial <= 0: # Defensive check
                raise ValueError("Initial pixel area must be positive.")

            # Data in erg/s/cm^2/Angstrom/arcsec^2 (true surface brightness)
            image_surface_brightness = image_data_initial_raw_units / pixel_area_arcsec2_initial
            print(f"  Initial {f_name} image converted to surface brightness (erg/s/cm^2/Angstrom/arcsec^2).")


            # --- 2. PSF Convolution ---
            print(f"  Convolving {f_name} image with PSF...")
            # PSF convolution is correctly applied to surface brightness
            psf_data = self._get_psf(f_name)
            convolved_surface_brightness = convolve_fft(image_surface_brightness, psf_data, boundary='fill', fill_value=0.0)
            print(f"  {f_name} image convolved.")

            # --- 3. Noise Simulation and Injection ---
            print(f"  Simulating noise for {f_name} image...")
            # Convert convolved surface brightness back to expected counts for noise calculation.
            # Here, we convert from surface brightness (per arcsec^2) to flux (per initial pixel)
            # as the noise model is per pixel counts.
            image_flux_per_initial_pixel = convolved_surface_brightness * pixel_area_arcsec2_initial


            # Get effective wavelength for this filter to convert flux
            _, filter_wave_pivot_data = self._load_filter_transmission_from_paths_local(self.filters, self.filter_transmission_path)
            wave_eff = filter_wave_pivot_data[f_name]

            # 1. Estimate background RMS (sigma_bg)
            C_aperture = self.exposure_time * (10**(0.4 * (self.mag_zp - self.limiting_magnitude)))
            sigma_bg_aperture_sq = (C_aperture / self.snr_limit)**2 - C_aperture
            sigma_bg_aperture_sq = np.maximum(0, sigma_bg_aperture_sq) # Ensure non-negative
            sigma_bg_aperture = np.sqrt(sigma_bg_aperture_sq)

            aperture_area_pix2_for_bg = np.pi * (self.aperture_radius_arcsec / self.initial_pixel_scale_arcsec)**2
            if aperture_area_pix2_for_bg <= 0:
                raise ValueError("Aperture area per pixel must be positive.")
            sigma_bg_per_pixel = sigma_bg_aperture / np.sqrt(aperture_area_pix2_for_bg) # Stddev of background counts per pixel

            # 2. Convert pixel flux (from convolved_surface_brightness * pixel_area) to AB magnitude, then to counts
            c_angstrom_s = 2.99792458e18 # speed of light in Angstrom/s
            
            # Convert flux per pixel (erg/s/cm2/A/pixel) to f_nu (erg/s/cm2/Hz/pixel)
            f_nu_erg_s_cm2_Hz_pixel = image_flux_per_initial_pixel * wave_eff**2 / c_angstrom_s
            
            # Convert f_nu (erg/s/cm2/Hz/pixel) to AB magnitude for each pixel
            pixel_mag_AB = -2.5 * np.log10(np.clip(f_nu_erg_s_cm2_Hz_pixel, 1e-50, None)) - 48.6

            source_counts_per_pixel_expected = self.exposure_time * (10**(0.4 * (self.mag_zp - pixel_mag_AB)))
            source_counts_per_pixel_expected = np.maximum(0, source_counts_per_pixel_expected) # Ensure non-negative counts

            mag_for_1_count = self.mag_zp - 2.5 * np.log10(1.0 / self.exposure_time)
            f_nu_erg_s_cm2_Hz_for_1_count = 10**((mag_for_1_count + 48.6)/(-2.5))
            flux_per_total_count_per_A_per_pixel = f_nu_erg_s_cm2_Hz_for_1_count * c_angstrom_s / wave_eff**2 # This is erg/s/cm^2/Angstrom/pixel for 1 count

            noisy_image_flux_per_initial_pixel = image_flux_per_initial_pixel.copy()

            if apply_noise_to_image:
                photon_shot_noise_sampled_counts = stats.poisson.rvs(source_counts_per_pixel_expected)
                total_noisy_counts = photon_shot_noise_sampled_counts + np.random.normal(0, sigma_bg_per_pixel, size=image_flux_per_initial_pixel.shape)
                total_noisy_counts = np.maximum(0, total_noisy_counts)
                
                # Convert noisy counts back to flux per initial pixel
                noisy_image_flux_per_initial_pixel = total_noisy_counts * flux_per_total_count_per_A_per_pixel
                noisy_image_flux_per_initial_pixel = np.maximum(1e-30, noisy_image_flux_per_initial_pixel) # Ensure non-negative flux

            total_variance_counts = source_counts_per_pixel_expected + sigma_bg_per_pixel**2
            total_rms_counts_per_pixel = np.sqrt(total_variance_counts)
            rms_image_flux_per_initial_pixel = total_rms_counts_per_pixel * flux_per_total_count_per_A_per_pixel
            print(f"  Noise simulated and injected for {f_name}.")

            # --- Convert back to Surface Brightness for Resampling ---
            # The `noisy_image_flux_per_initial_pixel` and `rms_image_flux_per_initial_pixel`
            # are currently "flux per pixel". For `reproject_adaptive` to work correctly
            # in conserving overall object flux while changing pixel scales, its input should be
            # surface brightness.
            final_noisy_surface_brightness = noisy_image_flux_per_initial_pixel / pixel_area_arcsec2_initial
            final_rms_surface_brightness = rms_image_flux_per_initial_pixel / pixel_area_arcsec2_initial


            # --- 4. Resampling to Desired Pixel Scale (Flux Conserving) ---
            print(f"  Resampling {f_name} image to desired pixel scale...")
            desired_pixel_scale = self.desired_pixel_scales.get(f_name)
            if desired_pixel_scale is None:
                raise ValueError(f"Desired pixel scale not provided for filter: {f_name}. "
                                 "Please ensure 'desired_pixel_scales' dictionary is complete.")

            if np.isclose(desired_pixel_scale, self.initial_pixel_scale_arcsec):
                print(f"  Desired pixel scale for {f_name} is already {desired_pixel_scale:.4f} arcsec. No resampling needed.")
                resampled_noisy_image_sb = final_noisy_surface_brightness
                resampled_rms_image_sb = final_rms_surface_brightness
            else:
                print(f"  Resampling from {self.initial_pixel_scale_arcsec:.4f} arcsec to {desired_pixel_scale:.4f} arcsec.")
                # Calculate new dimensions based on flux conservation
                old_ny, old_nx = final_noisy_surface_brightness.shape
                resampling_factor_x = self.initial_pixel_scale_arcsec / desired_pixel_scale
                resampling_factor_y = self.initial_pixel_scale_arcsec / desired_pixel_scale

                new_nx = int(np.round(old_nx * resampling_factor_x))
                new_ny = int(np.round(old_ny * resampling_factor_y))
                new_shape = (new_ny, new_nx)

                # Resample noisy image (input is surface brightness) using reproject_adaptive
                nddata_noisy_sb = NDData(final_noisy_surface_brightness)
                # reproject_adaptive returns (output_array, footprint). We only need output_array
                resampled_noisy_image_sb = reproject_adaptive(
                    nddata_noisy_sb, 
                    output_projection=None, # No WCS conversion needed, just reshape
                    shape_out=new_shape, 
                    fill_value=0.0, 
                    flux_conserving=True
                )[0]

                # Resample RMS image (input is surface brightness) using reproject_adaptive
                nddata_rms_sb = NDData(final_rms_surface_brightness)
                resampled_rms_image_sb = reproject_adaptive(
                    nddata_rms_sb, 
                    output_projection=None, # No WCS conversion needed, just reshape
                    shape_out=new_shape, 
                    fill_value=0.0, 
                    flux_conserving=True
                )[0]

            # Store the final resampled images (still in surface brightness units for now)
            # The convert_flux_map in save_results_to_fits will handle the final unit conversion
            # including applying the *new* pixel scale for units like MJy/sr.
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

        # Primary HDU (can be empty or hold a reference image)
        prihdr = self.hdul[0].header.copy()
        prihdr['COMMENT'] = 'Mock Observation Results'
        prihdr['NOISE_SIM'] = 'True'
        prihdr['ZP_MAG'] = self.mag_zp
        prihdr['LIM_MAG'] = self.limiting_magnitude
        prihdr['SNR_LIM'] = self.snr_limit
        prihdr['APER_RAD'] = self.aperture_radius_arcsec
        prihdr['EXP_TIME'] = self.exposure_time
        # Update the pixel scale in the header to the desired final pixel scale for the primary image
        # Assuming the first filter's desired pixel scale is representative for the primary HDU
        if self.filters and self.filters[0] in self.desired_pixel_scales:
            prihdr['PIXSIZE'] = self.desired_pixel_scales[self.filters[0]]
        else:
            prihdr['PIXSIZE'] = self.initial_pixel_scale_arcsec # Fallback

        # Determine a representative image for the primary HDU and convert it to the final unit
        if self.filters and f"dust_{self.filters[0]}_processed" in self.processed_images:
            # Data is currently in erg/s/cm2/Angstrom/arcsec^2 (surface brightness)
            primary_data_sb_erg_s_cm2_A_arcsec2 = self.processed_images[f"dust_{self.filters[0]}_processed"]

            # Convert to final flux unit using the *desired* pixel scale for this filter
            _, filter_wave_pivot_data = self._load_filter_transmission_from_paths_local(self.filters, self.filter_transmission_path)
            wave_eff = filter_wave_pivot_data[self.filters[0]]
            
            # Use the desired_pixel_scale for the conversion (e.g., to MJy/sr or nJy, which depend on pixel area)
            final_pixel_scale_for_conversion = self.desired_pixel_scales.get(self.filters[0], self.initial_pixel_scale_arcsec)
            
            # Convert surface brightness (erg/s/cm2/A/arcsec2) to flux per pixel (erg/s/cm2/A)
            # for `convert_flux_map`, which expects this as its `flux_map` input.
            primary_data_flux_per_pixel_erg_s_cm2_A = primary_data_sb_erg_s_cm2_A_arcsec2 * (final_pixel_scale_for_conversion**2)

            primary_data_final_unit = convert_flux_map(primary_data_flux_per_pixel_erg_s_cm2_A, wave_eff, to_unit=self.flux_unit, pixel_scale_arcsec=final_pixel_scale_for_conversion)

            # Apply the final scaling as done in generate_images (from galsyn_run_fsps)
            final_scaled_primary_data = primary_data_final_unit / self.flux_scale

            prihdr['BUNIT'] = self.flux_unit
            prihdr['SCALE'] = self.flux_scale # Store the scaling factor used
            hdul_out.append(fits.PrimaryHDU(data=final_scaled_primary_data, header=prihdr))
        else:
            hdul_out.append(fits.PrimaryHDU(header=prihdr))


        # Add processed (noisy and resampled) images
        for f_name in self.filters:
            key_processed = f"{'dust' if dust_attenuation else 'nodust'}_{f_name}_processed"
            if key_processed in self.processed_images:
                # Data is currently in erg/s/cm2/Angstrom/arcsec^2 (surface brightness)
                img_data_sb_erg_s_cm2_A_arcsec2 = self.processed_images[key_processed]

                # Convert to final desired flux unit using the *desired* pixel scale for this filter
                _, filter_wave_pivot_data = self._load_filter_transmission_from_paths_local(self.filters, self.filter_transmission_path)
                wave_eff = filter_wave_pivot_data[f_name]
                
                final_pixel_scale_for_conversion = self.desired_pixel_scales.get(f_name, self.initial_pixel_scale_arcsec)
                
                # Convert surface brightness (erg/s/cm2/A/arcsec2) back to flux per pixel (erg/s/cm2/A)
                # for `convert_flux_map`
                img_data_flux_per_pixel_erg_s_cm2_A = img_data_sb_erg_s_cm2_A_arcsec2 * (final_pixel_scale_for_conversion**2)

                img_data_final_unit = convert_flux_map(img_data_flux_per_pixel_erg_s_cm2_A, wave_eff, to_unit=self.flux_unit, pixel_scale_arcsec=final_pixel_scale_for_conversion)

                # Apply the final scaling as in generate_images
                final_scaled_data = img_data_final_unit / self.flux_scale

                ext_hdr = fits.Header()
                ext_hdr['EXTNAME'] = f"PROCESSED_IMG_{f_name.upper()}"
                ext_hdr['FILTER'] = f_name
                ext_hdr['COMMENT'] = f'Convolved, noise-injected, and resampled image for filter: {f_name}'
                ext_hdr['BUNIT'] = self.flux_unit
                ext_hdr['SCALE'] = self.flux_scale # Consistent scale factor
                # Add the final pixel scale to the extension header
                ext_hdr['PIXSIZE'] = final_pixel_scale_for_conversion
                hdul_out.append(fits.ImageHDU(data=final_scaled_data, header=ext_hdr))

        # Add resampled RMS images
        for f_name in self.filters:
            key_rms = f"{f_name}_{'dust' if dust_attenuation else 'nodust'}_rms"
            if key_rms in self.rms_images:
                # Data is currently in erg/s/cm2/Angstrom/arcsec^2 (surface brightness)
                rms_data_sb_erg_s_cm2_A_arcsec2 = self.rms_images[key_rms]

                # Convert to final desired flux unit using the *desired* pixel scale for this filter
                _, filter_wave_pivot_data = self._load_filter_transmission_from_paths_local(self.filters, self.filter_transmission_path)
                wave_eff = filter_wave_pivot_data[f_name]

                final_pixel_scale_for_conversion = self.desired_pixel_scales.get(f_name, self.initial_pixel_scale_arcsec)
                
                # Convert surface brightness (erg/s/cm2/A/arcsec2) back to flux per pixel (erg/s/cm2/A)
                # for `convert_flux_map`
                rms_data_flux_per_pixel_erg_s_cm2_A = rms_data_sb_erg_s_cm2_A_arcsec2 * (final_pixel_scale_for_conversion**2)

                rms_data_final_unit = convert_flux_map(rms_data_flux_per_pixel_erg_s_cm2_A, wave_eff, to_unit=self.flux_unit, pixel_scale_arcsec=final_pixel_scale_for_conversion)

                # Apply the final scaling
                final_scaled_rms = rms_data_final_unit / self.flux_scale

                ext_hdr = fits.Header()
                ext_hdr['EXTNAME'] = f"RMS_IMG_{f_name.upper()}"
                ext_hdr['FILTER'] = f_name
                ext_hdr['COMMENT'] = f'RMS image for filter: {f_name}'
                ext_hdr['BUNIT'] = self.flux_unit # RMS has same units as flux
                ext_hdr['SCALE'] = self.flux_scale # Consistent scale factor
                # Add the final pixel scale to the extension header
                ext_hdr['PIXSIZE'] = final_pixel_scale_for_conversion
                hdul_out.append(fits.ImageHDU(data=final_scaled_rms, header=ext_hdr))

        output_dir = os.path.dirname(output_fits_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        hdul_out.writeto(output_fits_path, overwrite=True, output_verify='fix')
        hdul_out.close()
        print(f"Results saved to {output_fits_path}")


# --- NEW CLASS: GalSynMockObservation_ifu ---

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
        mag_zp : float
            Magnitude zero-point of the observation system. This is crucial for converting
            between flux and counts in the noise model.
        limiting_magnitude_wave_func : callable
            A function that takes wavelength (in Angstrom) as input and returns
            the limiting magnitude at that wavelength. This magnitude is assumed
            to be per area of the `final_pixel_scale_arcsec`.
        snr_limit : float
            Signal-to-noise ratio corresponding to the limiting magnitude.
        final_pixel_scale_arcsec : float
            Desired final spatial pixel size in arcsec for the output data cube.
        exposure_time : float
            Exposure time in seconds.
        """
        self.fits_file_path = fits_file_path
        self.desired_wave_grid = desired_wave_grid
        self.psf_cube_path = psf_cube_path
        self.psf_pixel_scale = psf_pixel_scale
        self.spectral_resolution_R = spectral_resolution_R
        self.mag_zp = mag_zp # Added
        self.limiting_magnitude_wave_func = limiting_magnitude_wave_func
        self.snr_limit = snr_limit
        self.final_pixel_scale_arcsec = final_pixel_scale_arcsec
        self.exposure_time = exposure_time

        self.hdul = fits.open(fits_file_path)
        self.image_header = self.hdul[0].header
        # Initial pixel scale from the input FITS cube
        self.initial_pixel_scale_arcsec = self.image_header['PIXSIZE'] if 'PIXSIZE' in self.image_header else None
        if self.initial_pixel_scale_arcsec is None:
            raise ValueError("Input FITS file header must contain 'PIXSIZE' (initial pixel scale in arcsec).")

        # Load the original wavelength grid from the FITS file
        try:
            wavelength_hdu = self.hdul['WAVELENGTH_GRID']
            self.original_wave_grid = wavelength_hdu.data['WAVELENGTH']
        except KeyError:
            raise ValueError("Input FITS file must contain a 'WAVELENGTH_GRID' binary table extension.")

        # Load the initial data cube (assuming 'OBS_SPEC_NODUST' or 'OBS_SPEC_DUST')
        # We'll load both if they exist, and the user can choose which to process.
        self.initial_datacube_nodust = None
        self.initial_datacube_dust = None
        if 'OBS_SPEC_NODUST' in self.hdul:
            # Data is (wavelength, y, x)
            self.initial_datacube_nodust = self.hdul['OBS_SPEC_NODUST'].data
        if 'OBS_SPEC_DUST' in self.hdul:
            self.initial_datacube_dust = self.hdul['OBS_SPEC_DUST'].data

        # Internal storage for processed cubes
        self.processed_datacube = None
        self.rms_datacube = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.hdul.close()

    def _load_psf_cube(self):
        """
        Loads the 3D PSF cube and ensures its wavelength axis matches desired_wave_grid.
        Also resamples each 2D PSF slice to the initial image pixel scale.
        Returns a 3D PSF cube (wavelength, y, x) ready for convolution.
        """
        if not os.path.exists(self.psf_cube_path):
            raise FileNotFoundError(f"PSF cube file not found at: {self.psf_cube_path}")

        psf_hdu = fits.open(self.psf_cube_path)
        psf_cube_data = psf_hdu[0].data # Assuming primary HDU contains the cube
        psf_hdu.close()

        # Assuming PSF cube data is (wavelength, y, x)
        if psf_cube_data.ndim != 3:
            raise ValueError(f"PSF cube must be 3D (wavelength, y, x), but has {psf_cube_data.ndim} dimensions.")

        # For simplicity, assume the PSF cube's wavelength axis is implicitly aligned with desired_wave_grid.
        # In a more robust implementation, you might need to interpolate the PSF cube spectrally.
        # For now, we'll just check if the number of wavelength channels matches.
        if psf_cube_data.shape[0] != len(self.desired_wave_grid):
            raise ValueError(f"PSF cube wavelength channels ({psf_cube_data.shape[0]}) "
                             f"must match desired_wave_grid length ({len(self.desired_wave_grid)}).")

        resampled_psf_cube = np.zeros_like(psf_cube_data)
        for i_wave in range(psf_cube_data.shape[0]):
            psf_slice = psf_cube_data[i_wave, :, :]
            if not np.isclose(self.psf_pixel_scale, self.initial_pixel_scale_arcsec):
                resampled_psf = resize_psf(psf_slice, self.psf_pixel_scale, self.initial_pixel_scale_arcsec)
                resampled_psf /= np.sum(resampled_psf) # Normalize
            else:
                resampled_psf = psf_slice / np.sum(psf_slice) # Normalize
            resampled_psf_cube[i_wave, :, :] = resampled_psf

        return resampled_psf_cube


    def process_datacube(self, dust_attenuation=True, apply_noise_to_cube=True):
        """
        Executes the full pipeline of observational effects for the IFU data cube:
        1. Cuts/interpolates the data cube to the desired wavelength grid.
        2. Smooths each spectrum to the desired spectral resolution (R).
        3. Convolves each 2D wavelength slice with its corresponding spatial PSF.
        4. Simulates and injects noise per pixel per wavelength channel.
        5. Creates an RMS data cube.
        6. Resamples the data cube spatially to the final desired pixel scale.

        Parameters:
        -----------
        dust_attenuation : bool
            Whether to process the data cube with dust attenuation (True) or without (False).
        apply_noise_to_cube : bool
            If True, noise is added to the processed cube. Otherwise, only RMS is calculated.
        """
        print("\nStarting full IFU data cube processing pipeline...")

        if dust_attenuation and self.initial_datacube_dust is not None:
            input_datacube = self.initial_datacube_dust
            cube_name = "DUST"
        elif not dust_attenuation and self.initial_datacube_nodust is not None:
            input_datacube = self.initial_datacube_nodust
            cube_name = "NODUST"
        else:
            raise ValueError(f"Requested data cube (dust_attenuation={dust_attenuation}) not found in FITS file.")

        # Initial cube dimensions: (wavelength, y, x)
        initial_n_wave, initial_ny, initial_nx = input_datacube.shape
        print(f"  Initial data cube shape: {input_datacube.shape} (Angstrom, Y, X)")

        # --- 1. Cut/Interpolate Wavelength Grid ---
        print("  Cutting/interpolating data cube to desired wavelength grid...")
        # Create an interpolation function for each spatial pixel's spectrum
        # input_datacube is (wave, y, x)
        # We want to interpolate along the wave axis for each (y, x) pixel.
        
        # Reshape for interp1d: (y*x, wave)
        reshaped_cube = input_datacube.reshape(initial_n_wave, -1).T # (y*x, wave)
        
        # Interpolate each spectrum onto the desired_wave_grid
        interpolated_cube_reshaped = np.zeros((reshaped_cube.shape[0], len(self.desired_wave_grid)))
        for i in range(reshaped_cube.shape[0]):
            interp_func = interp1d(self.original_wave_grid, reshaped_cube[i, :], kind='linear',
                                   bounds_error=False, fill_value=0.0)
            interpolated_cube_reshaped[i, :] = interp_func(self.desired_wave_grid)
        
        # Reshape back to (desired_wave, y, x)
        processed_cube_wave_cut = interpolated_cube_reshaped.T.reshape(len(self.desired_wave_grid), initial_ny, initial_nx)
        print(f"  Data cube cut to wavelength grid. New shape: {processed_cube_wave_cut.shape}")

        # --- Convert to Surface Brightness for Internal Processing ---
        # Data from galsyn_run_fsps is erg/s/cm2/Angstrom (spectral flux density)
        # To handle spatial convolution and resampling correctly, we convert to
        # spectral surface brightness: erg/s/cm^2/Angstrom/arcsec^2
        pixel_area_arcsec2_initial = self.initial_pixel_scale_arcsec**2
        if pixel_area_arcsec2_initial <= 0:
            raise ValueError("Initial pixel area must be positive.")
        
        # Convert each slice to surface brightness
        processed_cube_sb = processed_cube_wave_cut / pixel_area_arcsec2_initial
        print(f"  Data cube converted to spectral surface brightness (erg/s/cm^2/Angstrom/arcsec^2).")


        # --- 3. Spectral Smoothing ---
        print(f"  Smoothing spectra to R={self.spectral_resolution_R}...")
        smoothed_cube_sb = np.zeros_like(processed_cube_sb)
        for i_y in range(processed_cube_sb.shape[1]):
            for i_x in range(processed_cube_sb.shape[2]):
                spectrum = processed_cube_sb[:, i_y, i_x]
                
                # Calculate FWHM of Gaussian kernel for desired R
                # FWHM = lambda / R
                # sigma_lambda = FWHM / (2 * sqrt(2 * ln(2)))
                # sigma_pixels = sigma_lambda / delta_lambda_per_pixel
                
                # Calculate delta_lambda (average spacing) for the desired_wave_grid
                if len(self.desired_wave_grid) > 1:
                    delta_lambda_per_pixel = np.mean(np.diff(self.desired_wave_grid))
                else:
                    delta_lambda_per_pixel = 1.0 # Fallback for single wavelength point

                # Calculate sigma in wavelength units
                # Using the desired_wave_grid to get per-channel sigma
                sigma_lambda_per_channel = self.desired_wave_grid / self.spectral_resolution_R / (2 * np.sqrt(2 * np.log(2)))
                sigma_pixels_per_channel = sigma_lambda_per_channel / delta_lambda_per_pixel

                # Apply Gaussian smoothing to each spectrum
                # For simplicity, let's use the mean sigma for the whole spectrum.
                mean_sigma_pixels = np.mean(sigma_pixels_per_channel)
                if mean_sigma_pixels > 0: # Only smooth if sigma is meaningful
                    gauss_kernel = Gaussian1DKernel(stddev=mean_sigma_pixels)
                    smoothed_spectrum = convolve_fft(spectrum, gauss_kernel, boundary='fill', fill_value=0.0)
                else:
                    smoothed_spectrum = spectrum # No smoothing if R is extremely high or grid is too coarse

                smoothed_cube_sb[:, i_y, i_x] = smoothed_spectrum
        print("  Spectra smoothed.")

        # --- 4. Spatial PSF Convolution ---
        print("  Convolving each spatial slice with PSF...")
        psf_cube = self._load_psf_cube() # (wave, y, x)
        convolved_cube_sb = np.zeros_like(smoothed_cube_sb)
        
        if psf_cube.shape[0] != smoothed_cube_sb.shape[0]:
            raise ValueError("Wavelength dimensions of PSF cube and data cube do not match after spectral cut/smoothing.")

        for i_wave in range(smoothed_cube_sb.shape[0]):
            spatial_slice_sb = smoothed_cube_sb[i_wave, :, :]
            psf_slice = psf_cube[i_wave, :, :]
            
            convolved_slice_sb = convolve_fft(spatial_slice_sb, psf_slice, boundary='fill', fill_value=0.0)
            convolved_cube_sb[i_wave, :, :] = convolved_slice_sb
        print("  Spatial PSF convolution complete.")


        # --- 5. Noise Simulation and Injection + RMS Cube Creation ---
        print("  Simulating and injecting noise...")
        noisy_cube_sb = np.zeros_like(convolved_cube_sb)
        rms_cube_sb = np.zeros_like(convolved_cube_sb)

        # Constant for AB magnitude to f_nu conversion
        c_angstrom_s = 2.99792458e18 # speed of light in Angstrom/s
        
        # Pixel area in arcsec^2 for the *final* desired pixel scale, for noise calculation
        pixel_area_arcsec2_final = self.final_pixel_scale_arcsec**2
        if pixel_area_arcsec2_final <= 0:
            raise ValueError("Final pixel area must be positive for noise calculation.")
        
        # Loop over each wavelength channel
        for i_wave in range(convolved_cube_sb.shape[0]):
            current_wave = self.desired_wave_grid[i_wave]
            
            # Limiting magnitude at this wavelength (per area of desired final pixel size)
            lim_mag_at_wave = self.limiting_magnitude_wave_func(current_wave)
            
            # This is the flux *in one pixel* (of `final_pixel_scale_arcsec` area)
            # that corresponds to 1 count/second, based on `self.mag_zp`.
            # Re-deriving `flux_per_total_count_per_A_per_pixel` for this wavelength using `self.mag_zp`
            # This factor represents the flux (erg/s/cm2/A/pixel) that corresponds to one total count
            # over the given exposure time, based on the ZP definition.
            mag_for_1_count = self.mag_zp - 2.5 * np.log10(1.0 / self.exposure_time)
            f_nu_erg_s_cm2_Hz_for_1_count = 10**((mag_for_1_count + 48.6)/(-2.5))
            flux_per_total_count_per_A_per_pixel = f_nu_erg_s_cm2_Hz_for_1_count * c_angstrom_s / current_wave**2


            # Convert current slice SB to flux per final pixel for noise calculation
            current_slice_flux_per_final_pixel = convolved_cube_sb[i_wave, :, :] * pixel_area_arcsec2_final

            # Expected source counts per pixel for this wavelength slice
            source_counts_per_pixel_expected = current_slice_flux_per_final_pixel / flux_per_total_count_per_A_per_pixel
            source_counts_per_pixel_expected = np.maximum(0, source_counts_per_pixel_expected)

            # Background counts variance (sigma_bg^2) derived from limiting magnitude and SNR
            # C_lim_at_wave is counts in a pixel at the limiting magnitude for this wavelength
            # C_lim_at_wave = self.exposure_time * 10**(0.4 * (self.mag_zp - lim_mag_at_wave))
            # The lim_mag_at_wave is already defined as per area of desired final pixel size.
            # So, this `lim_mag_at_wave` should be interpreted as the magnitude for a source
            # that produces `C_lim_at_wave` counts in a single pixel.
            
            # If `lim_mag_at_wave` is an AB magnitude, the flux density (f_nu) is:
            f_nu_lim_erg_s_cm2_Hz = 10**((lim_mag_at_wave + 48.6)/(-2.5))
            # The counts associated with this limiting flux density for *one pixel* over `exposure_time`
            # C_lim_per_pixel_from_flux = f_nu_lim_erg_s_cm2_Hz * (exposure_time in sec) / (conversion factor from f_nu to counts/sec)
            # A simpler way, consistent with imaging class, is to calculate `C_aperture` as counts for `lim_mag`
            # using the `mag_zp`.
            
            C_aperture_at_wave = self.exposure_time * (10**(0.4 * (self.mag_zp - lim_mag_at_wave)))
            C_aperture_at_wave = np.maximum(0, C_aperture_at_wave)

            sigma_bg_counts_sq_per_pixel = (C_aperture_at_wave / self.snr_limit)**2 - C_aperture_at_wave
            sigma_bg_counts_sq_per_pixel = np.maximum(0, sigma_bg_counts_sq_per_pixel)
            sigma_bg_counts_per_pixel = np.sqrt(sigma_bg_counts_sq_per_pixel)


            # Generate noisy counts
            noisy_counts_slice = source_counts_per_pixel_expected.copy()
            if apply_noise_to_cube:
                photon_shot_noise_sampled_counts = stats.poisson.rvs(source_counts_per_pixel_expected)
                total_noisy_counts_slice = photon_shot_noise_sampled_counts + np.random.normal(0, sigma_bg_counts_per_pixel, size=source_counts_per_pixel_expected.shape)
                total_noisy_counts_slice = np.maximum(0, total_noisy_counts_slice)
                noisy_counts_slice = total_noisy_counts_slice

            # Convert noisy counts back to flux per final pixel, then to surface brightness
            noisy_cube_sb[i_wave, :, :] = (noisy_counts_slice * flux_per_total_count_per_A_per_pixel) / pixel_area_arcsec2_final
            noisy_cube_sb[i_wave, :, :] = np.maximum(1e-30, noisy_cube_sb[i_wave, :, :]) # Ensure non-negative

            # Calculate RMS in surface brightness units
            total_variance_counts_slice = source_counts_per_pixel_expected + sigma_bg_counts_sq_per_pixel
            total_rms_counts_per_pixel_slice = np.sqrt(total_variance_counts_slice)
            rms_cube_sb[i_wave, :, :] = (total_rms_counts_per_pixel_slice * flux_per_total_count_per_A_per_pixel) / pixel_area_arcsec2_final
            rms_cube_sb[i_wave, :, :] = np.maximum(1e-30, rms_cube_sb[i_wave, :, :]) # Ensure non-negative
        print("  Noise simulated and injected.")


        # --- 6. Spatial Resampling to Final Pixel Scale ---
        print(f"  Resampling data cube spatially to {self.final_pixel_scale_arcsec:.4f} arcsec...")
        if np.isclose(self.final_pixel_scale_arcsec, self.initial_pixel_scale_arcsec):
            print(f"  Final pixel scale is already {self.final_pixel_scale_arcsec:.4f} arcsec. No spatial resampling needed.")
            resampled_processed_cube_sb = noisy_cube_sb
            resampled_rms_cube_sb = rms_cube_sb
        else:
            old_ny, old_nx = noisy_cube_sb.shape[1:]
            resampling_factor = self.initial_pixel_scale_arcsec / self.final_pixel_scale_arcsec
            new_nx = int(np.round(old_nx * resampling_factor))
            new_ny = int(np.round(old_ny * resampling_factor))
            new_spatial_shape = (new_ny, new_nx)

            resampled_processed_cube_sb = np.zeros((noisy_cube_sb.shape[0], new_ny, new_nx))
            resampled_rms_cube_sb = np.zeros((rms_cube_sb.shape[0], new_ny, new_nx))

            for i_wave in range(noisy_cube_sb.shape[0]):
                # Resample noisy image slice (input is surface brightness)
                nddata_noisy_slice_sb = NDData(noisy_cube_sb[i_wave, :, :])
                resampled_noisy_nddata = reproject_adaptive(
                    nddata_noisy_slice_sb, 
                    output_projection=None, 
                    shape_out=new_spatial_shape, 
                    fill_value=0.0, 
                    flux_conserving=True
                )[0]
                resampled_processed_cube_sb[i_wave, :, :] = resampled_noisy_nddata

                # Resample RMS image slice (input is surface brightness)
                nddata_rms_slice_sb = NDData(rms_cube_sb[i_wave, :, :])
                resampled_rms_nddata = reproject_adaptive(
                    nddata_rms_slice_sb, 
                    output_projection=None, 
                    shape_out=new_spatial_shape, 
                    fill_value=0.0, 
                    flux_conserving=True
                )[0]
                resampled_rms_cube_sb[i_wave, :, :] = resampled_rms_nddata
        print("  Spatial resampling complete.")

        self.processed_datacube = resampled_processed_cube_sb
        self.rms_datacube = resampled_rms_cube_sb
        print("\nFull IFU data cube processing pipeline complete.")


    def save_results_to_fits(self, output_fits_path, flux_unit='erg/s/cm2/A'):
        """
        Saves the processed (noise-injected, smoothed, convolved, resampled) data cube
        and the RMS data cube to a new FITS file.

        Parameters:
        -----------
        output_fits_path : str
            Path to the output FITS file.
        flux_unit : str, optional
            Desired flux unit for the saved data. Options are: 'MJy/sr', 'nJy',
            'AB magnitude', or 'erg/s/cm2/A'. Defaults to 'erg/s/cm2/A'.
        """
        print(f"\nSaving IFU results to FITS file: {output_fits_path}...")
        hdul_out = fits.HDUList()

        # Primary HDU (can be empty or hold a reference image)
        prihdr = self.image_header.copy() # Copy original header
        prihdr['COMMENT'] = 'Mock IFU Observation Results'
        prihdr['NOISE_SIM'] = 'True'
        prihdr['RES_R'] = self.spectral_resolution_R
        prihdr['SNR_LIM'] = self.snr_limit
        prihdr['EXP_TIME'] = self.exposure_time
        prihdr['PIXSIZE'] = self.final_pixel_scale_arcsec # Update to final pixel scale
        prihdr['ZP_MAG'] = self.mag_zp # Add magnitude zero-point to header

        # Assuming the original FITS file's primary HDU might have some data,
        # but for an IFU output, the main data is in extensions.
        # We can make a blank primary HDU or use a representative slice.
        # For simplicity, let's create a blank primary HDU.
        hdul_out.append(fits.PrimaryHDU(header=prihdr))

        if self.processed_datacube is not None:
            # Data is currently in erg/s/cm2/Angstrom/arcsec^2 (surface brightness)
            # We need to convert it to the desired `flux_unit`.
            # `convert_flux_map` expects `flux_map` in `erg/s/cm2/A` (spectral flux per pixel).
            # So, convert SB back to flux per pixel using the *final* pixel scale.
            
            # Transpose from (wave, y, x) to (y, x, wave) for easier per-pixel conversion
            # Then transpose back to (wave, y, x) for saving
            
            # Calculate pixel area for the final output
            pixel_area_arcsec2_final = self.final_pixel_scale_arcsec**2

            # Processed Data Cube
            processed_cube_flux_per_pixel = self.processed_datacube * pixel_area_arcsec2_final
            final_processed_cube = np.zeros_like(processed_cube_flux_per_pixel)

            for i_wave in range(processed_cube_flux_per_pixel.shape[0]):
                current_wave = self.desired_wave_grid[i_wave]
                # convert_flux_map expects a 2D array for flux_map
                converted_slice = convert_flux_map(processed_cube_flux_per_pixel[i_wave, :, :],
                                                   current_wave,
                                                   to_unit=flux_unit,
                                                   pixel_scale_arcsec=self.final_pixel_scale_arcsec)
                final_processed_cube[i_wave, :, :] = converted_slice
            
            # Apply the global flux_scale from the original imaging FITS header
            # This scale is usually 1.0 or 1e-20 depending on the flux_unit.
            # We need to ensure it's applied consistently.
            # The `convert_flux_map` already handles the unit conversion.
            # The `flux_scale` from the original header is usually for the *output* of galsyn_run_fsps.
            # Let's assume it should be applied to the final converted data.
            # If the output unit is 'erg/s/cm2/A', the scale is 1e-20. Otherwise 1.0.
            # We should derive it here to be safe.
            if flux_unit == 'erg/s/cm2/A':
                output_flux_scale = 1e-20
            else:
                output_flux_scale = 1.0
            
            final_processed_cube /= output_flux_scale

            ext_hdr_proc = fits.Header()
            ext_hdr_proc['EXTNAME'] = 'PROCESSED_IFU_CUBE'
            ext_hdr_proc['COMMENT'] = 'Processed IFU Data Cube (Smoothed, Convolved, Noisy, Resampled)'
            ext_hdr_proc['BUNIT'] = flux_unit
            ext_hdr_proc['SCALE'] = output_flux_scale
            # Add WCS information for the cube (wavelength, y, x)
            ext_hdr_proc['CRPIX1'] = 1.0 # Wavelength axis
            ext_hdr_proc['CRVAL1'] = self.desired_wave_grid[0]
            ext_hdr_proc['CDELT1'] = (self.desired_wave_grid[1] - self.desired_wave_grid[0]) if len(self.desired_wave_grid) > 1 else 0.0
            ext_hdr_proc['CUNIT1'] = 'Angstrom'

            ext_hdr_proc['CRPIX2'] = final_processed_cube.shape[1] / 2.0 + 0.5 # Y-axis
            ext_hdr_proc['CDELT2'] = self.final_pixel_scale_arcsec
            ext_hdr_proc['CUNIT2'] = 'arcsec'

            ext_hdr_proc['CRPIX3'] = final_processed_cube.shape[2] / 2.0 + 0.5 # X-axis
            ext_hdr_proc['CDELT3'] = self.final_pixel_scale_arcsec
            ext_hdr_proc['CUNIT3'] = 'arcsec'

            hdul_out.append(fits.ImageHDU(data=final_processed_cube, header=ext_hdr_proc))

        if self.rms_datacube is not None:
            # RMS cube is also in erg/s/cm2/Angstrom/arcsec^2 (surface brightness)
            rms_cube_flux_per_pixel = self.rms_datacube * pixel_area_arcsec2_final
            final_rms_cube = np.zeros_like(rms_cube_flux_per_pixel)

            for i_wave in range(rms_cube_flux_per_pixel.shape[0]):
                current_wave = self.desired_wave_grid[i_wave]
                converted_slice = convert_flux_map(rms_cube_flux_per_pixel[i_wave, :, :],
                                                   current_wave,
                                                   to_unit=flux_unit,
                                                   pixel_scale_arcsec=self.final_pixel_scale_arcsec)
                final_rms_cube[i_wave, :, :] = converted_slice
            
            final_rms_cube /= output_flux_scale

            ext_hdr_rms = fits.Header()
            ext_hdr_rms['EXTNAME'] = 'RMS_IFU_CUBE'
            ext_hdr_rms['COMMENT'] = 'RMS Data Cube'
            ext_hdr_rms['BUNIT'] = flux_unit
            ext_hdr_rms['SCALE'] = output_flux_scale
            # Add WCS information for the cube (wavelength, y, x)
            ext_hdr_rms['CRPIX1'] = 1.0 # Wavelength axis
            ext_hdr_rms['CRVAL1'] = self.desired_wave_grid[0]
            ext_hdr_rms['CDELT1'] = (self.desired_wave_grid[1] - self.desired_wave_grid[0]) if len(self.desired_wave_grid) > 1 else 0.0
            ext_hdr_rms['CUNIT1'] = 'Angstrom'

            ext_hdr_rms['CRPIX2'] = final_rms_cube.shape[1] / 2.0 + 0.5 # Y-axis
            ext_hdr_rms['CDELT2'] = self.final_pixel_scale_arcsec
            ext_hdr_rms['CUNIT2'] = 'arcsec'

            ext_hdr_rms['CRPIX3'] = final_rms_cube.shape[2] / 2.0 + 0.5 # X-axis
            ext_hdr_rms['CDELT3'] = self.final_pixel_scale_arcsec
            ext_hdr_rms['CUNIT3'] = 'arcsec'

            hdul_out.append(fits.ImageHDU(data=final_rms_cube, header=ext_hdr_rms))

        # Add the desired wavelength grid as a binary table extension
        if len(self.desired_wave_grid) > 0:
            col = fits.Column(name='WAVELENGTH', format='D', array=self.desired_wave_grid)
            cols = fits.ColDefs([col])
            wavelength_hdu = fits.BinTableHDU.from_columns(cols, name='WAVELENGTH_GRID_FINAL')
            wavelength_hdu.header['BUNIT'] = 'Angstrom'
            wavelength_hdu.header['COMMENT'] = 'Final wavelength grid for IFU cube'
            hdul_out.append(wavelength_hdu)
        else:
            print("Warning: No final wavelength grid data to save for WAVELENGTH_GRID_FINAL extension.")

        output_dir = os.path.dirname(output_fits_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        hdul_out.writeto(output_fits_path, overwrite=True, output_verify='fix')
        hdul_out.close()
        print(f"IFU results saved to {output_fits_path}")