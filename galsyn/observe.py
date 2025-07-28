import os
import numpy as np
from astropy.io import fits
from astropy.convolution import convolve_fft
from photutils.psf.matching import resize_psf
from scipy.ndimage import gaussian_filter
from scipy.integrate import simpson
from scipy import stats
from .imgutils import *

class GalSynMockObservation:
    """
    A class to simulate observational effects on synthetic galaxy images,
    including PSF convolution, noise injection, and RMS image generation.
    """

    def __init__(self, fits_file_path, filters, psf_paths, psf_pixel_scales,
                 mag_zp, limiting_magnitude, snr_limit, aperture_radius_arcsec, exposure_time,
                 filter_transmission_path):
        """
        Initializes the GalSynMockObservation with input parameters.

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

        self.hdul = fits.open(fits_file_path)
        self.image_header = self.hdul[0].header
        self.pixel_scale_kpc = self.image_header['PIX_KPC']
        self.pixel_scale_arcsec = self.image_header['PIXSIZE']
        self.flux_unit = self.image_header['BUNIT']
        self.flux_scale = self.image_header['SCALE']

        self.processed_images = {}
        self.rms_images = {}

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.hdul.close()

    def _get_flux_data(self, filter_name, dust_attenuation=True):
        """
        Retrieves flux data for a given filter from the FITS file.
        This function converts the stored flux (in its original `self.flux_unit`)
        back to `erg/s/cm^2/Angstrom` for internal calculations.
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
            # back to `erg/s/cm^2/Angstrom`.

            # Start with the original converted flux before the final scaling by `flux_scale`
            converted_flux_original = flux_data_in_fits_unit * self.flux_scale

            if self.flux_unit == 'erg/s/cm2/A':
                return converted_flux_original # Already in desired units

            elif self.flux_unit == 'nJy':
                # f_nu [erg/s/cm^2/Hz] = nJy / 1e9 * 1e-23
                f_nu_erg_s_cm2_Hz = converted_flux_original / 1e9 * 1e-23
                # erg/s/cm^2/Angstrom = f_nu * c / wave_eff^2
                flux_erg_s_cm2_A = f_nu_erg_s_cm2_Hz * c_angstrom_s / wave_eff**2
                return flux_erg_s_cm2_A

            elif self.flux_unit == 'AB magnitude':
                # For AB magnitude, the `flux_scale` in galsyn_run_fsps is 1.0, so `converted_flux_original` is just the AB mag.
                # f_nu [erg/s/cm^2/Hz] = 10^((AB_mag + 48.6)/(-2.5))
                f_nu_erg_s_cm2_Hz = 10**((converted_flux_original + 48.6)/(-2.5))
                flux_erg_s_cm2_A = f_nu_erg_s_cm2_Hz * c_angstrom_s / wave_eff**2
                return flux_erg_s_cm2_A

            elif self.flux_unit == 'MJy/sr':
                # f_nu_jy = MJy/sr * pixel_area_sr * 1e6
                pixel_area_sr = (self.pixel_scale_arcsec * np.pi / (180.0 * 3600.0))**2
                f_nu_jy = converted_flux_original * pixel_area_sr * 1e6
                # f_nu [erg/s/cm^2/Hz] = f_nu_jy * 1e-23
                f_nu_erg_s_cm2_Hz = f_nu_jy * 1e-23
                flux_erg_s_cm2_A = f_nu_erg_s_cm2_Hz * c_angstrom_s / wave_eff**2
                return flux_erg_s_cm2_A
            else:
                raise ValueError(f"Unsupported flux_unit for inverse conversion: {self.flux_unit}")

        except KeyError:
            raise ValueError(f"Filter {filter_name} with dust_attenuation={dust_attenuation} not found in FITS file. "
                             f"Available extensions: {[hdu.name for hdu in self.hdul]}.")


    def _get_psf(self, filter_name):
        """
        Loads and resamples the PSF image to match the synthetic image's pixel scale.
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

        if not np.isclose(psf_pixel_scale_arcsec, self.pixel_scale_arcsec):
            print(f"Resampling PSF for {filter_name}: PSF pixel scale ({psf_pixel_scale_arcsec:.4f} arcsec) "
                  f"differs from image pixel scale ({self.pixel_scale_arcsec:.4f} arcsec).")

            resampled_psf = resize_psf(psf_data, psf_pixel_scale_arcsec, self.pixel_scale_arcsec)

            # Ensure the resampled PSF sums to 1 (normalization)
            resampled_psf /= np.sum(resampled_psf)
            return resampled_psf
        else:
            # Ensure the PSF sums to 1 (normalization) if no resampling
            psf_data /= np.sum(psf_data)
            return psf_data

    def convolve_with_psf(self, dust_attenuation=True):
        """
        Convolves each image with its corresponding PSF.
        """
        print("\nStarting PSF convolution...")
        for f_name in self.filters:
            print(f"  Convolving {f_name} image...")
            image_data = self._get_flux_data(f_name, dust_attenuation)
            psf_data = self._get_psf(f_name)

            convolved_image = convolve_fft(image_data, psf_data, boundary='fill', fill_value=0.0)

            # Store convolved image, still in erg/s/cm2/A
            key = f"{'dust' if dust_attenuation else 'nodust'}_{f_name}_convolved"
            self.processed_images[key] = convolved_image
            print(f"  {f_name} image convolved.")
        print("PSF convolution complete.")

    def simulate_noise(self, dust_attenuation=True, apply_noise_to_image=True):
        """
        Simulates background noise and photon shot noise, then injects them into the images.
        Also calculates and stores RMS images.

        Parameters:
        -----------
        dust_attenuation : bool
            Whether to process images with dust attenuation (True) or without (False).
        apply_noise_to_image : bool
            If True, noise is added to the convolved image. Otherwise, only RMS is calculated.
        """
        print("\nStarting noise simulation and injection...")
        for f_name in self.filters:
            key_convolved = f"{'dust' if dust_attenuation else 'nodust'}_{f_name}_convolved"
            if key_convolved not in self.processed_images:
                raise ValueError(f"Convolved image for {f_name} (dust_attenuation={dust_attenuation}) not found. "
                                 f"Run convolve_with_psf() first.")

            image_flux_erg_s_cm2_A = self.processed_images[key_convolved]

            # Get effective wavelength for this filter to convert flux
            _, filter_wave_pivot_data = self._load_filter_transmission_from_paths_local(self.filters, self.filter_transmission_path)
            wave_eff = filter_wave_pivot_data[f_name]

            # 1. Estimate background RMS (sigma_bg)
            # C_aperture: Total counts in the aperture at limiting magnitude for exposure_time
            C_aperture = self.exposure_time * (10**(0.4 * (self.mag_zp - self.limiting_magnitude)))

            # sigma_bg_aperture: standard deviation of background counts over the aperture
            sigma_bg_aperture_sq = (C_aperture / self.snr_limit)**2 - C_aperture
            sigma_bg_aperture_sq = np.maximum(0, sigma_bg_aperture_sq) # Ensure non-negative
            sigma_bg_aperture = np.sqrt(sigma_bg_aperture_sq)

            # Convert sigma_bg from aperture area to per pixel
            aperture_area_pix2 = np.pi * (self.aperture_radius_arcsec / self.pixel_scale_arcsec)**2
            if aperture_area_pix2 <= 0:
                raise ValueError("Aperture area per pixel must be positive.")
            sigma_bg_per_pixel = sigma_bg_aperture / np.sqrt(aperture_area_pix2) # Stddev of background counts per pixel

            # 2. Convert pixel flux to AB magnitude, then to counts
            c_angstrom_s = 2.99792458e18 # speed of light in Angstrom/s

            # Convert flux (erg/s/cm2/A) to f_nu (erg/s/cm2/Hz)
            f_nu_erg_s_cm2_Hz_pixel = image_flux_erg_s_cm2_A * wave_eff**2 / c_angstrom_s

            # Convert f_nu (erg/s/cm2/Hz) to AB magnitude for each pixel
            # AB magnitude: -2.5 * log10(f_nu [erg/s/cm^2/Hz]) - 48.6
            pixel_mag_AB = -2.5 * np.log10(np.clip(f_nu_erg_s_cm2_Hz_pixel, 1e-50, None)) - 48.6

            # Calculate expected source counts per pixel using the provided formula
            # Counts = t * 10^(0.4*(ZP-mag))
            source_counts_per_pixel_expected = self.exposure_time * (10**(0.4 * (self.mag_zp - pixel_mag_AB)))
            source_counts_per_pixel_expected = np.maximum(0, source_counts_per_pixel_expected) # Ensure non-negative counts

            # Calculate the standard deviation of Poisson shot noise (for RMS calculation)
            # For Poisson distribution, variance = mean, so stddev = sqrt(mean)
            photon_shot_noise_sigma_counts = np.sqrt(source_counts_per_pixel_expected)

            # Determine the conversion factor from counts to flux (erg/s/cm2/A)
            # This factor represents the flux (erg/s/cm2/A) that corresponds to one count
            # over the given exposure time, based on the ZP definition.
            # We can derive it by inverting the counts formula:
            # If 1 count = exposure_time * 10^(0.4 * (ZP - mag_for_1_count))
            # Then mag_for_1_count = ZP - 2.5 * log10(1 / exposure_time)
            # And then convert mag_for_1_count back to flux (erg/s/cm2/A)
            mag_for_1_count = self.mag_zp - 2.5 * np.log10(1.0 / self.exposure_time)
            
            # Convert mag_for_1_count to f_nu (erg/s/cm2/Hz)
            f_nu_erg_s_cm2_Hz_for_1_count = 10**((mag_for_1_count + 48.6)/(-2.5))
            
            # Convert f_nu to flux (erg/s/cm2/A)
            flux_per_total_count = f_nu_erg_s_cm2_Hz_for_1_count * c_angstrom_s / wave_eff**2

            # Initialize noisy_image_flux with the original convolved image
            # This ensures that if apply_noise_to_image is False, it remains noiseless.
            noisy_image_flux = image_flux_erg_s_cm2_A.copy()

            if apply_noise_to_image:
                # 3. Simulate Poisson shot noise
                # Generate Poisson-distributed counts based on expected source counts
                photon_shot_noise_sampled_counts = stats.poisson.rvs(source_counts_per_pixel_expected)

                # Combine source counts (with Poisson noise) and background noise (Gaussian)
                # Background noise is modeled as zero-mean Gaussian fluctuations in counts.
                total_noisy_counts = photon_shot_noise_sampled_counts + np.random.normal(0, sigma_bg_per_pixel, size=image_flux_erg_s_cm2_A.shape)

                # Ensure non-negative total counts
                total_noisy_counts = np.maximum(0, total_noisy_counts)

                # Convert total noisy counts back to flux units (erg/s/cm2/A)
                noisy_image_flux = total_noisy_counts * flux_per_total_count

                # Ensure non-negative flux after conversion
                noisy_image_flux = np.maximum(1e-30, noisy_image_flux)

            # Store the noise-injected image (or noiseless if apply_noise_to_image is False)
            self.processed_images[f"{key_convolved}_noisy"] = noisy_image_flux

            # Calculate and store RMS image in flux units (erg/s/cm2/A)
            # Total variance in counts = variance_source (Poisson) + variance_background (Gaussian)
            # Variance_source (Poisson) = expected_source_counts
            # Variance_background (Gaussian) = sigma_bg_per_pixel^2
            total_variance_counts = source_counts_per_pixel_expected + sigma_bg_per_pixel**2
            total_rms_counts_per_pixel = np.sqrt(total_variance_counts)

            # Convert total RMS from counts to flux units
            total_rms_flux_units = total_rms_counts_per_pixel * flux_per_total_count
            self.rms_images[f"{f_name}_{'dust' if dust_attenuation else 'nodust'}_rms"] = total_rms_flux_units

            print(f"  Noise simulated and injected for {f_name}.")
        print("Noise simulation and injection complete.")

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


    def save_results_to_fits(self, output_fits_path, dust_attenuation=True):
        """
        Saves the noise-injected images and RMS images to a new FITS file.
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
        # BUNIT and SCALE will be derived from the final conversion

        # Determine a representative image for the primary HDU and convert it to the final unit
        if self.filters and f"dust_{self.filters[0]}_convolved_noisy" in self.processed_images:
            # Data is currently in erg/s/cm2/A
            primary_data_erg_s_cm2_A = self.processed_images[f"dust_{self.filters[0]}_convolved_noisy"]

            # Convert to final flux unit
            _, filter_wave_pivot_data = self._load_filter_transmission_from_paths_local(self.filters, self.filter_transmission_path)
            wave_eff = filter_wave_pivot_data[self.filters[0]]
            primary_data_final_unit = convert_flux_map(primary_data_erg_s_cm2_A, wave_eff, to_unit=self.flux_unit, pixel_scale_arcsec=self.pixel_scale_arcsec)

            # Apply the final scaling as done in generate_images
            # In generate_images, `map_flux = map_flux/flux_scale` (where flux_scale can be 1.0 or 1e-20)
            # So, the data saved in FITS is `converted_flux / flux_scale_from_galsyn`.
            # To be consistent, we should also divide by `self.flux_scale` here.
            final_scaled_primary_data = primary_data_final_unit / self.flux_scale

            prihdr['BUNIT'] = self.flux_unit
            prihdr['SCALE'] = self.flux_scale # Store the scaling factor used
            hdul_out.append(fits.PrimaryHDU(data=final_scaled_primary_data, header=prihdr))
        else:
            hdul_out.append(fits.PrimaryHDU(header=prihdr))


        # Add noisy images
        for f_name in self.filters:
            key_noisy = f"{'dust' if dust_attenuation else 'nodust'}_{f_name}_convolved_noisy"
            if key_noisy in self.processed_images:
                img_data_erg_s_cm2_A = self.processed_images[key_noisy]

                # Convert to final desired flux unit
                _, filter_wave_pivot_data = self._load_filter_transmission_from_paths_local(self.filters, self.filter_transmission_path)
                wave_eff = filter_wave_pivot_data[f_name]
                img_data_final_unit = convert_flux_map(img_data_erg_s_cm2_A, wave_eff, to_unit=self.flux_unit, pixel_scale_arcsec=self.pixel_scale_arcsec)

                # Apply the final scaling as in generate_images
                final_scaled_data = img_data_final_unit / self.flux_scale

                ext_hdr = fits.Header()
                ext_hdr['EXTNAME'] = f"NOISY_IMG_{f_name.upper()}"
                ext_hdr['FILTER'] = f_name
                ext_hdr['COMMENT'] = f'Convolved and noise-injected image for filter: {f_name}'
                ext_hdr['BUNIT'] = self.flux_unit
                ext_hdr['SCALE'] = self.flux_scale # Consistent scale factor
                hdul_out.append(fits.ImageHDU(data=final_scaled_data, header=ext_hdr))

        # Add RMS images
        for f_name in self.filters:
            key_rms = f"{f_name}_{'dust' if dust_attenuation else 'nodust'}_rms"
            if key_rms in self.rms_images:
                rms_data_erg_s_cm2_A = self.rms_images[key_rms]

                # Convert to final desired flux unit
                _, filter_wave_pivot_data = self._load_filter_transmission_from_paths_local(self.filters, self.filter_transmission_path)
                wave_eff = filter_wave_pivot_data[f_name]
                rms_data_final_unit = convert_flux_map(rms_data_erg_s_cm2_A, wave_eff, to_unit=self.flux_unit, pixel_scale_arcsec=self.pixel_scale_arcsec)

                # Apply the final scaling
                final_scaled_rms = rms_data_final_unit / self.flux_scale

                ext_hdr = fits.Header()
                ext_hdr['EXTNAME'] = f"RMS_IMG_{f_name.upper()}"
                ext_hdr['FILTER'] = f_name
                ext_hdr['COMMENT'] = f'RMS image for filter: {f_name}'
                ext_hdr['BUNIT'] = self.flux_unit # RMS has same units as flux
                ext_hdr['SCALE'] = self.flux_scale # Consistent scale factor
                hdul_out.append(fits.ImageHDU(data=final_scaled_rms, header=ext_hdr))

        output_dir = os.path.dirname(output_fits_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        hdul_out.writeto(output_fits_path, overwrite=True, output_verify='fix')
        hdul_out.close()
        print(f"Results saved to {output_fits_path}")
