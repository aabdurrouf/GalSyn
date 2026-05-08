import os
import numpy as np
from astropy.io import fits
from astropy.convolution import convolve_fft, Gaussian1DKernel
from photutils.psf_matching import resize_psf
from scipy.integrate import simpson
from scipy import stats
from scipy.interpolate import interp1d
from .imgutils import convert_flux_map
from astropy.wcs import WCS


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

        self.hdul = fits.open(fits_file_path)
        self.image_header = self.hdul[0].header
        self.initial_pixel_scale_arcsec = self.image_header['PIXSIZE']
        self.original_flux_unit = self.image_header['BUNIT']

        self.sci_images = {} 
        self.rms_images = {}

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.hdul.close()

    def _rebin_map_flux(self, data, initial_scale, final_scale):
        """
        Rebins a 2D array by an integer or fractional factor while conserving flux.
        Uses bilinear interpolation for fractional factors with empirical 
        flux normalization.
        """
        if np.isclose(initial_scale, final_scale):
            return data
        
        factor = final_scale / initial_scale
        
        # If integer factor, use optimized reshaping
        if np.isclose(factor % 1, 0):
            f = int(np.round(factor))
            y, x = data.shape
            new_y, new_x = y // f, x // f
            reshaped = data[:new_y*f, :new_x*f].reshape(new_y, f, new_x, f)
            return reshaped.sum(axis=(1, 3))
        else:
            # Fractional rebinning using bilinear interpolation
            from scipy.ndimage import zoom
            # Use order=1 for smoother sub-pixel area weighting
            resampled = zoom(data, 1/factor, order=1, prefilter=False)
            
            # Apply theoretical area scaling
            resampled *= (factor**2)
            
            # Enforce exact empirical flux conservation
            sum_in = np.nansum(data)
            sum_out = np.nansum(resampled)
            if sum_out != 0 and not np.isnan(sum_out):
                resampled *= (sum_in / sum_out)
                
            return resampled

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
        flux_data = self.hdul[ext_name].data

        _, filter_wave_pivot_data = self._load_filter_transmission_from_paths_local(self.filters, self.filter_transmission_path)
        wave_eff = filter_wave_pivot_data[filter_name]

        c_angstrom_s = 2.99792458e18     # speed of light in Angstrom/s

        if self.original_flux_unit == 'erg/s/cm2/A':
            return flux_data
        elif self.original_flux_unit == 'nJy':
            return (flux_data / 1e9 * 1e-23) * c_angstrom_s / wave_eff**2
        elif self.original_flux_unit == 'AB magnitude':
            f_nu = 10**((flux_data + 48.6)/(-2.5))
            return f_nu * c_angstrom_s / wave_eff**2
        elif self.original_flux_unit == 'MJy/sr':
            pixel_area_sr = (self.initial_pixel_scale_arcsec * np.pi / (180.0 * 3600.0))**2
            f_nu_per_pixel = flux_data * pixel_area_sr * 1e6
            return (f_nu_per_pixel * 1e-23) * c_angstrom_s / wave_eff**2
        else:
            raise ValueError(f"Unsupported unit: {self.original_flux_unit}")

    def _load_filter_transmission_from_paths_local(self, filters_list, filter_transmission_path_dict):
        filter_transmission_data = {}
        filter_wave_pivot_data = {}
        for f_name in filters_list:
            data = np.loadtxt(filter_transmission_path_dict[f_name])
            wave, trans = data[:, 0], data[:, 1]
            filter_transmission_data[f_name] = {'wave': wave, 'trans': trans}
            num = simpson(wave * trans, wave)
            den = simpson(trans / np.where(wave != 0, wave, 1e-10), wave)
            filter_wave_pivot_data[f_name] = np.sqrt(num / den) if den > 0 else np.nan
        return filter_transmission_data, filter_wave_pivot_data

    def process_images(self, dust_attenuation=None, apply_noise_to_image=True):
        print("\nStarting full image processing pipeline...")
        process_types = [True, False] if dust_attenuation is None else [dust_attenuation]
        
        for current_dust_attenuation in process_types:
            for f_name in self.filters:
                print(f"Processing Filter: {f_name}")

                # 1. Load and convert to erg/s/cm2/A per pixel
                flux_per_pixel = self._get_flux_data(f_name, current_dust_attenuation)

                # 2. Resample (Flux Conserving)
                desired_scale = self.desired_pixel_scales[f_name]
                resampled_flux = self._rebin_map_flux(flux_per_pixel, self.initial_pixel_scale_arcsec, desired_scale)

                # 3. PSF Convolution (at new pixel scale)
                psf_path = self.psf_paths[f_name]
                with fits.open(psf_path) as ph:
                    psf_data = ph[0].data
                
                # Ensure PSF pixel size matches image pixel size before convolution
                psf_resampled = resize_psf(psf_data, self.psf_pixel_scales[f_name], desired_scale)
                psf_resampled /= np.sum(psf_resampled)
                convolved_flux = convolve_fft(resampled_flux, psf_resampled, boundary='fill', fill_value=0.0)

                # 4. Noise Injection (at final resolution)
                _, filter_wave_pivot_data = self._load_filter_transmission_from_paths_local(self.filters, self.filter_transmission_path)
                wave_eff = filter_wave_pivot_data[f_name]
                c_angstrom_s = 2.99792458e18
                
                mag_zp = self.mag_zp[f_name]
                lim_mag = self.limiting_magnitude[f_name]
                snr_lim = self.snr_limit[f_name]
                exp_time = self.exposure_time[f_name]
                ap_rad = self.aperture_radius_arcsec[f_name]

                # Background noise calc
                C_ap = exp_time * (10**(0.4 * (mag_zp - lim_mag)))
                sigma_bg_ap_sq = np.maximum(0, (C_ap / snr_lim)**2 - C_ap)
                ap_area_pix2 = np.pi * (ap_rad / desired_scale)**2
                sigma_bg_per_pixel = np.sqrt(sigma_bg_ap_sq / ap_area_pix2)

                # Counts conversion
                f_nu_pixel = convolved_flux * wave_eff**2 / c_angstrom_s
                pix_mag_AB = -2.5 * np.log10(np.clip(f_nu_pixel, 1e-50, None)) - 48.6
                counts_expected = np.maximum(0, exp_time * (10**(0.4 * (mag_zp - pix_mag_AB))))

                flux_per_count = (10**((mag_zp - 2.5 * np.log10(1.0/exp_time) + 48.6)/(-2.5))) * c_angstrom_s / wave_eff**2

                if apply_noise_to_image:
                    noisy_counts = stats.poisson.rvs(counts_expected) + np.random.normal(0, sigma_bg_per_pixel, size=convolved_flux.shape)
                    final_flux_erg = noisy_counts * flux_per_count
                else:
                    final_flux_erg = convolved_flux

                final_rms_erg = np.sqrt(counts_expected + sigma_bg_per_pixel**2) * flux_per_count

                # 5. Convert back to original units
                key = f"{f_name}_{'dust' if current_dust_attenuation else 'nodust'}"
                self.sci_images[key] = convert_flux_map(final_flux_erg, wave_eff, to_unit=self.original_flux_unit, pixel_scale_arcsec=desired_scale)
                self.rms_images[f"{key}_rms"] = convert_flux_map(final_rms_erg, wave_eff, to_unit=self.original_flux_unit, pixel_scale_arcsec=desired_scale)

    def save_results_to_fits(self, output_fits_path):
        """
        Saves all processed (noise-injected and resampled) images and RMS images
        to a new FITS file, maintaining the original flux units.
        """
        hdul_out = fits.HDUList()
        prihdr = fits.Header()
        prihdr['BUNIT'] = self.original_flux_unit
        hdul_out.append(fits.PrimaryHDU(header=prihdr))

        for key, img_data in self.sci_images.items():
            ext_hdr = fits.Header()
            parts = key.rsplit('_', 1)
            ext_hdr['EXTNAME'] = f"SCI_{parts[1].upper()}_{parts[0].upper()}"
            ext_hdr['BUNIT'] = self.original_flux_unit
            hdul_out.append(fits.ImageHDU(data=img_data, header=ext_hdr))

        for key, rms_data in self.rms_images.items():
            ext_hdr = fits.Header()
            base_key = key.rsplit('_', 1)[0]
            parts = base_key.rsplit('_', 1)
            ext_hdr['EXTNAME'] = f"RMS_{parts[1].upper()}_{parts[0].upper()}"
            ext_hdr['BUNIT'] = self.original_flux_unit
            hdul_out.append(fits.ImageHDU(data=rms_data, header=ext_hdr))

        hdul_out.writeto(output_fits_path, overwrite=True)
        hdul_out.close()
        print(f"Results saved to {output_fits_path}")


class GalSynMockObservation_ifu:
    """
    A class to simulate observational effects on synthetic IFU data cubes,
    including wavelength cutting and regridding, spatial resampling, spectral smoothing, spatial PSF convolution,
    noise injection, and RMS data cube generation.

    Note: IFU input and output are strictly in erg/s/cm^2/Angstrom.

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
        
        self.hdul = fits.open(fits_file_path)
        self.image_header = self.hdul[0].header
        self.initial_pixel_scale_arcsec = self.image_header['PIXSIZE']
        
        # Synthetic IFU data is consistently erg/s/cm^2/A
        self.original_flux_unit = 'erg/s/cm2/A'
        self.original_wave_grid = self.hdul['WAVELENGTH_GRID'].data['WAVELENGTH']
        
        self.desired_wave_grid = desired_wave_grid
        self.psf_cube_path = psf_cube_path
        self.psf_pixel_scale = psf_pixel_scale
        self.spectral_resolution_R = spectral_resolution_R
        self.mag_zp = mag_zp
        self.limiting_magnitude_wave_func = limiting_magnitude_wave_func
        self.snr_limit = snr_limit
        self.final_pixel_scale_arcsec = final_pixel_scale_arcsec
        self.exposure_time = exposure_time

        self.initial_datacube_nodust = self.hdul['OBS_SPEC_NODUST'].data if 'OBS_SPEC_NODUST' in self.hdul else None
        self.initial_datacube_dust = self.hdul['OBS_SPEC_DUST'].data if 'OBS_SPEC_DUST' in self.hdul else None
        self.sci_datacubes, self.rms_datacubes = {}, {}

    def __enter__(self): return self
    def __exit__(self, exc_type, exc_val, exc_tb): self.hdul.close()

    def _get_wave_dependent_param(self, param, wavelength):
        return param(wavelength) if callable(param) else param

    def _rebin_map_flux(self, data, initial_scale, final_scale):
        """
        Improved spatial resampling using bicubic interpolation and 
        exact flux conservation, updated to resolve UserWarning.
        """
        if np.isclose(initial_scale, final_scale):
            return data
        
        factor = final_scale / initial_scale
        from scipy.ndimage import zoom
        
        # Updated with mode='grid-constant' to match grid_mode=True
        resampled = zoom(data, 1/factor, order=3, prefilter=True, 
                        grid_mode=True, mode='grid-constant')
        
        # Flux scaling based on pixel area change
        resampled *= (factor**2)
        
        # Final safety check for absolute conservation
        sum_in = np.nansum(data)
        sum_out = np.nansum(resampled)
        if sum_out != 0 and not np.isnan(sum_out):
            resampled *= (sum_in / sum_out)
                
        return resampled

    def process_datacube(self, dust_attenuation=None, apply_noise_to_cube=True):
        print("\nStarting IFU Pipeline...")
        process_types = [True, False] if dust_attenuation is None else [dust_attenuation]
        
        for current_dust_attenuation in process_types:
            cube_erg = self.initial_datacube_dust if current_dust_attenuation else self.initial_datacube_nodust
            if cube_erg is None: continue
            
            print(f"Processing IFU data for dust_attenuation={current_dust_attenuation}")

            # 1. Spectral Grid Alignment
            reshaped = cube_erg.reshape(cube_erg.shape[0], -1).T
            interp_cube_flat = np.zeros((reshaped.shape[0], len(self.desired_wave_grid)))
            for i in range(reshaped.shape[0]):
                interp_cube_flat[i] = interp1d(self.original_wave_grid, reshaped[i], 
                                              bounds_error=False, fill_value=0.0)(self.desired_wave_grid)
            cube_wave_cut = interp_cube_flat.T.reshape(len(self.desired_wave_grid), cube_erg.shape[1], cube_erg.shape[2])

            # 2. Improved Spatial Resampling
            print(f"  Resampling spatially to {self.final_pixel_scale_arcsec:.4f} arcsec...")
            sample_slice = self._rebin_map_flux(cube_wave_cut[0], self.initial_pixel_scale_arcsec, self.final_pixel_scale_arcsec)
            resampled_cube = np.zeros((len(self.desired_wave_grid), sample_slice.shape[0], sample_slice.shape[1]))
            
            for i in range(len(self.desired_wave_grid)):
                resampled_cube[i] = self._rebin_map_flux(cube_wave_cut[i], self.initial_pixel_scale_arcsec, self.final_pixel_scale_arcsec)

            # 3. Spectral Smoothing
            print(f"  Smoothing spectra to R={self.spectral_resolution_R}...")
            smoothed_cube = np.zeros_like(resampled_cube)
            if len(self.desired_wave_grid) > 1:
                delta_lambda = np.mean(np.diff(self.desired_wave_grid))
                sigma_lambda = self.desired_wave_grid / self.spectral_resolution_R / (2 * np.sqrt(2 * np.log(2)))
                mean_sigma_pix = np.mean(sigma_lambda / delta_lambda)
                if mean_sigma_pix > 0:
                    gauss_kernel = Gaussian1DKernel(stddev=mean_sigma_pix)
                    for iy in range(resampled_cube.shape[1]):
                        for ix in range(resampled_cube.shape[2]):
                            smoothed_cube[:, iy, ix] = convolve_fft(resampled_cube[:, iy, ix], gauss_kernel, boundary='fill', fill_value=0.0)
                else: smoothed_cube = resampled_cube
            else: smoothed_cube = resampled_cube

            # 4. PSF Convolution
            print("  Convolving each spatial slice with PSF...")
            with fits.open(self.psf_cube_path) as ph: psf_cube_data = ph[0].data
            convolved_cube = np.zeros_like(smoothed_cube)
            for i in range(len(self.desired_wave_grid)):
                psf_slice = resize_psf(psf_cube_data[i], self.psf_pixel_scale, self.final_pixel_scale_arcsec)
                psf_slice /= np.sum(psf_slice)
                convolved_cube[i] = convolve_fft(smoothed_cube[i], psf_slice, boundary='fill', fill_value=0.0)

            # 5. Noise Simulation
            print("  Simulating and injecting noise...")
            final_sci, final_rms = np.zeros_like(convolved_cube), np.zeros_like(convolved_cube)
            c_angstrom_s = 2.99792458e18
            factor = self.final_pixel_scale_arcsec / self.initial_pixel_scale_arcsec
            for i, wave in enumerate(self.desired_wave_grid):
                mag_zp, lim_mag, snr_lim, exp_time = self._get_wave_dependent_param(self.mag_zp, wave), self.limiting_magnitude_wave_func(wave), self._get_wave_dependent_param(self.snr_limit, wave), self._get_wave_dependent_param(self.exposure_time, wave)
                
                counts_expected = np.maximum(0, exp_time * (10**(0.4 * (mag_zp - (-2.5 * np.log10(np.clip(convolved_cube[i] * wave**2 / c_angstrom_s, 1e-50, None)) - 48.6)))))
                sigma_bg_sq = np.maximum(0, (exp_time * (10**(0.4 * (mag_zp - lim_mag))) / snr_lim / 2.3)**2 - exp_time * (10**(0.4 * (mag_zp - lim_mag)))) / (factor**2)
                flux_per_count = (10**((mag_zp - 2.5 * np.log10(1.0/exp_time) + 48.6)/(-2.5))) * c_angstrom_s / wave**2
                
                if apply_noise_to_cube: 
                    final_sci[i] = (stats.poisson.rvs(counts_expected) + np.random.normal(0, np.sqrt(sigma_bg_sq), size=counts_expected.shape)) * flux_per_count
                else: 
                    final_sci[i] = convolved_cube[i]
                final_rms[i] = np.sqrt(counts_expected + sigma_bg_sq) * flux_per_count

            key = 'dust' if current_dust_attenuation else 'nodust'
            self.sci_datacubes[key] = final_sci
            self.rms_datacubes[key] = final_rms

    def save_results_to_fits(self, output_fits_path):
        hdul_out = fits.HDUList()
        prihdr = fits.Header()
        prihdr['BUNIT'] = 'erg/s/cm2/A'
        prihdr['PIXSIZE'] = self.final_pixel_scale_arcsec
        hdul_out.append(fits.PrimaryHDU(header=prihdr))

        for key, cube in self.sci_datacubes.items():
            if cube is not None: 
                hdul_out.append(fits.ImageHDU(data=cube, header=self._create_ifu_header(f'SCI_{key.upper()}', cube.shape, 'erg/s/cm2/A')))
        for key, cube in self.rms_datacubes.items():
            if cube is not None: 
                hdul_out.append(fits.ImageHDU(data=cube, header=self._create_ifu_header(f'RMS_{key.upper()}', cube.shape, 'erg/s/cm2/A')))
        
        col = fits.Column(name='WAVELENGTH', format='D', array=self.desired_wave_grid)
        hdul_out.append(fits.BinTableHDU.from_columns([col], name='WAVELENGTH_GRID'))
        hdul_out.writeto(output_fits_path, overwrite=True)
        print(f"IFU results saved to {output_fits_path}")

    def _create_ifu_header(self, extname, shape, flux_unit):
        hdr = fits.Header()
        hdr['EXTNAME'], hdr['BUNIT'] = extname, flux_unit
        hdr['CTYPE1'], hdr['CUNIT1'] = 'WAVE', 'Angstrom'
        hdr['CRVAL1'] = self.desired_wave_grid[0]
        hdr['CDELT1'] = (self.desired_wave_grid[1] - self.desired_wave_grid[0]) if len(self.desired_wave_grid) > 1 else 0.0
        hdr['CRPIX2'], hdr['CDELT2'], hdr['CUNIT2'] = shape[1]/2.0 + 0.5, self.final_pixel_scale_arcsec, 'arcsec'
        hdr['CRPIX3'], hdr['CDELT3'], hdr['CUNIT3'] = shape[2]/2.0 + 0.5, self.final_pixel_scale_arcsec, 'arcsec'
        return hdr
    

class GalSynMockObservation_mosaic:
    """
    A class to simulate observational effects on synthetic mosaic images containing multiple galaxies, applying the same processing pipeline as
    GalSynMockObservation_imaging. Unlike GalSynMockObservation_imaging, which operates on a single
    multi-extension FITS file containing images for multiple filters, this class operates on one single-extension FITS file per filter, where the
    mosaic image is stored in the primary HDU. A separate input FITS file path must therefore be provided for each filter via `fits_file_paths`.

    Parameters
    ----------
    fits_file_paths : dict
        Keys are filter names, values are paths to the single-extension
        mosaic FITS file for that filter. The mosaic image is expected in
        the primary HDU (index 0).
    filters : list of str
        Filter names to process (e.g., ['jwst_nircam_f115w', 'jwst_nircam_f200w']).
        Each name must be a key in `fits_file_paths`.
    psf_paths : dict
        Keys are filter names, values are paths to 2D PSF FITS images.
    psf_pixel_scales : dict
        Keys are filter names, values are the pixel scale of the PSF image
        in arcsec/pixel.
    mag_zp : dict
        Keys are filter names, values are the instrumental magnitude
        zero-points.
    limiting_magnitude : dict
        Keys are filter names, values are the limiting AB magnitudes.
    snr_limit : dict
        Keys are filter names, values are the SNR achieved at the limiting
        magnitude within the specified aperture.
    aperture_radius_arcsec : dict
        Keys are filter names, values are the circular aperture radius in
        arcsec used to define the limiting magnitude.
    exposure_time : dict
        Keys are filter names, values are exposure times in seconds.
    filter_transmission_path : dict
        Keys are filter names, values are paths to ASCII files with two
        columns: wavelength (Angstrom) and transmission (0-1).
    desired_pixel_scales : dict
        Keys are filter names, values are the desired output pixel scale
        in arcsec/pixel. If equal to the input pixel scale, no spatial
        resampling is performed.

    Notes
    -----
    - The pixel scale for each mosaic is read from the 'PIXSIZE' keyword
      in its primary header. If absent, it is inferred from the WCS.
    - The flux unit is read from the 'BUNIT' keyword in the primary header.
      Supported units: 'erg/s/cm2/A', 'nJy', 'AB magnitude', 'MJy/sr'.
    - Output science images and RMS maps are stored in `self.sci_images`
      and `self.rms_images`, keyed by filter name, and can be written to
      disk with :meth:`save_results_to_fits`.

    """

    def __init__(
        self,
        fits_file_paths,
        filters,
        psf_paths,
        psf_pixel_scales,
        mag_zp,
        limiting_magnitude,
        snr_limit,
        aperture_radius_arcsec,
        exposure_time,
        filter_transmission_path,
        desired_pixel_scales,
    ):
        self.fits_file_paths        = fits_file_paths
        self.filters                = filters
        self.psf_paths              = psf_paths
        self.psf_pixel_scales       = psf_pixel_scales
        self.mag_zp                 = mag_zp
        self.limiting_magnitude     = limiting_magnitude
        self.snr_limit              = snr_limit
        self.aperture_radius_arcsec = aperture_radius_arcsec
        self.exposure_time          = exposure_time
        self.filter_transmission_path = filter_transmission_path
        self.desired_pixel_scales   = desired_pixel_scales

        # Output storage, keyed by filter name
        self.sci_images  = {}   # noise-added science images
        self.rms_images  = {}   # RMS error maps
        self.sci_headers = {}   # updated FITS headers for science images
        self.rms_headers = {}   # updated FITS headers for RMS maps

    def _read_mosaic(self, filter_name):
        """
        Opens the mosaic FITS file for *filter_name*, reads the primary
        image and header, and returns them together with the pixel scale
        and flux unit.

        The file is opened and closed within this method so that large
        mosaics are not held in memory across filters.

        Parameters
        ----------
        filter_name : str

        Returns
        -------
        flux_data          : 2D ndarray
        header             : fits.Header
        pixel_scale_arcsec : float
        flux_unit          : str
        """
        path = self.fits_file_paths[filter_name]
        with fits.open(path, memmap=False) as hdul:
            header    = hdul[0].header.copy()
            flux_data = hdul[0].data.copy().astype(float)

        # Pixel scale: prefer explicit PIXSIZE keyword, fall back to WCS
        if 'PIXSIZE' in header:
            pixel_scale_arcsec = float(header['PIXSIZE'])
        else:
            pixel_scale_arcsec = self._pixel_scale_from_wcs(header, filter_name)

        flux_unit = header.get('BUNIT', 'erg/s/cm2/A')

        return flux_data, header, pixel_scale_arcsec, flux_unit

    def _pixel_scale_from_wcs(self, header, filter_name):
        """
        Derives pixel scale in arcsec from the WCS stored in *header*.
        Raises a clear error if no usable WCS is found.
        """
        try:
            wcs = WCS(header)
            scales = wcs.proj_plane_pixel_scales()
            scale_arcsec = float(np.mean([abs(s.to('arcsec').value) for s in scales]))
            print(
                f"  [{filter_name}] 'PIXSIZE' not found — inferred pixel scale "
                f"from WCS: {scale_arcsec:.5f} arcsec/pixel"
            )
            return scale_arcsec
        except Exception as exc:
            raise ValueError(
                f"Cannot determine pixel scale for filter '{filter_name}' in "
                f"'{self.fits_file_paths[filter_name]}': 'PIXSIZE' keyword is "
                f"absent and WCS inference failed ({exc})."
            )

    def _load_filter_transmission_from_paths_local(self, filters_list, filter_transmission_path_dict):
        """
        Loads filter transmission curves and computes pivot wavelengths for every filter in *filters_list*.
        Identical to the implementation in GalSynMockObservation_imaging.

        Returns
        -------
        filter_transmission_data : dict  {filter_name: {'wave': ..., 'trans': ...}}
        filter_wave_pivot_data   : dict  {filter_name: pivot_wavelength_angstrom}
        """
        filter_transmission_data = {}
        filter_wave_pivot_data   = {}
        for f_name in filters_list:
            data = np.loadtxt(filter_transmission_path_dict[f_name])
            wave, trans = data[:, 0], data[:, 1]
            filter_transmission_data[f_name] = {'wave': wave, 'trans': trans}
            num = simpson(wave * trans, x=wave)
            den = simpson(trans / np.where(wave != 0, wave, 1e-10), x=wave)
            filter_wave_pivot_data[f_name] = np.sqrt(num / den) if den > 0 else np.nan
        return filter_transmission_data, filter_wave_pivot_data

    def _flux_to_erg_per_angstrom(self, flux_data, flux_unit, wave_eff, pixel_scale_arcsec):
        """
        Converts *flux_data* from *flux_unit* to erg/s/cm^2/Angstrom for
        internal noise calculations. Mirrors the unit handling in
        GalSynMockObservation_imaging._get_flux_data.

        Parameters
        ----------
        flux_data          : 2D ndarray
        flux_unit          : str
        wave_eff           : float   (pivot wavelength, Angstrom)
        pixel_scale_arcsec : float   (needed for MJy/sr conversion)

        Returns
        -------
        flux_erg : 2D ndarray  (erg/s/cm^2/Angstrom)
        """
        c_aa = 2.99792458e18  # speed of light in Angstrom/s

        if flux_unit == 'erg/s/cm2/A':
            return flux_data.copy()
        elif flux_unit == 'nJy':
            return (flux_data / 1e9 * 1e-23) * c_aa / wave_eff**2
        elif flux_unit == 'AB magnitude':
            f_nu = 10**((flux_data + 48.6) / (-2.5))
            return f_nu * c_aa / wave_eff**2
        elif flux_unit == 'MJy/sr':
            pixel_area_sr = (pixel_scale_arcsec * np.pi / (180.0 * 3600.0))**2
            f_nu_per_pixel = flux_data * pixel_area_sr * 1e6
            return (f_nu_per_pixel * 1e-23) * c_aa / wave_eff**2
        else:
            raise ValueError(
                f"Unsupported flux unit '{flux_unit}'. Supported: "
                "'erg/s/cm2/A', 'nJy', 'AB magnitude', 'MJy/sr'."
            )

    def _rebin_map_flux(self, data, initial_scale, final_scale):
        """
        Flux-conserving 2D resampling. Identical to the method used in
        GalSynMockObservation_imaging: exact integer block-sum for integer
        scale factors, bilinear zoom with empirical renormalisation for
        fractional factors.

        Parameters
        ----------
        data          : 2D ndarray
        initial_scale : float  (arcsec/pixel)
        final_scale   : float  (arcsec/pixel)

        Returns
        -------
        resampled : 2D ndarray  (flux conserved)
        """
        if np.isclose(initial_scale, final_scale):
            return data.copy()

        factor = final_scale / initial_scale

        if np.isclose(factor % 1, 0):
            # Exact integer downsampling via block summation
            f = int(np.round(factor))
            y, x = data.shape
            new_y, new_x = y // f, x // f
            reshaped = data[:new_y * f, :new_x * f].reshape(new_y, f, new_x, f)
            return reshaped.sum(axis=(1, 3))
        else:
            from scipy.ndimage import zoom
            resampled = zoom(data, 1.0 / factor, order=1, prefilter=False)
            resampled *= factor**2
            # Enforce exact flux conservation empirically
            sum_in  = np.nansum(data)
            sum_out = np.nansum(resampled)
            if sum_out != 0 and not np.isnan(sum_out):
                resampled *= sum_in / sum_out
            return resampled

    def _update_wcs_for_resampling(self, header, initial_scale, final_scale):
        """
        Returns a copy of *header* with WCS keywords updated to reflect
        the new pixel scale after resampling. Handles both CDELT-style
        and CD-matrix-style WCS representations. CRPIX values are scaled
        so that the sky reference position is preserved.

        Parameters
        ----------
        header        : fits.Header
        initial_scale : float  (arcsec/pixel)
        final_scale   : float  (arcsec/pixel)

        Returns
        -------
        new_header : fits.Header
        """
        new_header = header.copy()
        if np.isclose(initial_scale, final_scale):
            return new_header

        ratio = final_scale / initial_scale

        for key in ('CD1_1', 'CD1_2', 'CD2_1', 'CD2_2'):
            if key in new_header:
                new_header[key] = new_header[key] * ratio

        for key in ('CDELT1', 'CDELT2'):
            if key in new_header:
                new_header[key] = new_header[key] * ratio

        for key in ('CRPIX1', 'CRPIX2'):
            if key in new_header:
                new_header[key] = (new_header[key] - 0.5) / ratio + 0.5

        if 'PIXSIZE' in new_header:
            new_header['PIXSIZE'] = final_scale

        if 'NAXIS1' in new_header:
            new_header['NAXIS1'] = int(np.round(new_header['NAXIS1'] / ratio))
        if 'NAXIS2' in new_header:
            new_header['NAXIS2'] = int(np.round(new_header['NAXIS2'] / ratio))

        return new_header

    def process_images(self, apply_noise_to_image=True):
        """
        Runs the full mock-observation pipeline for every filter.
        Results are stored in ``self.sci_images`` and ``self.rms_images``
        (keyed by filter name) and can be written to disk with
        :meth:`save_results_to_fits`.

        Parameters
        ----------
        apply_noise_to_image : bool
            If True (default), Poisson shot noise and Gaussian background
            noise are injected into the science image following Eqs. 24-26
            of the GalSyn paper. If False, the convolved image is stored
            without noise; the RMS map is still computed so it can be used
            as a noise reference.
        """
        print("\nStarting mosaic observation pipeline...")

        _, filter_wave_pivot_data = self._load_filter_transmission_from_paths_local(
            self.filters, self.filter_transmission_path
        )

        for f_name in self.filters:
            print(f"\n  Processing filter: {f_name}")

            # 1. Load mosaic image from primary HDU
            flux_data, header, initial_scale, flux_unit = self._read_mosaic(f_name)
            print(
                f"    Input shape    : {flux_data.shape}"
                f"  |  pixel scale: {initial_scale:.5f} arcsec/pixel"
                f"  |  unit: {flux_unit}"
            )

            # 2. Convert to erg/s/cm^2/Angstrom for internal calculations
            wave_eff = filter_wave_pivot_data[f_name]
            print(f"    Pivot wavelength: {wave_eff:.1f} Angstrom")
            flux_erg = self._flux_to_erg_per_angstrom(
                flux_data, flux_unit, wave_eff, initial_scale
            )

            # 3. Flux-conserving resampling to desired pixel scale
            desired_scale = self.desired_pixel_scales[f_name]
            print(f"    Resampling     : {initial_scale:.5f} -> {desired_scale:.5f} arcsec/pixel")
            resampled_flux = self._rebin_map_flux(flux_erg, initial_scale, desired_scale)

            # Update WCS in header to reflect the new pixel scale
            updated_header = self._update_wcs_for_resampling(
                header, initial_scale, desired_scale
            )

            # 4. PSF convolution at the new pixel scale
            print(f"    Convolving with PSF: {self.psf_paths[f_name]}")
            with fits.open(self.psf_paths[f_name]) as ph:
                psf_data = ph[0].data.astype(float)

            psf_resampled = resize_psf(
                psf_data, self.psf_pixel_scales[f_name], desired_scale
            )
            psf_resampled /= psf_resampled.sum()   # normalise to unit sum
            convolved_flux = convolve_fft(
                resampled_flux, psf_resampled, boundary='fill', fill_value=0.0
            )

            # 5. Noise parameters
            c_aa     = 2.99792458e18
            mag_zp   = self.mag_zp[f_name]
            lim_mag  = self.limiting_magnitude[f_name]
            snr_lim  = self.snr_limit[f_name]
            exp_time = self.exposure_time[f_name]
            ap_rad   = self.aperture_radius_arcsec[f_name]

            # Background noise from limiting magnitude (Eq. 21-23)
            C_lim              = exp_time * (10**(0.4 * (mag_zp - lim_mag)))
            sigma_bg_ap_sq     = np.maximum(0, (C_lim / snr_lim)**2 - C_lim)
            ap_area_pix2       = np.pi * (ap_rad / desired_scale)**2
            sigma_bg_per_pixel = np.sqrt(sigma_bg_ap_sq / ap_area_pix2)

            # Source counts per pixel (Eq. 19-20)
            f_nu_pixel      = convolved_flux * wave_eff**2 / c_aa
            pix_mag_AB      = -2.5 * np.log10(np.clip(f_nu_pixel, 1e-50, None)) - 48.6
            counts_expected = np.maximum(0, exp_time * (10**(0.4 * (mag_zp - pix_mag_AB))))

            # Flux conversion factor: counts -> erg/s/cm^2/Angstrom
            flux_per_count = (
                10**((mag_zp - 2.5 * np.log10(1.0 / exp_time) + 48.6) / (-2.5))
                * c_aa / wave_eff**2
            )

            if apply_noise_to_image:
                print("    Injecting noise...")
                # Poisson shot noise (Eq. 24) + Gaussian background (Eq. 25)
                noisy_counts   = (stats.poisson.rvs(counts_expected)
                                  + np.random.normal(0, sigma_bg_per_pixel,
                                                     size=convolved_flux.shape))
                final_flux_erg = noisy_counts * flux_per_count               # Eq. 26
            else:
                print("    Skipping noise injection.")
                final_flux_erg = convolved_flux

            # RMS map (Eq. 27) — always computed regardless of noise flag
            final_rms_erg = (
                np.sqrt(counts_expected + sigma_bg_per_pixel**2) * flux_per_count
            )

            # 6. Convert back to original flux unit
            sci_out = convert_flux_map(
                final_flux_erg, wave_eff,
                to_unit=flux_unit, pixel_scale_arcsec=desired_scale
            )
            rms_out = convert_flux_map(
                final_rms_erg, wave_eff,
                to_unit=flux_unit, pixel_scale_arcsec=desired_scale
            )

            # 7. Store results keyed by filter name
            self.sci_images[f_name]  = sci_out
            self.rms_images[f_name]  = rms_out
            self.sci_headers[f_name] = updated_header
            self.rms_headers[f_name] = updated_header.copy()

            print(f"    Done. Output shape: {sci_out.shape}")

        print("\nMosaic pipeline complete.")

    def save_results_to_fits(self, output_fits_paths):
        """
        Saves the processed science image and RMS map for each filter to
        separate single-extension FITS files, one file per filter,
        mirroring the single-file-per-filter input structure.

        The science image is written to the primary HDU (index 0).
        The RMS map is appended as a named image extension ('RMS').

        Parameters
        ----------
        output_fits_paths : dict
            Keys are filter names, values are output file paths.
            Example::

                {
                    'jwst_nircam_f200w': 'mosaic_f200w_obs.fits',
                    'jwst_nircam_f356w': 'mosaic_f356w_obs.fits',
                }
        """
        for f_name in self.filters:
            if f_name not in self.sci_images:
                print(f"  [{f_name}] No processed data found — skipping save.")
                continue

            out_path = output_fits_paths[f_name]

            sci_hdr            = self.sci_headers[f_name].copy()
            sci_hdr['HISTORY'] = 'Processed by GalSynMockObservation_mosaic'

            hdul_out = fits.HDUList([
                fits.PrimaryHDU(data=self.sci_images[f_name], header=sci_hdr),
                fits.ImageHDU(
                    data=self.rms_images[f_name],
                    header=self.rms_headers[f_name],
                    name='RMS',
                ),
            ])
            hdul_out.writeto(out_path, overwrite=True)
            hdul_out.close()
            print(f"  [{f_name}] Saved to '{out_path}'")