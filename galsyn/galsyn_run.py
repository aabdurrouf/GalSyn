import h5py
import os, sys 
import numpy as np 
from astropy.io import fits
from .imgutils import *
from .utils import *
from .dust import *
from joblib import Parallel, delayed
import multiprocessing
from scipy.interpolate import interp1d

from tqdm.auto import tqdm
from tqdm_joblib import tqdm_joblib

import fsps # Keep FSPS import for on-the-fly option
from .ssp_generator import FSPS_Z_SUN # Import FSPS_Z_SUN for consistent metallicity handling

# Define the primordial solar metallicity value used in the simulation data conversion
PRIMORDIAL_Z_SUN_VALUE = 0.0127 

# Global variables for pre-computed SSP spectra
ssp_wave = None
ssp_ages_gyr = None
ssp_logzsol_grid = None
ssp_spectra_grid = None
ssp_stellar_mass_grid = None # New global variable for surviving stellar mass
ssp_fsps_z_sun = None # Store Z_sun used when generating SSPs

# Global variables for worker initialization
sp_instance = None  # FSPS instance, now conditionally initialized
igm_trans = None
snap_z = None
pix_area_kpc2 = None
mean_AV_unres = None
filters = None
filter_transmission = None
imf_type = None
add_neb_emission = None
gas_logu = None
add_igm_absorption = None
igm_type = None
dust_index_bc = None
dust_index = None
t_esc = None
scale_dust_tau = None
cosmo = None
dust_law = None
bump_amp = None
salim_a0 = None
salim_a1 = None
salim_a2 = None
salim_a3 = None
salim_RV = None
salim_B = None
dust_Alambda_per_AV = None
func_interp_dust_index = None # For dust_law 0 and 1
use_precomputed_ssp = False # New global flag

# New global variables for pixel spectra output
output_pixel_spectra_flag = False
global_output_obs_wave = None # The observed wavelength grid for output spectra


def init_worker(snap_z_val, pix_area_kpc2_val, mean_AV_unres_val, filters_val, 
                filter_transmission_val, imf_type_val, add_neb_emission_val, 
                gas_logu_val, add_igm_absorption_val, igm_type_val, dust_index_bc_val, 
                dust_index_val, t_esc_val, scale_dust_tau_val, cosmo_val, dust_law_val,
                bump_amp_val, dustindexAV_AV_val, dustindexAV_dust_index_val, salim_a0_val, 
                salim_a1_val, salim_a2_val, salim_a3_val, salim_RV_val, salim_B_val,
                use_precomputed_ssp_val, ssp_filepath_val=None,
                output_pixel_spectra_val=False, rest_wave_min_val=None, rest_wave_max_val=None): # Added new args
    
    global ssp_wave, ssp_ages_gyr, ssp_logzsol_grid, ssp_spectra_grid, ssp_stellar_mass_grid, ssp_fsps_z_sun
    global sp_instance, igm_trans, snap_z, pix_area_kpc2
    global mean_AV_unres, filters, filter_transmission, imf_type
    global add_neb_emission, gas_logu, add_igm_absorption, igm_type
    global dust_index_bc, dust_index, t_esc, scale_dust_tau, cosmo
    global dust_law, bump_amp, salim_a0, salim_a1, salim_a2, salim_a3 
    global salim_RV, salim_B, dust_Alambda_per_AV, func_interp_dust_index
    global use_precomputed_ssp # Update global flag
    global output_pixel_spectra_flag, global_output_obs_wave # New globals

    snap_z = snap_z_val
    pix_area_kpc2 = pix_area_kpc2_val
    mean_AV_unres = mean_AV_unres_val
    filters = filters_val
    filter_transmission = filter_transmission_val
    imf_type = imf_type_val
    add_neb_emission = add_neb_emission_val
    gas_logu = gas_logu_val
    add_igm_absorption = add_igm_absorption_val
    igm_type = igm_type_val
    dust_index_bc = dust_index_bc_val
    t_esc = t_esc_val
    scale_dust_tau = scale_dust_tau_val
    cosmo = cosmo_val
    dust_law = dust_law_val
    salim_a0 = salim_a0_val
    salim_a1 = salim_a1_val
    salim_a2 = salim_a2_val
    salim_a3 = salim_a3_val
    salim_RV = salim_RV_val
    salim_B = salim_B_val
    dust_index = dust_index_val
    bump_amp = bump_amp_val
    use_precomputed_ssp = use_precomputed_ssp_val # Set the flag

    # Set new global flags for pixel spectra output
    output_pixel_spectra_flag = output_pixel_spectra_val
    
    # Initialize FSPS instance or load SSP data to get ssp_wave
    if use_precomputed_ssp:
        # Load pre-computed SSP spectra and stellar masses
        if ssp_filepath_val is None:
            print("Error: ssp_filepath must be provided when use_precomputed_ssp is True.")
            sys.exit(1)
        try:
            with h5py.File(ssp_filepath_val, 'r') as f_ssp:
                ssp_wave = f_ssp['wavelength'][:]
                ssp_ages_gyr = f_ssp['ages_gyr'][:]
                ssp_logzsol_grid = f_ssp['logzsol'][:]
                ssp_spectra_grid = f_ssp['spectra'][:]
                ssp_stellar_mass_grid = f_ssp['stellar_mass'][:] # Load surviving stellar mass
                ssp_fsps_z_sun = f_ssp.attrs['fsps_z_sun']
                # Verify consistency of FSPS parameters used for SSP generation
                if f_ssp.attrs['imf_type'] != imf_type:
                    print(f"Warning: IMF type mismatch! SSP grid was generated with {f_ssp.attrs['imf_type']}, but current setting is {imf_type}.")
                if f_ssp.attrs['add_neb_emission'] != add_neb_emission:
                    print(f"Warning: Nebular emission setting mismatch! SSP grid was generated with {f_ssp.attrs['add_neb_emission']}, but current setting is {add_neb_emission}.")
                if f_ssp.attrs['gas_logu'] != gas_logu:
                    print(f"Warning: Gas LogU setting mismatch! SSP grid was generated with {f_ssp.attrs['gas_logu']}, but current setting is {gas_logu}.")
        except Exception as e:
            print(f"Error loading SSP grid from {ssp_filepath_val}: {e}")
            sys.exit(1) # Exit if SSP grid cannot be loaded
    else:
        # Initialize FSPS instance for on-the-fly generation
        sp_instance = fsps.StellarPopulation(zcontinuous=1)
        sp_instance.params['imf_type'] = imf_type
        sp_instance.params["add_dust_emission"] = False
        sp_instance.params["add_neb_emission"] = add_neb_emission
        sp_instance.params['gas_logu'] = gas_logu
        sp_instance.params["fagn"] = 0
        sp_instance.params["sfh"] = 0   # SSP
        sp_instance.params["dust1"] = 0.0
        sp_instance.params["dust2"] = 0.0   # optical depth
        # Get the wavelength array from FSPS for on-the-fly mode
        ssp_wave, _ = sp_instance.get_spectrum(peraa=True, tage=1.0) # Use ssp_wave for consistency in global scope

    # Now that ssp_wave is guaranteed to be populated, define global_output_obs_wave
    if output_pixel_spectra_flag:
        # Redshift the original SSP wavelength grid
        obs_ssp_wave_full = ssp_wave * (1.0 + snap_z)
        
        # Determine the observed wavelength boundaries corresponding to the requested rest-frame range
        obs_min_boundary = rest_wave_min_val * (1.0 + snap_z)
        obs_max_boundary = rest_wave_max_val * (1.0 + snap_z)
        
        # Select the portion of the observed SSP wavelength grid that falls within the boundaries
        idx_valid_obs_wave = np.where((obs_ssp_wave_full >= obs_min_boundary) & 
                                      (obs_ssp_wave_full <= obs_max_boundary))
        
        global_output_obs_wave = obs_ssp_wave_full[idx_valid_obs_wave]

        # Handle edge case where no wavelengths fall in the range
        if global_output_obs_wave.size == 0:
            print(f"Warning: No observed wavelengths from SSP grid fall within the requested rest-frame range "
                  f"[{rest_wave_min_val}-{rest_wave_max_val}] Angstroms at z={snap_z}. "
                  f"Observed range: [{obs_min_boundary:.1f}-{obs_max_boundary:.1f}] Angstroms. "
                  f"Output spectra will be empty for this pixel.")
            # Ensure global_output_obs_wave is an empty array if no points are found
            global_output_obs_wave = np.array([])


    # Dust law setup (depends on the wave array, which is now ssp_wave from either source)
    if dust_law <= 1:
        global func_interp_dust_index
        dustindexAV_AV = dustindexAV_AV_val
        dustindexAV_dust_index = dustindexAV_dust_index_val
        func_interp_dust_index = interp1d(dustindexAV_AV, dustindexAV_dust_index, bounds_error=False, fill_value='extrapolate')

    elif dust_law == 2:  # modified Calzetti+20 with Bump strength tied to dust_index and dust_index is free. Dust index is single value applied to all star particles irrespective of A_V
        bump_amp = bump_amp_from_dust_index(dust_index)
        dust_Alambda_per_AV = modified_calzetti_dust_Alambda_per_AV(ssp_wave, dust_index=dust_index, bump_amp=bump_amp)

    elif dust_law == 3:  # modified Calzetti+20 with free Bump strengh and dust_index.  Dust index is single value applied to all star particles irrespective of A_V
        dust_Alambda_per_AV = modified_calzetti_dust_Alambda_per_AV(ssp_wave, dust_index=dust_index, bump_amp=bump_amp)

    elif dust_law == 4:  # Salim et al. (2018)
        dust_Alambda_per_AV = salim18_dust_Alambda_per_AV(ssp_wave, salim_a0, salim_a1, salim_a2, salim_a3, salim_B, salim_RV)

    elif dust_law == 5:  # Calzetti et al. (2000)
        dust_Alambda_per_AV = calzetti_dust_Alambda_per_AV(ssp_wave)

    if add_igm_absorption == 1:
        if igm_type == 0:
            igm_trans = igm_att_madau(ssp_wave * (1.0+snap_z), snap_z)
        elif igm_type == 1:
            igm_trans = igm_att_inoue(ssp_wave * (1.0+snap_z), snap_z)
        else:
            print ('igm_type is not recognized! options are: 1 for Madau+1995 and Inoue+2014')
            sys.exit()
    else:
        igm_trans = 1


def dust_reddening_diffuse_ism(dust_AV, wave, dust_law):
    if dust_law == 0:  # modified Calzetti+20 with Bump strength tied to dust_index and dust_index is dependent on A_V
        dust_index1 = func_interp_dust_index(dust_AV)
        bump_amp1 = bump_amp_from_dust_index(dust_index1)
        Alambda = modified_calzetti_dust_Alambda_per_AV(wave, dust_index=dust_index1, bump_amp=bump_amp1) * dust_AV

    elif dust_law == 1:  # modified Calzetti+20 with free Bump strengh and dust_index is dependent on A_V
        dust_index1 = func_interp_dust_index(dust_AV)
        Alambda = modified_calzetti_dust_Alambda_per_AV(wave, dust_index=dust_index1, bump_amp=bump_amp) * dust_AV

    return Alambda


def _process_pixel_data(ii, jj, star_particle_membership_list, gas_particle_membership_list, # Renamed arguments
                        stars_mass, stars_age, stars_zsol, stars_init_mass, 
                        gas_mass, gas_sfr_inst, gas_zsol, gas_log_temp, gas_mass_H):
    """
    ii=y jj=x
    Worker function to process calculations for a single pixel (ii, jj).
    This function will be executed in parallel for each pixel.
    
    star_particle_membership_list: List of (original_particle_index, line_of_sight_distance) for THIS pixel.
    gas_particle_membership_list: List of (original_particle_index, line_of_sight_distance) for THIS pixel.
    """

    # Initialize a dictionary to store results for this pixel
    pixel_results = {
        'map_stars_mass': 0.0,
        'map_mw_age': 0.0,
        'map_stars_mw_zsol': 0.0,
        'map_sfr_100': 0.0,
        'map_sfr_30': 0.0,
        'map_sfr_10': 0.0,
        'map_gas_mass': 0.0,
        'map_sfr_inst': 0.0,
        'map_gas_mw_zsol': 0.0,
        'map_dust_mean_tauV': 0.0,
        'map_dust_mean_AV': 0.0,
        'map_flux': np.zeros(len(filters)),
        'map_flux_dust': np.zeros(len(filters)),
        'obs_spectra_nodust_igm': np.zeros(len(global_output_obs_wave)) if output_pixel_spectra_flag else None,
        'obs_spectra_dust_igm': np.zeros(len(global_output_obs_wave)) if output_pixel_spectra_flag else None
    }

    # Corrected: Directly use the passed lists for the current pixel
    star_ids0 = np.asarray([x[0] for x in star_particle_membership_list], dtype=int)
    star_los_dist0 = np.asarray([x[1] for x in star_particle_membership_list])

    gas_ids0 = np.asarray([x[0] for x in gas_particle_membership_list], dtype=int)
    gas_los_dist0 = np.asarray([x[1] for x in gas_particle_membership_list])


    idxs = np.where((np.isnan(stars_mass[star_ids0]) == False) &
                     (np.isnan(stars_age[star_ids0]) == False) &
                     (np.isnan(stars_zsol[star_ids0]) == False))[0]
    star_ids = star_ids0[idxs]
    
    star_los_dist = star_los_dist0[idxs]

    idxg = np.where(np.isnan(gas_mass[gas_ids0]) == False)[0]
    gas_ids = gas_ids0[idxg]
    
    gas_los_dist = gas_los_dist0[idxg]

    current_stars_mass_sum = np.nansum(stars_mass[star_ids])
    pixel_results['map_stars_mass'] = current_stars_mass_sum

    if current_stars_mass_sum > 0:
        pixel_results['map_mw_age'] = np.nansum(stars_mass[star_ids] * stars_age[star_ids]) / current_stars_mass_sum
        pixel_results['map_stars_mw_zsol'] = np.nansum(stars_mass[star_ids] * stars_zsol[star_ids]) / current_stars_mass_sum
    else:
        pixel_results['map_mw_age'] = np.nan
        pixel_results['map_stars_mw_zsol'] = np.nan

    idxs3_100 = np.where((np.isnan(stars_init_mass[star_ids]) == False) & (stars_age[star_ids] <= 0.1))[0]
    pixel_results['map_sfr_100'] = np.nansum(stars_init_mass[star_ids[idxs3_100]]) / 0.1 / 1e+9

    idxs3_30 = np.where((np.isnan(stars_init_mass[star_ids]) == False) & (stars_age[star_ids] <= 0.03))[0]
    pixel_results['map_sfr_30'] = np.nansum(stars_init_mass[star_ids[idxs3_30]]) / 0.03 / 1e+9

    idxs3_10 = np.where((np.isnan(stars_init_mass[star_ids]) == False) & (stars_age[star_ids] <= 0.01))[0]
    pixel_results['map_sfr_10'] = np.nansum(stars_init_mass[star_ids[idxs3_10]]) / 0.01 / 1e+9

    current_gas_mass_sum = np.nansum(gas_mass[gas_ids])
    pixel_results['map_gas_mass'] = current_gas_mass_sum
    pixel_results['map_sfr_inst'] = np.nansum(gas_sfr_inst[gas_ids])

    if current_gas_mass_sum > 0:
        pixel_results['map_gas_mw_zsol'] = np.nansum(gas_mass[gas_ids] * gas_zsol[gas_ids]) / current_gas_mass_sum
    else:
        pixel_results['map_gas_mw_zsol'] = np.nan

    # Model spectra of SSP, get total spectrum and photometric fluxes
    if len(star_ids) > 0:

        array_spec = []
        array_spec_dust = []
        array_AV = []
        array_tauV = []
        
        # Determine the wavelength array based on the chosen method
        wave = ssp_wave # ssp_wave is always populated in init_worker now

        for i_sid in range(len(star_ids)):
            star_id = star_ids[i_sid]

            if use_precomputed_ssp:
                # Convert stellar metallicity (in units of primordial solar metallicity)
                # to logzsol relative to FSPS_Z_SUN (0.019)
                # Absolute metallicity = stars_zsol[star_id] * PRIMORDIAL_Z_SUN_VALUE
                # logzsol for FSPS = log10(Absolute metallicity / FSPS_Z_SUN)
                particle_logzsol = np.log10( (stars_zsol[star_id] * PRIMORDIAL_Z_SUN_VALUE) / ssp_fsps_z_sun )
                
                # Find closest age and logzsol indices
                age_idx = np.argmin(np.abs(ssp_ages_gyr - stars_age[star_id]))
                z_idx = np.argmin(np.abs(ssp_logzsol_grid - particle_logzsol))
                
                # Retrieve the pre-computed spectrum (L_sun/AA) and its surviving stellar mass
                spec = ssp_spectra_grid[age_idx, z_idx, :]
                ssp_mass_formed = ssp_stellar_mass_grid[age_idx, z_idx] # Use surviving stellar mass from grid
            else:
                # On-the-fly FSPS spectrum generation
                # Convert stellar metallicity (in units of primordial solar metallicity)
                # to logzsol relative to FSPS_Z_SUN (0.019) for on-the-fly FSPS call
                # Absolute metallicity = stars_zsol[star_id] * PRIMORDIAL_Z_SUN_VALUE
                # logzsol for FSPS = log10(Absolute metallicity / FSPS_Z_SUN)
                logzsol = np.log10( (stars_zsol[star_id] * PRIMORDIAL_Z_SUN_VALUE) / FSPS_Z_SUN )
                sp_instance.params["logzsol"] = logzsol   
                sp_instance.params['gas_logz'] = logzsol # For nebular emission consistency

                # Get spectrum in L_sun/AA
                _, spec = sp_instance.get_spectrum(peraa=True, tage=stars_age[star_id])
                ssp_mass_formed = sp_instance.stellar_mass # Actual surviving stellar mass for this SSP from FSPS

            # calculate total cold hydrogen gas column density in front of the particle along the line of sight
            idxg_front = np.where(gas_los_dist < star_los_dist[i_sid])[0]
            front_gas_ids = gas_ids[idxg_front]
            idxg1 = np.where((gas_sfr_inst[front_gas_ids] > 0.0) | (gas_log_temp[front_gas_ids] < 3.9))[0]
            cold_front_gas_ids = front_gas_ids[idxg1]

            if len(cold_front_gas_ids) > 0:
                temp_mw_gas_zsol = np.nansum(gas_mass[cold_front_gas_ids]*gas_zsol[cold_front_gas_ids])/np.nansum(gas_mass[cold_front_gas_ids])
                nH = np.nansum(gas_mass_H[cold_front_gas_ids])*1.247914e+14/pix_area_kpc2      # number of hydrogen atom per cm^2
                tauV = scale_dust_tau * temp_mw_gas_zsol * nH / 2.1e+21
                dust_AV = -2.5*np.log10((1.0 - np.exp(-1.0*tauV))/tauV)

                if np.isnan(dust_AV)==True or dust_AV==0.0:
                    spec_dust = spec
                else:
                    # attenuation by resolved dust in the diffuse ISM
                    if dust_law <= 1:
                        Alambda = dust_reddening_diffuse_ism(dust_AV, wave, dust_law)
                    else:
                        Alambda = dust_Alambda_per_AV * dust_AV
                    
                    spec_dust = spec*np.power(10.0, -0.4*Alambda)
                    array_tauV.append(tauV)
                    array_AV.append(dust_AV)
            else:
                spec_dust = spec

            if stars_age[star_id] <= t_esc:    # age criterion defining young stars associated with birth clouds 
                # attenuation by unresolved dust in the birth cloud
                Alambda = unresolved_dust_birth_cloud_Alambda_per_AV(wave, dust_index_bc=dust_index_bc) * mean_AV_unres
                spec_dust = spec_dust*np.power(10.0, -0.4*Alambda)
    
            # Normalization by the SSP's surviving stellar mass
            norm = stars_mass[star_id] / ssp_mass_formed

            if len(np.asarray(spec_dust).shape) == 1:
                array_spec.append(spec*norm)
                array_spec_dust.append(spec_dust*norm)

        # average AV:
        array_AV, array_tauV = np.asarray(array_AV), np.asarray(array_tauV)
        if array_AV.size == 0:
            mean_AV = np.nan
            mean_tauV = np.nan
        else:
            mean_AV = np.nanmean(np.asarray(array_AV))
            mean_tauV = np.nanmean(np.asarray(array_tauV))

        redshift_flux, redshift_flux_dust = [], []
        
        if len(array_spec) > 0:
            spec_lum = np.nansum(array_spec, axis=0) # Total rest-frame luminosity (L_sun/AA)
            spec_lum_dust = np.nansum(array_spec_dust, axis=0) # Total dust-attenuated rest-frame luminosity (L_sun/AA)

            # Redshifting and IGM absorption for observed spectra output and broadband fluxes
            spec_wave_obs, spec_flux_obs = cosmo_redshifting(wave, spec_lum, snap_z, cosmo)              # in erg/s/cm^2/Ang.
            spec_wave_obs, spec_flux_dust_obs = cosmo_redshifting(wave, spec_lum_dust, snap_z, cosmo)    # in erg/s/cm^2/Ang.

            # Apply IGM transmission to observed fluxes
            spec_flux_obs_igm = spec_flux_obs * igm_trans
            spec_flux_dust_obs_igm = spec_flux_dust_obs * igm_trans

            # If user wants pixel spectra output, interpolate to the common observed grid
            if output_pixel_spectra_flag:
                # Only interpolate if the global_output_obs_wave is not empty
                if global_output_obs_wave.size > 0:
                    interp_func_nodust = interp1d(spec_wave_obs, spec_flux_obs_igm, kind='linear', 
                                                  bounds_error=False, fill_value=0.0)
                    interp_func_dust = interp1d(spec_wave_obs, spec_flux_dust_obs_igm, kind='linear', 
                                                bounds_error=False, fill_value=0.0)
                    
                    pixel_results['obs_spectra_nodust_igm'] = interp_func_nodust(global_output_obs_wave)
                    pixel_results['obs_spectra_dust_igm'] = interp_func_dust(global_output_obs_wave)
                else:
                    # If global_output_obs_wave is empty, output empty arrays of correct size
                    pixel_results['obs_spectra_nodust_igm'] = np.zeros(0)
                    pixel_results['obs_spectra_dust_igm'] = np.zeros(0)


            # filtering for broadband fluxes
            nbands = len(filters)
            redshift_flux = np.zeros(nbands)
            redshift_flux_dust = np.zeros(nbands)

            for i_band in range(nbands):
                redshift_flux[i_band] = filtering(spec_wave_obs, spec_flux_obs_igm, filter_transmission[filters[i_band]]['wave'], filter_transmission[filters[i_band]]['trans'])
                redshift_flux_dust[i_band] = filtering(spec_wave_obs, spec_flux_dust_obs_igm, filter_transmission[filters[i_band]]['wave'], filter_transmission[filters[i_band]]['trans'])

        if len(redshift_flux) > 0:
            pixel_results['map_flux'] = redshift_flux
            pixel_results['map_flux_dust'] = redshift_flux_dust

            pixel_results['map_dust_mean_tauV'] = mean_tauV
            pixel_results['map_dust_mean_AV'] = mean_AV

    return ii, jj, pixel_results


def generate_images(sim_file, z, filters, filter_transmission, filter_wave_eff, dim_kpc=None, 
                    pix_arcsec=0.02, flux_unit='MJy/sr', polar_angle_deg=0, azimuth_angle_deg=0,
                    name_out_img=None, n_jobs=-1, imf_type=1, add_neb_emission=1, gas_logu=-2.0, 
                    add_igm_absorption=1, igm_type=0, dust_index_bc=-0.7, dust_index=0.0, t_esc=0.01, 
                    norm_dust_z=[], norm_dust_tau=[], cosmo_str='Planck18', cosmo_h=0.6774, XH=0.76, 
                    dust_law=0, bump_amp=0.85, dustindexAV_AV=[], dustindexAV_dust_index=[], salim_a0=-4.30, 
                    salim_a1=2.71, salim_a2=-0.191, salim_a3=0.0121, salim_RV=3.15, salim_B=1.57, 
                    initdim_kpc=100, initdim_mass_fraction=0.92,
                    use_precomputed_ssp=True, ssp_filepath="ssp_spectra.hdf5",
                    output_pixel_spectra=False, rest_wave_min=1000.0, rest_wave_max=16000.0): # Added new args
    """
    Generates astrophysical images from HDF5 simulation data with parallelized pixel calculations.
    Allows choice between using pre-computed SSP spectra from an HDF5 file or
    generating them on-the-fly using FSPS.
    Optionally outputs observed-frame spectra (redshifted and IGM-transmitted) for each pixel.

    Parameters:
        sim_file (str): Path to the HDF5 simulation file.
        z (float): Redshift of the galaxy.
        filters (list): List of photometric filters.
        filter_transmission (dict): Dictionary with filter transmission curves.
        filter_wave_eff (dict): Dictionary with effective wavelengths for filters.
        dim_kpc (float, optional): Dimension of the image in kpc. If None, assigned automatically. Defaults to None.
        pix_arcsec (float, optional): Pixel size in arcseconds. Defaults to 0.02.
        flux_unit (string, optional): Desired flux unit for the generated images. Options are: 'MJy/sr', 'nJy', 'AB magnitude', or 'erg/s/cm2/A'. Default to 'MJy/sr'.
        polar_angle_deg (float, optional): Polar angle for projection. Defaults to 0.
        azimuth_angle_deg (float, optional): Azimuth angle for projection. Defaults to 0.
        name_out_img (str, optional): Output file name for images. Defaults to None.
        n_jobs (int, optional): Number of CPU cores to use for parallel processing. Defaults to -1 (all available).
        imf_type (int, optional): IMF type for FSPS (must match SSP grid if pre-computed). Defaults to 1.
        add_neb_emission (int, optional): Add nebular emission (must match SSP grid if pre-computed). Defaults to 1.
        gas_logu (float, optional): Log ionization parameter (must match SSP grid if pre-computed). Defaults to -2.0.
        add_igm_absorption (int, optional): Add IGM absorption. Defaults to 1.
        igm_type (int, optional): IGM absorption model type. Defaults to 0.
        dust_index_bc (float, optional): Dust index for birth clouds. Defaults to -0.7.
        dust_index (float, optional): Dust index for diffuse ISM (if applicable). Defaults to 0.0.
        t_esc (float, optional): Escape time for young stars. Defaults to 0.01.
        norm_dust_z (list, optional): Redshift array for dust normalization. Defaults to [].
        norm_dust_tau (list, optional): Tau array for dust normalization. Defaults to [].
        cosmo_str (str, optional): Cosmology string. Defaults to 'Planck18'.
        cosmo_h (float, optional): Hubble parameter h. Defaults to 0.6774.
        XH (float, optional): Hydrogen mass fraction. Defaults to 0.76.
        dust_law (int, optional): Dust attenuation law type. Defaults to 0.
        bump_amp (float, optional): UV bump amplitude. Defaults to 0.85.
        dustindexAV_AV (list, optional): A_V values for dust index interpolation. Defaults to [].
        dustindexAV_dust_index (list, optional): Dust index values for interpolation. Defaults to [].
        salim_a0, salim_a1, salim_a2, salim_a3, salim_RV, salim_B (float, optional): Parameters for Salim+2018 dust law.
        initdim_kpc (float, optional): Initial guess for image dimension in kpc. Defaults to 100.
        initdim_mass_fraction (float, optional): Mass fraction to determine initial image dimension. Defaults to 0.92.
        use_precomputed_ssp (bool, optional): If True, use pre-computed SSP spectra. If False, generate on-the-fly.
                                              Defaults to True.
        ssp_filepath (str, optional): Path to the pre-computed SSP spectra HDF5 file.
                                      Only used if `use_precomputed_ssp` is True. Defaults to "ssp_spectra.hdf5".
        output_pixel_spectra (bool, optional): If True, output observed-frame spectra for each pixel. Defaults to False.
        rest_wave_min (float, optional): Minimum rest-frame wavelength for output spectra (Angstrom). Defaults to 1000.0.
        rest_wave_max (float, optional): Maximum rest-frame wavelength for output spectra (Angstrom). Defaults to 16000.0.
    """

    cosmo = define_cosmo(cosmo_str)

    print ('Processing '+sim_file)
    # --- Data Loading and Initial Calculations (Sequential) ---

    snap_z = z
    snap_a = 1.0/(1.0 + snap_z)

    pix_kpc = angular_to_physical(snap_z, pix_arcsec, cosmo)
    pix_area_kpc2 = pix_kpc*pix_kpc
    print ('pixel size: %lf arcsec or %lf kpc' % (pix_arcsec,pix_kpc))

    f = h5py.File(sim_file,'r')

    stars_form_a = f['PartType4']['GFM_StellarFormationTime'][:]
    stars_form_z = (1.0/stars_form_a) - 1.0

    stars_init_mass = f['PartType4']['GFM_InitialMass'][:]*1e+10/cosmo_h
    stars_mass = f['PartType4']['Masses'][:]*1e+10/cosmo_h
    stars_zsol = f['PartType4']['GFM_Metallicity'][:]/PRIMORDIAL_Z_SUN_VALUE         # in solar metallicity

    coords = f['PartType4']['Coordinates'][:]
    coords_x = coords[:,0]*snap_a/cosmo_h             # x in kpc
    coords_y = coords[:,1]*snap_a/cosmo_h             # y in kpc
    coords_z = coords[:,2]*snap_a/cosmo_h             # z in kpc

    snap_univ_age = cosmo.age(snap_z).value
    stars_form_age_univ = interp_age_univ_from_z(stars_form_z, cosmo)
    stars_age = snap_univ_age - stars_form_age_univ                 # age in Gyr
    # select only stellar particles
    idx = np.where((stars_form_a>0) & (stars_age>=0))[0]
    stars_form_a = stars_form_a[idx]
    stars_form_z = stars_form_z[idx]
    stars_init_mass = stars_init_mass[idx]
    stars_mass = stars_mass[idx]
    stars_zsol = stars_zsol[idx]
    stars_coords_x = coords_x[idx]
    stars_coords_y = coords_y[idx]
    stars_coords_z = coords_z[idx]
    stars_age = stars_age[idx]

    # read gas particles data
    gas_mass = f['PartType0']['Masses'][:]*1e+10/cosmo_h
    gas_zsol = f['PartType0']['GFM_Metallicity'][:]/PRIMORDIAL_Z_SUN_VALUE  # in solar metallicity
    gas_coords = f['PartType0']['Coordinates'][:]
    gas_coords_x = gas_coords[:,0]*snap_a/cosmo_h                 # in kpc
    gas_coords_y = gas_coords[:,1]*snap_a/cosmo_h
    gas_coords_z = gas_coords[:,2]*snap_a/cosmo_h
    gas_sfr_inst = f['PartType0']['StarFormationRate'][:]   # in Msun/yr 
    gas_mass_H = gas_mass*XH                     # mass of hydrogen 
    # calculate gas temperature
    u = f['PartType0']['InternalEnergy'][:]      #  the Internal Energy
    Xe = f['PartType0']['ElectronAbundance'][:]  # xe (=ne/nH)  the electron abundance                
    gamma = 5.0/3.0          # the adiabatic index
    KB = 1.3807e-16          # the Boltzmann constant in CGS units  [cm^2 g s^-2 K^-1]
    mp = 1.6726e-24          # the proton mass  [g]
    mu = (4*mp)/(1+3*XH+4*XH*Xe)
    gas_log_temp = np.log10((gamma-1.0)*(u/KB)*mu*1e+10)  # log temperature in Kelvin

    f.close()

    star_coords = np.column_stack((stars_coords_x, stars_coords_y, stars_coords_z))
    gas_coords = np.column_stack((gas_coords_x, gas_coords_y, gas_coords_z))

    if dim_kpc is None:
        dim_kpc = determine_image_size(star_coords, stars_mass, pix_kpc, (initdim_kpc, initdim_kpc), 
                                       polar_angle_deg, azimuth_angle_deg, gas_coords, gas_mass, 
                                       mass_percentage=initdim_mass_fraction)

    output_dimension = (dim_kpc, dim_kpc)
    star_particle_membership, star_mass_density_map, central_pixel_coords, grid_info, gas_particle_membership, gas_mass_density_map = get_2d_density_projection_no_los_binning(
                                                                                                                                                star_coords, 
                                                                                                                                                stars_mass, 
                                                                                                                                                pix_kpc, 
                                                                                                                                                output_dimension, 
                                                                                                                                                polar_angle_deg=polar_angle_deg, 
                                                                                                                                                azimuth_angle_deg=azimuth_angle_deg, 
                                                                                                                                                gas_coords=gas_coords, 
                                                                                                                                                gas_masses=gas_mass)
    
    dimx, dimy = grid_info['num_pixels_x'], grid_info['num_pixels_y']
    print ('Cutout size: %d x %d pix or %d x %d kpc' % (dimx,dimy,dim_kpc,dim_kpc))

    # ------ estimate unresolved dust AV associated with the birth cloud ------- #
    idxg_global = np.where((gas_sfr_inst>0.0) | (gas_log_temp<3.9))[0]
    # calculate Hydrogen column density
    if np.nansum(gas_mass[idxg_global]) > 0:
        temp_mw_gas_zsol = np.nansum(gas_mass[idxg_global]*gas_zsol[idxg_global])/np.nansum(gas_mass[idxg_global])
    else:
        temp_mw_gas_zsol = 0.0 # Default if no gas mass
    nH = np.nansum(gas_mass_H[idxg_global])*1.247914e+14/dim_kpc/dim_kpc      # number of hydrogen atom per cm^2

    scale_dust_tau = tau_dust_given_z(snap_z, norm_dust_z, norm_dust_tau)
    mean_tauV_res = scale_dust_tau*temp_mw_gas_zsol*nH/2.1e+21 

    global mean_AV_unres
    if np.isnan(mean_tauV_res)==True or np.isinf(mean_tauV_res)==True:
        mean_tauV_res, mean_AV_unres = 0.0, 0.0
    else:
        mean_AV_unres = -2.5*np.log10(np.exp(-2.0*mean_tauV_res))                 # assumed twice of average tauV resolved (Vogelsberger+20)
    print ('mean_tauV_res=%lf mean_AV_unres=%lf' % (mean_tauV_res,mean_AV_unres))
    # -----------

    nbands = len(filters)

    # Initialize all result maps to zeros
    map_mw_age = np.zeros((dimy,dimx))
    map_stars_mw_zsol = np.zeros((dimy,dimx))
    map_stars_mass = np.zeros((dimy,dimx))
    map_sfr_100 = np.zeros((dimy,dimx))
    map_sfr_10 = np.zeros((dimy,dimx))
    map_sfr_30 = np.zeros((dimy,dimx))

    map_gas_mw_zsol = np.zeros((dimy,dimx))
    map_gas_mass = np.zeros((dimy,dimx))
    map_sfr_inst = np.zeros((dimy,dimx))

    map_dust_mean_tauV = np.zeros((dimy,dimx))
    map_dust_mean_AV = np.zeros((dimy,dimx))

    map_flux = np.zeros((dimy,dimx,nbands))
    map_flux_dust = np.zeros((dimy,dimx,nbands))

    # Initialize spectral maps if requested
    num_obs_wave_points = len(global_output_obs_wave) if output_pixel_spectra else 0
    map_spectra_nodust = np.zeros((dimy, dimx, num_obs_wave_points)) if output_pixel_spectra else None
    map_spectra_dust = np.zeros((dimy, dimx, num_obs_wave_points)) if output_pixel_spectra else None


    # --- Parallel Calculation Section ---
    tasks = []
    # Estimate complexity for each pixel and store it with the task
    for ii in range(dimy): # Iterate over rows (y-axis)
        for jj in range(dimx): # Iterate over columns (x-axis)
            num_stars_in_pixel = len(star_particle_membership[ii][jj])
            num_gas_in_pixel = len(gas_particle_membership[ii][jj]) if gas_particle_membership is not None else 0
            complexity = num_stars_in_pixel + num_gas_in_pixel # Simple sum as complexity proxy
            tasks.append({'coords': (ii, jj), 'complexity': complexity, 
                          'star_part_mem': star_particle_membership[ii][jj],
                          'gas_part_mem': gas_particle_membership[ii][jj] if gas_particle_membership is not None else []})

    # Sort tasks by complexity in descending order (heaviest tasks first)
    tasks.sort(key=lambda x: x['complexity'], reverse=True)

    # Reconstruct the task arguments for _process_pixel_data
    processed_tasks_args = []
    for task in tasks:
        ii, jj = task['coords']
        # Note: We pass the original full membership lists, but the _process_pixel_data
        # function will use the pre-filtered membership lists from the task dict.
        # This is a bit redundant but ensures the worker has access to the specific
        # particles for its pixel without re-indexing the global lists.
        processed_tasks_args.append((ii, jj, task['star_part_mem'], task['gas_part_mem'],
                                     stars_mass, stars_age, stars_zsol, stars_init_mass, 
                                     gas_mass, gas_sfr_inst, gas_zsol, gas_log_temp, gas_mass_H))


    # Determine the number of CPU cores to use
    num_cores = n_jobs
    if num_cores == -1:
        num_cores = multiprocessing.cpu_count() # Use all available cores

    print(f"\nStarting parallel pixel processing on {num_cores} cores...")

    with tqdm_joblib(total=len(processed_tasks_args), desc="Processing pixels") as progress_bar:
        results = Parallel(n_jobs=num_cores, verbose=0, initializer=init_worker,
                           initargs=(snap_z, pix_area_kpc2, mean_AV_unres, filters, filter_transmission, imf_type, add_neb_emission, 
                                     gas_logu, add_igm_absorption, igm_type, dust_index_bc, dust_index, t_esc, scale_dust_tau, cosmo, 
                                     dust_law, bump_amp, dustindexAV_AV, dustindexAV_dust_index, salim_a0, salim_a1, salim_a2, salim_a3, 
                                     salim_RV, salim_B,
                                     use_precomputed_ssp, ssp_filepath, # Existing args
                                     output_pixel_spectra, rest_wave_min, rest_wave_max))( # New args
            delayed(_process_pixel_data)(*task_args) for task_args in processed_tasks_args
        )
    print("\nFinished parallel pixel processing.")

    # --- Aggregate Results (Sequential) ---
    # Populate the pre-initialized maps with results from parallel processing
    # Need to map results back to their original (ii, jj) grid positions
    # Since tasks were sorted, results will be in sorted order.
    # We need to use the original (ii, jj) from the 'tasks' list to place results correctly.
    for k, pixel_result_tuple in enumerate(results): # results are now tuples of (ii, jj, pixel_data)
        # Unpack the tuple: the first two elements are ii, jj, the third is pixel_data
        ii, jj, pixel_data = pixel_result_tuple 
        
        # Get original coordinates from the sorted tasks list.
        # This is crucial because `results` are ordered by `tasks` which are sorted by complexity.
        # The `ii` and `jj` passed *into* _process_pixel_data are the original grid coordinates.
        # So we use those directly, no need to lookup from `tasks[k]['coords']`
        original_ii, original_jj = ii, jj 
        
        map_stars_mass[original_ii][original_jj] = pixel_data['map_stars_mass']
        map_mw_age[original_ii][original_jj] = pixel_data['map_mw_age']
        map_stars_mw_zsol[original_ii][original_jj] = pixel_data['map_stars_mw_zsol']
        map_sfr_100[original_ii][original_jj] = pixel_data['map_sfr_100']
        map_sfr_30[original_ii][original_jj] = pixel_data['map_sfr_30']
        map_sfr_10[original_ii][original_jj] = pixel_data['map_sfr_10']

        map_gas_mass[original_ii][original_jj] = pixel_data['map_gas_mass']
        map_sfr_inst[original_ii][original_jj] = pixel_data['map_sfr_inst']
        map_gas_mw_zsol[original_ii][original_jj] = pixel_data['map_gas_mw_zsol']

        map_dust_mean_tauV[original_ii][original_jj] = pixel_data['map_dust_mean_tauV']
        map_dust_mean_AV[original_ii][original_jj] = pixel_data['map_dust_mean_AV']

        map_flux[original_ii][original_jj] = pixel_data['map_flux']
        map_flux_dust[original_ii][original_jj] = pixel_data['map_flux_dust']

        if output_pixel_spectra:
            map_spectra_nodust[original_ii][original_jj] = pixel_data['obs_spectra_nodust_igm']
            map_spectra_dust[original_ii][original_jj] = pixel_data['obs_spectra_dust_igm']


    print("All calculations complete. Maps populated.")

    # convert flux units
    for i_band in range(nbands):
        map_flux[:,:,i_band] = convert_flux_map(map_flux[:,:,i_band], filter_wave_eff[filters[i_band]], to_unit=flux_unit, pixel_scale_arcsec=pix_arcsec)
        map_flux_dust[:,:,i_band] = convert_flux_map(map_flux_dust[:,:,i_band], filter_wave_eff[filters[i_band]], to_unit=flux_unit, pixel_scale_arcsec=pix_arcsec)

    # rescaling maps if flux_unit is 'erg/s/cm2/A'
    if flux_unit == 'erg/s/cm2/A':
        flux_scale = 1e-20
    else:
        flux_scale = 1.0
    
    map_flux = map_flux/flux_scale
    map_flux_dust = map_flux_dust/flux_scale

    # --- Save results to FITS file (New Block) ---
    if name_out_img is not None:
        try:
            # Create a list of HDUs
            hdul = fits.HDUList()

            # Primary HDU: Can be empty or hold a basic header
            if map_flux.shape[2] > 0:
                primary_data = map_flux[:, :, 0] # First band for primary
                prihdr = fits.Header()
                prihdr['COMMENT'] = 'Primary Image: First band (no dust)'
                prihdr['CRPIX1'] = dimx / 2.0 + 0.5 # Reference pixel X (center)
                prihdr['CRPIX2'] = dimy / 2.0 + 0.5 # Reference pixel Y (center)
                prihdr['CDELT1'] = pix_kpc # Pixel scale X (kpc/pixel)
                prihdr['CDELT2'] = pix_kpc # Pixel scale Y (kpc/pixel)
                prihdr['CUNIT1'] = 'kpc'
                prihdr['CUNIT2'] = 'kpc'
                prihdr['REDSHIFT'] = snap_z
                prihdr['POLAR'] = polar_angle_deg
                prihdr['AZIMUTH'] = azimuth_angle_deg
                prihdr['DIM_KPC'] = dim_kpc
                prihdr['PIX_KPC'] = pix_kpc
                prihdr['PIXSIZE'] = pix_arcsec
                prihdr['BUNIT'] = flux_unit
                prihdr['SCALE'] = flux_scale

                primary_hdu = fits.PrimaryHDU(data=primary_data, header=prihdr)
                hdul.append(primary_hdu)

                # Add extensions for other bands (no dust)
                for i_band in range(nbands):
                    ext_hdr = fits.Header()
                    ext_hdr['EXTNAME'] = 'NODUST_'+filters[i_band].upper()
                    ext_hdr['FILTER'] = filters[i_band]
                    ext_hdr['COMMENT'] = f'Flux for filter: {filters[i_band]}'
                    ext_hdu = fits.ImageHDU(data=map_flux[:, :, i_band], header=ext_hdr)
                    hdul.append(ext_hdu)

                # Add extensions for other bands (with dust)
                for i_band in range(nbands):
                    ext_hdr = fits.Header()
                    ext_hdr['EXTNAME'] = 'DUST_'+filters[i_band].upper()
                    ext_hdr['FILTER'] = filters[i_band]
                    ext_hdr['COMMENT'] = f'Flux (with dust) for filter: {filters[i_band]}'
                    ext_hdu = fits.ImageHDU(data=map_flux_dust[:, :, i_band], header=ext_hdr)
                    hdul.append(ext_hdu)
            else:
                # If no flux bands, create an empty primary HDU
                hdul.append(fits.PrimaryHDU())

            # Add other maps as ImageHDU extensions
            map_data_to_save = {
                'STARS_MASS': map_stars_mass,
                'MW_AGE': map_mw_age,
                'STARS_MW_ZSOL': map_stars_mw_zsol,
                'SFR_100MYR': map_sfr_100,
                'SFR_30MYR': map_sfr_30,
                'SFR_10MYR': map_sfr_10,
                'GAS_MASS': map_gas_mass,
                'SFR_INST': map_sfr_inst,
                'GAS_MW_ZSOL': map_gas_mw_zsol,
                'DUST_MEAN_TAUV': map_dust_mean_tauV,
                'DUST_MEAN_AV': map_dust_mean_AV,
            }

            for map_name, data_array in map_data_to_save.items():
                if data_array is not None: # Ensure the array exists
                    ext_hdr = fits.Header()
                    ext_hdr['EXTNAME'] = map_name
                    ext_hdr['COMMENT'] = f'Map of {map_name.replace("_", " ").title()}'
                    hdul.append(fits.ImageHDU(data=data_array, header=ext_hdr))

            # Add observed-frame spectra extensions if requested
            if output_pixel_spectra:
                # No-dust spectra
                ext_hdr_nodust_spec = fits.Header()
                ext_hdr_nodust_spec['EXTNAME'] = 'OBS_SPEC_NODUST' # Changed EXTNAME
                ext_hdr_nodust_spec['COMMENT'] = 'Observed-frame spectra (no dust attenuation, with IGM)' # Changed comment
                ext_hdr_nodust_spec['CRPIX1'] = dimx / 2.0 + 0.5 # Reference pixel X (center)
                ext_hdr_nodust_spec['CRPIX2'] = dimy / 2.0 + 0.5 # Reference pixel Y (center)
                ext_hdr_nodust_spec['CDELT1'] = pix_kpc # Pixel scale X (kpc/pixel)
                ext_hdr_nodust_spec['CDELT2'] = pix_kpc # Pixel scale Y (kpc/pixel)
                ext_hdr_nodust_spec['CUNIT1'] = 'kpc'
                ext_hdr_nodust_spec['CUNIT2'] = 'kpc'
                # Add wavelength axis info for observed frame
                ext_hdr_nodust_spec['CRPIX3'] = 1.0 # Reference pixel for wavelength axis (first point)
                ext_hdr_nodust_spec['CDELT3'] = (global_output_obs_wave[1] - global_output_obs_wave[0]) if global_output_obs_wave.size > 1 else 0.0 # Wavelength step
                ext_hdr_nodust_spec['CRVAL3'] = global_output_obs_wave[0] if global_output_obs_wave.size > 0 else 0.0 # Starting wavelength
                ext_hdr_nodust_spec['CUNIT3'] = 'Angstrom'
                ext_hdr_nodust_spec['BUNIT'] = 'erg/s/cm2/Angstrom' # Units of the spectra
                hdul.append(fits.ImageHDU(data=map_spectra_nodust, header=ext_hdr_nodust_spec))

                # Dust-attenuated spectra
                ext_hdr_dust_spec = fits.Header()
                ext_hdr_dust_spec['EXTNAME'] = 'OBS_SPEC_DUST' # Changed EXTNAME
                ext_hdr_dust_spec['COMMENT'] = 'Observed-frame spectra (with dust attenuation, with IGM)' # Changed comment
                ext_hdr_dust_spec['CRPIX1'] = dimx / 2.0 + 0.5 # Reference pixel X (center)
                ext_hdr_dust_spec['CRPIX2'] = dimy / 2.0 + 0.5 # Reference pixel Y (center)
                ext_hdr_dust_spec['CDELT1'] = pix_kpc # Pixel scale X (kpc/pixel)
                ext_hdr_dust_spec['CDELT2'] = pix_kpc # Pixel scale Y (kpc/pixel)
                ext_hdr_dust_spec['CUNIT1'] = 'kpc'
                ext_hdr_dust_spec['CUNIT2'] = 'kpc'
                # Add wavelength axis info for observed frame
                ext_hdr_dust_spec['CRPIX3'] = 1.0
                ext_hdr_dust_spec['CDELT3'] = (global_output_obs_wave[1] - global_output_obs_wave[0]) if global_output_obs_wave.size > 1 else 0.0
                ext_hdr_dust_spec['CRVAL3'] = global_output_obs_wave[0] if global_output_obs_wave.size > 0 else 0.0
                ext_hdr_dust_spec['CUNIT3'] = 'Angstrom'
                ext_hdr_dust_spec['BUNIT'] = 'erg/s/cm2/Angstrom'
                hdul.append(fits.ImageHDU(data=map_spectra_dust, header=ext_hdr_dust_spec))


            # Ensure the directory exists
            output_dir = os.path.dirname(name_out_img)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)

            # Write the HDUList to the FITS file
            hdul.writeto(name_out_img, overwrite=True, output_verify='fix')
            print(f"Galaxy image synthesis completed successfully and results saved to FITS file: {name_out_img}")

        except Exception as e:
            print(f"Error saving FITS file {name_out_img}: {e}")
