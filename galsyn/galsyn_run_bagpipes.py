import h5py
import os, sys
import numpy as np
from astropy.io import fits
from .imgutils import *
from .utils import *
from .dust import *
from joblib import Parallel, delayed
from tqdm.auto import tqdm
from tqdm_joblib import tqdm_joblib
import multiprocessing
from scipy.interpolate import interp1d, RegularGridInterpolator
from scipy.integrate import simpson
import importlib.resources

import bagpipes as pipes

# Constants for solar metallicity (from Bagpipes documentation or common usage)
BAGPIPES_Z_SUN = 0.02
L_SUN_ERG_S = 3.828e33 # Solar luminosity in erg/s
PRIMORDIAL_Z_SUN_VALUE = 0.0127 # This is likely a constant for the simulation data's metallicity definition

# Global variables for SSP data (loaded once per worker if use_precomputed_ssp is True)
ssp_wave = None
ssp_ages_gyr = None
ssp_logzsol_grid = None
ssp_spectra_grid = None
ssp_stellar_mass_grid = None
ssp_code_z_sun = None

_global_ssp_spectra_interpolator = None
_global_ssp_stellar_mass_interpolator = None

# Global variables for Bagpipes instance components (initialized once per worker if use_precomputed_ssp is False)
_ssp_worker_bagpipes_model_components = None

# Other global worker variables
igm_trans = None
snap_z = None
pix_area_kpc2 = None
mean_AV_unres = None
# add_neb_emission = None # Removed as it's always True
gas_logu = None
# add_igm_absorption = None # Removed as it's always True
igm_type = None
dust_index_bc = None
dust_index = None
t_esc = None
dust_law = None
bump_amp = None
salim_a0 = None
salim_a1 = None
salim_a2 = None
salim_a3 = None
salim_RV = None
salim_B = None
dust_Alambda_per_AV = None
func_interp_dust_index = None
use_precomputed_ssp = False
ssp_interpolation_method = 'nearest'

# Global variables for dustindexAV_AV and dustindexAV_dust_index
dustindexAV_AV = None
dustindexAV_dust_index = None

_worker_scale_dust_tau = None

output_pixel_spectra_flag = False
# Renamed from global_output_obs_wave to _worker_output_obs_wave_grid
# as it's defined and used within each worker for interpolation.
# The actual common output grid is 'fixed_global_output_obs_wave'
# determined in the main process.
_worker_output_obs_wave_grid = None 

_worker_filters = None
_worker_filter_transmission = None
_worker_filter_wave_eff = None
_worker_cosmo = None

# Global variables for light-weighted calculation wavelength range
_lw_wave_min_rest = 1000.0 # Angstrom
_lw_wave_max_rest = 30000.0 # Angstrom


def _load_filter_transmission_from_paths(filters_list, filter_transmission_path_dict):
    filter_transmission_data = {}
    filter_wave_pivot_data = {}

    for f_name in filters_list:
        if f_name not in filter_transmission_path_dict:
            raise ValueError(f"Path for filter '{f_name}' not provided in filter_transmission_path.")
        
        file_path = filter_transmission_path_dict[f_name]
        try:
            data = np.loadtxt(file_path)
            if data.shape[1] != 2:
                raise ValueError(f"Filter file {file_path} must have exactly two columns (wavelength, transmission).")
            
            wave = data[:, 0]
            trans = data[:, 1]

            filter_transmission_data[f_name] = {'wave': wave, 'trans': trans}

            numerator = simpson(wave * trans, wave)
            integrand_denominator = np.where(wave != 0, trans / wave, 0)
            denominator = simpson(integrand_denominator, wave)

            if denominator > 0:
                pivot_wavelength = np.sqrt(numerator / denominator)
            else:
                pivot_wavelength = np.nan

            filter_wave_pivot_data[f_name] = pivot_wavelength

        except Exception as e:
            print(f"Error loading or processing filter file {file_path} for filter {f_name}: {e}")
            raise

    return filter_transmission_data, filter_wave_pivot_data


def init_worker(ssp_code_val, snap_z_val, pix_area_kpc2_val, mean_AV_unres_val, 
                filters_list_val, filter_transmission_path_val,
                gas_logu_val, # Removed add_neb_emission_val
                igm_type_val, dust_index_bc_val, # Removed add_igm_absorption_val
                dust_index_val, t_esc_val, precomputed_scale_dust_tau_val,
                cosmo_str_val, cosmo_h_val, XH_val, 
                dust_law_val, bump_amp_val, relation_AVslope_val, salim_a0_val, 
                salim_a1_val, salim_a2_val, salim_a3_val, salim_RV_val, salim_B_val,
                use_precomputed_ssp_val, ssp_filepath_val=None, ssp_interpolation_method_val='nearest', 
                output_pixel_spectra_val=False, output_obs_wave_grid_val=None): 
    
    global ssp_wave, ssp_ages_gyr, ssp_logzsol_grid, ssp_spectra_grid, ssp_stellar_mass_grid, ssp_code_z_sun
    global _global_ssp_spectra_interpolator, _global_ssp_stellar_mass_interpolator
    global _ssp_worker_bagpipes_model_components, igm_trans, snap_z, pix_area_kpc2
    global mean_AV_unres, gas_logu, igm_type # Removed add_neb_emission, add_igm_absorption
    global dust_index_bc, dust_index, t_esc, dust_law, bump_amp, salim_a0, salim_a1, salim_a2, salim_a3
    global salim_RV, salim_B, dust_Alambda_per_AV, func_interp_dust_index
    global use_precomputed_ssp, ssp_interpolation_method
    global output_pixel_spectra_flag, _worker_output_obs_wave_grid
    global _worker_filters, _worker_filter_transmission, _worker_filter_wave_eff, _worker_cosmo
    global dustindexAV_AV, dustindexAV_dust_index
    global _worker_scale_dust_tau

    snap_z = snap_z_val
    pix_area_kpc2 = pix_area_kpc2_val
    mean_AV_unres = mean_AV_unres_val
    
    # These are now fixed to True (1)
    # add_neb_emission = add_neb_emission_val
    gas_logu = gas_logu_val
    # add_igm_absorption = add_igm_absorption_val
    igm_type = igm_type_val
    dust_index_bc = dust_index_bc_val
    t_esc = t_esc_val
    _worker_scale_dust_tau = precomputed_scale_dust_tau_val
    
    _worker_cosmo = define_cosmo(cosmo_str_val)
    
    dust_law = dust_law_val
    salim_a0 = salim_a0_val
    salim_a1 = salim_a1_val
    salim_a2 = salim_a2_val
    salim_a3 = salim_a3_val
    salim_RV = salim_RV_val
    salim_B = salim_B_val
    dust_index = dust_index_val
    bump_amp = bump_amp_val
    use_precomputed_ssp = use_precomputed_ssp_val
    ssp_interpolation_method = ssp_interpolation_method_val

    _worker_filters = filters_list_val
    _worker_filter_transmission, _worker_filter_wave_eff = _load_filter_transmission_from_paths(
        _worker_filters, filter_transmission_path_val
    )

    output_pixel_spectra_flag = output_pixel_spectra_val
    _worker_output_obs_wave_grid = output_obs_wave_grid_val # Set the global worker wave grid
    
    if use_precomputed_ssp:
        if ssp_filepath_val is None:
            print("Error: ssp_filepath must be provided when use_precomputed_ssp is True.")
            sys.exit(1)
        try:
            with h5py.File(ssp_filepath_val, 'r') as f_ssp:
                ssp_wave = f_ssp['wavelength'][:]
                ssp_ages_gyr = f_ssp['ages_gyr'][:]
                ssp_logzsol_grid = f_ssp['logzsol'][:]
                ssp_spectra_grid = f_ssp['spectra'][:]
                ssp_stellar_mass_grid = f_ssp['stellar_mass'][:]
                ssp_code_z_sun = f_ssp.attrs['z_sun']
                
                if ssp_interpolation_method == 'linear':
                    _global_ssp_spectra_interpolator = RegularGridInterpolator(
                        (ssp_ages_gyr, ssp_logzsol_grid), ssp_spectra_grid, 
                        method='linear', bounds_error=False, fill_value=0.0
                    )
                    _global_ssp_stellar_mass_interpolator = RegularGridInterpolator(
                        (ssp_ages_gyr, ssp_logzsol_grid), ssp_stellar_mass_grid, 
                        method='linear', bounds_error=False, fill_value=0.0
                    )
                elif ssp_interpolation_method == 'cubic':
                    _global_ssp_spectra_interpolator = RegularGridInterpolator(
                        (ssp_ages_gyr, ssp_logzsol_grid), ssp_spectra_grid, 
                        method='cubic', bounds_error=False, fill_value=0.0
                    )
                    _global_ssp_stellar_mass_interpolator = RegularGridInterpolator(
                        (ssp_ages_gyr, ssp_logzsol_grid), ssp_stellar_mass_grid, 
                        method='cubic', bounds_error=False, fill_value=0.0
                    )

        except Exception as e:
            print(f"Error loading SSP grid from {ssp_filepath_val}: {e}")
            sys.exit(1)
    else:
        dust = {}
        dust["type"] = "Calzetti"
        dust["Av"] = 0.0
        dust["eta"] = 1.0

        nebular = {}
        nebular["logU"] = gas_logu_val

        model_components = {}
        model_components["redshift"] = 0.0
        model_components["veldisp"] = 0
        model_components["dust"] = dust
        model_components["nebular"] = nebular
        
        _ssp_worker_bagpipes_model_components = model_components

        # Remove spec_wavs from the dummy call to let Bagpipes use its default.
        dummy_burst = {"age": 0.01, "massformed": 1.0, "metallicity": 1.0}
        dummy_model_components = _ssp_worker_bagpipes_model_components.copy()
        dummy_model_components["burst"] = dummy_burst
        dummy_model = pipes.model_galaxy(dummy_model_components)
        ssp_wave = dummy_model.wavelengths
        ssp_code_z_sun = BAGPIPES_Z_SUN

    # Handle relation_AVslope_val
    if isinstance(relation_AVslope_val, str):
        data_file_name = f"{relation_AVslope_val}_AV_dust_index.txt"
        try:
            data_path = str(importlib.resources.files('galsyn.data').joinpath(data_file_name))
            data = np.loadtxt(data_path)
            dustindexAV_AV = data[:, 0]
            dustindexAV_dust_index = data[:, 1]
        except Exception as e:
            print(f"Error loading dust relation data from {data_file_name}: {e}")
            sys.exit(1)
    elif isinstance(relation_AVslope_val, dict):
        dustindexAV_AV = np.asarray(relation_AVslope_val["AV"])
        dustindexAV_dust_index = np.asarray(relation_AVslope_val["dust_index"])
    else:
        print("Error: Invalid relation_AVslope_val type passed to init_worker.")
        dustindexAV_AV = np.array([])
        dustindexAV_dust_index = np.array([])
        sys.exit(1)


    if dust_law <= 1:
        global func_interp_dust_index
        func_interp_dust_index = interp1d(dustindexAV_AV, dustindexAV_dust_index, bounds_error=False, fill_value='extrapolate')

    elif dust_law == 2:
        bump_amp = bump_amp_from_dust_index(dust_index)
        dust_Alambda_per_AV = modified_calzetti_dust_Alambda_per_AV(ssp_wave, dust_index=dust_index, bump_amp=bump_amp)

    elif dust_law == 3:
        dust_Alambda_per_AV = modified_calzetti_dust_Alambda_per_AV(ssp_wave, dust_index=dust_index, bump_amp=bump_amp)

    elif dust_law == 4:
        dust_Alambda_per_AV = salim18_dust_Alambda_per_AV(ssp_wave, salim_a0, salim_a1, salim_a2, salim_a3, salim_B, salim_RV)

    elif dust_law == 5:
        dust_Alambda_per_AV = calzetti_dust_Alambda_per_AV(ssp_wave)

    # Always add IGM absorption
    # if add_igm_absorption == 1: # This check is removed
    if igm_type == 0:
        igm_trans = igm_att_madau(ssp_wave * (1.0+snap_z), snap_z)
    elif igm_type == 1:
        igm_trans = igm_att_inoue(ssp_wave * (1.0+snap_z), snap_z)
    else:
        print ('igm_type is not recognized! options are: 1 for Madau+1995 and Inoue+2014')
        sys.exit()
    # else: # This else block is removed
    #     igm_trans = 1


def dust_reddening_diffuse_ism(dust_AV, wave, dust_law):
    if dust_law == 0:
        dust_index1 = func_interp_dust_index(dust_AV)
        bump_amp1 = bump_amp_from_dust_index(dust_index1)
        Alambda = modified_calzetti_dust_Alambda_per_AV(wave, dust_index=dust_index1, bump_amp=bump_amp1) * dust_AV

    elif dust_law == 1:
        dust_index1 = func_interp_dust_index(dust_AV)
        Alambda = modified_calzetti_dust_Alambda_per_AV(wave, dust_index=dust_index1, bump_amp=bump_amp) * dust_AV

    return Alambda


def _process_pixel_data(ii, jj, star_particle_membership_list, gas_particle_membership_list, 
                        stars_mass, stars_age, stars_zsol, stars_init_mass, stars_vel_los_proj,
                        gas_mass, gas_sfr_inst, gas_zsol, gas_log_temp, gas_mass_H, gas_vel_los_proj):
    """
    ii=y jj=x
    Worker function to process calculations for a single pixel (ii, jj).
    This function will be executed in parallel.
    
    star_particle_membership_list: List of (original_particle_index, line_of_sight_distance) for THIS pixel.
    gas_particle_membership_list: List of (original_particle_index, line_of_sight_distance) for THIS pixel.
    """
    # Use the pre-defined _worker_output_obs_wave_grid for the output spectra dimensions
    current_num_obs_wave_points = len(_worker_output_obs_wave_grid) if output_pixel_spectra_flag else 0

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
        'map_flux': np.zeros(len(_worker_filters)),
        'map_flux_dust': np.zeros(len(_worker_filters)),
        'obs_spectra_nodust_igm': np.zeros(current_num_obs_wave_points),
        'obs_spectra_dust_igm': np.zeros(current_num_obs_wave_points),
        'map_lw_age_nodust': np.nan,
        'map_lw_age_dust': np.nan,
        'map_lw_zsol_nodust': np.nan,
        'map_lw_zsol_dust': np.nan,
        'map_stars_mw_vel_los': np.nan,
        'map_gas_mw_vel_los': np.nan,
        'map_stars_vel_disp_los': np.nan,
        'map_gas_vel_disp_los': np.nan,
        'map_lw_vel_los_nodust': np.nan,
        'map_lw_vel_los_dust': np.nan
    }

    star_ids0 = np.asarray([x[0] for x in star_particle_membership_list], dtype=int)
    star_los_dist0 = np.asarray([x[1] for x in star_particle_membership_list])

    idx_valid_stars_in_pixel = np.where((np.isnan(stars_mass[star_ids0]) == False) &
                                        (np.isnan(stars_age[star_ids0]) == False) &
                                        (np.isnan(stars_zsol[star_ids0]) == False) &
                                        (np.isnan(stars_vel_los_proj[star_ids0]) == False))[0]
    star_ids = star_ids0[idx_valid_stars_in_pixel]
    star_los_dist = star_los_dist0[idx_valid_stars_in_pixel]

    gas_ids0 = np.asarray([x[0] for x in gas_particle_membership_list], dtype=int)
    gas_los_dist0 = np.asarray([x[1] for x in gas_particle_membership_list])
    
    idxg = np.where((np.isnan(gas_mass[gas_ids0]) == False) &
                    (np.isnan(gas_vel_los_proj[gas_ids0]) == False))[0]
    gas_ids = gas_ids0[idxg]
    gas_los_dist = gas_los_dist0[idxg]

    current_stars_mass_sum = np.nansum(stars_mass[star_ids])
    pixel_results['map_stars_mass'] = current_stars_mass_sum

    if current_stars_mass_sum > 0:
        pixel_results['map_mw_age'] = np.nansum(stars_mass[star_ids] * stars_age[star_ids]) / current_stars_mass_sum
        pixel_results['map_stars_mw_zsol'] = np.nansum(stars_mass[star_ids] * stars_zsol[star_ids]) / current_stars_mass_sum
        pixel_results['map_stars_mw_vel_los'] = np.nansum(stars_mass[star_ids] * stars_vel_los_proj[star_ids]) / current_stars_mass_sum
        pixel_results['map_stars_vel_disp_los'] = np.sqrt(np.nansum(stars_mass[star_ids] * (stars_vel_los_proj[star_ids] - pixel_results['map_stars_mw_vel_los'])**2) / current_stars_mass_sum)
    else:
        pixel_results['map_mw_age'] = np.nan
        pixel_results['map_stars_mw_zsol'] = np.nan
        pixel_results['map_stars_mw_vel_los'] = np.nan
        pixel_results['map_stars_vel_disp_los'] = np.nan

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
        pixel_results['map_gas_mw_vel_los'] = np.nansum(gas_mass[gas_ids] * gas_vel_los_proj[gas_ids]) / current_gas_mass_sum
        pixel_results['map_gas_vel_disp_los'] = np.sqrt(np.nansum(gas_mass[gas_ids] * (gas_vel_los_proj[gas_ids] - pixel_results['map_gas_mw_vel_los'])**2) / current_gas_mass_sum)
    else:
        pixel_results['map_gas_mw_zsol'] = np.nan
        pixel_results['map_gas_mw_vel_los'] = np.nan
        pixel_results['map_gas_vel_disp_los'] = np.nan

    if len(star_ids) > 0:

        array_spec = []
        array_spec_dust = []
        array_AV = []
        array_tauV = []
        
        # New arrays for light-weighted calculations
        array_L_nodust = []
        array_L_dust = []
        array_vel_los = [] # For light-weighted velocity

        wave = ssp_wave

        # Define the rest-frame wavelength range for light-weighting
        lw_wave_idx = np.where((wave >= _lw_wave_min_rest) & (wave <= _lw_wave_max_rest))[0]
        if lw_wave_idx.size == 0:
            # If no wavelengths fall in range, light-weighted quantities will be NaN
            pass

        for i_sid in range(len(star_ids)):
            star_id = star_ids[i_sid]
            star_vel_los_current = stars_vel_los_proj[star_id]

            if use_precomputed_ssp:
                particle_logzsol_ssp_code = np.log10( (stars_zsol[star_id] * PRIMORDIAL_Z_SUN_VALUE) / ssp_code_z_sun )
                
                points = np.array([[stars_age[star_id], particle_logzsol_ssp_code]])

                if ssp_interpolation_method == 'linear':
                    spec = _global_ssp_spectra_interpolator(points)[0]
                    ssp_mass_formed = _global_ssp_stellar_mass_interpolator(points)[0]
                elif ssp_interpolation_method == 'cubic':
                    spec = _global_ssp_spectra_interpolator(points)[0]
                    ssp_mass_formed = _global_ssp_stellar_mass_interpolator(points)[0]
                else: # 'nearest' method
                    age_idx = np.argmin(np.abs(ssp_ages_gyr - stars_age[star_id]))
                    z_idx = np.argmin(np.abs(ssp_logzsol_grid - particle_logzsol_ssp_code))
                    
                    spec = ssp_spectra_grid[age_idx, z_idx, :]
                    ssp_mass_formed = ssp_stellar_mass_grid[age_idx, z_idx]
            else: # On-the-fly Bagpipes
                metallicity_z_zsun_bagpipes = (stars_zsol[star_id] * PRIMORDIAL_Z_SUN_VALUE) / BAGPIPES_Z_SUN

                burst = {}
                burst["age"] = stars_age[star_id]
                burst["massformed"] = 1.0
                burst["metallicity"] = metallicity_z_zsun_bagpipes

                current_model_components = _ssp_worker_bagpipes_model_components.copy()
                current_model_components["burst"] = burst

                # Use the global ssp_wave determined in init_worker for consistent wavelength grid
                model = pipes.model_galaxy(current_model_components, spec_wavs=wave)
                
                rest_frame_fluxes_erg_s_aa = model.spectrum_full

                spec = rest_frame_fluxes_erg_s_aa / L_SUN_ERG_S

                surv_stellar_mass = model.sfh.stellar_mass
                ssp_mass_formed = surv_stellar_mass

            # Apply Doppler shift to the spectrum
            wave_doppler, spec_doppler = doppler_shift_spectrum(wave, spec, star_vel_los_current)
            spec = interp1d(wave_doppler, spec_doppler, kind='linear', bounds_error=False, fill_value=0.0)(wave)

            idxg_front = np.where(gas_los_dist < star_los_dist[i_sid])[0]
            front_gas_ids = gas_ids[idxg_front]
            idxg1 = np.where((gas_sfr_inst[front_gas_ids] > 0.0) | (gas_log_temp[front_gas_ids] < 3.9))[0]
            cold_front_gas_ids = front_gas_ids[idxg1]

            spec_dust = spec.copy() # Initialize spec_dust with original spec

            if len(cold_front_gas_ids) > 0:
                temp_mw_gas_zsol = np.nansum(gas_mass[cold_front_gas_ids]*gas_zsol[cold_front_gas_ids])/np.nansum(gas_mass[cold_front_gas_ids])
                nH = np.nansum(gas_mass_H[cold_front_gas_ids])*1.247914e+14/pix_area_kpc2
                # Use the new global variable _worker_scale_dust_tau
                tauV = _worker_scale_dust_tau * temp_mw_gas_zsol * nH / 2.1e+21
                dust_AV = -2.5*np.log10((1.0 - np.exp(-1.0*tauV))/tauV)

                if not (np.isnan(dust_AV) or dust_AV == 0.0):
                    if dust_law <= 1:
                        Alambda = dust_reddening_diffuse_ism(dust_AV, wave, dust_law)
                    else:
                        Alambda = dust_Alambda_per_AV * dust_AV
                    
                    spec_dust = spec_dust*np.power(10.0, -0.4*Alambda)
                    array_tauV.append(tauV)
                    array_AV.append(dust_AV)
            
            if stars_age[star_id] <= t_esc:
                Alambda = unresolved_dust_birth_cloud_Alambda_per_AV(wave, dust_index_bc=dust_index_bc) * mean_AV_unres
                spec_dust = spec_dust*np.power(10.0, -0.4*Alambda)
    
            norm = stars_mass[star_id] / ssp_mass_formed

            if len(np.asarray(spec_dust).shape) == 1:
                array_spec.append(spec*norm)
                array_spec_dust.append(spec_dust*norm)

                # Calculate luminosity for light-weighting
                # Integrate over specified rest-frame wavelength range
                if lw_wave_idx.size > 1: # Need at least 2 points to integrate
                    L_nodust_particle = simpson(spec[lw_wave_idx]*norm, wave[lw_wave_idx])
                    L_dust_particle = simpson(spec_dust[lw_wave_idx]*norm, wave[lw_wave_idx])
                else: # No valid wavelength range for integration
                    L_nodust_particle = 0.0
                    L_dust_particle = 0.0
                
                array_L_nodust.append(L_nodust_particle)
                array_L_dust.append(L_dust_particle)
                array_vel_los.append(star_vel_los_current)

        array_AV, array_tauV = np.asarray(array_AV), np.asarray(array_tauV)
        if array_AV.size == 0:
            mean_AV = np.nan
            mean_tauV = np.nan
        else:
            mean_AV = np.nanmean(np.asarray(array_AV))
            mean_tauV = np.nanmean(np.asarray(array_tauV))

        redshift_flux, redshift_flux_dust = [], []
        
        if len(array_spec) > 0:
            spec_lum = np.nansum(array_spec, axis=0)
            spec_lum_dust = np.nansum(array_spec_dust, axis=0)

            spec_wave_obs, spec_flux_obs = cosmo_redshifting(wave, spec_lum, snap_z, _worker_cosmo)
            spec_wave_obs, spec_flux_dust_obs = cosmo_redshifting(wave, spec_lum_dust, snap_z, _worker_cosmo)

            spec_flux_obs_igm = spec_flux_obs * igm_trans
            spec_flux_dust_obs_igm = spec_flux_dust_obs * igm_trans

            if output_pixel_spectra_flag:
                # Interpolate onto the fixed output observed-frame wavelength grid
                if _worker_output_obs_wave_grid.size > 0:
                    interp_func_nodust = interp1d(spec_wave_obs, spec_flux_obs_igm, kind='linear', 
                                                  bounds_error=False, fill_value=0.0)
                    interp_func_dust = interp1d(spec_wave_obs, spec_flux_dust_obs_igm, kind='linear', 
                                                bounds_error=False, fill_value=0.0)
                    
                    pixel_results['obs_spectra_nodust_igm'] = interp_func_nodust(_worker_output_obs_wave_grid)
                    pixel_results['obs_spectra_dust_igm'] = interp_func_dust(_worker_output_obs_wave_grid)
                else:
                    # If the target output grid is empty, ensure the output spectra arrays are also empty
                    pixel_results['obs_spectra_nodust_igm'] = np.zeros(0)
                    pixel_results['obs_spectra_dust_igm'] = np.zeros(0)

            nbands = len(_worker_filters)
            redshift_flux = np.zeros(nbands)
            redshift_flux_dust = np.zeros(nbands)

            for i_band in range(nbands):
                redshift_flux[i_band] = filtering(spec_wave_obs, spec_flux_obs_igm, _worker_filter_transmission[_worker_filters[i_band]]['wave'], _worker_filter_transmission[_worker_filters[i_band]]['trans'])
                redshift_flux_dust[i_band] = filtering(spec_wave_obs, spec_flux_dust_obs_igm, _worker_filter_transmission[_worker_filters[i_band]]['wave'], _worker_filter_transmission[_worker_filters[i_band]]['trans'])

            if len(redshift_flux) > 0:
                pixel_results['map_flux'] = redshift_flux
                pixel_results['map_flux_dust'] = redshift_flux_dust

                pixel_results['map_dust_mean_tauV'] = mean_tauV
                pixel_results['map_dust_mean_AV'] = mean_AV

            # Calculate Light-weighted age and metallicity
            total_L_nodust = np.nansum(array_L_nodust)
            total_L_dust = np.nansum(array_L_dust)
            array_vel_los = np.asarray(array_vel_los)

            if total_L_nodust > 0:
                pixel_results['map_lw_age_nodust'] = np.nansum(np.asarray(array_L_nodust) * stars_age[star_ids]) / total_L_nodust
                pixel_results['map_lw_zsol_nodust'] = np.nansum(np.asarray(array_L_nodust) * stars_zsol[star_ids]) / total_L_nodust
                pixel_results['map_lw_vel_los_nodust'] = np.nansum(np.asarray(array_L_nodust) * array_vel_los) / total_L_nodust
            else:
                pixel_results['map_lw_age_nodust'] = np.nan
                pixel_results['map_lw_zsol_nodust'] = np.nan
                pixel_results['map_lw_vel_los_nodust'] = np.nan

            if total_L_dust > 0:
                pixel_results['map_lw_age_dust'] = np.nansum(np.asarray(array_L_dust) * stars_age[star_ids]) / total_L_dust
                pixel_results['map_lw_zsol_dust'] = np.nansum(np.asarray(array_L_dust) * stars_zsol[star_ids]) / total_L_dust
                pixel_results['map_lw_vel_los_dust'] = np.nansum(np.asarray(array_L_dust) * array_vel_los) / total_L_dust
            else:
                pixel_results['map_lw_age_dust'] = np.nan
                pixel_results['map_lw_zsol_dust'] = np.nan
                pixel_results['map_lw_vel_los_dust'] = np.nan

    return ii, jj, pixel_results


def generate_images(sim_file, z, filters, filter_transmission_path, dim_kpc=None,
                    pix_arcsec=0.02, flux_unit='MJy/sr', polar_angle_deg=0, azimuth_angle_deg=0,
                    name_out_img=None, n_jobs=-1, ssp_code='Bagpipes', gas_logu=-2.0, # Removed add_neb_emission
                    igm_type=0, dust_index_bc=-0.7, dust_index=0.0, t_esc=0.01, # Removed add_igm_absorption
                    scale_dust_redshift="Vogelsberger20", cosmo_str='Planck18', cosmo_h=0.6774, XH=0.76,
                    dust_law=0, bump_amp=0.85, relation_AVslope="Salim18", salim_a0=-4.30, 
                    salim_a1=2.71, salim_a2= -0.191, salim_a3=0.0121, salim_RV=3.15, salim_B=1.57, 
                    initdim_kpc=200, initdim_mass_fraction=0.99, use_precomputed_ssp=True, 
                    ssp_filepath=None, ssp_interpolation_method='nearest', 
                    output_pixel_spectra=False, rest_wave_min=1000.0, rest_wave_max=16000.0): 

    cosmo = define_cosmo(cosmo_str)

    print ('Processing '+sim_file)

    filter_transmission_data_global, filter_wave_pivot_data_global = _load_filter_transmission_from_paths(filters, filter_transmission_path)

    snap_z = z
    snap_a = 1.0/(1.0 + snap_z)
    snap_univ_age = cosmo.age(snap_z).value

    pix_kpc = angular_to_physical(snap_z, pix_arcsec, cosmo)
    pix_area_kpc2 = pix_kpc*pix_kpc
    print ('pixel size: %lf arcsec or %lf kpc' % (pix_arcsec,pix_kpc))

    f = h5py.File(sim_file,'r')

    # load star particles data
    stars_form_a = f['PartType4']['GFM_StellarFormationTime'][:]
    stars_init_mass = f['PartType4']['GFM_InitialMass'][:] * 1e+10 / cosmo_h
    stars_mass = f['PartType4']['Masses'][:] * 1e+10 / cosmo_h
    stars_zsol = f['PartType4']['GFM_Metallicity'][:] / PRIMORDIAL_Z_SUN_VALUE
    stars_coords = f['PartType4']['Coordinates'][:] * snap_a / cosmo_h  # in kpc
    stars_vel = f['PartType4']['Velocities'][:] * np.sqrt(snap_a)  # peculiar velocity in km/s

    stars_form_z = (1.0/stars_form_a) - 1.0
    stars_form_age_univ = interp_age_univ_from_z(stars_form_z, cosmo)
    stars_age = snap_univ_age - stars_form_age_univ                 # age in Gyr

    idx = np.where((stars_form_a>0) & (stars_age>=0))[0]
    stars_init_mass = stars_init_mass[idx]
    stars_mass = stars_mass[idx]
    stars_zsol = stars_zsol[idx]
    stars_age = stars_age[idx]
    stars_coords = stars_coords[idx,:]
    stars_vel = stars_vel[idx,:]

    # load gas particles data
    gas_mass = f['PartType0']['Masses'][:] * 1e+10 / cosmo_h
    gas_zsol = f['PartType0']['GFM_Metallicity'][:] / PRIMORDIAL_Z_SUN_VALUE
    gas_coords = f['PartType0']['Coordinates'][:] * snap_a / cosmo_h           # in kpc
    gas_sfr_inst = f['PartType0']['StarFormationRate'][:]   # in Msun/yr
    gas_vel = f['PartType0']['Velocities'][:] * np.sqrt(snap_a)   ## peculiar velocity in km/s
    u = f['PartType0']['InternalEnergy'][:]
    Xe = f['PartType0']['ElectronAbundance'][:]
    gas_mass_H = gas_mass*XH
    gamma = 5.0/3.0
    KB = 1.3807e-16
    mp = 1.6726e-24
    mu = (4*mp)/(1 + (3*XH) + (4*XH*Xe))
    gas_log_temp = np.log10((gamma-1.0)*(u/KB)*mu*1e+10)

    f.close()

    if dim_kpc is None:
        dim_kpc = determine_image_size(stars_coords, stars_mass, pix_kpc, (initdim_kpc, initdim_kpc), 
                                       polar_angle_deg, azimuth_angle_deg, gas_coords, gas_mass, 
                                       mass_percentage=initdim_mass_fraction, max_img_dim=initdim_kpc)

    output_dimension = (dim_kpc, dim_kpc)
    star_particle_membership, star_mass_density_map, central_pixel_coords, grid_info, gas_particle_membership, gas_mass_density_map, stars_vel_los_proj, gas_vel_los_proj = get_2d_density_projection_no_los_binning(
                                                                                                                                                stars_coords, 
                                                                                                                                                stars_mass, 
                                                                                                                                                pix_kpc, 
                                                                                                                                                output_dimension, 
                                                                                                                                                polar_angle_deg=polar_angle_deg, 
                                                                                                                                                azimuth_angle_deg=azimuth_angle_deg, 
                                                                                                                                                gas_coords=gas_coords, 
                                                                                                                                                gas_masses=gas_mass,
                                                                                                                                                star_vels=stars_vel,
                                                                                                                                                gas_vels=gas_vel)
    
    dimx, dimy = grid_info['num_pixels_x'], grid_info['num_pixels_y']
    print ('Cutout size: %d x %d pix or %d x %d kpc' % (dimx,dimy,dim_kpc,dim_kpc))

    idxg_global = np.where((gas_sfr_inst>0.0) | (gas_log_temp<3.9))[0]
    if np.nansum(gas_mass[idxg_global]) > 0:
        temp_mw_gas_zsol = np.nansum(gas_mass[idxg_global]*gas_zsol[idxg_global])/np.nansum(gas_mass[idxg_global])
    else:
        temp_mw_gas_zsol = 0.0
    nH = np.nansum(gas_mass_H[idxg_global])*1.247914e+14/dim_kpc/dim_kpc

    # --- Moved the dust normalization loading here ---
    norm_dust_z = None
    norm_dust_tau = None
    if isinstance(scale_dust_redshift, str):
        if scale_dust_redshift == "Vogelsberger20":
            data_file_name = "Vogelsberger20_scale_dust.txt"
            try:
                data_path = str(importlib.resources.files('galsyn.data').joinpath(data_file_name))
                data = np.loadtxt(data_path)
                norm_dust_z = data[:,0]
                norm_dust_tau = data[:,1]
            except Exception as e:
                print(f"Error loading dust normalization data from {data_file_name}: {e}")
                sys.exit(1)
        else:
            print(f"Error: Unknown string option for scale_dust_redshift: {scale_dust_redshift}")
            sys.exit(1)
    elif isinstance(scale_dust_redshift, dict):
        norm_dust_z = np.asarray(scale_dust_redshift["z"])
        norm_dust_tau = np.asarray(scale_dust_redshift["tau_dust"])
    else:
        print("Error: Invalid scale_dust_redshift type passed to generate_images.")
        sys.exit(1)

    # Calculate scale_dust_tau here in the main process
    scale_dust_tau = tau_dust_given_z(snap_z, norm_dust_z, norm_dust_tau)
    # --- End of dust normalization loading and calculation ---

    mean_tauV_res = scale_dust_tau*temp_mw_gas_zsol*nH/2.1e+21 

    global mean_AV_unres
    if np.isnan(mean_tauV_res)==True or np.isinf(mean_tauV_res)==True:
        mean_tauV_res, mean_AV_unres = 0.0, 0.0
    else:
        mean_AV_unres = -2.5*np.log10(np.exp(-2.0*mean_tauV_res))
    print ('mean_tauV_res=%lf mean_AV_unres=%lf' % (mean_tauV_res,mean_AV_unres))

    nbands = len(filters)

    # --- Define the fixed observed-frame wavelength grid for output spectra ---
    fixed_global_output_obs_wave = np.array([])
    if output_pixel_spectra:
        obs_min_boundary = rest_wave_min * (1.0 + snap_z)
        obs_max_boundary = rest_wave_max * (1.0 + snap_z)
        # Create a linear wavelength grid with 5 Angstrom increment
        fixed_global_output_obs_wave = np.arange(obs_min_boundary, obs_max_boundary + 5.0, 5.0)
        if fixed_global_output_obs_wave.size == 0:
            print(f"Warning: Calculated output observed wavelength grid is empty for rest-frame range "
                  f"[{rest_wave_min}-{rest_wave_max}] Angstroms at z={snap_z}. "
                  f"Observed range: [{obs_min_boundary:.1f}-{obs_max_boundary:.1f}] Angstroms. "
                  f"Output spectra will be empty for all pixels.")

    num_obs_wave_points = fixed_global_output_obs_wave.size if output_pixel_spectra else 0

    map_mw_age = np.zeros((dimy,dimx))
    map_stars_mw_zsol = np.zeros((dimy,dimx))
    map_stars_mass = np.zeros((dimy,dimx))
    map_sfr_100 = np.zeros((dimy,dimx))
    map_sfr_30 = np.zeros((dimy,dimx))
    map_sfr_10 = np.zeros((dimy,dimx))
    map_gas_mw_zsol = np.zeros((dimy,dimx))
    map_gas_mass = np.zeros((dimy,dimx))
    map_sfr_inst = np.zeros((dimy,dimx))
    map_dust_mean_tauV = np.zeros((dimy,dimx))
    map_dust_mean_AV = np.zeros((dimy,dimx))
    map_flux = np.zeros((dimy,dimx,nbands))
    map_flux_dust = np.zeros((dimy,dimx,nbands))
    
    # Initialize spectra maps directly with the fixed size
    map_spectra_nodust = np.zeros((dimy, dimx, num_obs_wave_points)) if output_pixel_spectra else None
    map_spectra_dust = np.zeros((dimy, dimx, num_obs_wave_points)) if output_pixel_spectra else None

    # New maps for light-weighted age and metallicity
    map_lw_age_nodust = np.full((dimy, dimx), np.nan, dtype=np.float32)
    map_lw_age_dust = np.full((dimy, dimx), np.nan, dtype=np.float32)
    map_lw_zsol_nodust = np.full((dimy, dimx), np.nan, dtype=np.float32)
    map_lw_zsol_dust = np.full((dimy, dimx), np.nan, dtype=np.float32)

    # New maps for velocity
    map_stars_mw_vel_los = np.full((dimy, dimx), np.nan, dtype=np.float32)
    map_gas_mw_vel_los = np.full((dimy, dimx), np.nan, dtype=np.float32)
    map_stars_vel_disp_los = np.full((dimy, dimx), np.nan, dtype=np.float32)
    map_gas_vel_disp_los = np.full((dimy, dimx), np.nan, dtype=np.float32)
    map_lw_vel_los_nodust = np.full((dimy, dimx), np.nan, dtype=np.float32)
    map_lw_vel_los_dust = np.full((dimy, dimx), np.nan, dtype=np.float32)

    tasks = []
    for ii in range(dimy):
        for jj in range(dimx):
            num_stars_in_pixel = len(star_particle_membership[ii][jj])
            num_gas_in_pixel = len(gas_particle_membership[ii][jj]) if gas_particle_membership is not None else 0
            complexity = num_stars_in_pixel + num_gas_in_pixel
            tasks.append({'coords': (ii, jj), 'complexity': complexity, 
                          'star_part_mem': star_particle_membership[ii][jj],
                          'gas_part_mem': gas_particle_membership[ii][jj] if gas_particle_membership is not None else []})

    tasks.sort(key=lambda x: x['complexity'], reverse=True)

    processed_tasks_args = []
    for task in tasks:
        ii, jj = task['coords']
        processed_tasks_args.append((ii, jj, task['star_part_mem'], task['gas_part_mem'],
                                     stars_mass, stars_age, stars_zsol, stars_init_mass, stars_vel_los_proj,
                                     gas_mass, gas_sfr_inst, gas_zsol, gas_log_temp, gas_mass_H, gas_vel_los_proj))

    num_cores = n_jobs
    if num_cores == -1:
        num_cores = multiprocessing.cpu_count()

    print(f"\nStarting parallel pixel processing on {num_cores} cores...")

    with tqdm_joblib(total=len(processed_tasks_args), desc="Processing pixels") as progress_bar:
        results = Parallel(n_jobs=num_cores, verbose=0, initializer=init_worker,
                           initargs=(ssp_code, snap_z, pix_area_kpc2, mean_AV_unres,
                                     filters, filter_transmission_path,
                                     gas_logu, # Removed add_neb_emission_val
                                     igm_type, dust_index_bc, # Removed add_igm_absorption_val
                                     dust_index, t_esc, scale_dust_tau,
                                     cosmo_str, cosmo_h, XH, 
                                     dust_law, bump_amp, relation_AVslope, 
                                     salim_a0, salim_a1, salim_a2, salim_a3, salim_RV, salim_B,
                                     use_precomputed_ssp, ssp_filepath, ssp_interpolation_method, 
                                     output_pixel_spectra, fixed_global_output_obs_wave))( # Pass the fixed grid
            delayed(_process_pixel_data)(*task_args) for task_args in processed_tasks_args
        )
    print("\nFinished parallel pixel processing.")

    for k, pixel_result_tuple in enumerate(results):
        ii, jj, pixel_data = pixel_result_tuple
        
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
            # Directly assign as map_spectra_nodust is pre-allocated with correct size
            map_spectra_nodust[original_ii][original_jj] = pixel_data['obs_spectra_nodust_igm']
            map_spectra_dust[original_ii][original_jj] = pixel_data['obs_spectra_dust_igm']

        # Populate new light-weighted maps
        map_lw_age_nodust[original_ii][original_jj] = pixel_data['map_lw_age_nodust']
        map_lw_age_dust[original_ii][original_jj] = pixel_data['map_lw_age_dust']
        map_lw_zsol_nodust[original_ii][original_jj] = pixel_data['map_lw_zsol_nodust']
        map_lw_zsol_dust[original_ii][original_jj] = pixel_data['map_lw_zsol_dust']

        # Populate new velocity maps
        map_stars_mw_vel_los[original_ii][original_jj] = pixel_data['map_stars_mw_vel_los']
        map_gas_mw_vel_los[original_ii][original_jj] = pixel_data['map_gas_mw_vel_los']
        map_stars_vel_disp_los[original_ii][original_jj] = pixel_data['map_stars_vel_disp_los']
        map_gas_vel_disp_los[original_ii][original_jj] = pixel_data['map_gas_vel_disp_los']
        map_lw_vel_los_nodust[original_ii][original_jj] = pixel_data['map_lw_vel_los_nodust']
        map_lw_vel_los_dust[original_ii][original_jj] = pixel_data['map_lw_vel_los_dust']


    print("All calculations complete. Maps populated.")

    for i_band in range(len(filters)):
        map_flux[:,:,i_band] = convert_flux_map(map_flux[:,:,i_band], filter_wave_pivot_data_global[filters[i_band]], to_unit=flux_unit, pixel_scale_arcsec=pix_arcsec)
        map_flux_dust[:,:,i_band] = convert_flux_map(map_flux_dust[:,:,i_band], filter_wave_pivot_data_global[filters[i_band]], to_unit=flux_unit, pixel_scale_arcsec=pix_arcsec)

    if flux_unit == 'erg/s/cm2/A':
        flux_scale = 1e-20
    else:
        flux_scale = 1.0
    
    map_flux = map_flux/flux_scale
    map_flux_dust = map_flux_dust/flux_scale

    if name_out_img is not None:
        try:
            hdul = fits.HDUList()

            if map_flux.shape[2] > 0:
                primary_data = map_flux[:, :, 0]
                prihdr = fits.Header()
                prihdr['COMMENT'] = 'Primary Image: First band (no dust)'
                prihdr['CRPIX1'] = dimx / 2.0 + 0.5
                prihdr['CRPIX2'] = dimy / 2.0 + 0.5
                prihdr['CDELT1'] = pix_kpc
                prihdr['CDELT2'] = pix_kpc
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
                prihdr['SSP_CODE'] = ssp_code

                primary_hdu = fits.PrimaryHDU(data=primary_data, header=prihdr)
                hdul.append(primary_hdu)

                for i_band in range(len(filters)):
                    ext_hdr = fits.Header()
                    ext_hdr['EXTNAME'] = 'NODUST_'+filters[i_band].upper()
                    ext_hdr['FILTER'] = filters[i_band]
                    ext_hdr['COMMENT'] = f'Flux for filter: {filters[i_band]}'
                    ext_hdu = fits.ImageHDU(data=map_flux[:, :, i_band], header=ext_hdr)
                    hdul.append(ext_hdu)

                for i_band in range(len(filters)):
                    ext_hdr = fits.Header()
                    ext_hdr['EXTNAME'] = 'DUST_'+filters[i_band].upper()
                    ext_hdr['FILTER'] = filters[i_band]
                    ext_hdr['COMMENT'] = f'Flux (with dust) for filter: {filters[i_band]}'
                    ext_hdu = fits.ImageHDU(data=map_flux_dust[:, :, i_band], header=ext_hdr)
                    hdul.append(ext_hdu)
            else:
                hdul.append(fits.PrimaryHDU())

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
                'LW_AGE_NODUST': map_lw_age_nodust,
                'LW_AGE_DUST': map_lw_age_dust,
                'LW_ZSOL_NODUST': map_lw_zsol_nodust,
                'LW_ZSOL_DUST': map_lw_zsol_dust,
                'STARS_MW_VEL_LOS': map_stars_mw_vel_los,
                'GAS_MW_VEL_LOS': map_gas_mw_vel_los,
                'STARS_VEL_DISP_LOS': map_stars_vel_disp_los,
                'GAS_VEL_DISP_LOS': map_gas_vel_disp_los,
                'LW_VEL_LOS_NODUST': map_lw_vel_los_nodust,
                'LW_VEL_LOS_DUST': map_lw_vel_los_dust
            }

            for map_name, data_array in map_data_to_save.items():
                if data_array is not None:
                    ext_hdr = fits.Header()
                    ext_hdr['EXTNAME'] = map_name
                    ext_hdr['COMMENT'] = f'Map of {map_name.replace("_", " ").title()}'
                    if 'AGE' in map_name:
                        ext_hdr['BUNIT'] = 'Gyr'
                    elif 'ZSOL' in map_name:
                        ext_hdr['BUNIT'] = 'Z/Zsun'
                    elif 'VEL' in map_name:
                        ext_hdr['BUNIT'] = 'km/s'
                    hdul.append(fits.ImageHDU(data=data_array, header=ext_hdr))

            if output_pixel_spectra:
                # Transpose the data cube from (dim_y, dim_x, wavelength) to (wavelength, dim_y, dim_x)
                transposed_map_spectra_nodust = map_spectra_nodust.transpose((2, 0, 1))
                transposed_map_spectra_dust = map_spectra_dust.transpose((2, 0, 1))

                ext_hdr_nodust_spec = fits.Header()
                ext_hdr_nodust_spec['EXTNAME'] = 'OBS_SPEC_NODUST'
                ext_hdr_nodust_spec['COMMENT'] = 'Observed-frame spectra (no dust attenuation)'
                
                # Update CRPIX, CDELT, CUNIT to reflect new axis order
                ext_hdr_nodust_spec['CRPIX1'] = 1.0 # Wavelength axis becomes the first axis
                ext_hdr_nodust_spec['CRVAL1'] = fixed_global_output_obs_wave[0] if fixed_global_output_obs_wave.size > 0 else 0.0
                ext_hdr_nodust_spec['CDELT1'] = (fixed_global_output_obs_wave[1] - fixed_global_output_obs_wave[0]) if fixed_global_output_obs_wave.size > 1 else 0.0
                ext_hdr_nodust_spec['CUNIT1'] = 'Angstrom'

                ext_hdr_nodust_spec['CRPIX2'] = dimy / 2.0 + 0.5 # dim_y becomes the second axis
                ext_hdr_nodust_spec['CDELT2'] = pix_kpc
                ext_hdr_nodust_spec['CUNIT2'] = 'kpc'

                ext_hdr_nodust_spec['CRPIX3'] = dimx / 2.0 + 0.5 # dim_x becomes the third axis
                ext_hdr_nodust_spec['CDELT3'] = pix_kpc
                ext_hdr_nodust_spec['CUNIT3'] = 'kpc'

                ext_hdr_nodust_spec['BUNIT'] = 'erg/s/cm2/Angstrom'
                hdul.append(fits.ImageHDU(data=transposed_map_spectra_nodust, header=ext_hdr_nodust_spec))

                ext_hdr_dust_spec = fits.Header()
                ext_hdr_dust_spec['EXTNAME'] = 'OBS_SPEC_DUST'
                ext_hdr_dust_spec['COMMENT'] = 'Observed-frame spectra (with dust attenuation)'
                
                # Update CRPIX, CDELT, CUNIT to reflect new axis order
                ext_hdr_dust_spec['CRPIX1'] = 1.0 # Wavelength axis becomes the first axis
                ext_hdr_dust_spec['CRVAL1'] = fixed_global_output_obs_wave[0] if fixed_global_output_obs_wave.size > 0 else 0.0
                ext_hdr_dust_spec['CDELT1'] = (fixed_global_output_obs_wave[1] - fixed_global_output_obs_wave[0]) if fixed_global_output_obs_wave.size > 1 else 0.0
                ext_hdr_dust_spec['CUNIT1'] = 'Angstrom'

                ext_hdr_dust_spec['CRPIX2'] = dimy / 2.0 + 0.5 # dim_y becomes the second axis
                ext_hdr_dust_spec['CDELT2'] = pix_kpc
                ext_hdr_dust_spec['CUNIT2'] = 'kpc'

                ext_hdr_dust_spec['CRPIX3'] = dimx / 2.0 + 0.5 # dim_x becomes the third axis
                ext_hdr_dust_spec['CDELT3'] = pix_kpc
                ext_hdr_dust_spec['CUNIT3'] = 'kpc'

                ext_hdr_dust_spec['BUNIT'] = 'erg/s/cm2/Angstrom'
                hdul.append(fits.ImageHDU(data=transposed_map_spectra_dust, header=ext_hdr_dust_spec))
                
                # --- Add wavelength array as a binary table extension ---
                if fixed_global_output_obs_wave.size > 0:
                    col = fits.Column(name='WAVELENGTH', format='D', array=fixed_global_output_obs_wave)
                    cols = fits.ColDefs([col])
                    wavelength_hdu = fits.BinTableHDU.from_columns(cols, name='WAVELENGTH_GRID')
                    wavelength_hdu.header['BUNIT'] = 'Angstrom'
                    wavelength_hdu.header['COMMENT'] = 'Wavelength grid for OBS_SPEC_NODUST and OBS_SPEC_DUST'
                    hdul.append(wavelength_hdu)
                else:
                    print("Warning: No wavelength grid data to save for WAVELENGTH_GRID extension.")


            output_dir = os.path.dirname(name_out_img)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)

            hdul.writeto(name_out_img, overwrite=True, output_verify='fix')
            print(f"Galaxy image synthesis completed successfully and results saved to FITS file: {name_out_img}")

        except Exception as e:
            print(f"Error saving FITS file {name_out_img}: {e}")