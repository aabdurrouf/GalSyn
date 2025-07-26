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

import fsps

FSPS_Z_SUN = 0.019
PRIMORDIAL_Z_SUN_VALUE = 0.0127 # This is likely a constant for the simulation data's metallicity definition

# Global variables for SSP data (loaded once per worker if use_precomputed_ssp is True)
ssp_wave = None
ssp_ages_gyr = None
ssp_logzsol_grid = None
ssp_spectra_grid = None
ssp_stellar_mass_grid = None
ssp_code_z_sun = None # Will store FSPS_Z_SUN or BAGPIPES_Z_SUN

_global_ssp_spectra_interpolator = None
_global_ssp_stellar_mass_interpolator = None

# Global variables for FSPS instance (initialized once per worker if use_precomputed_ssp is False)
sp_instance = None

# Other global worker variables
igm_trans = None
snap_z = None
pix_area_kpc2 = None
mean_AV_unres = None
add_neb_emission = None
gas_logu = None
add_igm_absorption = None
igm_type = None
dust_index_bc = None
dust_index = None
t_esc = None
scale_dust_tau = None
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

output_pixel_spectra_flag = False
global_output_obs_wave = None

_worker_filters = None
_worker_filter_transmission = None
_worker_filter_wave_eff = None
_worker_imf_type = None
_worker_cosmo = None
_worker_imf_upper_limit = None
_worker_imf_lower_limit = None
_worker_imf1 = None
_worker_imf2 = None
_worker_imf3 = None
_worker_vdmc = None
_worker_mdave = None

def _load_filter_transmission_from_paths(filters_list, filter_transmission_path_dict):
    """
    Loads filter transmission data from specified text files and calculates
    pivot wavelengths.
    """
    filter_transmission_data = {}
    filter_wave_pivot_data = {}

    for f_name in filters_list:
        if f_name not in filter_transmission_path_dict:
            raise ValueError(f"Path for filter '{f_name}' not provided in filter_transmission_path.")
        
        file_path = filter_transmission_path_dict[f_name]
        try:
            # Load data from the text file (assuming 2 columns: wavelength, transmission)
            data = np.loadtxt(file_path)
            if data.shape[1] != 2:
                raise ValueError(f"Filter file {file_path} must have exactly two columns (wavelength, transmission).")
            
            wave = data[:, 0]
            trans = data[:, 1]

            filter_transmission_data[f_name] = {'wave': wave, 'trans': trans}

            # Calculate pivot wavelength
            # Lambda_p = sqrt(integral(lambda * T(lambda) d_lambda) / integral(T(lambda) / lambda d_lambda))
            # Numerator: integral(lambda * T(lambda) d_lambda)
            numerator = simpson(wave * trans, wave)
            # Denominator: integral(T(lambda) / lambda d_lambda)
            # Avoid division by zero if trans is zero or wave is zero at some points
            integrand_denominator = np.where(wave != 0, trans / wave, 0)
            denominator = simpson(integrand_denominator, wave)

            if denominator > 0:
                pivot_wavelength = np.sqrt(numerator / denominator)
            else:
                pivot_wavelength = np.nan # Or handle as an error/default value

            filter_wave_pivot_data[f_name] = pivot_wavelength # Storing pivot wavelength

        except Exception as e:
            print(f"Error loading or processing filter file {file_path} for filter {f_name}: {e}")
            raise

    return filter_transmission_data, filter_wave_pivot_data


def init_worker(ssp_code_val, snap_z_val, pix_area_kpc2_val, mean_AV_unres_val, 
                filters_list_val,
                filter_transmission_path_val,
                imf_type_val, imf_upper_limit_val, imf_lower_limit_val, 
                imf1_val, imf2_val, imf3_val, vdmc_val, mdave_val,     
                add_neb_emission_val, gas_logu_val, 
                add_igm_absorption_val, igm_type_val, dust_index_bc_val, 
                dust_index_val, t_esc_val, scale_dust_tau_val, 
                cosmo_str_val, cosmo_h_val, XH_val, 
                dust_law_val, bump_amp_val, dustindexAV_AV_val, dustindexAV_dust_index_val, salim_a0_val, 
                salim_a1_val, salim_a2_val, salim_a3_val, salim_RV_val, salim_B_val,
                use_precomputed_ssp_val, ssp_filepath_val=None, ssp_interpolation_method_val='nearest', 
                output_pixel_spectra_val=False, rest_wave_min_val=None, rest_wave_max_val=None): 
    
    global ssp_wave, ssp_ages_gyr, ssp_logzsol_grid, ssp_spectra_grid, ssp_stellar_mass_grid, ssp_code_z_sun
    global _global_ssp_spectra_interpolator, _global_ssp_stellar_mass_interpolator 
    global sp_instance, igm_trans, snap_z, pix_area_kpc2
    global mean_AV_unres, add_neb_emission, gas_logu, add_igm_absorption, igm_type
    global dust_index_bc, dust_index, t_esc, scale_dust_tau, dust_law, bump_amp, salim_a0, salim_a1, salim_a2, salim_a3 
    global salim_RV, salim_B, dust_Alambda_per_AV, func_interp_dust_index
    global use_precomputed_ssp, ssp_interpolation_method 
    global output_pixel_spectra_flag, global_output_obs_wave 
    global _worker_filters, _worker_filter_transmission, _worker_filter_wave_eff, _worker_imf_type, _worker_cosmo
    global _worker_imf_upper_limit, _worker_imf_lower_limit, _worker_imf1, _worker_imf2, _worker_imf3, _worker_vdmc, _worker_mdave

    snap_z = snap_z_val
    pix_area_kpc2 = pix_area_kpc2_val
    mean_AV_unres = mean_AV_unres_val
    
    _worker_imf_type = imf_type_val
    _worker_imf_upper_limit = imf_upper_limit_val
    _worker_imf_lower_limit = imf_lower_limit_val
    _worker_imf1 = imf1_val
    _worker_imf2 = imf2_val
    _worker_imf3 = imf3_val
    _worker_vdmc = vdmc_val
    _worker_mdave = mdave_val

    add_neb_emission = add_neb_emission_val
    gas_logu = gas_logu_val
    add_igm_absorption = add_igm_absorption_val
    igm_type = igm_type_val
    dust_index_bc = dust_index_bc_val
    t_esc = t_esc_val
    scale_dust_tau = scale_dust_tau_val
    
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
    
    if use_precomputed_ssp:
        if ssp_filepath_val is None:
            print("Error: ssp_filepath must be provided when use_precomputed_ssp is True.")
            sys.exit(1)
        try:
            with h5py.File(ssp_filepath_val, 'r') as f_ssp:
                # Check if the SSP file is for FSPS
                if f_ssp.attrs.get('code') != 'FSPS':
                    print(f"Error: SSP grid file '{ssp_filepath_val}' was generated with '{f_ssp.attrs.get('code', 'unknown')}' but 'FSPS' is selected for ssp_code.")
                    sys.exit(1)

                ssp_wave = f_ssp['wavelength'][:]
                ssp_ages_gyr = f_ssp['ages_gyr'][:]
                ssp_logzsol_grid = f_ssp['logzsol'][:]
                ssp_spectra_grid = f_ssp['spectra'][:]
                ssp_stellar_mass_grid = f_ssp['stellar_mass'][:]
                ssp_code_z_sun = f_ssp.attrs['z_sun'] # This will be FSPS_Z_SUN

                # Consistency checks for FSPS-specific parameters
                if f_ssp.attrs['imf_type'] != _worker_imf_type:
                    print(f"Warning: IMF type mismatch! SSP grid was generated with {f_ssp.attrs['imf_type']}, but current setting is {_worker_imf_type}.")
                if f_ssp.attrs['add_neb_emission'] != add_neb_emission:
                    print(f"Warning: Nebular emission setting mismatch! SSP grid was generated with {f_ssp.attrs['add_neb_emission']}, but current setting is {add_neb_emission}.")
                if f_ssp.attrs['gas_logu'] != gas_logu:
                    print(f"Warning: Gas LogU setting mismatch! SSP grid was generated with {f_ssp.attrs['gas_logu']}, but current setting is {gas_logu}.")
                
                if 'imf_upper_limit' in f_ssp.attrs and f_ssp.attrs['imf_upper_limit'] != _worker_imf_upper_limit:
                    print(f"Warning: IMF upper limit mismatch! SSP grid was generated with {f_ssp.attrs['imf_upper_limit']}, but current setting is {_worker_imf_upper_limit}.")
                if 'imf_lower_limit' in f_ssp.attrs and f_ssp.attrs['imf_lower_limit'] != _worker_imf_lower_limit:
                    print(f"Warning: IMF lower limit mismatch! SSP grid was generated with {f_ssp.attrs['imf_lower_limit']}, but current setting is {_worker_imf_lower_limit}.")
                if 'imf1' in f_ssp.attrs and f_ssp.attrs['imf1'] != _worker_imf1:
                    print(f"Warning: IMF1 mismatch! SSP grid was generated with {f_ssp.attrs['imf1']}, but current setting is {_worker_imf1}.")
                if 'imf2' in f_ssp.attrs and f_ssp.attrs['imf2'] != _worker_imf2:
                    print(f"Warning: IMF2 mismatch! SSP grid was generated with {f_ssp.attrs['imf2']}, but current setting is {_worker_imf2}.")
                if 'imf3' in f_ssp.attrs and f_ssp.attrs['imf3'] != _worker_imf3:
                    print(f"Warning: IMF3 mismatch! SSP grid was generated with {f_ssp.attrs['imf3']}, but current setting is {_worker_imf3}.")
                if 'vdmc' in f_ssp.attrs and f_ssp.attrs['vdmc'] != _worker_vdmc:
                    print(f"Warning: VDMC mismatch! SSP grid was generated with {f_ssp.attrs['vdmc']}, but current setting is {_worker_vdmc}.")
                if 'mdave' in f_ssp.attrs and f_ssp.attrs['mdave'] != _worker_mdave:
                    print(f"Warning: MDAVE mismatch! SSP grid was generated with {f_ssp.attrs['mdave']}, but current setting is {_worker_mdave}.")

                if ssp_interpolation_method == 'linear': # Add 'cubic' if supported by RegularGridInterpolator
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
    else: # Generate on-the-fly using FSPS
        sp_instance = fsps.StellarPopulation(zcontinuous=1)
        sp_instance.params['imf_type'] = _worker_imf_type
        sp_instance.params['imf_upper_limit'] = _worker_imf_upper_limit
        sp_instance.params['imf_lower_limit'] = _worker_imf_lower_limit
        sp_instance.params['imf1'] = imf1_val # Corrected from _imf1_val
        sp_instance.params['imf2'] = imf2_val # Corrected from _imf2_val
        sp_instance.params['imf3'] = imf3_val # Corrected from _imf3_val
        sp_instance.params['vdmc'] = vdmc_val # Corrected from _vdmc_val
        sp_instance.params['mdave'] = mdave_val # Corrected from _mdave_val
        sp_instance.params["add_dust_emission"] = False
        sp_instance.params["add_neb_emission"] = add_neb_emission
        sp_instance.params['gas_logu'] = gas_logu
        sp_instance.params["fagn"] = 0
        sp_instance.params["sfh"] = 0
        sp_instance.params["dust1"] = 0.0
        sp_instance.params["dust2"] = 0.0
        ssp_wave, _ = sp_instance.get_spectrum(peraa=True, tage=1.0)
        ssp_code_z_sun = FSPS_Z_SUN # Set Z_sun for on-the-fly FSPS

    if output_pixel_spectra_flag:
        obs_ssp_wave_full = ssp_wave * (1.0 + snap_z)
        
        obs_min_boundary = rest_wave_min_val * (1.0 + snap_z)
        obs_max_boundary = rest_wave_max_val * (1.0 + snap_z)
        
        idx_valid_obs_wave = np.where((obs_ssp_wave_full >= obs_min_boundary) & 
                                      (obs_ssp_wave_full <= obs_max_boundary))
        
        global_output_obs_wave = obs_ssp_wave_full[idx_valid_obs_wave]

        if global_output_obs_wave.size == 0:
            print(f"Warning: No observed wavelengths from SSP grid fall within the requested rest-frame range "
                  f"[{rest_wave_min_val}-{rest_wave_max_val}] Angstroms at z={snap_z}. "
                  f"Observed range: [{obs_min_boundary:.1f}-{obs_max_boundary:.1f}] Angstroms. "
                  f"Output spectra will be empty for this pixel.")
            global_output_obs_wave = np.array([])

    if dust_law <= 1:
        global func_interp_dust_index
        dustindexAV_AV = dustindexAV_AV_val
        dustindexAV_dust_index = dustindexAV_dust_index_val
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
    if dust_law == 0:
        dust_index1 = func_interp_dust_index(dust_AV)
        bump_amp1 = bump_amp_from_dust_index(dust_index1)
        Alambda = modified_calzetti_dust_Alambda_per_AV(wave, dust_index=dust_index1, bump_amp=bump_amp1) * dust_AV

    elif dust_law == 1:
        dust_index1 = func_interp_dust_index(dust_AV)
        Alambda = modified_calzetti_dust_Alambda_per_AV(wave, dust_index=dust_index1, bump_amp=bump_amp) * dust_AV

    return Alambda


def _process_pixel_data(ii, jj, star_particle_membership_list, gas_particle_membership_list, 
                        stars_mass, stars_age, stars_zsol, stars_init_mass, 
                        gas_mass, gas_sfr_inst, gas_zsol, gas_log_temp, gas_mass_H):
    """
    ii=y jj=x
    Worker function to process calculations for a single pixel (ii, jj).
    This function will be executed in parallel.
    
    star_particle_membership_list: List of (original_particle_index, line_of_sight_distance) for THIS pixel.
    gas_particle_membership_list: List of (original_particle_index, line_of_sight_distance) for THIS pixel.
    """

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
        'obs_spectra_nodust_igm': np.zeros(len(global_output_obs_wave)) if output_pixel_spectra_flag and global_output_obs_wave.size > 0 else np.zeros(0),
        'obs_spectra_dust_igm': np.zeros(len(global_output_obs_wave)) if output_pixel_spectra_flag and global_output_obs_wave.size > 0 else np.zeros(0)
    }

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

    if len(star_ids) > 0:

        array_spec = []
        array_spec_dust = []
        array_AV = []
        array_tauV = []
        
        wave = ssp_wave

        for i_sid in range(len(star_ids)):
            star_id = star_ids[i_sid]

            if use_precomputed_ssp:
                # Convert simulation metallicity (Z/Z_primordial_solar) to log(Z/Z_ssp_code_solar)
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
            else: # On-the-fly FSPS
                logzsol = np.log10( (stars_zsol[star_id] * PRIMORDIAL_Z_SUN_VALUE) / FSPS_Z_SUN )
                sp_instance.params["logzsol"] = logzsol   
                sp_instance.params['gas_logz'] = logzsol

                _, spec = sp_instance.get_spectrum(peraa=True, tage=stars_age[star_id])
                ssp_mass_formed = sp_instance.stellar_mass

            idxg_front = np.where(gas_los_dist < star_los_dist[i_sid])[0]
            front_gas_ids = gas_ids[idxg_front]
            idxg1 = np.where((gas_sfr_inst[front_gas_ids] > 0.0) | (gas_log_temp[front_gas_ids] < 3.9))[0]
            cold_front_gas_ids = front_gas_ids[idxg1]

            if len(cold_front_gas_ids) > 0:
                temp_mw_gas_zsol = np.nansum(gas_mass[cold_front_gas_ids]*gas_zsol[cold_front_gas_ids])/np.nansum(gas_mass[cold_front_gas_ids])
                nH = np.nansum(gas_mass_H[cold_front_gas_ids])*1.247914e+14/pix_area_kpc2
                tauV = scale_dust_tau * temp_mw_gas_zsol * nH / 2.1e+21
                dust_AV = -2.5*np.log10((1.0 - np.exp(-1.0*tauV))/tauV)

                if np.isnan(dust_AV)==True or dust_AV==0.0:
                    spec_dust = spec
                else:
                    if dust_law <= 1:
                        Alambda = dust_reddening_diffuse_ism(dust_AV, wave, dust_law)
                    else:
                        Alambda = dust_Alambda_per_AV * dust_AV
                    
                    spec_dust = spec*np.power(10.0, -0.4*Alambda)
                    array_tauV.append(tauV)
                    array_AV.append(dust_AV)
            else:
                spec_dust = spec

            if stars_age[star_id] <= t_esc:
                Alambda = unresolved_dust_birth_cloud_Alambda_per_AV(wave, dust_index_bc=dust_index_bc) * mean_AV_unres
                spec_dust = spec_dust*np.power(10.0, -0.4*Alambda)
    
            norm = stars_mass[star_id] / ssp_mass_formed

            if len(np.asarray(spec_dust).shape) == 1:
                array_spec.append(spec*norm)
                array_spec_dust.append(spec_dust*norm)

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
                if global_output_obs_wave.size > 0:
                    interp_func_nodust = interp1d(spec_wave_obs, spec_flux_obs_igm, kind='linear', 
                                                  bounds_error=False, fill_value=0.0)
                    interp_func_dust = interp1d(spec_wave_obs, spec_flux_dust_obs_igm, kind='linear', 
                                                bounds_error=False, fill_value=0.0)
                    
                    pixel_results['obs_spectra_nodust_igm'] = interp_func_nodust(global_output_obs_wave)
                    pixel_results['obs_spectra_dust_igm'] = interp_func_dust(global_output_obs_wave)

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

    return ii, jj, pixel_results


def generate_images(sim_file, z, filters, filter_transmission_path, dim_kpc=None,
                    pix_arcsec=0.02, flux_unit='MJy/sr', polar_angle_deg=0, azimuth_angle_deg=0,
                    name_out_img=None, n_jobs=-1, ssp_code='FSPS', imf_type=1, imf_upper_limit=120.0, imf_lower_limit=0.08,
                    imf1=1.3, imf2=2.3, imf3=2.3, vdmc=0.08, mdave=0.5, add_neb_emission=1, gas_logu=-2.0, 
                    add_igm_absorption=1, igm_type=0, dust_index_bc=-0.7, dust_index=0.0, t_esc=0.01, 
                    norm_dust_z=[], norm_dust_tau=[], cosmo_str='Planck18', cosmo_h=0.6774, XH=0.76, 
                    dust_law=0, bump_amp=0.85, dustindexAV_AV=[], dustindexAV_dust_index=[], salim_a0=-4.30, 
                    salim_a1=2.71, salim_a2= -0.191, salim_a3=0.0121, salim_RV=3.15, salim_B=1.57, 
                    initdim_kpc=200, initdim_mass_fraction=0.99, use_precomputed_ssp=True, 
                    ssp_filepath=None, ssp_interpolation_method='nearest', 
                    output_pixel_spectra=False, rest_wave_min=1000.0, rest_wave_max=16000.0): 
    """
    Generates astrophysical images from HDF5 simulation data with parallelized pixel calculations.
    Allows choice between using pre-computed SSP spectra from an HDF5 file or
    generating them on-the-fly using FSPS.
    Optionally outputs observed-frame spectra (redshifted and IGM-transmitted) for each pixel.

    Parameters:
        sim_file (str): Path to the HDF5 simulation file.
        z (float): Redshift of the galaxy.
        filters (list): List of photometric filters.
        filter_transmission_path (dict): Dictionary of paths to text files containing the transmission function.
                                         Keys are filter names, values are file paths. Each text file has
                                         two columns: wavelength and transmission.
        dim_kpc (float, optional): Dimension of the image in kpc. If None, assigned automatically. Defaults to None.
        pix_arcsec (float, optional): Pixel size in arcseconds. Defaults to 0.02.
        flux_unit (string, optional): Desired flux unit for the generated images. Options are: 'MJy/sr', 'nJy', 'AB magnitude', or 'erg/s/cm2/A'. Default to 'MJy/sr'.
        polar_angle_deg (float, optional): Polar angle for projection. Defaults to 0.
        azimuth_angle_deg (float, optional): Azimuth angle for projection. Defaults to 0.
        name_out_img (str, optional): Output file name for images. Defaults to None.
        n_jobs (int, optional): Number of CPU cores to use for parallel processing. Defaults to -1 (all available).
        ssp_code (str, optional): The SSP code to use ('FSPS' or 'Bagpipes'). Defaults to 'FSPS'.
        imf_type (int, optional): IMF type for FSPS (must match SSP grid if pre-computed). Defaults to 1.
        imf_upper_limit (float, optional): The upper limit of the IMF, in solar masses. Only used if `use_precomputed_ssp` is False or for consistency check if `use_precomputed_ssp` is True. Defaults to 120.0.
        imf_lower_limit (float, optional): The lower limit of the IMF, in solar masses. Only used if `use_precomputed_ssp` is False or for consistency check if `use_precomputed_ssp` is True. Defaults to 0.08.
        imf1 (float, optional): Logarithmic slope of the IMF over the range. Only used if `imf_type=2`. Only used if `use_precomputed_ssp` is False or for consistency check if `use_precomputed_ssp` is True. Defaults to 1.3.
        imf2 (float, optional): Logarithmic slope of the IMF over the range. Only used if `imf_type=2`. Only used if `use_precomputed_ssp` is False or for consistency check if `use_precomputed_ssp` is True. Defaults to 2.3.
        imf3 (float, optional): Logarithmic slope of the IMF over the range. Only used if `imf_type=2`. Only used if `use_precomputed_ssp` is False or for consistency check if `use_precomputed_ssp` is True. Defaults to 2.3.
        vdmc (float, optional): IMF parameter defined in van Dokkum (2008). Only used if `imf_type=3`. Only used if `use_precomputed_ssp` is False or for consistency check if `use_precomputed_ssp` is True. Defaults to 0.08.
        mdave (float, optional): IMF parameter defined in Dave (2008). Only used if `imf_type=4`. Only used if `use_precomputed_ssp` is False or for consistency check if `use_precomputed_ssp` is True. Defaults to 0.5.
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
        initdim_mass_fraction (float, optional): Mass fraction to determine initial image dimension. Defaults to 0.99. 
        use_precomputed_ssp (bool, optional): If True, use pre-computed SSP spectra. If False, generate on-the-fly.
                                              Defaults to True.
        ssp_filepath (str, optional): Path to the pre-computed SSP spectra HDF5 file.
                                      Only used if `use_precomputed_ssp` is True. Defaults to "ssp_spectra.hdf5".
        ssp_interpolation_method (str, optional): Method for interpolating SSPs if `use_precomputed_ssp` is True.
                                                  Options: 'nearest', 'linear', 'cubic'. Defaults to 'nearest'.
        output_pixel_spectra (bool, optional): If True, output observed-frame spectra for each pixel. Defaults to False.
        rest_wave_min (float, optional): Minimum rest-frame wavelength for output spectra (Angstrom). Defaults to 1000.0.
        rest_wave_max (float, optional): Maximum rest-frame wavelength for output spectra (Angstrom). Defaults to 16000.0.
    """

    cosmo = define_cosmo(cosmo_str)

    print ('Processing '+sim_file)
    # --- Data Loading and Initial Calculations (Sequential) ---

    # Load filter transmission and pivot wavelengths once for the main process
    filter_transmission_data_global, filter_wave_pivot_data_global = _load_filter_transmission_from_paths(filters, filter_transmission_path)

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
    stars_zsol = f['PartType4']['GFM_Metallicity'][:]/PRIMORDIAL_Z_SUN_VALUE

    coords = f['PartType4']['Coordinates'][:]
    coords_x = coords[:,0]*snap_a/cosmo_h
    coords_y = coords[:,1]*snap_a/cosmo_h
    coords_z = coords[:,2]*snap_a/cosmo_h

    snap_univ_age = cosmo.age(snap_z).value
    stars_form_age_univ = interp_age_univ_from_z(stars_form_z, cosmo)
    stars_age = snap_univ_age - stars_form_age_univ
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

    gas_mass = f['PartType0']['Masses'][:]*1e+10/cosmo_h
    gas_zsol = f['PartType0']['GFM_Metallicity'][:]/PRIMORDIAL_Z_SUN_VALUE
    gas_coords = f['PartType0']['Coordinates'][:]
    gas_coords_x = gas_coords[:,0]*snap_a/cosmo_h
    gas_coords_y = gas_coords[:,1]*snap_a/cosmo_h
    gas_coords_z = gas_coords[:,2]*snap_a/cosmo_h
    gas_sfr_inst = f['PartType0']['StarFormationRate'][:]
    gas_mass_H = gas_mass*XH
    u = f['PartType0']['InternalEnergy'][:]
    Xe = f['PartType0']['ElectronAbundance'][:]
    gamma = 5.0/3.0
    KB = 1.3807e-16
    mp = 1.6726e-24
    mu = (4*mp)/(1+3*XH+4*XH*Xe)
    gas_log_temp = np.log10((gamma-1.0)*(u/KB)*mu*1e+10)

    f.close()

    star_coords = np.column_stack((stars_coords_x, stars_coords_y, stars_coords_z))
    gas_coords = np.column_stack((gas_coords_x, gas_coords_y, gas_coords_z))

    if dim_kpc is None:
        dim_kpc = determine_image_size(star_coords, stars_mass, pix_kpc, (initdim_kpc, initdim_kpc), 
                                       polar_angle_deg, azimuth_angle_deg, gas_coords, gas_mass, 
                                       mass_percentage=initdim_mass_fraction, max_img_dim=initdim_kpc)

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

    idxg_global = np.where((gas_sfr_inst>0.0) | (gas_log_temp<3.9))[0]
    if np.nansum(gas_mass[idxg_global]) > 0:
        temp_mw_gas_zsol = np.nansum(gas_mass[idxg_global]*gas_zsol[idxg_global])/np.nansum(gas_mass[idxg_global])
    else:
        temp_mw_gas_zsol = 0.0
    nH = np.nansum(gas_mass_H[idxg_global])*1.247914e+14/dim_kpc/dim_kpc

    scale_dust_tau = tau_dust_given_z(snap_z, norm_dust_z, norm_dust_tau)
    mean_tauV_res = scale_dust_tau*temp_mw_gas_zsol*nH/2.1e+21 

    global mean_AV_unres
    if np.isnan(mean_tauV_res)==True or np.isinf(mean_tauV_res)==True:
        mean_tauV_res, mean_AV_unres = 0.0, 0.0
    else:
        mean_AV_unres = -2.5*np.log10(np.exp(-2.0*mean_tauV_res))
    print ('mean_tauV_res=%lf mean_AV_unres=%lf' % (mean_tauV_res,mean_AV_unres))

    nbands = len(filters)

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
    num_obs_wave_points = len(global_output_obs_wave) if output_pixel_spectra else 0
    map_spectra_nodust = np.zeros((dimy, dimx, num_obs_wave_points)) if output_pixel_spectra else None
    map_spectra_dust = np.zeros((dimy, dimx, num_obs_wave_points)) if output_pixel_spectra else None

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
                                     stars_mass, stars_age, stars_zsol, stars_init_mass, 
                                     gas_mass, gas_sfr_inst, gas_zsol, gas_log_temp, gas_mass_H))

    num_cores = n_jobs
    if num_cores == -1:
        num_cores = multiprocessing.cpu_count()

    print(f"\nStarting parallel pixel processing on {num_cores} cores...")

    with tqdm_joblib(total=len(processed_tasks_args), desc="Processing pixels") as progress_bar:
        results = Parallel(n_jobs=num_cores, verbose=0, initializer=init_worker,
                           initargs=(ssp_code, snap_z, pix_area_kpc2, mean_AV_unres, # Pass ssp_code
                                     filters, filter_transmission_path,
                                     imf_type, imf_upper_limit, imf_lower_limit, 
                                     imf1, imf2, imf3, vdmc, mdave, add_neb_emission, 
                                     gas_logu, add_igm_absorption, igm_type, dust_index_bc, 
                                     dust_index, t_esc, scale_dust_tau, cosmo_str, cosmo_h, XH, 
                                     dust_law, bump_amp, list(dustindexAV_AV), list(dustindexAV_dust_index), 
                                     salim_a0, salim_a1, salim_a2, salim_a3, salim_RV, salim_B,
                                     use_precomputed_ssp, ssp_filepath, ssp_interpolation_method, 
                                     output_pixel_spectra, rest_wave_min, rest_wave_max))( 
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
            map_spectra_nodust[original_ii][original_jj] = pixel_data['obs_spectra_nodust_igm']
            map_spectra_dust[original_ii][original_jj] = pixel_data['obs_spectra_dust_igm']

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
                prihdr['SSP_CODE'] = ssp_code # Add SSP code to FITS header

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
            }

            for map_name, data_array in map_data_to_save.items():
                if data_array is not None:
                    ext_hdr = fits.Header()
                    ext_hdr['EXTNAME'] = map_name
                    ext_hdr['COMMENT'] = f'Map of {map_name.replace("_", " ").title()}'
                    hdul.append(fits.ImageHDU(data=data_array, header=ext_hdr))

            if output_pixel_spectra:
                ext_hdr_nodust_spec = fits.Header()
                ext_hdr_nodust_spec['EXTNAME'] = 'OBS_SPEC_NODUST' 
                ext_hdr_nodust_spec['COMMENT'] = 'Observed-frame spectra (no dust attenuation)' 
                ext_hdr_nodust_spec['CRPIX1'] = dimx / 2.0 + 0.5 
                ext_hdr_nodust_spec['CRPIX2'] = dimy / 2.0 + 0.5 
                ext_hdr_nodust_spec['CDELT1'] = pix_kpc 
                ext_hdr_nodust_spec['CDELT2'] = pix_kpc 
                ext_hdr_nodust_spec['CUNIT1'] = 'kpc'
                ext_hdr_nodust_spec['CUNIT2'] = 'kpc'
                ext_hdr_nodust_spec['CRPIX3'] = 1.0 
                ext_hdr_nodust_spec['CDELT3'] = (global_output_obs_wave[1] - global_output_obs_wave[0]) if global_output_obs_wave.size > 1 else 0.0 
                ext_hdr_nodust_spec['CRVAL3'] = global_output_obs_wave[0] if global_output_obs_wave.size > 0 else 0.0 
                ext_hdr_nodust_spec['CUNIT3'] = 'Angstrom'
                ext_hdr_nodust_spec['BUNIT'] = 'erg/s/cm2/Angstrom' 
                hdul.append(fits.ImageHDU(data=map_spectra_nodust, header=ext_hdr_nodust_spec))

                ext_hdr_dust_spec = fits.Header()
                ext_hdr_dust_spec['EXTNAME'] = 'OBS_SPEC_DUST' 
                ext_hdr_dust_spec['COMMENT'] = 'Observed-frame spectra (with dust attenuation)' 
                ext_hdr_dust_spec['CRPIX1'] = dimx / 2.0 + 0.5 
                ext_hdr_dust_spec['CRPIX2'] = dimy / 2.0 + 0.5 
                ext_hdr_dust_spec['CDELT1'] = pix_kpc 
                ext_hdr_dust_spec['CDELT2'] = pix_kpc 
                ext_hdr_dust_spec['CUNIT1'] = 'kpc'
                ext_hdr_dust_spec['CUNIT2'] = 'kpc'
                ext_hdr_dust_spec['CRPIX3'] = 1.0
                ext_hdr_dust_spec['CDELT3'] = (global_output_obs_wave[1] - global_output_obs_wave[0]) if global_output_obs_wave.size > 1 else 0.0
                ext_hdr_dust_spec['CRVAL3'] = global_output_obs_wave[0] if global_output_obs_wave.size > 0 else 0.0
                ext_hdr_dust_spec['CUNIT3'] = 'Angstrom'
                ext_hdr_dust_spec['BUNIT'] = 'erg/s/cm2/Angstrom'
                hdul.append(fits.ImageHDU(data=map_spectra_dust, header=ext_hdr_dust_spec))

            output_dir = os.path.dirname(name_out_img)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)

            hdul.writeto(name_out_img, overwrite=True, output_verify='fix')
            print(f"Galaxy image synthesis completed successfully and results saved to FITS file: {name_out_img}")

        except Exception as e:
            print(f"Error saving FITS file {name_out_img}: {e}")

